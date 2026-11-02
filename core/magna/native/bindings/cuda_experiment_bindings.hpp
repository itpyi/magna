// Python conversion of the CUDA sampler's plans (runtime/cuda_plan.hpp) and
// the native references that the CUDA tests compare against.

struct CudaExperimentBinding {
    std::shared_ptr<CompiledExperiment> run;
    std::shared_ptr<const CudaExperimentPlan> plan;
};
CudaExperimentBinding* cuda_experiment_handle(PyObject* cap) {
    return static_cast<CudaExperimentBinding*>(PyCapsule_GetPointer(cap, "magna-cuda-experiment-v1"));
}
void cuda_experiment_destructor(PyObject* cap) { delete cuda_experiment_handle(cap); }
template <class T> PyObject* cuda_output_bytes(const std::vector<T>& values) {
    if(values.size() > size_t(PY_SSIZE_T_MAX) / sizeof(T)) throw std::length_error("CUDA byte buffer too large");
    return PyBytes_FromStringAndSize(reinterpret_cast<const char*>(values.data()), Py_ssize_t(values.size() * sizeof(T)));
}
template <class T> std::vector<T> cuda_read_bytes(PyObject* raw) {
    if(!PyBytes_CheckExact(raw)) throw std::invalid_argument("CUDA bridge requires immutable bytes buffers");
    auto bytes = size_t(PyBytes_GET_SIZE(raw));
    if(bytes % sizeof(T)) throw std::invalid_argument("CUDA byte buffer has an incomplete integer");
    std::vector<T> out(bytes / sizeof(T));
    if(bytes) std::memcpy(out.data(), PyBytes_AS_STRING(raw), bytes);
    return out;
}
void cuda_validate_run(const CudaExperimentBinding& bridge) { cuda_check_current(*bridge.run, *bridge.plan); }
PyObject* make_cuda_experiment(PyObject*, PyObject* cap) {
    try {
        const uint16_t endian = 1;
        if(*reinterpret_cast<const uint8_t*>(&endian) != 1)
            throw std::invalid_argument("CUDA experiment buffers require a little-endian host");
        auto* owner = execution_handle(cap);
        if(!owner) return nullptr;
        auto bridge = std::make_unique<CudaExperimentBinding>();
        bridge->run = owner->run;
        auto& run = *bridge->run;
        CompiledExperiment::ExclusiveUse use(run);
        native_work([&]() {
            bridge->plan = std::make_shared<const CudaExperimentPlan>(build_cuda_experiment(run));
            return 0;
        });
        const auto& plan = *bridge->plan;
        Object metadata(Py_BuildValue("{s:n,s:n,s:n,s:n,s:n}", "frame_bits", Py_ssize_t(plan.frame_bits),
                "words", Py_ssize_t(plan.words), "actions", Py_ssize_t(run.noise.actions.size()),
                "num_sites", Py_ssize_t(run.noise.categories.size()), "max_shots", Py_ssize_t(cuda_max_shots)));
        phase_set(metadata.p, "columns", cuda_output_bytes(plan.columns));
        phase_set(metadata.p, "base_frame", cuda_output_bytes(plan.base));
        phase_set(metadata.p, "target_normalization", output_scalar(run.target_normalization));
        phase_set(metadata.p, "has_importance_weights", PyBool_FromLong(plan.has_importance_weights));
        Object offsets(PyTuple_New(run.noise.offsets.size()));
        for(size_t j = 0; j < run.noise.offsets.size(); j++)
            PyTuple_SET_ITEM(offsets.p, j, PyLong_FromSize_t(run.noise.offsets[j]));
        phase_set(metadata.p, "offsets", Py_NewRef(offsets.p));
        Object handle(PyCapsule_New(bridge.get(), "magna-cuda-experiment-v1", cuda_experiment_destructor));
        bridge.release();
        return PyTuple_Pack(2, handle.p, metadata.p);
    } catch(...) { return execution_error(); }
}
struct CudaNoiseBatch {
    std::vector<uint64_t> offsets;
    std::vector<uint32_t> indices;
    size_t shots() const { return offsets.size() - 1; }
};
CudaNoiseBatch cuda_read_batch(const CudaExperimentBinding& bridge, PyObject* raw_offsets, PyObject* raw_indices) {
    CudaNoiseBatch batch{cuda_read_bytes<uint64_t>(raw_offsets), cuda_read_bytes<uint32_t>(raw_indices)};
    if(batch.offsets.empty() || batch.shots() > cuda_max_shots || batch.offsets.front() != 0 ||
            batch.offsets.back() != batch.indices.size())
        throw std::invalid_argument("invalid CUDA CSR offsets or shot limit");
    for(size_t shot = 0; shot < batch.shots(); shot++) {
        auto first = batch.offsets[shot], end = batch.offsets[shot + 1];
        if(first > end || end > batch.indices.size()) throw std::invalid_argument("CUDA CSR offsets are not monotone");
        size_t previous_site = 0;
        for(auto k = first; k < end; k++) {
            auto index = batch.indices[size_t(k)];
            if(index >= bridge.plan->action_sites.size()) throw std::invalid_argument("CUDA action index outside plan");
            auto site = bridge.plan->action_sites[index];
            if(k != first && site <= previous_site)
                throw std::invalid_argument("CUDA actions must be ordered and select at most one outcome per noise site");
            previous_site = site;
        }
    }
    return batch;
}
PyObject* cuda_draw_noise(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *raw_seeds;
        if(!PyArg_ParseTuple(args, "OO", &cap, &raw_seeds)) return nullptr;
        auto* bridge = cuda_experiment_handle(cap);
        if(!bridge) return nullptr;
        Seq input(raw_seeds);
        if(input.size() > cuda_max_shots) throw std::invalid_argument("CUDA noise batch exceeds shot limit");
        std::vector<uint64_t> seeds;
        seeds.reserve(input.size());
        for(size_t j = 0; j < input.size(); j++) {
            if(!PyLong_CheckExact(input[j])) throw std::invalid_argument("CUDA seeds must be uint64 integers");
            auto seed = PyLong_AsUnsignedLongLong(input[j]);
            if(PyErr_Occurred()) throw std::invalid_argument("CUDA seed outside uint64");
            seeds.push_back(uint64_t(seed));
        }
        CompiledExperiment::ExclusiveUse use(*bridge->run);
        cuda_validate_run(*bridge);
        CudaNoiseBatch batch;
        batch.offsets.push_back(0);
        native_work([&]() {
            SignalCheck signals;
            for(auto seed : seeds) {
                auto draw = draw_noise(bridge->run->noise, seed, true);
                for(auto index : draw.selected) batch.indices.push_back(uint32_t(index));
                batch.offsets.push_back(uint64_t(batch.indices.size()));
                signals();
            }
            return 0;
        });
        Object offsets(cuda_output_bytes(batch.offsets)), indices(cuda_output_bytes(batch.indices));
        return PyTuple_Pack(2, offsets.p, indices.p);
    } catch(...) { return execution_error(); }
}
PyObject* cuda_map_frames(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *raw_offsets, *raw_indices;
        if(!PyArg_ParseTuple(args, "OOO", &cap, &raw_offsets, &raw_indices)) return nullptr;
        auto* bridge = cuda_experiment_handle(cap);
        if(!bridge) return nullptr;
        auto batch = cuda_read_batch(*bridge, raw_offsets, raw_indices);
        CompiledExperiment::ExclusiveUse use(*bridge->run);
        cuda_validate_run(*bridge);
        size_t stride = 2 * bridge->plan->words;
        std::vector<uint64_t> output;
        if(batch.shots() > output.max_size() / stride) throw std::length_error("CUDA frame output too large");
        output.resize(batch.shots() * stride);
        native_work([&]() {
            SignalCheck signals;
            for(size_t shot = 0; shot < batch.shots(); shot++) {
                auto* row = output.data() + shot * stride;
                std::copy(bridge->plan->base.begin(), bridge->plan->base.end(), row);
                for(auto k = batch.offsets[shot]; k < batch.offsets[shot + 1]; k++) {
                    const auto* column = bridge->plan->columns.data() + size_t(batch.indices[size_t(k)]) * stride;
                    for(size_t w = 0; w < stride; w++) row[w] ^= column[w];
                }
                signals();
            }
            return 0;
        });
        return cuda_output_bytes(output);
    } catch(...) { return execution_error(); }
}
PyObject* cuda_output_masses(const std::vector<HistoryMasses>& results) {
    Object output(PyTuple_New(results.size()));
    for(size_t j = 0; j < results.size(); j++) {
        const auto& masses = results[j];
        Object history(output_scalar(masses.history)), acceptance(output_scalar(masses.acceptance));
        Object error(masses.error ? output_scalar(*masses.error) : Py_NewRef(Py_None));
        auto* row = PyTuple_Pack(3, history.p, acceptance.p, error.p);
        if(!row) throw std::bad_alloc();
        PyTuple_SET_ITEM(output.p, j, row);
    }
    return Py_NewRef(output.p);
}
PyObject* cuda_evaluate_frames(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *raw_frames;
        if(!PyArg_ParseTuple(args, "OO", &cap, &raw_frames)) return nullptr;
        auto* bridge = cuda_experiment_handle(cap);
        if(!bridge) return nullptr;
        auto frames = cuda_read_bytes<uint64_t>(raw_frames);
        auto stride = 2 * bridge->plan->words;
        if(frames.size() % stride || frames.size() / stride > cuda_max_shots)
            throw std::invalid_argument("CUDA frame buffer shape or shot limit");
        if(bridge->plan->frame_bits % 64 || !bridge->plan->frame_bits) {
            auto valid = bridge->plan->frame_bits ? (uint64_t(1) << (bridge->plan->frame_bits % 64)) - 1 : uint64_t(0);
            for(size_t j = 0; j < frames.size(); j += bridge->plan->words)
                if(frames[j + bridge->plan->words - 1] & ~valid)
                    throw std::invalid_argument("CUDA frame contains nonzero padding bits");
        }
        CompiledExperiment::ExclusiveUse use(*bridge->run);
        cuda_validate_run(*bridge);
        std::vector<HistoryMasses> results;
        results.reserve(frames.size() / stride);
        native_work([&]() {
            SignalCheck signals;
            for(size_t first = 0; first < frames.size(); first += stride) {
                Frame frame;
                frame.x.w.assign(frames.begin() + first, frames.begin() + first + bridge->plan->words);
                frame.z.w.assign(frames.begin() + first + bridge->plan->words, frames.begin() + first + stride);
                frame.x.trim();
                frame.z.trim();
                results.push_back(bridge->run->evaluate({}, {}, {}, {}, {}, {}, nullptr, nullptr, nullptr, &frame));
                signals();
            }
            return 0;
        });
        return cuda_output_masses(results);
    } catch(...) { return execution_error(); }
}
PyObject* cuda_evaluate_noise(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *raw_offsets, *raw_indices;
        if(!PyArg_ParseTuple(args, "OOO", &cap, &raw_offsets, &raw_indices)) return nullptr;
        auto* bridge = cuda_experiment_handle(cap);
        if(!bridge) return nullptr;
        auto batch = cuda_read_batch(*bridge, raw_offsets, raw_indices);
        CompiledExperiment::ExclusiveUse use(*bridge->run);
        cuda_validate_run(*bridge);
        std::vector<HistoryMasses> results;
        results.reserve(batch.shots());
        native_work([&]() {
            ExecutionHooks hooks;
            hooks.seed = []() -> uint64_t { throw std::logic_error("unexpected random draw in fixed-frame evaluation"); };
            SignalCheck signals;
            for(size_t shot = 0; shot < batch.shots(); shot++) {
                FileNoiseDraw selected;
                for(auto k = batch.offsets[shot]; k < batch.offsets[shot + 1]; k++)
                    selected.selected.push_back(batch.indices[size_t(k)]);
                auto noise = noise_actions(bridge->run->noise, selected);
                results.push_back(bridge->run->sample(hooks, std::move(noise)).masses);
                signals();
            }
            return 0;
        });
        return cuda_output_masses(results);
    } catch(...) { return execution_error(); }
}
