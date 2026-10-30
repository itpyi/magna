// Python conversion of the CUDA sampler's exact query plans (runtime/cuda_plan.hpp).
#pragma once

struct CudaQueriesBinding {
    std::shared_ptr<CompiledExperiment> run;
    std::shared_ptr<const CudaExperimentPlan> experiment;
    std::vector<CudaQueryPlan> queries;
};

inline PyObject* cuda_query_plan_output(const CudaQueryPlan& p) {
    if(p.mode == 6)
        return Py_BuildValue("{s:i,s:n,s:n,s:n,s:n}", "mode", 6, "n", Py_ssize_t(p.n),
                "frame_words", Py_ssize_t(p.frame_words), "base_index", Py_ssize_t(p.base_index),
                "disabled_caps", Py_ssize_t(p.disabled_caps));
    Object out(Py_BuildValue("{s:n,s:n,s:i,s:i,s:i,s:i,s:n,s:n}", "n", Py_ssize_t(p.n),
            "frame_words", Py_ssize_t(p.frame_words), "mode", p.mode, "constant", p.constant,
            "power_sqrt2", p.power_sqrt2, "output_power", p.output_power,
            "phase_rank", Py_ssize_t(p.phase_rank), "feature_count", Py_ssize_t(p.features.size())));
    std::vector<uint32_t> offsets{0}, words;
    std::vector<uint64_t> masks;
    std::vector<uint8_t> constants;
    for(const auto& feature : p.features) {
        for(size_t w = 0; w < feature.row.w.size(); w++)
            if(feature.row.w[w]) { words.push_back(uint32_t(w)); masks.push_back(feature.row.w[w]); }
        offsets.push_back(uint32_t(words.size()));
        constants.push_back(uint8_t(feature.constant));
    }
    phase_set(out.p, "feature_offsets", cuda_output_bytes(offsets));
    phase_set(out.p, "feature_words", cuda_output_bytes(words));
    phase_set(out.p, "feature_masks", cuda_output_bytes(masks));
    phase_set(out.p, "feature_constants", cuda_output_bytes(constants));
    phase_set(out.p, "support", cuda_output_bytes(p.support));
    phase_set(out.p, "guards", cuda_output_bytes(p.guards));
    phase_set(out.p, "linear_shift", cuda_output_bytes(p.linear_shift));
    phase_set(out.p, "constant_linear", cuda_output_bytes(p.constant_linear));
    phase_set(out.p, "constant_pairs", cuda_output_bytes(p.constant_pairs));
    phase_set(out.p, "magic", cuda_output_bytes(p.magic));
    phase_set(out.p, "magic_masks", cuda_output_bytes(p.magic_masks));
    phase_set(out.p, "upper", cuda_output_bytes(p.upper));
    phase_set(out.p, "linear", cuda_output_bytes(p.linear));
    phase_set(out.p, "phase_table", cuda_output_bytes(p.phase_table));
    phase_set(out.p, "steps", cuda_output_bytes(p.steps));
    phase_set(out.p, "step_offsets", cuda_output_bytes(p.step_offsets));
    phase_set(out.p, "step_indices", cuda_output_bytes(p.step_indices));
    phase_set(out.p, "step_tags", cuda_output_bytes(p.step_tags));
    if(p.mode == 5) {
        Object components(PyTuple_New(p.components.size())), weights(PyTuple_New(p.weights.size()));
        for(size_t j = 0; j < p.components.size(); j++) {
            PyTuple_SET_ITEM(components.p, j, cuda_query_plan_output(p.components[j]));
            PyTuple_SET_ITEM(weights.p, j, output_scalar(p.weights[j]));
        }
        phase_set(out.p, "components", Py_NewRef(components.p));
        phase_set(out.p, "weights", Py_NewRef(weights.p));
    }
    return Py_NewRef(out.p);
}

inline CudaQueriesBinding* cuda_queries_handle(PyObject* cap) {
    return static_cast<CudaQueriesBinding*>(PyCapsule_GetPointer(cap, "magna-cuda-queries-v1"));
}
inline void cuda_queries_destructor(PyObject* cap) { delete cuda_queries_handle(cap); }

PyObject* make_cuda_queries(PyObject*, PyObject* cap) {
    try {
        auto* bridge = cuda_experiment_handle(cap);
        if(!bridge) return nullptr;
        CompiledExperiment::ExclusiveUse use(*bridge->run);
        cuda_validate_run(*bridge);
        auto owner = std::make_unique<CudaQueriesBinding>();
        owner->run = bridge->run;
        owner->experiment = bridge->plan;
        native_work([&]() {
            owner->queries = build_cuda_queries(*owner->run, *owner->experiment);
            return 0;
        });
        Object metadata(PyTuple_New(owner->queries.size()));
        for(size_t j = 0; j < owner->queries.size(); j++)
            PyTuple_SET_ITEM(metadata.p, j, cuda_query_plan_output(owner->queries[j]));
        Object handle(PyCapsule_New(owner.get(), "magna-cuda-queries-v1", cuda_queries_destructor));
        owner.release();
        return PyTuple_Pack(2, handle.p, metadata.p);
    } catch(...) { return execution_error(); }
}

PyObject* cuda_query_cpu_evaluate(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *raw;
        if(!PyArg_ParseTuple(args, "OO", &cap, &raw)) return nullptr;
        auto* owner = cuda_queries_handle(cap);
        if(!owner) return nullptr;
        auto frames = cuda_read_bytes<uint64_t>(raw);
        const auto& experiment = *owner->experiment;
        size_t stride = 2 * experiment.words, shots = frames.size() / stride, queries = owner->queries.size();
        if(frames.size() % stride || shots > cuda_max_shots) throw std::invalid_argument("CUDA query frame shape/shot limit");
        if(experiment.frame_bits % 64 || !experiment.frame_bits) {
            uint64_t valid = experiment.frame_bits ? (uint64_t(1) << (experiment.frame_bits % 64)) - 1 : 0;
            for(size_t first = 0; first < frames.size(); first += experiment.words)
                if(frames[first + experiment.words - 1] & ~valid) throw std::invalid_argument("CUDA query frame padding is nonzero");
        }
        CompiledExperiment::ExclusiveUse use(*owner->run);
        cuda_check_current(*owner->run, *owner->experiment);
        std::vector<int64_t> coefficients(shots * queries * 4);
        std::vector<int32_t> powers(shots * queries), status(shots * queries);
        native_work([&]() {
            SignalCheck signals;
            for(size_t shot = 0; shot < shots; shot++) {
                for(size_t query = 0; query < queries; query++) {
                    auto value = cuda_query_cpu_value(owner->queries[query], frames.data() + shot * stride);
                    size_t index = shot * queries + query;
                    powers[index] = int32_t(value.power);
                    for(size_t k = 0; k < 4; k++) {
                        const auto& c = value.c[k];
                        if(c.d.size() > 2 || (c.d.size() == 2 && c.d[1] >= (uint32_t(1) << 31)))
                            throw std::overflow_error("CUDA exact query coefficient exceeds int64 proof bound");
                        uint64_t v = c.d.empty() ? 0 : c.d[0];
                        if(c.d.size() == 2) v |= uint64_t(c.d[1]) << 32;
                        coefficients[4 * index + k] = c.neg ? -int64_t(v) : int64_t(v);
                    }
                }
                signals();
            }
            return 0;
        });
        Object c(cuda_output_bytes(coefficients)), p(cuda_output_bytes(powers)), s(cuda_output_bytes(status));
        return PyTuple_Pack(3, c.p, p.p, s.p);
    } catch(...) { return execution_error(); }
}
