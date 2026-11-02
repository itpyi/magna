// Fused noise tables and the CPU reference of fused CUDA noise sampling.

PyObject* cuda_noise_metadata(PyObject*, PyObject* cap) {
    try {
        auto* bridge = cuda_experiment_handle(cap);
        if(!bridge) return nullptr;
        CompiledExperiment::ExclusiveUse use(*bridge->run);
        cuda_validate_run(*bridge);
        const auto& plan = *bridge->plan;
        const auto& tables = plan.noise;
        Object out(Py_BuildValue("{s:n,s:n}", "sites", Py_ssize_t(tables.sites),
                                "blocks", Py_ssize_t(tables.thresholds.size())));
        phase_set(out.p, "thresholds", cuda_output_bytes(tables.thresholds));
        phase_set(out.p, "certain", cuda_output_bytes(tables.certain));
        phase_set(out.p, "block_offsets", cuda_output_bytes(tables.block_offsets));
        phase_set(out.p, "block_sites", cuda_output_bytes(tables.block_sites));
        phase_set(out.p, "choice_offsets", cuda_output_bytes(tables.choice_offsets));
        phase_set(out.p, "choices", cuda_output_bytes(tables.choices));
        phase_set(out.p, "uniform", cuda_output_bytes(tables.uniform));
        phase_set(out.p, "conditional_thresholds", cuda_output_bytes(tables.conditional_thresholds));
        phase_set(out.p, "conditional_certain", cuda_output_bytes(tables.conditional_certain));
        phase_set(out.p, "has_importance_weights", PyBool_FromLong(plan.has_importance_weights));
        phase_set(out.p, "weighted_sites", cuda_output_bytes(plan.weighted_sites));
        phase_set(out.p, "likelihood_values", cuda_output_bytes(plan.likelihood_values));
        return Py_NewRef(out.p);
    } catch(...) { return execution_error(); }
}

PyObject* cuda_noise_validate(PyObject*, PyObject* cap) {
    try {
        auto* bridge = cuda_experiment_handle(cap);
        if(!bridge) return nullptr;
        CompiledExperiment::ExclusiveUse use(*bridge->run);
        cuda_validate_run(*bridge);
        Py_RETURN_NONE;
    } catch(...) { return execution_error(); }
}

uint64_t cuda_noise_uint64(PyObject* value) {
    if(!PyLong_CheckExact(value)) throw std::invalid_argument("CUDA noise counters must be uint64 integers");
    auto result = PyLong_AsUnsignedLongLong(value);
    if(PyErr_Occurred()) throw std::invalid_argument("CUDA noise counter outside uint64");
    return uint64_t(result);
}

PyObject* cuda_noise_cpu_sample(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *raw_shots, *raw_seed, *raw_offset;
        int record = 0, weighted = 0;
        if(!PyArg_ParseTuple(args, "OOOO|pp", &cap, &raw_shots, &raw_seed, &raw_offset, &record, &weighted)) return nullptr;
        auto* bridge = cuda_experiment_handle(cap);
        if(!bridge) return nullptr;
        auto shots = cuda_noise_uint64(raw_shots), seed = cuda_noise_uint64(raw_seed);
        auto offset = cuda_noise_uint64(raw_offset);
        if(shots > cuda_max_shots || (shots && offset > UINT64_MAX - (shots - 1)))
            throw std::invalid_argument("CUDA noise shot count or offset limit");
        CompiledExperiment::ExclusiveUse use(*bridge->run);
        cuda_validate_run(*bridge);
        const auto& plan = *bridge->plan;
        if(plan.has_importance_weights && !weighted)
            throw std::invalid_argument("importance draws require the weighted noise interface");
        size_t stride = 2 * plan.words;
        std::vector<uint64_t> frames(size_t(shots) * stride), offsets;
        std::vector<uint32_t> indices;
        std::vector<double> likelihoods;
        if(weighted) likelihoods.resize(size_t(shots));
        if(record) offsets.push_back(0);
        native_work([&]() {
            SignalCheck signals;
            for(uint64_t shot = 0; shot < shots; shot++) {
                size_t first_index = indices.size();
                auto count = cuda_reference_draw(bridge->run->noise, plan, seed, offset + shot,
                                                 frames.data() + size_t(shot) * stride, record ? &indices : nullptr);
                if(record) {
                    std::sort(indices.begin() + first_index, indices.end());
                    offsets.push_back(indices.size());
                }
                if(weighted) likelihoods[size_t(shot)] = plan.likelihood_values.at(count);
                signals();
            }
            return 0;
        });
        Object output(cuda_output_bytes(frames));
        if(weighted) {
            Object weights(cuda_output_bytes(likelihoods));
            if(!record) return PyTuple_Pack(2, output.p, weights.p);
            Object raw_offsets(cuda_output_bytes(offsets)), raw_indices(cuda_output_bytes(indices));
            return PyTuple_Pack(4, output.p, weights.p, raw_offsets.p, raw_indices.p);
        }
        if(!record) return Py_NewRef(output.p);
        Object raw_offsets(cuda_output_bytes(offsets)), raw_indices(cuda_output_bytes(indices));
        return PyTuple_Pack(3, output.p, raw_offsets.p, raw_indices.p);
    } catch(...) { return execution_error(); }
}
