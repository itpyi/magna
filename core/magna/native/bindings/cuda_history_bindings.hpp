// Conversion only: causal plans and replay live in runtime/cuda_history.hpp.
#pragma once
struct CudaHistoryBinding {
    std::shared_ptr<CompiledExperiment> run;
    std::shared_ptr<const CudaExperimentPlan> experiment;
    CudaHistoryPlan plan;
};
inline CudaHistoryBinding* cuda_history_handle(PyObject* cap) {
    return static_cast<CudaHistoryBinding*>(PyCapsule_GetPointer(cap, "magna-cuda-history-v1"));
}
inline void cuda_history_destructor(PyObject* cap) { delete cuda_history_handle(cap); }
inline PyObject* cuda_history_rows(const std::vector<Bits>& rows, size_t words) {
    std::vector<uint64_t> packed(rows.size() * words);
    for(size_t j = 0; j < rows.size(); j++) {
        if(rows[j].w.size() > words) throw std::logic_error("CUDA history row width mismatch");
        std::copy(rows[j].w.begin(), rows[j].w.end(), packed.begin() + j * words);
    }
    return cuda_output_bytes(packed);
}
inline PyObject* cuda_history_affines(const std::vector<CudaQueryAffine>& values, size_t words) {
    std::vector<Bits> rows;
    std::vector<uint8_t> constants;
    for(const auto& value : values) { rows.push_back(value.row); constants.push_back(uint8_t(value.constant)); }
    Object result(Py_BuildValue("{s:n}", "count", Py_ssize_t(values.size())));
    phase_set(result.p, "rows", cuda_history_rows(rows, words));
    phase_set(result.p, "constants", cuda_output_bytes(constants));
    return Py_NewRef(result.p);
}
inline PyObject* cuda_history_query_list(const std::vector<CudaQueryPlan>& plans) {
    Object result(PyTuple_New(plans.size()));
    for(size_t j = 0; j < plans.size(); j++) PyTuple_SET_ITEM(result.p, j, cuda_query_plan_output(plans[j]));
    return Py_NewRef(result.p);
}
inline PyObject* cuda_history_plan_output(const CudaHistoryPlan& p, const CudaExperimentPlan& experiment) {
    size_t words = 2 * experiment.words + p.auxiliary_words, open_words = (p.open_bits + 63) / 64;
    Object result(Py_BuildValue("{s:n,s:n,s:n,s:n,s:n,s:n,s:n,s:n}",
            "frame_words", Py_ssize_t(words), "auxiliary_bits", Py_ssize_t(p.auxiliary_bits),
            "auxiliary_words", Py_ssize_t(p.auxiliary_words), "sector_bits", Py_ssize_t(p.sector_bits),
            "sector_rank", Py_ssize_t(p.sector_rank), "observations", Py_ssize_t(p.observations),
            "controls", Py_ssize_t(p.controls), "open_bits", Py_ssize_t(p.open_bits)));
    phase_set(result.p, "sector_base", cuda_history_rows({p.sector_base}, (p.sector_bits + 63) / 64));
    phase_set(result.p, "normalization", Py_BuildValue("(dL)", p.normalization.first, (long long)p.normalization.second));
    phase_set(result.p, "sector_columns", cuda_history_rows(p.sector_columns, (p.sector_bits + 63) / 64));
    phase_set(result.p, "mother", cuda_query_plan_output(p.mother));
    phase_set(result.p, "open_base", cuda_history_affines(p.open_base, words));
    Object entries(PyTuple_New(p.entries.size()));
    for(size_t j = 0; j < p.entries.size(); j++) {
        const auto& entry = p.entries[j];
        size_t n = entry.reference.linear.size();
        Object value(Py_BuildValue("{s:n}", "n", Py_ssize_t(n)));
        phase_set(value.p, "shift", cuda_history_rows({entry.shift}, (p.mother.n + 63) / 64));
        phase_set(value.p, "change", cuda_history_rows(entry.change, (p.mother.n + 63) / 64));
        phase_set(value.p, "columns", cuda_history_rows(entry.columns, open_words));
        phase_set(value.p, "base", cuda_history_rows({entry.base}, open_words));
        std::vector<uint8_t> linear(entry.reference.linear.begin(), entry.reference.linear.end());
        phase_set(value.p, "linear", cuda_output_bytes(linear));
        phase_set(value.p, "upper", cuda_history_rows(entry.reference.upper, (n + 63) / 64));
        PyTuple_SET_ITEM(entries.p, j, Py_NewRef(value.p));
    }
    phase_set(result.p, "entries", Py_NewRef(entries.p));
    Object sites(PyTuple_New(p.sites.size()));
    for(size_t j = 0; j < p.sites.size(); j++) {
        const auto& site = p.sites[j];
        Object value(Py_BuildValue("{s:s,s:n,s:n,s:n,s:n,s:n,s:n,s:n}",
                "stage", site.stage.c_str(), "index", Py_ssize_t(site.index),
                "observation_first", Py_ssize_t(site.observation_first), "history_count", Py_ssize_t(site.history_count),
                "input_count", Py_ssize_t(site.inputs.size()), "output_count", Py_ssize_t(site.outputs.size()),
                "pending_count", Py_ssize_t(site.pending.size()), "event_count", Py_ssize_t(site.events.size())));
        phase_set(value.p, "inputs", cuda_history_affines(site.inputs, words));
        phase_set(value.p, "input_for_pending", cuda_output_bytes(site.input_for_pending));
        phase_set(value.p, "keep", cuda_history_rows(site.keep, open_words));
        phase_set(value.p, "offsets", cuda_history_affines(site.offsets, words));
        phase_set(value.p, "events", cuda_history_rows(site.events, open_words));
        phase_set(value.p, "outcomes", cuda_history_affines(site.outcomes, words));
        phase_set(value.p, "outputs", cuda_output_bytes(site.outputs));
        phase_set(value.p, "columns", cuda_output_bytes(site.columns));
        phase_set(value.p, "prefixes", cuda_history_query_list(site.prefixes));
        PyTuple_SET_ITEM(sites.p, j, Py_NewRef(value.p));
    }
    phase_set(result.p, "sites", Py_NewRef(sites.p));
    phase_set(result.p, "terminal", cuda_history_query_list(p.terminal));
    Object bases(PyTuple_New(p.query_bases.size()));
    for(size_t j = 0; j < p.query_bases.size(); j++)
        PyTuple_SET_ITEM(bases.p, j, cuda_query_plan_output(*p.query_bases[j]));
    phase_set(result.p, "query_bases", Py_NewRef(bases.p));
    return Py_NewRef(result.p);
}
PyObject* make_cuda_history(PyObject*, PyObject* cap) {
    try {
        auto* bridge = cuda_experiment_handle(cap);
        if(!bridge) return nullptr;
        CompiledExperiment::ExclusiveUse use(*bridge->run);
        cuda_validate_run(*bridge);
        auto owner = std::make_unique<CudaHistoryBinding>();
        owner->run = bridge->run;
        owner->experiment = bridge->plan;
        native_work([&]() { owner->plan = build_cuda_history(*owner->run, *owner->experiment); return 0; });
        Object metadata(cuda_history_plan_output(owner->plan, *owner->experiment));
        Object handle(PyCapsule_New(owner.get(), "magna-cuda-history-v1", cuda_history_destructor));
        owner.release();
        return PyTuple_Pack(2, handle.p, metadata.p);
    } catch(...) { return execution_error(); }
}
inline std::vector<std::vector<int>> cuda_history_bits(PyObject* input) {
    Seq sequence(input);
    std::vector<std::vector<int>> result;
    for(size_t j = 0; j < sequence.size(); j++) {
        auto row = cuda_read_bytes<uint8_t>(sequence[j]);
        for(auto bit : row) if(bit > 1) throw std::invalid_argument("CUDA history requires binary bytes");
        result.emplace_back(row.begin(), row.end());
    }
    return result;
}
inline PyObject* cuda_history_inputs_output(const std::vector<std::vector<int>>& values) {
    Object result(PyTuple_New(values.size()));
    for(size_t j = 0; j < values.size(); j++) {
        std::vector<uint8_t> row(values[j].begin(), values[j].end());
        PyTuple_SET_ITEM(result.p, j, cuda_output_bytes(row));
    }
    return Py_NewRef(result.p);
}
inline PyObject* cuda_history_contribution_output(const HistoryResult& value) {
    auto contribution = value.contribution();
    Object result(PyTuple_New(contribution.values.size()));
    for(size_t j = 0; j < contribution.values.size(); j++) {
        const auto& number = contribution.values[j].scientific;
        PyTuple_SET_ITEM(result.p, j, number ? Py_BuildValue("(dL)", number->first, (long long)number->second) : Py_NewRef(Py_None));
    }
    return Py_NewRef(result.p);
}
PyObject* cuda_history_replay(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *raw_inputs, *raw_outputs;
        unsigned long long seed, shot;
        if(!PyArg_ParseTuple(args, "OKKOO", &cap, &seed, &shot, &raw_inputs, &raw_outputs)) return nullptr;
        auto* owner = cuda_history_handle(cap);
        if(!owner) return nullptr;
        auto inputs = cuda_history_bits(raw_inputs), outputs = cuda_history_bits(raw_outputs);
        CompiledExperiment::ExclusiveUse use(*owner->run);
        cuda_check_current(*owner->run, *owner->experiment);
        auto result = native_work([&]() {
            return cuda_replay_history(*owner->run, *owner->experiment, owner->plan, seed, shot, outputs.size(), inputs, outputs);
        });
        if(result.result) return cuda_history_contribution_output(*result.result);
        std::vector<uint8_t> row(result.inputs.back().begin(), result.inputs.back().end());
        return cuda_output_bytes(row);
    } catch(...) { return execution_error(); }
}
PyObject* cuda_history_cpu_reference(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *raw_outputs;
        unsigned long long seed, shot;
        if(!PyArg_ParseTuple(args, "OKKO", &cap, &seed, &shot, &raw_outputs)) return nullptr;
        auto* owner = cuda_history_handle(cap);
        if(!owner) return nullptr;
        auto outputs = cuda_history_bits(raw_outputs);
        CompiledExperiment::ExclusiveUse use(*owner->run);
        cuda_check_current(*owner->run, *owner->experiment);
        auto result = native_work([&]() {
            return cuda_replay_history(*owner->run, *owner->experiment, owner->plan, seed, shot, outputs.size(), {}, outputs);
        });
        Object out(Py_BuildValue("{s:n}", "proposal_rank", Py_ssize_t(result.proposal_rank)));
        phase_set(out.p, "inputs", cuda_history_inputs_output(result.inputs));
        phase_set(out.p, "frame", cuda_output_bytes(result.frame));
        if(result.result) {
            phase_set(out.p, "contribution", cuda_history_contribution_output(*result.result));
            phase_set(out.p, "masses", cuda_output_masses({result.result->masses}));
            Object values(PyTuple_New(owner->plan.terminal.size()));
            for(size_t j = 0; j < owner->plan.terminal.size(); j++)
                PyTuple_SET_ITEM(values.p, j, output_scalar(cuda_query_cpu_value(owner->plan.terminal[j], result.frame.data())));
            phase_set(out.p, "queries", Py_NewRef(values.p));
        }
        return Py_NewRef(out.p);
    } catch(...) { return execution_error(); }
}
PyObject* cuda_history_support_reference(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *raw;
        Py_ssize_t site;
        if(!PyArg_ParseTuple(args, "OnO", &cap, &site, &raw)) return nullptr;
        auto* owner = cuda_history_handle(cap);
        if(!owner) return nullptr;
        auto frame = cuda_read_bytes<uint64_t>(raw);
        if(site < 0 || size_t(site) >= owner->plan.sites.size() || frame.size() != owner->plan.mother.frame_words)
            throw std::invalid_argument("CUDA history support shape mismatch");
        CompiledExperiment::ExclusiveUse use(*owner->run);
        cuda_check_current(*owner->run, *owner->experiment);
        auto result = native_work([&]() { return cuda_history_support(owner->plan, size_t(site), frame.data()); });
        size_t words = (result.width + 63) / 64;
        Object out(Py_BuildValue("{s:O,s:n,s:n}", "zero", result.zero ? Py_True : Py_False,
                "width", Py_ssize_t(result.width), "rank", Py_ssize_t(result.columns.size())));
        phase_set(out.p, "base", cuda_history_rows({result.base}, words));
        phase_set(out.p, "columns", cuda_history_rows(result.columns, words));
        return Py_NewRef(out.p);
    } catch(...) { return execution_error(); }
}
PyObject* cuda_history_prefix_cpu_evaluate(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *raw;
        Py_ssize_t site, prefix;
        if(!PyArg_ParseTuple(args, "OnnO", &cap, &site, &prefix, &raw)) return nullptr;
        auto* owner = cuda_history_handle(cap);
        if(!owner) return nullptr;
        auto frame = cuda_read_bytes<uint64_t>(raw);
        if(site < 0 || size_t(site) >= owner->plan.sites.size() || prefix < 0 ||
                size_t(prefix) >= owner->plan.sites[size_t(site)].prefixes.size() || frame.size() != owner->plan.mother.frame_words)
            throw std::invalid_argument("CUDA history prefix shape mismatch");
        CompiledExperiment::ExclusiveUse use(*owner->run);
        cuda_check_current(*owner->run, *owner->experiment);
        auto result = native_work([&]() {
            return cuda_query_cpu_value(owner->plan.sites[size_t(site)].prefixes[size_t(prefix)], frame.data());
        });
        return output_scalar(result);
    } catch(...) { return execution_error(); }
}
