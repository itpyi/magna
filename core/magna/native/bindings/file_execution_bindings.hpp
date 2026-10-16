// Only explicit language callbacks and value adapters live in this layer.
struct CallbackGIL {
    PyGILState_STATE state = PyGILState_Ensure();
    ~CallbackGIL() { PyGILState_Release(state); }
};
// A Python exception belongs to its thread state. Fetch it before releasing a
// worker's GIL and restore it on the calling thread after every worker has joined.
struct PythonCallbackError : std::exception {
    struct Value {
        PyObject *type = nullptr, *value = nullptr, *traceback = nullptr;
        Value() { PyErr_Fetch(&type, &value, &traceback); }
        ~Value() {
            CallbackGIL gil;
            Py_XDECREF(type);
            Py_XDECREF(value);
            Py_XDECREF(traceback);
        }
    };
    std::shared_ptr<Value> captured = std::make_shared<Value>();
    void restore() const {
        PyErr_Restore(Py_XNewRef(captured->type), Py_XNewRef(captured->value), Py_XNewRef(captured->traceback));
    }
};
struct FileExecutionBinding {
    std::shared_ptr<CompiledExperiment> run;
    std::vector<std::unique_ptr<CompiledExperiment>> workers;
    ParallelFileExecutor parallel;
    PyObject* callbacks = nullptr;
    ~FileExecutionBinding() { Py_XDECREF(callbacks); }
};
FileExecutionBinding* execution_handle(PyObject* cap) {
    return static_cast<FileExecutionBinding*>(PyCapsule_GetPointer(cap, "magic-file-execution-v1"));
}
void execution_destructor(PyObject* cap) { delete execution_handle(cap); }
PyObject* execution_error() {
    try {
        throw;
    } catch(const PythonCallbackError& e) {
        e.restore();
        return nullptr;
    } catch(const ResourceLimitError&) { return error(); } catch(...) {
        if(PyErr_Occurred()) return nullptr;
        return error();
    }
}
RuntimeLimits read_runtime_limits(PyObject* raw) {
    RuntimeLimits limits;
    limits.phase = phase_limits(raw);
    if(auto* value = PyDict_GetItemString(raw, "history_sampler")) {
        auto method = file_text(value);
        if(method != "affine_support" && method != "exact_marginal")
            throw std::invalid_argument("history_sampler must be affine_support or exact_marginal");
        limits.capped_affine = method != "exact_marginal";
    }
    if(auto* value = PyDict_GetItemString(raw, "max_path_bits")) limits.paths = natural(value);
    if(auto* value = PyDict_GetItemString(raw, "max_doubled_path_bits")) limits.doubled = natural(value);
    if(auto* value = PyDict_GetItemString(raw, "max_plans")) limits.plans = natural(value);
    if(auto* value = PyDict_GetItemString(raw, "max_queries")) limits.queries = natural(value);
    if(auto* value = PyDict_GetItemString(raw, "max_seconds")) {
        limits.seconds = PyFloat_AsDouble(value);
        if(PyErr_Occurred()) throw std::invalid_argument("seconds must be numeric");
    }
    if(!limits.plans) throw std::invalid_argument("positive plan cache bound required");
    return limits;
}
PyObject* make_file_execution(PyObject*, PyObject* args) {
    try {
        PyObject *raw_source, *raw_recovery, *raw_target, *raw_task, *raw_noise, *raw_limits, *raw_database, *callbacks;
        if(!PyArg_ParseTuple(args, "OOOOOOOO", &raw_source, &raw_recovery, &raw_target, &raw_task, &raw_noise,
                   &raw_limits, &raw_database, &callbacks))
            return nullptr;
        auto* source = circuit_handle(raw_source);
        auto* noise = file_noise_owner(raw_noise);
        if(!source || !noise) return nullptr;
        auto* recovery = raw_recovery == Py_None ? nullptr : circuit_handle(raw_recovery);
        if(raw_recovery != Py_None && !recovery) return nullptr;
        auto* target = raw_target == Py_None ? nullptr : ideal_target_handle(raw_target);
        if(raw_target != Py_None && !target) return nullptr;
        auto outputs = read_sizes(field(raw_task, "outputs"));
        HistoryTask task;
        task.boundaries = read_query_task(raw_task);
        task.boundaries.has_target = target != nullptr;
        task.outcomes = intvec(field(raw_task, "outcomes"));
        auto limits = read_runtime_limits(raw_limits);
        auto database = phase_database(raw_database);
        auto owner = std::make_unique<FileExecutionBinding>();
        owner->run = native_work([&]() {
            return std::make_shared<CompiledExperiment>(
                    FileSource(*source, outputs, recovery ? std::make_optional(*recovery) : std::nullopt,
                            target ? std::make_optional(*target) : std::nullopt),
                    task, *noise, limits, database);
        });
        owner->callbacks = Py_NewRef(callbacks);
        return PyCapsule_New(owner.release(), "magic-file-execution-v1", execution_destructor);
    } catch(...) { return execution_error(); }
}
PyObject* prepare_file_execution(PyObject*, PyObject* cap) {
    try {
        auto* owner = execution_handle(cap);
        if(!owner) return nullptr;
        CompiledExperiment::ExclusiveUse use(*owner->run);
        native_work([&]() {
            owner->run->prepare();
            return 0;
        });
        Py_RETURN_NONE;
    } catch(...) { return execution_error(); }
}
PyObject* file_execution_state(PyObject*, PyObject* args) {
    try {
        PyObject* cap;
        int details = 1;
        if(!PyArg_ParseTuple(args, "O|p", &cap, &details)) return nullptr;
        auto* owner = execution_handle(cap);
        if(!owner) return nullptr;
        CompiledExperiment::ExclusiveUse use(*owner->run);
        auto& run = *owner->run;
        const auto& c = run.context;
        size_t plans = run.plans.size(), structures = run.plans.structure_size();
        for(const auto& worker : owner->workers) {
            plans += worker->plans.size();
            structures += worker->plans.structure_size();
        }
        Object out(Py_BuildValue("{s:n,s:n,s:d,s:d,s:d,s:n,s:n}", "queries", Py_ssize_t(c.queries), "compilations",
                Py_ssize_t(c.compilations), "compile_seconds", c.compile_seconds, "kernel_seconds", c.kernel_seconds,
                "decoder_seconds", c.decoder_seconds, "plans", Py_ssize_t(plans), "structures",
                Py_ssize_t(structures)));
        Object resources(PyDict_New());
        phase_set(out.p, "route", PyUnicode_FromString(run.nonlinear() ? "nonlinear" : "linear"));
        phase_set(out.p, "conditional_noise", PyBool_FromLong(!run.source.guarded_ids.empty()));
        phase_set(out.p, "sampled_history", PyBool_FromLong(run.needs_history()));
        phase_set(out.p, "shared_path_gates", PyLong_FromSize_t(c.shared_path_gates));
        phase_set(out.p, "support_updates", PyLong_FromSize_t(c.support_updates));
        phase_set(out.p, "support_seconds", PyFloat_FromDouble(c.support_seconds));
        phase_set(out.p, "history_sampler",
                PyUnicode_FromString(run.limits.capped_affine ? "affine_support" : "exact_marginal"));
        phase_set(out.p, "support_geometry_updates", PyLong_FromSize_t(c.support_geometry_updates));
        phase_set(out.p, "support_geometry_reuses", PyLong_FromSize_t(c.support_geometry_reuses));
        phase_set(out.p, "support_term_visits", PyLong_FromSize_t(c.support_term_visits));
        phase_set(out.p, "support_tensor_expansions", PyLong_FromSize_t(c.support_tensor_expansions));
        phase_set(out.p, "support_method",
                PyUnicode_FromString(!run.proposal         ? "none"
                                : run.history->contraction ? "prepared_clifford"
                                                           : "affine_envelope"));
        Object evaluations(PyDict_New());
        for(const auto& [purpose, count] : c.reference_network_evaluations)
            phase_set(evaluations.p, purpose.c_str(), PyLong_FromSize_t(count));
        phase_set(out.p, "reference_network_evaluations", Py_NewRef(evaluations.p));
        phase_set(out.p, "reference_network_preparations", PyLong_FromSize_t(c.reference_network_preparations));
        phase_set(out.p, "reference_network_resource_constructions",
                PyLong_FromSize_t(c.reference_network_resource_constructions));
        phase_set(out.p, "reference_network_coordinate_constructions",
                PyLong_FromSize_t(c.reference_network_coordinate_constructions));
        phase_set(out.p, "commuting_control_parities", PyLong_FromSize_t(run.sector_rows.size()));
        phase_set(out.p, "affine_control_dimension", PyLong_FromSize_t(run.sector_support.columns.size()));
        phase_set(out.p, "target_normalization", output_scalar(run.target_normalization));
        if(details) {
            // The per-draw counter sync omits these fixed names.
            Object targets(PyTuple_New(run.task.boundaries.target_records.size()));
            for(size_t j = 0; j < run.task.boundaries.target_records.size(); j++)
                PyTuple_SET_ITEM(targets.p, j, PyUnicode_FromString(run.task.boundaries.target_records[j].c_str()));
            phase_set(out.p, "target_records", Py_NewRef(targets.p));
            auto summary = run.resource_summary();
            Object aggregate(Py_BuildValue("{s:n,s:n,s:n}", "queries", Py_ssize_t(summary.queries),
                    "max_reference_phase_coordinates", Py_ssize_t(summary.reference_coordinates), "max_reference_terms",
                    Py_ssize_t(summary.reference_terms)));
            phase_set(aggregate.p, "max_uniform_term_bound",
                    summary.all_uniform && summary.queries ? PyLong_FromSize_t(summary.uniform_terms)
                                                           : Py_NewRef(Py_None));
            phase_set(aggregate.p, "max_decomposition_entries", PyLong_FromSize_t(summary.max_decomposition_entries));
            phase_set(aggregate.p, "sum_decomposition_entries", PyLong_FromSize_t(summary.sum_decomposition_entries));
            phase_set(out.p, "resource_summary", Py_NewRef(aggregate.p));
        }
        phase_set(out.p, "completed_history_reuses", PyLong_FromSize_t(run.context.completed_history_reuses));
        phase_set(out.p, "target_frame_reuses", PyLong_FromSize_t(run.context.target_frame_reuses));
        if(details) {
            auto payload = run.payload_report();
            for(const auto& worker : owner->workers) {
                auto extra = worker->payload_report();
                payload.persistent_networks += extra.persistent_networks;
                payload.networks += extra.networks;
                payload.proposal += extra.proposal;
                payload.protocol_transport += extra.protocol_transport;
                payload.branch_transport += extra.branch_transport;
                payload.branch_locations += extra.branch_locations;
            }
            phase_set(out.p, "persistent_networks", PyLong_FromSize_t(payload.persistent_networks));
            phase_set(out.p, "protocol_transport_arithmetic_payload_bytes",
                    PyLong_FromSize_t(payload.protocol_transport));
            phase_set(out.p, "branch_transport_arithmetic_payload_bytes", PyLong_FromSize_t(payload.branch_transport));
            phase_set(out.p, "branch_original_locations", PyLong_FromSize_t(payload.branch_locations));
            phase_set(out.p, "proposal_arithmetic_payload_bytes", PyLong_FromSize_t(payload.proposal));
            phase_set(
                    out.p, "network_arithmetic_payload_bytes", PyLong_FromSize_t(payload.networks + payload.proposal));
        }
        if(details) {
            Object compilers(PyDict_New());
            for(const auto& [stage, engine] : run.plans.prepared) {
                const auto& compiled = engine->plan->compiled;
                Object row(Py_BuildValue("{s:n,s:n,s:n,s:n,s:n,s:n}", "rotation_layers",
                        Py_ssize_t(compiled.rotation_layers), "reference_layer_slots",
                        Py_ssize_t(compiled.reference_layer_slots), "phase_ancillas",
                        Py_ssize_t(compiled.ancilla_count), "shared_phase_coordinates",
                        Py_ssize_t(compiled.phase_coordinates), "clifford_gate_count", Py_ssize_t(compiled.gate_count),
                        "path_variables", Py_ssize_t(compiled.path_variables)));
                phase_set(row.p, "reference_layer", PyLong_FromLong(compiled.reference_layer));
                phase_set(row.p, "layer_slots", compiler_intvec(compiled.layer_slots));
                phase_set(compilers.p, stage.c_str(), Py_NewRef(row.p));
            }
            phase_set(out.p, "compiler_diagnostics", Py_NewRef(compilers.p));
        }
        if(details)
            for(const auto& [stage, queries] : run.resources) {
                Object rows(PyList_New(queries.size()));
                for(size_t j = 0; j < queries.size(); j++) {
                    Object report(compiled_network_info(*queries[j].second));
                    if(queries[j].first == "sampling_capped") {
                        phase_set(report.p, "transport_labels",
                                PyLong_FromSize_t(queries[j].second->boundary.keep.size()));
                        phase_set(report.p, "retained_labels", PyLong_FromLong(0));
                        phase_set(report.p, "resource_reference",
                                PyUnicode_FromString("shared unselected parent; complete stabilizer cap view"));
                    }
                    const auto* fixed = queries[j].second->fixed_certificate();
                    phase_set(report.p, "fixed_update_domain", PyBool_FromLong(fixed != nullptr));
                    if(fixed) {
                        phase_set(report.p, "prepared_update_coordinates",
                                PyLong_FromSize_t(fixed->reference.quotient.axes.size()));
                        phase_set(report.p, "prepared_update_entries", PyLong_FromSize_t(fixed->term_bound()));
                        phase_set(report.p, "entry_change_groups",
                                PyLong_FromSize_t(fixed->reference.entry_plans
                                                ? fixed->reference.entry_plans->changes.size()
                                                : 0));
                    }
                    phase_set(report.p, "purpose", PyUnicode_FromString(queries[j].first.c_str()));
                    PyList_SET_ITEM(rows.p, j, Py_NewRef(report.p));
                }
                if(PyDict_SetItemString(resources.p, stage.c_str(), rows.p) < 0) throw std::bad_alloc();
            }
        phase_set(out.p, "query_families", Py_NewRef(resources.p));
        phase_set(out.p, "complete", PyBool_FromLong(run.complete));
        return Py_NewRef(out.p);
    } catch(...) { return execution_error(); }
}
PyObject* configure_file_execution(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *options;
        if(!PyArg_ParseTuple(args, "OO", &cap, &options)) return nullptr;
        auto* owner = execution_handle(cap);
        if(!owner) return nullptr;
        auto limits = read_runtime_limits(options);
        CompiledExperiment::ExclusiveUse use(*owner->run);
        for(auto& worker : owner->workers) worker->configure(limits);
        owner->run->configure(std::move(limits));
        Py_RETURN_NONE;
    } catch(...) { return execution_error(); }
}
PyObject* begin_file_execution_run(PyObject*, PyObject* cap) {
    try {
        auto* owner = execution_handle(cap);
        if(!owner) return nullptr;
        CompiledExperiment::ExclusiveUse use(*owner->run);
        owner->run->context.begin_run();
        Py_RETURN_NONE;
    } catch(...) { return execution_error(); }
}
ActualNoise read_actual_noise(PyObject* raw) {
    Seq data(raw);
    if(data.size() != 3) throw std::invalid_argument("noise assignment shape");
    ActualNoise noise;
    noise.actions = read_noise_actions(data[0]);
    Seq flips(data[1]);
    for(size_t j = 0; j < flips.size(); j++) noise.flips.push_back(file_text(flips[j]));
    Seq weight(data[2]);
    if(weight.size() != 2) throw std::invalid_argument("noise weight shape");
    noise.weight = {big(weight[0]), big(weight[1])};
    return noise;
}
// Lets Python signal handlers (Ctrl-C) run between draws, at most every 100 ms.
struct SignalCheck {
    std::chrono::steady_clock::time_point next = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
    void operator()() {
        auto now = std::chrono::steady_clock::now();
        if(now < next) return;
        next = now + std::chrono::milliseconds(100);
        CallbackGIL gil;
        if(PyErr_CheckSignals() < 0) throw PendingPythonError();
    }
};
// The caller owns the GIL. Parallel windows acquire it once for all of their
// ordinary getrandbits(64) calls, preserving the existing draw seed stream.
uint64_t execution_seed(PyObject* rng) {
    Object value(PyObject_CallMethod(rng, "getrandbits", "i", 64));
    if(!value.p) throw PendingPythonError();
    auto result = PyLong_AsUnsignedLongLong(value.p);
    if(PyErr_Occurred()) throw std::invalid_argument("RNG uint64 required");
    return uint64_t(result);
}
std::vector<uint64_t> execution_seeds(PyObject* rng, size_t count) {
    CallbackGIL gil;
    std::vector<uint64_t> seeds;
    seeds.reserve(count);
    for(size_t j = 0; j < count; j++) seeds.push_back(execution_seed(rng));
    return seeds;
}
ExecutionHooks execution_hooks(FileExecutionBinding& owner, PyObject* rng) {
    ExecutionHooks hooks;
    hooks.seed = [rng]() {
        CallbackGIL gil;
        return execution_seed(rng);
    };
    hooks.controller = [&owner](const std::string& stage, size_t index, const std::vector<int>& bits, size_t count) {
        CallbackGIL gil;
        try {
            auto* group = field(owner.callbacks, stage.c_str());
            Seq callbacks(group);
            Object inputs(output_int_tuple(bits)), width(PyLong_FromSize_t(count));
            Object result(PyObject_CallFunctionObjArgs(callbacks[index], inputs.p, width.p, nullptr));
            return intvec(result.p);
        } catch(...) {
            if(PyErr_Occurred()) throw PythonCallbackError();
            throw;
        }
    };
    return hooks;
}
PyObject* output_history(const HistoryResult& result) {
    const auto& m = result.masses;
    Object masses(PyDict_New()), moments(PyList_New(m.moments.size()));
    phase_set(masses.p, "history", output_scalar(m.history));
    phase_set(masses.p, "acceptance", output_scalar(m.acceptance));
    phase_set(masses.p, "error", m.error ? output_scalar(*m.error) : Py_NewRef(Py_None));
    phase_set(masses.p, "target", m.target_success ? output_scalar(*m.target_success) : Py_NewRef(Py_None));
    phase_set(masses.p, "target_normalization", output_scalar(m.target_normalization));
    for(size_t j = 0; j < m.moments.size(); j++) PyList_SET_ITEM(moments.p, j, output_scalar(m.moments[j]));
    phase_set(masses.p, "moments", Py_NewRef(moments.p));
    Object samples(output_int_tuple(result.samples)), n(output_big(result.noise.weight.n)),
            d(output_big(result.noise.weight.d));
    auto contribution = result.contribution();
    Object statistics(output_history_contribution(contribution));
    phase_set(statistics.p, "noise_weight_scientific", output_scientific(contribution.noise_weight));
    Object denominator(result.proposal_denominator ? output_big(*result.proposal_denominator) : Py_NewRef(Py_None));
    phase_set(masses.p, "proposal",
            result.proposal_denominator ? Py_BuildValue("(iO)", 1, denominator.p) : Py_NewRef(Py_None));
    return Py_BuildValue("{s:O,s:O,s:(OO),s:O,s:n,s:n,s:n}", "masses", masses.p, "samples", samples.p, "weight", n.p,
            d.p, "statistics", statistics.p, "decoder_calls", Py_ssize_t(result.decoder_calls),
            "recovery_decoder_calls", Py_ssize_t(result.recovery_decoder_calls), "sampled_parities",
            Py_ssize_t(result.sampled_parities));
}
PyObject* sample_file_execution(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *rng, *noise;
        if(!PyArg_ParseTuple(args, "OOO", &cap, &rng, &noise)) return nullptr;
        auto* owner = execution_handle(cap);
        if(!owner) return nullptr;
        auto hooks = execution_hooks(*owner, rng);
        std::optional<ActualNoise> supplied;
        CompiledExperiment::ExclusiveUse use(*owner->run);
        if(noise != Py_None) {
            supplied = read_actual_noise(noise);
            owner->run->source.validate_noise(*supplied);
        }
        auto result = native_work([&]() { return owner->run->sample(hooks, supplied); });
        return output_history(result);
    } catch(...) { return execution_error(); }
}
FileBatch* file_batch_handle(PyObject* cap) {
    return static_cast<FileBatch*>(PyCapsule_GetPointer(cap, "magic-file-batch-v1"));
}
void file_batch_destructor(PyObject* cap) { delete file_batch_handle(cap); }
PyObject* make_file_batch(PyObject*, PyObject*) {
    try {
        auto batch = std::make_unique<FileBatch>();
        return PyCapsule_New(batch.release(), "magic-file-batch-v1", file_batch_destructor);
    } catch(...) { return execution_error(); }
}
PyObject* advance_file_batch(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *raw_batch, *rng, *raw_count, *raw_workers = nullptr;
        if(!PyArg_ParseTuple(args, "OOOO|O", &cap, &raw_batch, &rng, &raw_count, &raw_workers)) return nullptr;
        auto* owner = execution_handle(cap);
        auto* batch = file_batch_handle(raw_batch);
        if(!owner || !batch) return nullptr;
        auto count = natural(raw_count);
        auto workers = raw_workers ? natural(raw_workers) : 1;
        if(!workers) throw std::invalid_argument("workers must be a positive integer");
        auto hooks = execution_hooks(*owner, rng);
        CompiledExperiment::ExclusiveUse use(*owner->run);
        native_work([&]() {
            SignalCheck signals;
            if(workers > 1)
                advance_parallel_file_batch(
                        *owner->run, owner->workers, owner->parallel, *batch, count, workers, hooks, signals,
                        [rng](size_t size) { return execution_seeds(rng, size); });
            else
                for(size_t j = 0; j < count; j++) {
                    batch->add(owner->run->sample(hooks));
                    signals();
                }
            return 0;
        });
        return PyLong_FromSize_t(batch->count());
    } catch(...) { return execution_error(); }
}
PyObject* file_batch_efficiency(PyObject*, PyObject* cap) {
    try {
        auto* batch = file_batch_handle(cap);
        if(!batch) return nullptr;
        return output_efficiency_summary(batch->efficiency);
    } catch(...) { return execution_error(); }
}
PyObject* file_batch_summary(PyObject*, PyObject* cap) {
    try {
        auto* batch = file_batch_handle(cap);
        if(!batch) return nullptr;
        auto summary = batch->values.summary();
        auto weights = batch->weights.summary();
        Object out(output_scientific_summary(summary)), records(PyList_New(batch->records.size()));
        for(size_t j = 0; j < batch->records.size(); j++)
            PyList_SET_ITEM(records.p, j, output_int_tuple(batch->records[j]));
        phase_set(out.p, "completed", PyLong_FromSize_t(batch->count()));
        phase_set(out.p, "sampled_records", Py_NewRef(records.p));
        phase_set(out.p, "weights",
                Py_BuildValue("{s:d,s:(dL),s:n}", "ess", weights.ess, "mean_scientific", weights.mantissa,
                        (long long)weights.exponent, "completed", Py_ssize_t(weights.completed)));
        phase_set(out.p, "efficiency", output_efficiency_summary(batch->efficiency));
        return Py_NewRef(out.p);
    } catch(...) { return execution_error(); }
}
