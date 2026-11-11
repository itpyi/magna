// Test-only oracle bindings. They are compiled into the _native_testing module by
// tests/support/native_testing.py, never into the installed package. Included after
// core/magna/native/bindings/module.hpp, whose helpers they use.
#pragma once
namespace {
PyObject* cuda_history_rng_words(PyObject*, PyObject* args) {
    try {
        unsigned long long seed, shot, domain;
        if(!PyArg_ParseTuple(args, "KKK", &seed, &shot, &domain)) return nullptr;
        CudaPhilox rng(shot, cuda_history_seed(seed, domain));
        std::vector<uint64_t> words(32);
        for(auto& word : words) word = rng();
        return cuda_output_bytes(words);
    } catch(...) { return execution_error(); }
}
PyObject* cuda_oversized_weight_plan(PyObject*, PyObject*) {
    try {
        CudaQueryPlan base;
        base.mode = 1;
        Scalar weight;
        weight.c[0] = Integer(1).shift(63) + Integer(1);
        return cuda_query_plan_output(cuda_query_scale(base, weight));
    } catch(...) { return execution_error(); }
}
// Helpers used only by the oracles below.
DecisionValues read_decisions(PyObject* raw) {
    Seq rows(raw);
    DecisionValues out;
    for(size_t j = 0; j < rows.size(); j++) {
        Seq row(rows[j]);
        if(row.size() != 2) throw std::invalid_argument("decision shape");
        int bit = index_int(row[1]);
        if(bit < 0 || bit > 1) throw std::invalid_argument("decision bit required");
        out[read_target(row[0])] = bit;
    }
    return out;
}
PyObject* output_declared_expression(const Expression& expression) {
    const auto& e = *expression;
    static const char* names[] = {"const", "record", "not", "xor", "and", "or", "add", "mul"};
    if(e.opcode == 0) return Py_BuildValue("(s(i))", "const", e.value);
    if(e.opcode == 1) return Py_BuildValue("(s(s))", "record", e.record.c_str());
    Object children(PyTuple_New(e.children.size()));
    for(size_t j = 0; j < e.children.size(); j++)
        PyTuple_SET_ITEM(children.p, j, output_declared_expression(e.children[j]));
    return Py_BuildValue("(sO)", names[e.opcode], children.p);
}
PyObject* output_lowered_file(const LoweredFile& lowered) {
    Object ops(PyTuple_New(lowered.operations.size())), controls(PyTuple_New(lowered.controls.size()));
    for(size_t j = 0; j < lowered.operations.size(); j++)
        PyTuple_SET_ITEM(ops.p, j, output_operation(lowered.operations[j]));
    for(size_t j = 0; j < lowered.controls.size(); j++) {
        const auto& [anchor, rule] = lowered.controls[j];
        Object guard(output_declared_expression(guard_expression(rule.guard))),
                parents(PyTuple_New(rule.guard.records.size())), gates(PyTuple_New(rule.gates.size()));
        for(size_t k = 0; k < rule.guard.records.size(); k++)
            PyTuple_SET_ITEM(parents.p, k, PyUnicode_FromString(("m" + std::to_string(rule.guard.records[k])).c_str()));
        for(size_t k = 0; k < rule.gates.size(); k++)
            PyTuple_SET_ITEM(gates.p, k, output_operation(rule.gates[k], false));
        PyTuple_SET_ITEM(controls.p, j, Py_BuildValue("(sOOO)", anchor.c_str(), parents.p, guard.p, gates.p));
    }
    return PyTuple_Pack(2, ops.p, controls.p);
}
PyObject* make_file_noise(PyObject*, PyObject* args);
PyObject* draw_file_noise(PyObject*, PyObject* args);
FeedbackTemplate* feedback_template_handle(PyObject* cap) {
    return static_cast<FeedbackTemplate*>(PyCapsule_GetPointer(cap, "magic-feedback-template-v1"));
}
using SharedFileBoundary = std::shared_ptr<FileBoundary>;
SharedFileBoundary* shared_file_boundary(PyObject* cap) {
    return static_cast<SharedFileBoundary*>(PyCapsule_GetPointer(cap, "magic-shared-file-boundary-v1"));
}
PyObject* output_guard(const RecordGuard& guard) {
    Object rows(output_sizes(guard.records));
    return Py_BuildValue("(Oi)", rows.p, guard.constant);
}
void feedback_template_destructor(PyObject* cap) { delete feedback_template_handle(cap); }
void shared_file_boundary_destructor(PyObject* cap) { delete shared_file_boundary(cap); }
void shared_boundary_destructor(PyObject* cap) {
    delete static_cast<std::shared_ptr<BoundaryTracePlan>*>(PyCapsule_GetContext(cap));
}
// Formerly bindings/file_runtime_bindings.hpp.
// Included inside bindings.cpp's anonymous namespace, after boundary bindings.
// All draws address completed global probability tensors, never quantum states.
// <random> and <cmath> must be included before that namespace.

PyObject* sample_boundary_terms(PyObject*, PyObject* args) {
    try {
        PyObject *raw, *width_raw, *seed_raw;
        if(!PyArg_ParseTuple(args, "OOO", &raw, &width_raw, &seed_raw)) return nullptr;
        size_t width = natural(width_raw);
        if(width > 1000000) throw std::invalid_argument("boundary sample width budget");
        if(!PyLong_CheckExact(seed_raw)) throw std::invalid_argument("seed must be a uint64 integer");
        uint64_t seed = PyLong_AsUnsignedLongLong(seed_raw);
        if(PyErr_Occurred()) throw std::invalid_argument("seed outside uint64 range");
        Seq items(raw);
        std::vector<std::pair<Scalar, Ket>> terms;
        for(size_t j = 0; j < items.size(); j++) {
            Seq row(items[j]);
            if(row.size() != 2) throw std::invalid_argument("probability term shape");
            auto c = scalar(row[0]);
            auto k = ket(row[1]);
            if(k.width != width) throw std::invalid_argument("probability term width mismatch");
            terms.push_back({std::move(c), std::move(k)});
        }
        Bits word;
        std::exception_ptr failure;
        PyThreadState* state = PyEval_SaveThread();
        try {
            word = sample_boundary(terms, width, seed);
        } catch(...) { failure = std::current_exception(); }
        PyEval_RestoreThread(state);
        if(failure) std::rethrow_exception(failure);
        return output_bits(word);
    } catch(...) { return error(); }
}

// Compatibility entry point; temporary preparation uses the same production
// sampler. The pre-optimization sampling implementation lives only in Git.
PyObject* sample_file_categories(PyObject*, PyObject* args) {
    try {
        PyObject *raw, *seed, *factors = Py_None;
        if(!PyArg_ParseTuple(args, "OO|O", &raw, &seed, &factors)) return nullptr;
        Object prepare_args(PyTuple_Pack(2, raw, factors));
        Object plan(make_file_noise(nullptr, prepare_args.p));
        Object seeds(PyTuple_Pack(1, seed)), draw_args(PyTuple_Pack(2, plan.p, seeds.p));
        Object results(draw_file_noise(nullptr, draw_args.p));
        auto* row = PyList_GET_ITEM(results.p, 0);
        auto* result = factors == Py_None ? PyTuple_GET_ITEM(row, 0) : row;
        Py_INCREF(result);
        return result;
    } catch(...) { return error(); }
}

double file_finite_number(PyObject* value) {
    double result = PyFloat_AsDouble(value);
    if(PyErr_Occurred() || !std::isfinite(result)) throw std::invalid_argument("finite draw value required");
    return result;
}
PyObject* file_optional_float(bool exists, long double value) {
    if(!exists) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    double result = double(value);
    if(!std::isfinite(result)) throw std::invalid_argument("summary exceeds finite floating-point range");
    return PyFloat_FromDouble(result);
}

PyObject* summarize_file_draws(PyObject*, PyObject* args) {
    try {
        PyObject* raw;
        if(!PyArg_ParseTuple(args, "O", &raw)) return nullptr;
        Seq rows(raw);
        if(!rows.size()) return Py_BuildValue("{s:O,s:O}", "estimates", Py_None, "standard_errors", Py_None);
        FileStatistics statistics;
        for(size_t j = 0; j < rows.size(); j++) {
            Seq row(rows[j]);
            if(row.size() != 3) throw std::invalid_argument("summary row requires acceptance, error, moments");
            FileContribution value{file_finite_number(row[0]), row[1] != Py_None, 0, {}};
            if(value.has_error) value.error = file_finite_number(row[1]);
            Seq ms(row[2]);
            for(size_t k = 0; k < ms.size(); k++) value.moments.push_back(file_finite_number(ms[k]));
            statistics.add(value);
        }
        const auto& acceptance = statistics.acceptance;
        const auto& error = statistics.error;
        const auto& moments = statistics.moments;
        bool has_error = statistics.has_error, has_se = rows.size() > 1, has_ratio = statistics.has_ratio();
        auto ratio = statistics.ratio(), ratio_se = statistics.ratio_error();
        Object estimates(PyDict_New()), ses(PyDict_New()), mean_moments(PyList_New(moments.size())),
                se_moments(PyList_New(moments.size()));
        auto put = [&](PyObject* d, const char* key, bool exists, long double value) {
            Object v(file_optional_float(exists, value));
            if(PyDict_SetItemString(d, key, v.p) < 0) throw std::bad_alloc();
        };
        put(estimates.p, "acceptance", true, acceptance.mean);
        put(estimates.p, "error", has_error, error.mean);
        put(estimates.p, "conditional_error", has_ratio, ratio);
        put(ses.p, "acceptance", has_se, has_se ? acceptance.standard_error() : 0);
        put(ses.p, "error", has_error && has_se, has_error && has_se ? error.standard_error() : 0);
        put(ses.p, "conditional_error", has_ratio && has_se, ratio_se);
        for(size_t k = 0; k < moments.size(); k++) {
            PyList_SET_ITEM(mean_moments.p, k, file_optional_float(true, moments[k].mean));
            PyList_SET_ITEM(se_moments.p, k, file_optional_float(has_se, has_se ? moments[k].standard_error() : 0));
        }
        if(PyDict_SetItemString(estimates.p, "moments", mean_moments.p) < 0 ||
                PyDict_SetItemString(ses.p, "moments", se_moments.p) < 0)
            throw std::bad_alloc();
        return Py_BuildValue("{s:O,s:O}", "estimates", estimates.p, "standard_errors", ses.p);
    } catch(...) { return error(); }
}

PyObject* file_scalar_ratios(PyObject*, PyObject* args) {
    try {
        PyObject *raw, *weight_raw = Py_None;
        if(!PyArg_ParseTuple(args, "O|O", &raw, &weight_raw)) return nullptr;
        Scalar wn(1), wd(1);
        if(weight_raw != Py_None) {
            Seq weight(weight_raw);
            if(weight.size() != 2) throw std::invalid_argument("weight requires numerator and denominator");
            auto n = big(weight[0]), d = big(weight[1]);
            if(n.neg || d.neg || d.zero()) throw std::invalid_argument("invalid nonnegative importance weight");
            wn = file_integer_scalar(n);
            wd = file_integer_scalar(d);
        }
        Seq items(raw);
        std::vector<std::pair<Scalar, Scalar>> input;
        for(size_t j = 0; j < items.size(); j++) {
            Seq row(items[j]);
            if(row.size() != 2) throw std::invalid_argument("scalar ratio requires numerator and denominator");
            input.push_back({scalar(row[0]), scalar(row[1])});
        }
        std::vector<std::pair<double, int64_t>> result;
        result.reserve(input.size());
        std::exception_ptr failure;
        PyThreadState* state = PyEval_SaveThread();
        try {
            for(auto& pair : input) result.push_back(scaled_real_ratio(pair.first * wn, pair.second * wd));
        } catch(...) { failure = std::current_exception(); }
        PyEval_RestoreThread(state);
        if(failure) std::rethrow_exception(failure);
        Object output(PyTuple_New(result.size()));
        for(size_t j = 0; j < result.size(); j++) {
            auto value = Py_BuildValue("(dL)", result[j].first, (long long)result[j].second);
            if(!value) throw std::bad_alloc();
            PyTuple_SET_ITEM(output.p, j, value);
        }
        Py_INCREF(output.p);
        return output.p;
    } catch(...) { return error(); }
}

PyObject* summarize_file_weights(PyObject*, PyObject* args) {
    try {
        PyObject* raw;
        if(!PyArg_ParseTuple(args, "O", &raw)) return nullptr;
        Seq rows(raw);
        std::vector<std::pair<double, int64_t>> values;
        for(size_t j = 0; j < rows.size(); j++) {
            Seq row(rows[j]);
            if(row.size() != 2) throw std::invalid_argument("scientific weight shape");
            values.push_back({file_finite_number(row[0]), integer(row[1])});
        }
        auto summary = summarize_weights(values);
        return Py_BuildValue("{s:d,s:(dL),s:n}", "ess", summary.ess, "mean_scientific", summary.mantissa,
                (long long)summary.exponent, "completed", Py_ssize_t(summary.completed));
    } catch(...) { return error(); }
}
// From bindings/task_bindings.hpp.
PyObject* file_execution_faults(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *raw, *values, *recovery;
        const char* stage;
        if(!PyArg_ParseTuple(args, "OOOOs", &cap, &raw, &values, &recovery, &stage)) return nullptr;
        auto* owner = execution_handle(cap);
        if(!owner) return nullptr;
        auto noise = read_actual_noise(raw);
        auto actual = read_decisions(values), terminal = read_decisions(recovery);
        auto faults = std::string(stage) == "original"
                ? owner->run->source.original_faults(noise)
                : owner->run->source.feedback_faults(actual, terminal, noise, owner->run->source.stage(stage));
        Object out(PyTuple_New(faults.size()));
        for(size_t j = 0; j < faults.size(); j++) {
            const auto& f = faults[j];
            std::string pauli(1, f.pauli);
            PyTuple_SET_ITEM(out.p, j,
                    Py_BuildValue("{s:s,s:n,s:s}", "after", f.anchor.c_str(), "qubit", Py_ssize_t(f.wire), "pauli",
                            pauli.c_str()));
        }
        return Py_NewRef(out.p);
    } catch(...) { return execution_error(); }
}
// From bindings/file_statistics_bindings.hpp.
PyObject* summarize_scientific_draws(PyObject*, PyObject* args) {
    try {
        PyObject* raw;
        if(!PyArg_ParseTuple(args, "O", &raw)) return nullptr;
        Seq rows(raw);
        std::vector<std::vector<std::optional<ScientificValue>>> values;
        for(size_t j = 0; j < rows.size(); j++) {
            Seq row(rows[j]);
            std::vector<std::optional<ScientificValue>> converted;
            for(size_t k = 0; k < row.size(); k++) {
                if(row[k] == Py_None) converted.push_back({});
                else {
                    Seq value(row[k]);
                    if(value.size() != 2) throw std::invalid_argument("scientific value shape");
                    converted.push_back(ScientificValue{file_finite_number(value[0]), integer(value[1])});
                }
            }
            values.push_back(std::move(converted));
        }
        auto summary = native_work([&]() { return summarize_scientific(values); });
        return output_scientific_summary(summary);
    } catch(...) { return error(); }
}
// From bindings/low_weight_bindings.hpp.
PyObject* enumerate_noise_categories(PyObject*, PyObject* args) {
    try {
        PyObject* raw;
        unsigned long long cap, requested;
        if(!PyArg_ParseTuple(args, "OKK", &raw, &requested, &cap)) return nullptr;
        Seq groups(raw);
        size_t n = groups.size();
        std::vector<std::vector<Integer>> units;
        Integer denominator(1);
        for(size_t i = 0; i < n; i++) {
            Seq row(groups[i]);
            if(row.size() != 2) throw std::invalid_argument("category requires integer masses and denominator");
            Seq vals(row[0]);
            auto d = big(row[1]);
            if(d.neg || d.zero() || !vals.size()) throw std::invalid_argument("invalid category denominator");
            Integer sum;
            std::vector<Integer> u;
            for(size_t j = 0; j < vals.size(); j++) {
                auto v = big(vals[j]);
                if(v.neg) throw std::invalid_argument("negative category mass");
                sum = sum + v;
                u.push_back(v);
            }
            if(!(sum == d)) throw std::invalid_argument("category masses do not sum to denominator");
            units.push_back(std::move(u));
            denominator = denominator * d;
        }
        auto patterns = enumerate_noise(units, requested, cap);
        Object output(PyList_New(patterns.size()));
        for(size_t j = 0; j < patterns.size(); j++) {
            const auto& pattern = patterns[j];
            Object selected(PyTuple_New(pattern.selected.size()));
            for(size_t i = 0; i < pattern.selected.size(); i++)
                PyTuple_SET_ITEM(selected.p, i,
                        Py_BuildValue("(KK)", (unsigned long long)pattern.selected[i].first,
                                (unsigned long long)pattern.selected[i].second));
            Object numerator(output_big(pattern.numerator));
            PyList_SET_ITEM(
                    output.p, j, Py_BuildValue("(OOK)", selected.p, numerator.p, (unsigned long long)pattern.weight));
        }
        Object d(output_big(denominator));
        return Py_BuildValue("(OO)", output.p, d.p);
    } catch(...) { return error(); }
}
// From bindings/file_noise_bindings.hpp.
PyObject* make_file_noise(PyObject*, PyObject* args) {
    try {
        PyObject *raw, *factors = Py_None, *actions = Py_None;
        if(!PyArg_ParseTuple(args, "O|OO", &raw, &factors, &actions)) return nullptr;
        auto plan = std::make_unique<FileNoisePlan>();
        Seq groups(raw);
        for(size_t j = 0; j < groups.size(); j++) {
            Seq group(groups[j]);
            std::vector<FileFraction> probabilities;
            for(size_t k = 0; k < group.size(); k++) {
                Seq pair(group[k]);
                if(pair.size() != 2)
                    throw std::invalid_argument("rational probability requires numerator and denominator");
                probabilities.emplace_back(big(pair[0]), big(pair[1]));
            }
            plan->categories.push_back(std::make_shared<const FileNoiseCategory>(std::move(probabilities)));
        }
        if(factors != Py_None) {
            Seq rows(factors);
            if(rows.size() != groups.size()) throw std::invalid_argument("noise weight factor count");
            for(size_t j = 0; j < rows.size(); j++) {
                Seq row(rows[j]);
                if(row.size() != 2) throw std::invalid_argument("noise weight needs absent/present factors");
                std::array<FileFraction, 2> values;
                for(size_t k = 0; k < 2; k++) {
                    Seq pair(row[k]);
                    if(pair.size() != 2) throw std::invalid_argument("noise weight rational shape");
                    values[k] = FileFraction(big(pair[0]), big(pair[1]));
                    if(values[k].n.neg || values[k].d.neg || values[k].d.zero())
                        throw std::invalid_argument("invalid noise weight factor");
                }
                plan->factors.push_back(std::move(values));
            }
        }
        if(actions != Py_None) {
            Seq rows(actions);
            if(rows.size() != groups.size()) throw std::invalid_argument("noise action category count");
            plan->has_actions = true;
            for(size_t j = 0; j < rows.size(); j++) {
                Seq row(rows[j]);
                if(row.size() != plan->categories[j]->size) throw std::invalid_argument("noise action count");
                plan->offsets.push_back(plan->actions.size());
                for(size_t k = 0; k < row.size(); k++) {
                    Seq item(row[k]);
                    if(item.size() != 2) throw std::invalid_argument("noise action shape");
                    const char* anchor = PyUnicode_AsUTF8(item[0]);
                    if(!anchor) throw std::invalid_argument("noise anchor string required");
                    FileNoiseAction a;
                    a.anchor = anchor;
                    if(a.anchor == "report") {
                        const char* report = PyUnicode_AsUTF8(item[1]);
                        if(!report) throw std::invalid_argument("noise report string required");
                        a.report = report;
                    } else {
                        Seq gates(item[1]);
                        for(size_t i = 0; i < gates.size(); i++) {
                            Seq gate(gates[i]);
                            if(gate.size() != 2) throw std::invalid_argument("noise gate shape");
                            const char* pauli = PyUnicode_AsUTF8(gate[0]);
                            if(!pauli || !pauli[0] || pauli[1] ||
                                    std::string("XYZ").find(pauli[0]) == std::string::npos)
                                throw std::invalid_argument("noise Pauli required");
                            a.gates.push_back({pauli[0], natural(gate[1])});
                        }
                    }
                    plan->actions.push_back(std::move(a));
                }
            }
        }
        plan->prepare_weights();
        return own_file_noise(std::move(plan));
    } catch(...) { return error(); }
}
PyObject* draw_file_noise_impl(PyObject* args, bool sparse) {
    try {
        PyObject *cap, *raw;
        if(!PyArg_ParseTuple(args, "OO", &cap, &raw)) return nullptr;
        auto* p = file_noise_handle(cap);
        if(!p) return nullptr;
        Seq seeds(raw);
        if(sparse && !p->has_actions) throw std::invalid_argument("noise action table required");
        if(seeds.size() > 65536) throw std::length_error("noise batch limit");
        std::vector<uint64_t> input;
        for(size_t j = 0; j < seeds.size(); j++) {
            if(!PyLong_CheckExact(seeds[j])) throw std::invalid_argument("seed must be a uint64 integer");
            auto value = PyLong_AsUnsignedLongLong(seeds[j]);
            if(PyErr_Occurred()) throw std::invalid_argument("seed outside uint64 range");
            input.push_back(value);
        }
        std::vector<std::vector<int64_t>> output;
        std::vector<FileFraction> weights;
        std::exception_ptr failure;
        PyThreadState* state = PyEval_SaveThread();
        try {
            for(auto seed : input) {
                auto draw = draw_noise(*p, seed, sparse);
                output.push_back(std::move(draw.selected));
                weights.push_back(std::move(draw.weight));
            }
        } catch(...) { failure = std::current_exception(); }
        PyEval_RestoreThread(state);
        if(failure) std::rethrow_exception(failure);
        Object result(PyList_New(output.size()));
        for(size_t j = 0; j < output.size(); j++) {
            Object selected(PyTuple_New(output[j].size()));
            for(size_t k = 0; k < output[j].size(); k++)
                PyTuple_SET_ITEM(selected.p, k,
                        sparse ? output_file_noise_action(p->actions[size_t(output[j][k])])
                               : PyLong_FromLongLong(output[j][k]));
            Object n(output_big(weights[j].n)), d(output_big(weights[j].d));
            PyList_SET_ITEM(result.p, j, Py_BuildValue("(O(OO))", selected.p, n.p, d.p));
        }
        Py_INCREF(result.p);
        return result.p;
    } catch(...) { return error(); }
}
PyObject* draw_file_noise(PyObject*, PyObject* args) { return draw_file_noise_impl(args, false); }
PyObject* draw_file_noise_actions(PyObject*, PyObject* args) { return draw_file_noise_impl(args, true); }
// From bindings/file_lowering_bindings.hpp.
PyObject* make_feedback_template(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *width;
        if(!PyArg_ParseTuple(args, "OO", &cap, &width)) return nullptr;
        auto* circuit = circuit_handle(cap);
        if(!circuit) return nullptr;
        size_t minimum = natural(width);
        auto plan =
                std::make_unique<FeedbackTemplate>(native_work([&]() { return FeedbackTemplate(*circuit, minimum); }));
        Object data(PyDict_New()), guards(PyDict_New()), virtuals(PyDict_New()), selections(PyDict_New()),
                offsets(PyDict_New());
        for(const auto& [anchor, guard] : plan->guards) {
            Object value(output_guard(guard));
            if(PyDict_SetItemString(guards.p, anchor.c_str(), value.p) < 0) throw std::bad_alloc();
        }
        for(const auto& [anchor, v] : plan->virtual_controls) {
            Object wires(output_sizes(v.targets)),
                    value(Py_BuildValue("(nsO)", Py_ssize_t(v.wire), v.gate.c_str(), wires.p));
            if(PyDict_SetItemString(virtuals.p, anchor.c_str(), value.p) < 0) throw std::bad_alloc();
        }
        for(const auto& [anchor, s] : plan->selections) {
            Object target(output_target(s.target)),
                    value(Py_BuildValue("(nOsi)", Py_ssize_t(s.wire), target.p, s.record.c_str(), 0));
            if(PyDict_SetItemString(selections.p, anchor.c_str(), value.p) < 0) throw std::bad_alloc();
        }
        for(const auto& [anchor, bit] : plan->selection_offsets) {
            Object value(PyLong_FromLong(bit));
            if(PyDict_SetItemString(offsets.p, anchor.c_str(), value.p) < 0) throw std::bad_alloc();
        }
        Object updates(PyTuple_New(plan->updates.size()));
        for(size_t j = 0; j < plan->updates.size(); j++) {
            const auto& u = plan->updates[j];
            char p[2] = {u.pauli, 0};
            Object terms(output_targets(u.terms));
            PyTuple_SET_ITEM(updates.p, j, Py_BuildValue("(snsO)", u.anchor.c_str(), Py_ssize_t(u.wire), p, terms.p));
        }
        for(const auto& [name, value] :
                std::vector<std::pair<const char*, PyObject*>>{{"guards", guards.p}, {"virtual", virtuals.p},
                        {"selections", selections.p}, {"selection_offsets", offsets.p}, {"updates", updates.p}})
            if(PyDict_SetItemString(data.p, name, value) < 0) throw std::bad_alloc();
        phase_set(data.p, "wire_count", PyLong_FromSize_t(plan->wire_count));
        Object handle(PyCapsule_New(plan.release(), "magic-feedback-template-v1", feedback_template_destructor));
        return PyTuple_Pack(2, handle.p, data.p);
    } catch(...) { return error(); }
}
PyObject* feedback_template_faults(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *decisions, *noise, *prefix, *wires;
        if(!PyArg_ParseTuple(args, "OOOOO", &cap, &decisions, &noise, &prefix, &wires)) return nullptr;
        auto* plan = feedback_template_handle(cap);
        if(!plan) return nullptr;
        auto values = read_decisions(decisions);
        auto actions = read_noise_actions(noise);
        auto start = file_text(prefix);
        std::optional<std::map<size_t, size_t>> wiremap;
        if(wires != Py_None) {
            wiremap.emplace();
            Seq rows(wires);
            for(size_t j = 0; j < rows.size(); j++) {
                Seq row(rows[j]);
                if(row.size() != 2) throw std::invalid_argument("wire map shape");
                (*wiremap)[natural(row[0])] = natural(row[1]);
            }
        }
        auto faults = native_work([&]() { return plan->faults(values, start, wiremap); });
        Object result(PyTuple_New(faults.size()));
        for(size_t j = 0; j < faults.size(); j++) {
            const auto& f = faults[j];
            char p[2] = {f.pauli, 0};
            PyTuple_SET_ITEM(result.p, j,
                    Py_BuildValue("{s:s,s:n,s:s}", "after", f.anchor.c_str(), "qubit", Py_ssize_t(f.wire), "pauli", p));
        }
        return Py_NewRef(result.p);
    } catch(...) { return error(); }
}
PyObject* lower_circuit_file(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *outputs, *values, *noise, *feedback;
        if(!PyArg_ParseTuple(args, "OOOOO", &cap, &outputs, &values, &noise, &feedback)) return nullptr;
        auto* circuit = circuit_handle(cap);
        if(!circuit) return nullptr;
        auto wires = read_sizes(outputs);
        auto decisions = read_decisions(values);
        auto actions = read_noise_actions(noise);
        auto* plan = feedback == Py_None ? nullptr : feedback_template_handle(feedback);
        if(feedback != Py_None && !plan) return nullptr;
        auto lowered = native_work([&]() { return lower_file(*circuit, wires, decisions, actions, plan); });
        return output_lowered_file(lowered);
    } catch(...) { return error(); }
}
// From bindings/history_efficiency_bindings.hpp.
HistoryEfficiency* efficiency_handle(PyObject* cap) {
    return static_cast<HistoryEfficiency*>(PyCapsule_GetPointer(cap, "magic-history-efficiency-v1"));
}
void efficiency_destructor(PyObject* cap) { delete efficiency_handle(cap); }
PyObject* make_file_efficiency(PyObject*, PyObject*) {
    try {
        auto value = std::make_unique<HistoryEfficiency>();
        return PyCapsule_New(value.release(), "magic-history-efficiency-v1", efficiency_destructor);
    } catch(...) { return error(); }
}
ScientificValue read_efficiency_value(PyObject* raw) {
    Seq pair(raw);
    if(pair.size() != 2) throw std::invalid_argument("scientific diagnostic value shape");
    ScientificValue value{file_finite_number(pair[0]), integer(pair[1])};
    if(value.first < 0 || value.first >= 1 || (value.first > 0 && value.first < .5) || value.second < -1000000000 ||
            value.second > 1000000000)
        throw std::invalid_argument("normalized nonnegative scientific diagnostic value required");
    return value;
}
PyObject* add_file_efficiency(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *a, *e, *w, *noise;
        if(!PyArg_ParseTuple(args, "OOOOO", &cap, &a, &e, &w, &noise)) return nullptr;
        auto* state = efficiency_handle(cap);
        if(!state) return nullptr;
        auto error_value =
                e == Py_None ? std::optional<ScientificValue>{} : std::make_optional(read_efficiency_value(e));
        state->add(read_efficiency_value(a), error_value, read_efficiency_value(w), read_efficiency_value(noise));
        Py_RETURN_NONE;
    } catch(...) { return error(); }
}
PyObject* file_efficiency_summary(PyObject*, PyObject* cap) {
    try {
        auto* s = efficiency_handle(cap);
        if(!s) return nullptr;
        return output_efficiency_summary(*s);
    } catch(...) { return error(); }
}
// From bindings/compiler_bindings.hpp.
PyObject* debug_prepare_compiler(PyObject*, PyObject* d) {
    try {
        return output_compiler(prepare_compiler(parse_compiler(d)));
    } catch(...) { return error(); }
}
// From bindings/boundary_trace_bindings.hpp.
PyObject* trace_probability_tensor(PyObject*, PyObject* args) {
    try {
        PyObject *raw, *kept, *events, *outcomes, *px, *pz;
        unsigned long long paths;
        if(!PyArg_ParseTuple(args, "OOOOOOK", &raw, &kept, &events, &outcomes, &px, &pz, &paths)) return nullptr;
        auto t = phase_tensor(raw);
        auto keep = bitvec(kept), rows = bitvec(events);
        auto values = intvec(outcomes);
        auto x = bits(px), z = bits(pz);
        TraceFactorCache support, pairing;
        PhaseLimits limits;
        Workspace ws;
        auto result = single_trace_query(
                std::move(t), keep, rows, values, x, z, true, ws, support, pairing, size_t(paths), limits, nullptr);
        return phase_output_tensor(result.tensor);
    } catch(...) { return error(); }
}
PyObject* boundary_commuting_rows(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *raw;
        if(!PyArg_ParseTuple(args, "OO", &cap, &raw)) return nullptr;
        auto* p = boundary_handle(cap);
        if(!p) return nullptr;
        auto rows = bitvec(raw);
        std::vector<Bits> result;
        std::exception_ptr failure;
        PyThreadState* state = PyEval_SaveThread();
        try {
            result = p->commuting_rows(rows);
        } catch(...) { failure = std::current_exception(); }
        PyEval_RestoreThread(state);
        if(failure) std::rethrow_exception(failure);
        Object out(phase_output_bits(result));
        return Py_BuildValue("(OO)", out.p, p->common_axes ? Py_True : Py_False);
    } catch(...) { return error(); }
}
// From bindings/phase_bindings.hpp.
PyObject* phase_database_test(PyObject*, PyObject* arg) {
    try {
        return PyLong_FromSize_t(phase_database(arg).entries.size());
    } catch(...) { return error(); }
}
// From bindings/file_execution_bindings.hpp.
PyObject* get_file_execution_boundary(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *raw;
        const char* stage;
        if(!PyArg_ParseTuple(args, "OsO", &cap, &stage, &raw)) return nullptr;
        auto* owner = execution_handle(cap);
        if(!owner) return nullptr;
        auto noise = read_actual_noise(raw);
        CompiledExperiment::ExclusiveUse use(*owner->run);
        auto engine = native_work([&]() { return owner->run->plans.get(stage); });
        Object id(PyLong_FromVoidPtr(engine.get()));
        auto shared = std::make_unique<SharedFileBoundary>(engine);
        Object result(
                PyCapsule_New(shared.release(), "magic-shared-file-boundary-v1", shared_file_boundary_destructor));
        return PyTuple_Pack(2, id.p, result.p);
    } catch(...) { return execution_error(); }
}
PyObject* inspect_file_boundary(PyObject*, PyObject* cap) {
    try {
        auto* ptr = shared_file_boundary(cap);
        if(!ptr) return nullptr;
        const auto& engine = **ptr;
        Object metadata(PyDict_New()), locations(PyDict_New()), ops(PyTuple_New(engine.deferred.operations.size())),
                records(PyTuple_New(engine.deferred.records.size())),
                anchors(PyTuple_New(engine.deferred.report_anchors.size()));
        for(const auto& [id, index] : engine.locations) phase_set(locations.p, id.c_str(), PyLong_FromSize_t(index));
        for(size_t j = 0; j < engine.deferred.operations.size(); j++)
            PyTuple_SET_ITEM(ops.p, j, output_operation(engine.deferred.operations[j].value));
        for(size_t j = 0; j < engine.deferred.records.size(); j++) {
            const auto& [name, wire] = engine.deferred.records[j];
            PyTuple_SET_ITEM(records.p, j, Py_BuildValue("(sn)", name.c_str(), Py_ssize_t(wire)));
        }
        for(size_t j = 0; j < engine.deferred.report_anchors.size(); j++) {
            const auto& [name, id, wire] = engine.deferred.report_anchors[j];
            PyTuple_SET_ITEM(anchors.p, j, Py_BuildValue("(ssn)", name.c_str(), id.c_str(), Py_ssize_t(wire)));
        }
        for(const auto& [name, value] : std::vector<std::pair<const char*, PyObject*>>{{"locations", locations.p},
                    {"operations", ops.p}, {"records", records.p}, {"report_anchors", anchors.p}})
            phase_set(metadata.p, name, Py_NewRef(value));
        phase_set(metadata.p, "inputs", output_sizes(engine.inputs));
        phase_set(metadata.p, "outputs", output_sizes(engine.outputs));
        auto owner = std::make_unique<std::shared_ptr<BoundaryTracePlan>>(engine.plan);
        Object handle(PyCapsule_New(engine.plan.get(), "magic-boundary-trace-v1", shared_boundary_destructor));
        if(PyCapsule_SetContext(handle.p, owner.get()) < 0) throw std::invalid_argument("shared boundary owner");
        owner.release();
        return PyTuple_Pack(2, handle.p, metadata.p);
    } catch(...) { return execution_error(); }
}
// Formerly production bindings whose only remaining callers are tests, with the
// native helpers only they use.
Bits indexed_record_row(const std::vector<std::string>& records, size_t offset, const std::vector<std::string>& names) {
    std::map<std::string, size_t> indices;
    for(size_t j = 0; j < records.size(); j++) indices[records[j]] = offset + j;
    return named_record_row(indices, names);
}
std::vector<ActualNoise> draw_noise_random(
        const FileNoisePlan& plan, size_t count, const std::function<uint64_t()>& seed) {
    if(count > 65536) throw std::length_error("noise batch limit");
    std::vector<uint64_t> seeds;
    if(!plan.categories.empty())
        for(size_t j = 0; j < count; j++) seeds.push_back(seed());
    std::vector<ActualNoise> out;
    if(seeds.empty()) {
        out.resize(count);
        return out;
    }
    for(auto value : seeds) out.push_back(noise_actions(plan, draw_noise(plan, value, true)));
    return out;
}
PyObject* phase_output_quotient(const PolynomialQuotient& q) {
    Object axes(phase_output_bits(q.axes)), core(phase_output_terms(q.core)), linear(phase_output_ints(q.linear)),
            upper(phase_output_bits(q.upper)), lifts(phase_output_bits(q.lifts));
    return Py_BuildValue("(OOOOO)", axes.p, core.p, linear.p, upper.p, lifts.p);
}
PyObject* phase_polynomial_test(PyObject*, PyObject* args) {
    try {
        PyObject *n, *terms;
        if(!PyArg_ParseTuple(args, "OO", &n, &terms)) return nullptr;
        return phase_output_quotient(polynomial_quotient(natural(n), phase_parse_terms(terms)));
    } catch(...) { return error(); }
}
PyObject* overlap_test(PyObject*, PyObject* args) {
    try {
        PyObject *a, *b;
        if(!PyArg_ParseTuple(args, "OO", &a, &b)) return nullptr;
        Workspace ws;
        return output_scalar(overlap(ket(a), ket(b), ws));
    } catch(...) { return error(); }
}
PyObject* quadratic_restrict(PyObject*, PyObject* args) {
    try {
        PyObject *l, *u, *a, *b;
        if(!PyArg_ParseTuple(args, "OOOO", &l, &u, &a, &b)) return nullptr;
        auto linear = intvec(l);
        auto upper = bitvec(u);
        if(linear.size() != upper.size()) throw std::invalid_argument("quadratic shape");
        validate_masks(upper, linear.size());
        for(size_t j = 0; j < upper.size(); j++)
            if(upper[j] != upper[j].above(j) || linear[j] < 0 || linear[j] > 3)
                throw std::invalid_argument("quadratic convention");
        auto basis = bitvec(b);
        validate_masks(basis, linear.size());
        auto q = restrict_q(linear, upper, bits(a), basis);
        Object ln(output_int_tuple(q.linear)), un(output_bit_tuple(q.upper));
        return Py_BuildValue("(iOO)", q.constant, ln.p, un.p);
    } catch(...) { return error(); }
}
PyObject* algebra_bernoulli(PyObject*, PyObject* args) {
    try {
        PyObject *a, *b, *rng;
        if(!PyArg_ParseTuple(args, "OOO", &a, &b, &rng)) return nullptr;
        auto value = exact_bernoulli(scalar(a), scalar(b), [&]() {
            Object result(PyObject_CallMethod(rng, "getrandbits", "i", 1));
            return index_int(result.p);
        });
        return PyLong_FromLong(value);
    } catch(...) { return error(); }
}
PyObject* model_weighted_choice(PyObject*, PyObject* args) {
    try {
        PyObject *rng, *raw;
        if(!PyArg_ParseTuple(args, "OO", &rng, &raw)) return nullptr;
        return PyLong_FromLongLong(weighted_choice(input_rationals(raw), input_randrange(rng)));
    } catch(...) { return error(); }
}
PyObject* native_boundary_coordinates(PyObject*, PyObject* args) {
    try {
        PyObject *records, *rows, *outcomes, *feedback;
        if(!PyArg_ParseTuple(args, "OOOO", &records, &rows, &outcomes, &feedback)) return nullptr;
        auto names = input_names(records);
        BoundaryCoordinates layout(names, read_named_rows(rows), intvec(outcomes), input_names(feedback));
        return output_boundary_coordinates(names, layout);
    } catch(...) { return error(); }
}
PyObject* boundary_service_counts(PyObject*, PyObject* cap) {
    try {
        auto* owner = boundary_service(cap);
        if(!owner) return nullptr;
        auto& service = *owner->service;
        std::lock_guard<std::mutex> lock(service.mutex);
        return Py_BuildValue("(nn)", Py_ssize_t(service.row_cache.size()), Py_ssize_t(service.total_cache.size()));
    } catch(...) { return error(); }
}
PyObject* indexed_record_row_binding(PyObject*, PyObject* args) {
    try {
        PyObject *records, *offset, *names;
        if(!PyArg_ParseTuple(args, "OOO", &records, &offset, &names)) return nullptr;
        return output_bits(indexed_record_row(input_names(records), natural(offset), input_names(names)));
    } catch(...) { return error(); }
}
PyObject* output_grouped_noise(const ActualNoise& noise) {
    Object actions(PyTuple_New(noise.actions.size()));
    size_t j = 0;
    for(const auto& [anchor, gates] : noise.actions) {
        FileNoiseAction action;
        action.anchor = anchor;
        action.gates = gates;
        PyTuple_SET_ITEM(actions.p, j++, output_file_noise_action(action));
    }
    Object flips(output_names(noise.flips)), n(output_big(noise.weight.n)), d(output_big(noise.weight.d)),
            weight(PyTuple_Pack(2, n.p, d.p));
    return PyTuple_Pack(3, actions.p, flips.p, weight.p);
}
PyObject* draw_file_noise_random(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *rng, *count;
        if(!PyArg_ParseTuple(args, "OOO", &cap, &rng, &count)) return nullptr;
        auto* plan = file_noise_handle(cap);
        if(!plan) return nullptr;
        auto size = natural(count);
        auto seed = [&]() {
            CallbackGIL gil;
            Object raw(PyObject_CallMethod(rng, "getrandbits", "i", 64));
            auto value = PyLong_AsUnsignedLongLong(raw.p);
            if(PyErr_Occurred()) throw std::invalid_argument("RNG uint64 required");
            return uint64_t(value);
        };
        auto values = native_work([&]() { return draw_noise_random(*plan, size, seed); });
        Object out(PyList_New(values.size()));
        for(size_t j = 0; j < values.size(); j++) PyList_SET_ITEM(out.p, j, output_grouped_noise(values[j]));
        return Py_NewRef(out.p);
    } catch(...) { return execution_error(); }
}
PyObject* affine_rule_check(PyObject*, PyObject* raw) {
    try {
        return PyBool_FromLong(is_affine_rule(read_declared_gates(raw)));
    } catch(...) { return error(); }
}
PyObject* attach_ideal_target(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *raw_outputs, *raw_offset, *raw_first = Py_None;
        if(!PyArg_ParseTuple(args, "OOO|O", &cap, &raw_outputs, &raw_offset, &raw_first)) return nullptr;
        auto* target = ideal_target_handle(cap);
        if(!target) return nullptr;
        auto outputs = read_sizes(raw_outputs);
        auto offset = natural(raw_offset);
        size_t first = raw_first == Py_None
                ? (outputs.empty() ? 0 : *std::max_element(outputs.begin(), outputs.end()) + 1)
                : natural(raw_first);
        auto result = native_work([&]() { return target->attach(outputs, offset, first); });
        Object ops(PyTuple_New(result.first.size())), records(PyTuple_New(result.second.size()));
        for(size_t j = 0; j < result.first.size(); j++) PyTuple_SET_ITEM(ops.p, j, output_operation(result.first[j]));
        for(size_t j = 0; j < result.second.size(); j++)
            PyTuple_SET_ITEM(records.p, j, PyUnicode_FromString(result.second[j].c_str()));
        return PyTuple_Pack(2, ops.p, records.p);
    } catch(...) { return error(); }
}
PyObject* attach_terminal_recovery(PyObject*, PyObject* args) {
    try {
        PyObject *cap, *raw_width, *raw_outputs, *raw_offset, *raw_first, *raw_values;
        if(!PyArg_ParseTuple(args, "OOOOOO", &cap, &raw_width, &raw_outputs, &raw_offset, &raw_first, &raw_values))
            return nullptr;
        auto* circuit = circuit_handle(cap);
        if(!circuit) return nullptr;
        auto width = natural(raw_width), offset = natural(raw_offset), first = natural(raw_first);
        auto outputs = read_sizes(raw_outputs);
        auto values = read_decisions(raw_values);
        auto local = native_work([&]() { return attach_recovery(*circuit, width, outputs, offset, first, values); });
        return output_lowered_file(local);
    } catch(...) { return error(); }
}
PyObject* check_file_execution(PyObject*, PyObject* cap) {
    try {
        auto* owner = execution_handle(cap);
        if(!owner) return nullptr;
        CompiledExperiment::ExclusiveUse use(*owner->run);
        owner->run->context.check(owner->run->limits);
        Py_RETURN_NONE;
    } catch(...) { return execution_error(); }
}
PyObject* file_noise_factors(PyObject*, PyObject* cap) {
    try {
        auto* plan = file_noise_handle(cap);
        if(!plan) return nullptr;
        Object out(PyTuple_New(plan->categories.size()));
        for(size_t j = 0; j < plan->categories.size(); j++) {
            Object row(PyTuple_New(2));
            for(size_t k = 0; k < 2; k++) {
                FileFraction f = plan->factors.empty() ? FileFraction(Integer(1)) : plan->factors[j][k];
                Object n(output_big(f.n)), d(output_big(f.d));
                PyTuple_SET_ITEM(row.p, k, PyTuple_Pack(2, n.p, d.p));
            }
            PyTuple_SET_ITEM(out.p, j, Py_NewRef(row.p));
        }
        return Py_NewRef(out.p);
    } catch(...) { return error(); }
}
PyObject* branch_schedule_stats(PyObject*, PyObject* cap) {
    try {
        auto* owner = branch_schedule(cap);
        if(!owner) return nullptr;
        const auto& s = owner->schedule->statistics;
        return Py_BuildValue("{s:n,s:n,s:n,s:n,s:n}", "extractions", Py_ssize_t(s.extractions.load()),
                "callback_invocations", Py_ssize_t(s.callbacks.load()), "rule_evaluations", Py_ssize_t(s.rules.load()),
                "native_rule_evaluations", Py_ssize_t(s.native_rules.load()), "full_branch_materializations",
                Py_ssize_t(s.materializations.load()));
    } catch(...) { return execution_error(); }
}
PyObject* gf2_solve(PyObject*, PyObject* args) {
    try {
        PyObject *rows, *rhs, *n;
        if(!PyArg_ParseTuple(args, "OOO", &rows, &rhs, &n)) return nullptr;
        auto out = solve_low_pivots(bitvec(rows), intvec(rhs), natural(n));
        if(!out) { Py_RETURN_NONE; }
        Object particular(output_bits(out->particular)), kernel(output_bit_tuple(out->kernel)),
                reduced(output_bit_tuple(out->rows));
        return Py_BuildValue("(OOO)", particular.p, kernel.p, reduced.p);
    } catch(...) { return error(); }
}
PyObject* gf2_right_inverse(PyObject*, PyObject* args) {
    try {
        PyObject *raw, *n;
        if(!PyArg_ParseTuple(args, "OO", &raw, &n)) return nullptr;
        auto rows = bitvec(raw);
        auto width = natural(n);
        validate_masks(rows, width);
        return output_bit_tuple(phase_right_inverse(rows, width));
    } catch(...) { return error(); }
}
PyObject* gaussian_scalar(PyObject*, PyObject* args) {
    try {
        PyObject *a, *b;
        if(!PyArg_ParseTuple(args, "OO", &a, &b)) return nullptr;
        Seq gaussian(a), phase(b);
        if(gaussian.size() != 2 || phase.size() != 2) throw std::invalid_argument("Gaussian scalar shape");
        Scalar value;
        value.c[0] = big(gaussian[0]);
        value.c[2] = big(gaussian[1]);
        value.normalize();
        return output_scalar(value * omega(int(integer(phase[0]) % 8), integer(phase[1])));
    } catch(...) { return error(); }
}
PyObject* real_sign_oracle(PyObject*, PyObject* value) {
    try {
        return PyLong_FromLong(real_sign(scalar(value)));
    } catch(...) { return error(); }
}
PyObject* phase_feedback_metadata(PyObject*, PyObject* raw) {
    try {
        auto family = read_feedback_family(raw);
        Object axes(PyTuple_New(family.axes.size())), baseline(output_int_tuple(family.baseline));
        for(size_t j = 0; j < family.axes.size(); j++) PyTuple_SET_ITEM(axes.p, j, output_int_tuple(family.axes[j]));
        return PyTuple_Pack(2, axes.p, baseline.p);
    } catch(...) { return error(); }
}
// Checks actual feedback gates against a declared family with the runtime's own resolution.
PyObject* resolve_phase_feedback(PyObject*, PyObject* args) {
    try {
        PyObject *raw, *gates;
        if(!PyArg_ParseTuple(args, "OO", &raw, &gates)) return nullptr;
        auto family = read_feedback_family(raw);
        Seq rows(gates);
        std::vector<Operation> operations;
        for(size_t j = 0; j < rows.size(); j++) operations.push_back(read_boundary_operation(rows[j], false));
        try {
            auto result = resolve_feedback_operations(std::move(family), operations);
            Object phases(output_int_tuple(result.phases)), flips(output_int_tuple(result.flips));
            return Py_BuildValue("(OOi)", phases.p, flips.p, result.phase);
        } catch(const ControlMismatch& e) { return Py_BuildValue("(Os)", Py_None, e.what()); }
    } catch(...) { return error(); }
}
PyObject* certificate_cache_info(PyObject*, PyObject* cap) {
    try {
        auto* cache = certificate_cache(cap);
        if(!cache) return nullptr;
        std::lock_guard<std::mutex> lock(cache->mutex);
        return Py_BuildValue("{s:n,s:n,s:n,s:n,s:n,s:n,s:n,s:d}", "hits", Py_ssize_t(cache->hits), "misses",
                Py_ssize_t(cache->misses), "evictions", Py_ssize_t(cache->evictions), "oversize",
                Py_ssize_t(cache->oversize), "entries", Py_ssize_t(cache->entries.size()), "serialized_bytes",
                Py_ssize_t(cache->bytes), "byte_budget", Py_ssize_t(cache->budget), "avoided_prior_compute_seconds",
                cache->saved_seconds);
    } catch(...) { return error(); }
}
// Keys of all entries, and of the entries that store a probability.
PyObject* certificate_cache_keys(PyObject*, PyObject* cap) {
    try {
        auto* cache = certificate_cache(cap);
        if(!cache) return nullptr;
        std::lock_guard<std::mutex> lock(cache->mutex);
        Object keys(PyList_New(0)), values(PyList_New(0));
        for(const auto& entry : cache->entries) {
            Object name(PyUnicode_FromStringAndSize(entry.key.data(), entry.key.size()));
            if(PyList_Append(keys.p, name.p) < 0) throw std::bad_alloc();
            if(entry.probability && PyList_Append(values.p, name.p) < 0) throw std::bad_alloc();
        }
        return PyTuple_Pack(2, keys.p, values.p);
    } catch(...) { return error(); }
}
PyObject* certificate_cache_clear(PyObject*, PyObject* cap) {
    try {
        auto* cache = certificate_cache(cap);
        if(!cache) return nullptr;
        cache->clear();
        Py_RETURN_NONE;
    } catch(...) { return error(); }
}
// From bindings/module.hpp.
PyObject* scalar_test(PyObject*, PyObject* args) {
    try {
        PyObject *a, *b;
        if(!PyArg_ParseTuple(args, "OO", &a, &b)) return nullptr;
        auto x = scalar(a), y = scalar(b);
        Object sum(output_scalar(x + y)), prod(output_scalar(x * y));
        return Py_BuildValue("(OO)", sum.p, prod.p);
    } catch(...) { return error(); }
}
} // namespace
