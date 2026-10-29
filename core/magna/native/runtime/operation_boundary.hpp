// Existing operation-interface causal sampler over completed whole-circuit boundaries.
#pragma once
#include "branch.hpp"
#include "boundary_service.hpp"
#include "../foundation/bounded_cache.hpp"
#include "../compile/deferred.hpp"
#include "../compile/program.hpp"
namespace magic_native {
struct OperationBoundaryTask {
    NamedRows selection;
    std::vector<int> outcomes;
    std::vector<std::pair<Bits, Bits>> observables;
};
struct OperationBoundaryEngine {
    std::vector<Operation> operations;
    BoundaryTracePlan plan;
    BoundaryService service{&plan};
    OperationBoundaryEngine(std::vector<Operation> operations, const std::vector<size_t>& inputs,
            const std::vector<size_t>& outputs, const Ket& initial,
            const std::map<std::string, std::optional<std::vector<DeclaredGate>>>& declarations,
            const BoundaryLimits& limits, const PhaseDatabase& database)
        : operations(std::move(operations)) {
        auto deferred = defer_instrument(inputs, outputs, this->operations, declarations, {}, true, {});
        std::vector<ProgramOperation> program;
        for(const auto& op : deferred.operations) program.push_back(program_operation(op.value));
        std::vector<int> in(inputs.begin(), inputs.end()), out(outputs.begin(), outputs.end());
        for(const auto& [name, wire] : deferred.records) out.push_back(int(wire));
        auto compiled = compile_input(initial, in, out, program, {}, limits.paths);
        service.locations = std::move(compiled.locations);
        for(size_t j = 0; j < deferred.records.size(); j++)
            service.records[deferred.records[j].first] = outputs.size() + j;
        for(const auto& [name, id, wire] : deferred.report_anchors) service.report_anchors[name] = {id, int(wire)};
        plan.ir = std::move(compiled.ir);
        plan.limits = limits.phase;
        plan.database = database;
        plan.max_doubled = limits.doubled;
        plan.prepare();
    }
};
inline bool same_operations(const std::vector<Operation>& a, const std::vector<Operation>& b) {
    if(a.size() != b.size()) return false;
    for(size_t j = 0; j < a.size(); j++) {
        const auto& x = a[j];
        const auto& y = b[j];
        if(std::tie(x.id, x.kind, x.qubits, x.exponent, x.sign, x.record, x.pauli, x.paulis) !=
                std::tie(y.id, y.kind, y.qubits, y.exponent, y.sign, y.record, y.pauli, y.paulis))
            return false;
    }
    return true;
}
struct OperationBoundaryDraw {
    struct Stage {
        std::string id;
        std::vector<std::string> records;
        BoundarySampleRows sample;
    };
    Scalar mass{1}, acceptance;
    std::optional<Scalar> error;
    std::vector<Scalar> moments;
    std::vector<std::pair<std::string, int>> reports, truth;
    std::vector<Stage> stages;
    std::vector<BoundarySampleRows::Query> resources;
    size_t controls = 0;
};
struct OperationBoundarySampler {
    InstrumentLayout layout;
    Ket initial;
    std::vector<size_t> inputs, outputs;
    std::vector<Operation> operations;
    std::map<std::string, std::optional<std::vector<DeclaredGate>>> declarations, direct_rules;
    std::map<std::string, std::optional<NativeControl>> families;
    OperationBoundaryTask task;
    BoundaryLimits limits;
    PhaseDatabase database;
    std::vector<std::string> direct, complex, required;
    std::optional<BoundaryCoordinates> selection;
    BoundedCache<std::shared_ptr<OperationBoundaryEngine>> plans;
    std::mutex mutex;
    using Resolved = std::map<std::string, std::vector<Operation>>;
    using Callback =
            std::function<std::vector<Operation>(const std::string&, const std::vector<std::pair<std::string, int>>&)>;
    OperationBoundarySampler(InstrumentLayout layout, Ket initial, std::vector<size_t> inputs,
            std::vector<size_t> outputs, std::vector<Operation> operations,
            std::map<std::string, std::optional<std::vector<DeclaredGate>>> declarations,
            std::map<std::string, std::optional<NativeControl>> families, OperationBoundaryTask task,
            BoundaryLimits limits, PhaseDatabase database)
        : layout(std::move(layout)), initial(std::move(initial)), inputs(std::move(inputs)),
          outputs(std::move(outputs)), operations(std::move(operations)), declarations(std::move(declarations)),
          families(std::move(families)), task(std::move(task)), limits(std::move(limits)),
          database(std::move(database)) {
        if(!this->limits.plans) throw std::invalid_argument("positive plan cache bound required");
        for(const auto& [id, rule] : this->declarations)
            if(rule && is_affine_rule(*rule)) {
                direct.push_back(id);
                direct_rules[id] = rule;
            }
        std::set<std::string> needed;
        for(const auto& op : this->operations)
            if(op.kind == "CONTROL" && !direct_rules.count(op.id)) {
                complex.push_back(op.id);
                if(!this->declarations.at(op.id) && !this->families.at(op.id))
                    throw UnsupportedFeedback(
                            "complex callback needs a declared unitary PhaseFeedback family: " + op.id);
                const auto& parents = this->layout.controls.at(op.id).parents;
                needed.insert(parents.begin(), parents.end());
            }
        for(const auto& name : this->layout.record_names)
            if(needed.count(name)) required.push_back(name);
        selection.emplace(this->layout.record_names, this->task.selection, this->task.outcomes, required);
        for(const auto& [x, z] : this->task.observables)
            if(x.size() > this->outputs.size() || z.size() > this->outputs.size())
                throw std::invalid_argument("Pauli outside physical output");
    }
    std::vector<Operation> resolved_operations(const Resolved& resolved) const {
        std::vector<Operation> out;
        std::set<std::string> ids;
        for(const auto& op : operations) ids.insert(op.id);
        for(const auto& op : operations) {
            if(op.kind != "CONTROL" || direct_rules.count(op.id)) {
                out.push_back(op);
                continue;
            }
            auto found = resolved.find(op.id);
            std::vector<Operation> gates;
            if(found != resolved.end()) gates = found->second;
            else if(const auto& family = families.at(op.id); family) {
                for(size_t j = 0; j < family->axes.size(); j++)
                    if(family->baseline[j]) {
                        std::vector<size_t> qs(family->axes[j].begin(), family->axes[j].end());
                        gates.push_back(operation("PHASE", qs, family->baseline[j]));
                    }
            }
            for(size_t j = 0; j < gates.size(); j++) {
                auto id = "__boundary/" + op.id + "/" + std::to_string(j);
                while(ids.count(id)) id = "_" + id;
                ids.insert(id);
                gates[j].id = std::move(id);
                out.push_back(std::move(gates[j]));
            }
            auto anchor = operation("I");
            anchor.id = op.id;
            out.push_back(std::move(anchor));
        }
        return out;
    }
    std::shared_ptr<OperationBoundaryEngine> engine(const Resolved& resolved) {
        auto ops = resolved_operations(resolved);
        std::lock_guard<std::mutex> lock(mutex);
        if(auto* cached = plans.find([&](const auto& engine) { return same_operations(engine->operations, ops); }))
            return *cached;
        auto value = std::make_shared<OperationBoundaryEngine>(
                std::move(ops), inputs, outputs, initial, direct_rules, limits, database);
        plans.insert(value, limits.plans);
        return value;
    }
    std::pair<std::vector<Bits>, std::vector<int>> selected(
            const OperationBoundaryEngine& engine, const BoundaryCoordinates& coordinates) const {
        if(!coordinates.compatible) return {{Bits()}, {1}};
        std::vector<Bits> rows;
        for(size_t j = 0; j < coordinates.selection; j++) {
            std::vector<std::string> names;
            for(auto k : coordinates.transform[j].indices()) names.push_back(layout.record_names[k]);
            rows.push_back(named_record_row(engine.service.records, names));
        }
        return {rows, coordinates.values};
    }
    OperationBoundaryDraw sample(const std::vector<OriginalFault>& faults, const std::vector<std::string>& flips,
            const RandomBits& random, const Callback& callback) {
        std::vector<std::pair<std::string, int>> zeros;
        for(const auto& name : layout.record_names) zeros.push_back({name, 0});
        auto validated = layout.validate(zeros, flips, faults);
        OperationBoundaryDraw out;
        Resolved resolved;
        std::map<std::string, int> reports;
        auto known = [&]() {
            NamedRows rows;
            std::vector<int> values;
            for(const auto& [name, bit] : out.reports) {
                rows.push_back({name});
                values.push_back(bit);
            }
            return std::make_pair(rows, values);
        };
        for(const auto& id : complex) {
            const auto& node = layout.controls.at(id);
            std::vector<std::string> missing;
            for(const auto& name : node.parents)
                if(!reports.count(name)) missing.push_back(name);
            if(!missing.empty()) {
                auto plan = engine(resolved);
                auto packed = plan->service.pack(faults, flips);
                auto [names, values] = known();
                NamedRows pending;
                for(const auto& name : missing) pending.push_back({name});
                auto draw = sample_boundary_rows(
                        plan->plan, packed, plan->service.rows(pending), plan->service.rows(names), values, random);
                out.mass = draw.mass;
                for(size_t j = 0; j < missing.size(); j++) {
                    int bit = draw.word.bit(j);
                    if(!reports.count(missing[j])) out.reports.push_back({missing[j], bit});
                    reports[missing[j]] = bit;
                }
                out.stages.push_back({id, missing, std::move(draw)});
            }
            std::vector<std::pair<std::string, int>> parents;
            for(const auto& name : node.parents) parents.push_back({name, reports.at(name)});
            auto gates = declarations.at(id) ? evaluated_declared_operations(*declarations.at(id),
                                                       std::map<std::string, int>(parents.begin(), parents.end()))
                                             : callback(id, parents);
            for(const auto& gate : gates)
                if(gate.record || gate.kind == "CONTROL")
                    throw std::invalid_argument("control expansion must be measurement-free");
            if(families.at(id)) resolve_feedback_operations(*families.at(id), gates);
            resolved[id] = std::move(gates);
        }
        auto plan = engine(resolved);
        auto packed = plan->service.pack(faults, flips);
        auto [names, values] = known();
        auto known_rows = plan->service.rows(names);
        Workspace ws;
        auto total = [&](const std::vector<Bits>& rows, const std::vector<int>& rhs, const Bits& x = Bits(),
                             const Bits& z = Bits()) {
            auto start = std::chrono::steady_clock::now();
            auto result = plan->plan.query(packed, {}, rows, rhs, x, z, false, ws);
            auto value = boundary_sum(result.terms, {}, {}, ws);
            out.resources.push_back(measure_boundary_query(
                    result, 0, std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()));
            return value;
        };
        // Validated unitary feedback preserves the mass of the last sampled
        // history. With no observations the normalized history has unit mass.
        auto [rows, rhs] = selected(*plan, *selection);
        auto events = known_rows;
        events.insert(events.end(), rows.begin(), rows.end());
        auto bits = values;
        bits.insert(bits.end(), rhs.begin(), rhs.end());
        out.acceptance = total(events, bits);
        if(real_sign(out.acceptance) < 0 || real_sign(out.mass + out.acceptance * Scalar(-1)) < 0)
            throw std::runtime_error("postselection mass outside history mass");
        if(out.acceptance.zero()) out.moments.resize(task.observables.size());
        else
            for(const auto& [x, z] : task.observables) out.moments.push_back(total(events, bits, x, z));
        for(const auto& [name, bit] : out.reports)
            out.truth.push_back({name, bit ^ int(validated.flip_word.bit(layout.record_index.at(name)))});
        out.controls = resolved.size();
        return out;
    }
};
} // namespace magic_native
