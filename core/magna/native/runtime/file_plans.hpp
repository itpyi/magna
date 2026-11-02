// Source protocol, run budgets and the per-stage prepared boundary plans of a
// whole fixed-input file experiment.
#pragma once
#include "../compile/file_program.hpp"
#include "causal.hpp"
#include "../evaluate/boundary_trace.hpp"
#include "../evaluate/reference_network.hpp"
#include "noise_compile.hpp"
#include "file_statistics.hpp"
#include "../foundation/bounded_cache.hpp"
#include <chrono>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
namespace magic_native {
struct RuntimeLimits : BoundaryLimits {
    bool capped_affine = true;
    std::optional<size_t> queries;
    std::optional<double> seconds;
};
struct RunContext {
    std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
    size_t queries = 0, query_base = 0, compilations = 0, shared_path_gates = 0;
    double compile_seconds = 0, kernel_seconds = 0, decoder_seconds = 0;
    size_t support_updates = 0, support_geometry_updates = 0, support_geometry_reuses = 0, support_term_visits = 0,
           support_tensor_expansions = 0;
    double support_seconds = 0;
    size_t completed_history_reuses = 0, target_frame_reuses = 0;
    std::map<std::string, size_t> reference_network_evaluations;
    size_t reference_network_preparations = 0, reference_network_resource_constructions = 0,
           reference_network_coordinate_constructions = 0;
    // Budgets count from the start of the current run; counters stay cumulative for reporting.
    void begin_run() {
        started = std::chrono::steady_clock::now();
        query_base = queries;
    }
    void check(const RuntimeLimits& limits) const {
        if(limits.seconds &&
                std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >= *limits.seconds)
            throw ResourceLimitError(
                    "run reached its time budget between quantum queries", "max_seconds", *limits.seconds);
        if(limits.queries && queries - query_base >= *limits.queries)
            throw ResourceLimitError("run reached its quantum-query budget", "max_queries", double(*limits.queries));
    }
};
struct FileBoundary {
    std::shared_ptr<BoundaryTracePlan> plan = std::make_shared<BoundaryTracePlan>();
    std::map<std::string, size_t> locations, records;
    std::map<std::string, std::pair<std::string, size_t>> reports;
    DeferredProgram deferred;
    std::vector<size_t> inputs;
    std::vector<size_t> outputs;
    std::map<std::string, std::shared_ptr<const ReferenceNetwork>> reference_networks;
    std::map<std::string, ReferenceRecordMap> reference_records;
    Frame feedback_constant;
    std::vector<Frame> protocol_frames, recovery_frames;
    explicit FileBoundary(const FileProgram& program, const RuntimeLimits& limits, const PhaseDatabase& database,
            CompilerPathPrefix* prefix = nullptr) {
        auto compiled = program.compile(limits.paths);
        locations = std::move(compiled.second.locations);
        for(size_t j = 0; j < compiled.first.records.size(); j++)
            records[compiled.first.records[j].first] = program.outputs.size() + j;
        for(const auto& [record, anchor, wire] : compiled.first.report_anchors) reports[record] = {anchor, wire};
        deferred = std::move(compiled.first);
        inputs = program.inputs;
        outputs = program.outputs;
        plan->ir = std::move(compiled.second.ir);
        plan->limits = limits.phase;
        plan->max_doubled = limits.doubled;
        plan->database = database;
        plan->prepare(prefix, prefix && !prefix->gates.empty());
    }
    std::vector<Bits> rows(const NamedRows& names) const {
        std::vector<Bits> result;
        for(const auto& names_row : names) {
            Bits row;
            for(const auto& name : names_row) row.flip(records.at(name));
            result.push_back(std::move(row));
        }
        return result;
    }
    Assignment pack(const std::vector<FeedbackFault>& faults, const std::vector<std::string>& flips) const {
        Assignment result;
        for(const auto& fault : faults) {
            auto code = std::string("IXZY").find(fault.pauli);
            if(code == std::string::npos) throw std::invalid_argument("invalid original fault");
            result.faults.push_back({locations.at(fault.anchor), int(fault.wire), int(code)});
        }
        for(const auto& name : flips) {
            const auto& [anchor, wire] = reports.at(name);
            result.faults.push_back({locations.at(anchor), int(wire), 1});
        }
        return result;
    }
    // Suffix images of each template noise site's targets (a report site: its
    // reported wire), resolved once on a boundary without continuation. Every
    // site resolves: file qubits stay live through the noisy protocol body, and
    // recovery and target circuits are noiseless. Drawn template actions XOR
    // these images directly; explicit actions keep the named path.
    std::vector<std::array<const Suffix*, 2>> noise_images;
    void prepare_noise_images(const FileNoiseActions& actions) {
        using Template = FileNoiseActions::Template;
        const auto& transport = plan->compiled.plan;
        noise_images.assign(actions.sites().size(), {nullptr, nullptr});
        // Consecutive sites of one instruction share its anchor and location.
        const std::string* anchor = nullptr;
        size_t location = 0;
        for(size_t j = 0; j < noise_images.size(); j++) {
            const auto& site = actions.sites()[j];
            auto& images = noise_images[j];
            if(site.channel == Template::Report) {
                const auto& [reported, wire] = reports.at(site.report);
                images[0] = &transport.fault_image(locations.at(reported), int(wire));
                continue;
            }
            if(!anchor || *anchor != site.anchor) {
                location = locations.at(site.anchor);
                anchor = &site.anchor;
            }
            images[0] = &transport.fault_image(location, int(site.targets[0]));
            if(site.channel == Template::Two) images[1] = &transport.fault_image(location, int(site.targets[1]));
        }
    }
    // XORs the untracked-phase frame of drawn action-table indices into frame.
    void transport_noise(const FileNoiseActions& actions, const std::vector<int64_t>& drawn, Frame& frame) const {
        std::vector<FeedbackFault> named;
        std::vector<std::string> flips;
        for(auto index : drawn) {
            auto j = size_t(index);
            // A boundary with a continuation has no image table.
            if(!noise_images.empty() && !actions.custom(j)) {
                auto [site, paulis] = actions.choice(j);
                const auto& images = noise_images[site];
                if(actions.sites()[site].channel == FileNoiseActions::Template::Report)
                    Plan::xor_fault(*images[0], 1, frame);
                else
                    for(size_t k = 0; k < 2; k++)
                        if(paulis[k] != 'I')
                            Plan::xor_fault(*images[k], paulis[k] == 'X' ? 1 : paulis[k] == 'Z' ? 2 : 3, frame);
                continue;
            }
            auto action = actions[j];
            if(!action.report.empty()) flips.push_back(action.report);
            else
                for(const auto& [pauli, wire] : action.gates) named.push_back({action.anchor, wire, pauli});
        }
        if(named.empty() && flips.empty()) return;
        auto delta = plan->assignment_transport().fault_frame(pack(named, flips).faults, false);
        frame.x ^= delta.x;
        frame.z ^= delta.z;
    }
};
// Named actions and report flips, as supplied or enumerated. A native draw keeps
// its action-table indices in drawn instead, which prepared boundaries transport.
struct ActualNoise {
    NoiseActions actions;
    std::vector<std::string> flips;
    std::vector<int64_t> drawn;
    FileFraction weight{Integer(1)};
    std::optional<ScientificValue> scientific_weight;
};
inline ActualNoise noise_actions(const FileNoisePlan& plan, const FileNoiseDraw& selected) {
    ActualNoise out;
    out.weight = selected.weight;
    out.scientific_weight = selected.scientific_weight;
    for(auto index : selected.selected) {
        const auto& action = plan.actions.at(size_t(index));
        if(!action.report.empty()) out.flips.push_back(action.report);
        else {
            auto& gates = out.actions[action.anchor];
            gates.insert(gates.end(), action.gates.begin(), action.gates.end());
        }
    }
    return out;
}
struct FileSource {
    CircuitData protocol;
    std::optional<CircuitData> recovery;
    std::optional<IdealTarget> target;
    std::vector<size_t> outputs;
    FeedbackTemplate feedback;
    std::optional<FeedbackTemplate> recovery_feedback;
    FileControlMetadata controls;
    std::optional<FileControlMetadata> recovery_controls;
    std::set<std::string> noise_ids, guarded_ids;
    FileSource(CircuitData circuit, std::vector<size_t> outputs, std::optional<CircuitData> terminal,
            std::optional<IdealTarget> ideal)
        : protocol(std::move(circuit)), recovery(std::move(terminal)), target(std::move(ideal)),
          outputs(std::move(outputs)), feedback(protocol), controls(protocol) {
        if(recovery) {
            validate_recovery(*recovery);
            recovery_feedback.emplace(*recovery, this->outputs.size());
            recovery_controls.emplace(*recovery);
        }
        for(const auto& inst : protocol.instructions)
            if(file_noise_instruction(inst.name)) (inst.guard ? guarded_ids : noise_ids).insert(inst.op_id);
    }
    std::string stage(std::string value) const { return value == "recovery" && !recovery ? "protocol" : value; }
    // The target projection records follow the protocol and recovery records.
    std::vector<std::string> target_records() const {
        std::vector<std::string> names;
        size_t first = protocol.num_records + (recovery ? recovery->num_records : 0);
        for(size_t j = 0; j < (target ? target->preparation_width : 0); j++)
            names.push_back("m" + std::to_string(first + j));
        return names;
    }
    std::vector<FeedbackFault> feedback_faults(const DecisionValues& values, const DecisionValues& recovery_values,
            const ActualNoise&, const std::string& stage) const {
        auto result = feedback.faults(values);
        if(stage != "protocol" && recovery) {
            std::map<size_t, size_t> wires;
            for(size_t j = 0; j < outputs.size(); j++) wires[j] = outputs[j];
            for(size_t j = outputs.size(); j < recovery_feedback->wire_count; j++)
                wires[j] = feedback.wire_count + j - outputs.size();
            auto extra = recovery_feedback->faults(recovery_values, "__recovery/", wires);
            result.insert(result.end(), extra.begin(), extra.end());
        }
        return result;
    }
    // A caller-supplied assignment is checked here once; drawn assignments are valid by construction.
    void validate_noise(const ActualNoise& noise) const {
        for(const auto& action : noise.actions) {
            const auto& anchor = action.first;
            const auto& gates = action.second;
            auto inst = std::find_if(protocol.instructions.begin(), protocol.instructions.end(),
                    [&](const FileInstruction& item) { return item.op_id == anchor; });
            if(inst == protocol.instructions.end() || !file_noise_instruction(inst->name))
                throw std::invalid_argument("noise anchor '" + anchor + "' is not a noise instruction of this circuit");
            std::set<size_t> wires;
            for(const auto& target : inst->targets)
                if(target.kind == "qubit" || target.kind == "pauli") wires.insert(target.index);
            for(const auto& [pauli, wire] : gates)
                if(!wires.count(wire))
                    throw std::invalid_argument("noise at '" + anchor + "' acts on qubit " + std::to_string(wire) +
                            ", which that instruction does not target");
        }
        for(const auto& name : noise.flips) {
            bool record = name.size() > 1 && name[0] == 'm' &&
                    name.find_first_not_of("0123456789", 1) == std::string::npos && name.size() < 20 &&
                    std::stoull(name.substr(1)) < protocol.num_records;
            if(!record)
                throw std::invalid_argument("report flip '" + name + "' is not a measurement record of this circuit");
        }
    }
    std::vector<FeedbackFault> original_faults(const ActualNoise& noise) const {
        std::vector<FeedbackFault> result;
        for(const auto& [anchor, gates] : noise.actions)
            for(const auto& [pauli, wire] : gates) result.push_back({anchor, wire, pauli});
        return result;
    }
    std::vector<FeedbackFault> faults(const DecisionValues& values, const DecisionValues& recovery_values,
            const ActualNoise& noise, const std::string& stage) const {
        auto result = feedback_faults(values, recovery_values, noise, stage), original = original_faults(noise);
        result.insert(result.end(), original.begin(), original.end());
        return result;
    }
};
class PlanRepository {
    struct Alias {
        std::string key;
        std::shared_ptr<FileBoundary> value;
    };
    struct Structure {
        std::string key;
        std::weak_ptr<FileBoundary> value;
    };
    const FileSource& source;
    const RuntimeLimits& limits;
    const PhaseDatabase& database;
    RunContext& context;
    BoundedCache<Alias> aliases;
    BoundedCache<Structure> structures;

public:
    std::map<std::string, std::shared_ptr<FileBoundary>> prepared;
    PlanRepository(
            const FileSource& source, const RuntimeLimits& limits, const PhaseDatabase& database, RunContext& context)
        : source(source), limits(limits), database(database), context(context) {}
    std::shared_ptr<FileBoundary> get(std::string stage, CompilerPathPrefix* prefix = nullptr) {
        context.check(limits);
        stage = source.stage(stage);
        auto ready = prepared.find(stage);
        if(ready != prepared.end()) return ready->second;
        if(auto* alias = aliases.find([&](const Alias& row) { return row.key == stage; })) return alias->value;
        FileProgram program(source.protocol, source.outputs, source.feedback,
                source.recovery ? &*source.recovery : nullptr,
                source.recovery_feedback ? &*source.recovery_feedback : nullptr,
                source.target ? &*source.target : nullptr, stage);
        auto identity = program.identity();
        auto* structure = structures.find([&](const Structure& row) { return row.key == identity; });
        auto value = structure ? structure->value.lock() : nullptr;
        if(!value) {
            auto tick = std::chrono::steady_clock::now();
            value = std::make_shared<FileBoundary>(program, limits, database, prefix);
            context.shared_path_gates += value->plan->compiled.shared_path_gates;
            context.compile_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - tick).count();
            context.compilations++;
        }
        if(structure) structure->value = value;
        else structures.insert({std::move(identity), value}, limits.plans);
        aliases.insert({stage, value}, limits.plans);
        return value;
    }
    size_t size() const { return aliases.size(); }
    size_t structure_size() const { return structures.size(); }
    void clear() {
        prepared.clear();
        aliases.clear();
        structures.clear();
    }
};
} // namespace magic_native
