// Native ownership and execution of whole fixed-input file query families.
#pragma once
#include "file_plans.hpp"
#include "affine_proposal.hpp"
#include "observation.hpp"
#include "sampling.hpp"
#include "marginal.hpp"
#include <atomic>
#include <functional>
#include <thread>
namespace magic_native {
struct HistoryTask {
    QueryTask boundaries;
    std::vector<int> outcomes;
};
struct HistoryMasses {
    Scalar history, acceptance;
    std::optional<Scalar> error, target_success;
    std::vector<Scalar> moments;
    // Terminal cap rows in the sampling parent's certified label basis.
    std::map<std::string, std::vector<Bits>> completed_caps;
    Scalar target_normalization{1};
    std::vector<std::optional<Scalar>> exact_row() const {
        std::vector<std::optional<Scalar>> row{history, acceptance, error};
        for(const auto& v : moments) row.push_back(v);
        return row;
    }
};
struct HistoryResult {
    HistoryMasses masses;
    ActualNoise noise;
    std::vector<int> samples;
    std::optional<Integer> proposal_denominator;
    size_t decoder_calls = 0, recovery_decoder_calls = 0, sampled_parities = 0;
    HistoryContribution contribution() const {
        const auto& m = masses;
        std::vector<std::optional<Scalar>> numerators{m.acceptance * m.target_normalization, m.error};
        for(const auto& value : m.moments) numerators.push_back(value * m.target_normalization);
        auto inverse = proposal_denominator ? file_integer_scalar(*proposal_denominator) : Scalar(1);
        if(proposal_denominator)
            for(auto& value : numerators)
                if(value) *value = *value * inverse;
        auto result = history_contribution((proposal_denominator ? Scalar(1) : m.history) * m.target_normalization,
                numerators, noise.weight, noise.scientific_weight);
        if(proposal_denominator)
            result.weight = scaled_real_ratio(
                    m.history * inverse * file_integer_scalar(noise.weight.n), file_integer_scalar(noise.weight.d));
        return result;
    }
};
struct ExecutionHooks {
    std::function<uint64_t()> seed;
    std::function<std::vector<int>(const std::string&, size_t, const std::vector<int>&, size_t)> controller;
    std::function<Bits(const StabilizerTerms&, size_t)> branch;
};
struct CompiledExperiment {
    // One thread at a time uses a compiled experiment; nested use on that thread
    // (a decoder calling back into the same task) is allowed. Language bindings
    // hold an ExclusiveUse around every entry point that reads or updates it.
    std::atomic<std::thread::id> user{};
    size_t user_depth = 0;
    struct ExclusiveUse {
        CompiledExperiment& run;
        explicit ExclusiveUse(CompiledExperiment& value) : run(value) {
            auto self = std::this_thread::get_id();
            auto none = std::thread::id();
            if(run.user.load() == self) {
                run.user_depth++;
                return;
            }
            if(!run.user.compare_exchange_strong(none, self))
                throw std::runtime_error(
                        "this compiled circuit is in use by another thread; compile one circuit per thread");
            run.user_depth = 1;
        }
        ~ExclusiveUse() {
            if(--run.user_depth == 0) run.user.store(std::thread::id());
        }
        ExclusiveUse(const ExclusiveUse&) = delete;
        ExclusiveUse& operator=(const ExclusiveUse&) = delete;
    };
    FileSource source;
    HistoryTask task;
    // The public noise schedule and execution share a prepared, immutable plan.
    // The reference keeps draw-time access unchanged; the owner pins its lifetime.
    std::shared_ptr<const FileNoisePlan> noise_owner;
    const FileNoisePlan& noise;
    RuntimeLimits limits;
    PhaseDatabase database;
    RunContext context;
    PlanRepository plans;
    FileQueryFamily family;
    NamedRows sector_rows;
    Ket sector_support;
    std::unique_ptr<SamplingNetworkPlan> history;
    bool proposal = false;
    // Terminal cap rows in the sampling parent's certified label basis.
    std::map<std::string, std::vector<Bits>> completed_caps;
    // Fixed causal observation schedule; draws supply only reported values.
    ObservationTemplate observation;
    Scalar target_normalization{1};
    bool target_normalization_ready = false;
    std::map<std::string, std::vector<std::pair<std::string, std::shared_ptr<const CompiledNetwork>>>> resources;
    bool complete = false;
    bool nonlinear() const {
        return !source.controls.callbacks.empty() ||
                (source.recovery_controls && !source.recovery_controls->callbacks.empty());
    }
    // Arithmetic payload retained by the prepared networks, proposal and transport, in bytes.
    struct PayloadReport {
        size_t persistent_networks = 0, networks = 0, proposal = 0, protocol_transport = 0, branch_transport = 0,
               branch_locations = 0;
    };
    PayloadReport payload_report() const {
        PayloadReport out;
        NetworkPayload payload;
        for(const auto& [stage, engine] : plans.prepared)
            for(const auto& [spec, cached] : engine->plan->networks) {
                auto before = payload.shared.size();
                out.networks += payload.network(*cached.network);
                out.persistent_networks += payload.shared.size() != before;
            }
        out.proposal = history ? history->payload_bytes() : 0;
        for(const auto& [purpose, rows] : completed_caps) out.proposal += NetworkPayload::rows(rows);
        auto continuation_bytes = [](const auto& map) {
            size_t bytes = 24;
            for(const auto* images : {&map.x, &map.z})
                for(const auto& image : *images) bytes += image.x.payload_bytes() + image.z.payload_bytes() + 4;
            return bytes;
        };
        for(const auto& [stage, engine] : plans.prepared)
            if(engine->plan->compiled.plan.continuation)
                out.proposal += continuation_bytes(*engine->plan->compiled.plan.continuation);
        std::set<const Plan*> counted;
        for(const auto& [stage, engine] : plans.prepared) {
            const auto& plan = engine->plan->compiled.plan;
            if(!counted.insert(&plan).second) continue;
            if(plan.continuation) {
                out.branch_locations += plan.locations.size();
                out.branch_transport += continuation_bytes(*plan.continuation);
                continue;
            }
            size_t transport = plan.locations.size() * sizeof(Location);
            for(const auto& [wire, history] : plan.histories)
                transport += sizeof(wire) + history.size() * sizeof(std::pair<int, int>);
            for(const auto& [wire, suffix] : plan.suffix) {
                transport += sizeof(wire);
                for(const auto& image : suffix)
                    transport += sizeof(image.position) + image.x.x.payload_bytes() + image.x.z.payload_bytes() +
                            image.z.x.payload_bytes() + image.z.z.payload_bytes() + 8;
            }
            transport += engine->noise_images.size() * sizeof(engine->noise_images[0]);
            out.protocol_transport += transport;
        }
        return out;
    }
    QueryResourceSummary resource_summary() const {
        QueryResourceSummary summary;
        for(const auto& [stage, queries] : resources) {
            for(const auto& [purpose, network] : queries) summary.include(*network);
        }
        return summary;
    }
    CompiledExperiment(FileSource source, HistoryTask task, std::shared_ptr<const FileNoisePlan> noise,
            RuntimeLimits limits, PhaseDatabase database)
        : source(std::move(source)), task(std::move(task)), noise_owner(std::move(noise)), noise(*noise_owner),
          limits(std::move(limits)), database(std::move(database)),
          plans(this->source, this->limits, this->database, context) {
        if(this->task.boundaries.has_target) this->task.boundaries.target_records = this->source.target_records();
        const auto& input = this->task;
        validate_trace_outcomes(input.boundaries.selection.size(), input.outcomes);
        family = plan_file_queries(this->source.controls,
                this->source.recovery_controls ? &*this->source.recovery_controls : nullptr, this->task.boundaries);
    }
    void prepare() {
        prepare_target_normalization();
        auto stage = source.stage("recovery");
        std::shared_ptr<FileBoundary> engine, target;
        {
            CompilerPathPrefix prefix;
            engine = plans.get(stage, &prefix);
            plans.prepared[stage] = engine;
            if(source.target) {
                target = plans.get("target", &prefix);
                plans.prepared["target"] = target;
            }
        }
        prepare_sector(*engine, target.get());
        prepare_reference_network(*engine, "acceptance");
        if(needs_history()) {
            auto sampling = prepare_reference_network(*engine, "sampling");
            auto tick = std::chrono::steady_clock::now();
            history = std::make_unique<SamplingNetworkPlan>(sampling);
            proposal = limits.capped_affine;
            if(proposal) {
                for(auto& item : resources[stage])
                    if(item.first == "sampling") item.first = "sampling_capped";
            }
            context.compile_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - tick).count();
        }
        if(target) prepare_reference_network(*target, "fidelity");
        if(history) {
            auto prepare_caps = [&](FileBoundary& terminal, const std::string& purpose) {
                const auto& parent = *terminal.reference_networks.at(purpose);
                std::map<size_t, size_t> records;
                for(const auto& [name, index] : terminal.records) {
                    auto found = engine->records.find(name);
                    if(found != engine->records.end()) records[index] = found->second;
                }
                auto& caps = completed_caps[purpose];
                caps.clear();
                for(size_t j = 0; j < parent.closed.size() + parent.capped_history; j++) {
                    Bits physical;
                    parent.labels.basis[j].each([&](size_t bit) { physical.flip(records.at(bit)); });
                    auto coordinates = history->parent->labels(physical);
                    if(!coordinates) throw std::logic_error("terminal history outside sampling label span");
                    caps.push_back(std::move(*coordinates));
                }
            };
            prepare_caps(*engine, "acceptance");
            if(target) prepare_caps(*target, "fidelity");
            auto tick = std::chrono::steady_clock::now();
            observation = prepare_observation(*history->parent, engine->reference_records.at("sampling"),
                    source.controls, source.recovery_controls ? &*source.recovery_controls : nullptr,
                    source.protocol.num_records, task.boundaries.sample_records, completed_caps);
            context.compile_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - tick).count();
        }
        for(const auto& [stage, engine] : plans.prepared) {
            if(!engine->plan->compiled.plan.continuation) {
                prepare_feedback_frames(*engine);
                if(noise.has_actions) {
                    auto tick = std::chrono::steady_clock::now();
                    engine->prepare_noise_images(noise.actions);
                    context.compile_seconds +=
                            std::chrono::duration<double>(std::chrono::steady_clock::now() - tick).count();
                }
            }
            engine->plan->finish_preparation();
        }
        complete = true;
    }

private:
    void prepare_feedback_frames(FileBoundary& engine) {
        auto tick = std::chrono::steady_clock::now();
        auto transport = [&](const std::vector<FeedbackFault>& faults) {
            return engine.plan->compiled.plan.fault_frame(engine.pack(faults, {}).faults, false);
        };
        engine.feedback_constant = transport(source.feedback_faults({}, {}, {}, "recovery"));
        auto prepare = [&](const FeedbackTemplate& feedback, const FileControlMetadata& controls,
                               const std::string& prefix, const std::map<size_t, size_t>* wires) {
            std::vector<Frame> frames;
            frames.reserve(controls.decision_order.size());
            for(const auto& decision : controls.decision_order) {
                std::vector<FeedbackFault> faults;
                auto found = feedback.consumers.find(decision);
                if(found != feedback.consumers.end())
                    for(auto j : found->second) {
                        const auto& update = feedback.updates[j];
                        faults.push_back(
                                {prefix + update.anchor, wires ? wires->at(update.wire) : update.wire, update.pauli});
                    }
                frames.push_back(transport(faults));
            }
            return frames;
        };
        engine.protocol_frames = prepare(source.feedback, source.controls, "", nullptr);
        if(source.recovery) {
            std::map<size_t, size_t> wires;
            for(size_t j = 0; j < source.outputs.size(); j++) wires[j] = source.outputs[j];
            for(size_t j = source.outputs.size(); j < source.recovery_feedback->wire_count; j++)
                wires[j] = source.feedback.wire_count + j - source.outputs.size();
            engine.recovery_frames =
                    prepare(*source.recovery_feedback, *source.recovery_controls, "__recovery/", &wires);
        }
        context.compile_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - tick).count();
    }
    void evaluate_fidelity(HistoryMasses& result, const ActualNoise& noise, const DecisionValues& values,
            const DecisionValues& recovery_values, const NamedRows& known, const std::vector<int>& outcomes,
            const Bits& sector, const std::vector<Bits>* numerical = nullptr, const Frame* supplied_frame = nullptr,
            SamplingNetworkSession* session = nullptr) {
        auto target = plans.get("target");
        Frame local;
        if(!supplied_frame) {
            local = actual_frame(
                    *target, target->plan->assignment_transport(), noise, values, recovery_values, "target");
            supplied_frame = &local;
        }
        result.target_normalization = target_normalization;
        auto rows = session ? ReferenceRows{} : history_coordinates(*target, "fidelity", known, numerical);
        result.target_success =
                (session ? completed_total(*target, "fidelity", *supplied_frame, *session)
                         : reference_total_coordinates(*target, "fidelity", *supplied_frame, rows, outcomes, sector)) *
                Scalar(1, int64_t(source.target->traced.size()));
        result.error = result.acceptance * target_normalization + *result.target_success * Scalar(-1);
    }
    void constrain_sector(FileBoundary& boundary, const NamedRows& history, std::vector<Bits>& constraints,
            const BoundaryTracePlan* prefix = nullptr) {
        auto rows = boundary.rows(history);
        boundary.plan->prepare_axes(prefix);
        if(!boundary.plan->common_axes) {
            for(size_t j = 0; j < history.size(); j++) constraints.push_back(Bits::unit(j));
            return;
        }
        for(const auto& axis : boundary.plan->axes) {
            Bits anti;
            for(size_t j = 0; j < rows.size(); j++)
                if(axis.x.parity_and(rows[j])) anti.set(j);
            constraints.push_back(std::move(anti));
        }
    }
    void prepare_target_normalization() {
        if(source.target && !target_normalization_ready) {
            const auto& ideal = *source.target;
            if(!ideal.postselection.empty()) {
                std::vector<size_t> outputs(ideal.preparation_width);
                std::iota(outputs.begin(), outputs.end(), 0);
                FeedbackTemplate feedback(ideal.preparation);
                FileProgram program(ideal.preparation, outputs, feedback, nullptr, nullptr, nullptr, "protocol");
                FileBoundary boundary(program, limits, database);
                std::vector<Bits> caps;
                std::vector<int> values;
                for(auto [q, bit] : ideal.postselection) {
                    caps.push_back(Bits::unit(q));
                    values.push_back(bit);
                }
                Workspace ws;
                auto reference = boundary.plan->query({}, {}, caps, values, {}, {}, false, ws);
                target_normalization = boundary_sum(reference.terms, {}, {}, ws);
                if(real_sign(target_normalization) <= 0)
                    throw std::invalid_argument("target postselection has zero probability");
            }
            target_normalization_ready = true;
        }
    }

public:
    bool needs_history() const {
        return nonlinear() || !source.guarded_ids.empty() || !task.boundaries.sample_records.empty();
    }
    void prepare_sector(FileBoundary& engine, FileBoundary* target) {
        sector_rows.clear();
        sector_support = {};
        if(!needs_history()) return;
        const auto& history = family.terminal_history;
        std::vector<Bits> constraints;
        constrain_sector(engine, history, constraints);
        if(target) constrain_sector(*target, history, constraints, engine.plan.get());
        for(const auto& mask : Factor(constraints, history.size()).kernel) {
            std::set<std::string> names;
            for(auto j : mask.indices())
                for(const auto& name : history[j])
                    if(!names.erase(name)) names.insert(name);
            sector_rows.emplace_back(names.begin(), names.end());
        }
        if(!sector_rows.empty()) sector_support = engine.plan->affine_control_support(engine.rows(sector_rows));
    }
    std::shared_ptr<const ReferenceNetwork> prepare_reference_network(
            FileBoundary& engine, const std::string& purpose) {
        auto found = engine.reference_networks.find(purpose);
        if(found != engine.reference_networks.end()) return found->second;
        auto tick = std::chrono::steady_clock::now();
        auto kept = family.terminal_history;
        NamedRows events;
        std::vector<int> outcomes;
        if(purpose != "sampling") {
            events = task.boundaries.selection;
            outcomes = task.outcomes;
        }
        if(purpose == "fidelity") {
            for(const auto& record : task.boundaries.target_records) {
                events.push_back({record});
                outcomes.push_back(0);
            }
        }
        auto reference_network =
                std::make_shared<const ReferenceNetwork>(*engine.plan, engine.rows(kept), engine.rows(events), outcomes,
                        engine.rows(sector_rows), purpose == "sampling" ? 0 : family.terminal_history.size());
        std::vector<size_t> record_indices;
        size_t count = source.protocol.num_records + (source.recovery ? source.recovery->num_records : 0);
        record_indices.reserve(count);
        for(size_t j = 0; j < count; j++) record_indices.push_back(engine.records.at("m" + std::to_string(j)));
        engine.reference_records.emplace(purpose,
                ReferenceRecordMap(reference_network, std::move(record_indices), engine.rows(family.terminal_history)));
        engine.reference_networks[purpose] = reference_network;
        context.reference_network_preparations++;
        auto stage = purpose == "fidelity" ? "target" : source.stage("recovery");
        if(!complete) resources[stage].push_back({purpose, reference_network->reference});
        context.compile_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - tick).count();
        return reference_network;
    }
    ReferenceRows history_coordinates(FileBoundary& engine, const std::string& purpose, const NamedRows& names,
            const std::vector<Bits>* numerical) {
        auto parent = prepare_reference_network(engine, purpose);
        return numerical ? engine.reference_records.at(purpose)(*numerical)
                         : ReferenceRows{parent->coordinates(engine.rows(names))};
    }
    BoundaryResult reference_query_coordinates(FileBoundary& engine, const std::string& purpose, const Frame& frame,
            const ReferenceRows& keep, const ReferenceRows& events, const std::vector<int>& outcomes,
            const Bits& sector = {}) {
        auto reference_network = prepare_reference_network(engine, purpose);
        context.check(limits);
        context.queries++;
        auto tick = std::chrono::steady_clock::now();
        Workspace ws;
        auto result = reference_network->query_coordinates(*engine.plan, frame, keep, events, outcomes, ws, sector);
        context.reference_network_evaluations[purpose]++;
        context.reference_network_resource_constructions += result.diagnostics.resource_constructions;
        context.reference_network_coordinate_constructions += result.diagnostics.phase_coordinate_constructions;
        context.kernel_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - tick).count();
        engine.plan->finish_preparation();
        return result;
    }
    Scalar reference_total_coordinates(FileBoundary& engine, const std::string& purpose, const Frame& frame,
            const ReferenceRows& events, const std::vector<int>& outcomes, const Bits& sector = {}) {
        auto result = reference_query_coordinates(engine, purpose, frame, {}, events, outcomes, sector);
        Workspace ws;
        return boundary_sum(result.terms, {}, {}, ws);
    }
    Scalar completed_total(
            FileBoundary& engine, const std::string& purpose, const Frame& frame, SamplingNetworkSession& session) {
        context.check(limits);
        context.queries++;
        auto tick = std::chrono::steady_clock::now();
        Workspace ws;
        const auto& parent = *engine.reference_networks.at(purpose);
        context.completed_history_reuses++;
        Scalar total;
        if(!session.zero && !session.committed.boundary.zero) {
            auto caps = parent.selection;
            for(const auto& recipe : observation.caps.at(purpose)) {
                if(!recipe) throw std::invalid_argument("terminal reference requires the complete causal history");
                caps.push_back(session.committed.value(*recipe));
            }
            auto result = parent.reference->evaluate(frame, engine.plan->compiled.plan.magic, caps, false, ws,
                    engine.plan->limits, &engine.plan->database, true,
                    purpose != "fidelity" || engine.plan->compiled.plan.continuation ? &session.common : nullptr);
            total = boundary_sum(result.terms, {}, {}, ws);
            context.reference_network_resource_constructions += result.diagnostics.resource_constructions;
            context.reference_network_coordinate_constructions += result.diagnostics.phase_coordinate_constructions;
        }
        context.reference_network_evaluations[purpose]++;
        context.kernel_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - tick).count();
        return total;
    }
    void configure(RuntimeLimits updated) {
        auto key = [](const RuntimeLimits& v) {
            return std::make_tuple(v.paths, v.doubled, v.phase.max_quotient_rank, v.phase.max_terms,
                    v.phase.max_table_bytes, v.capped_affine);
        };
        if(key(limits) != key(updated)) {
            history.reset();
            observation = {};
            proposal = false;
            plans.clear();
            resources.clear();
            complete = false;
        }
        limits = std::move(updated);
    }
    ActualNoise draw(const ExecutionHooks& hooks) const {
        if(noise.categories.empty()) return {};
        auto selected = draw_noise(noise, hooks.seed(), true, true);
        ActualNoise out;
        out.drawn = std::move(selected.selected);
        out.weight = std::move(selected.weight);
        out.scientific_weight = selected.scientific_weight;
        return out;
    }
    BoundaryResult query(const FileBoundary& engine, const Assignment& assignment, const NamedRows& keep,
            const NamedRows& events, const std::vector<int>& outcomes, const Bits& x = {}, const Bits& z = {},
            const Frame* frame = nullptr) {
        context.check(limits);
        context.queries++;
        auto tick = std::chrono::steady_clock::now();
        Workspace ws;
        auto result = engine.plan->query(
                assignment, engine.rows(keep), engine.rows(events), outcomes, x, z, false, ws, true, frame);
        auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - tick).count();
        context.kernel_seconds += elapsed;
        return result;
    }
    Scalar total(const FileBoundary& engine, const Assignment& assignment, const NamedRows& events,
            const std::vector<int>& outcomes, const Bits& x = {}, const Bits& z = {}, const Frame* frame = nullptr) {
        auto result = query(engine, assignment, {}, events, outcomes, x, z, frame);
        Workspace ws;
        return boundary_sum(result.terms, {}, {}, ws);
    }
    // Feedback and named noise only; drawn indices are transported by actual_frame.
    Assignment assignment(const FileBoundary& engine, const ActualNoise& noise, const DecisionValues& values,
            const DecisionValues& recovery_values, const std::string& stage) const {
        return engine.pack(source.faults(values, recovery_values, noise, source.stage(stage)), noise.flips);
    }
    Frame actual_frame(const FileBoundary& engine, const Plan& transport, const ActualNoise& noise,
            const DecisionValues& values, const DecisionValues& recovery_values, const std::string& stage) const {
        auto frame = transport.fault_frame(assignment(engine, noise, values, recovery_values, stage).faults, false);
        engine.transport_noise(this->noise.actions, noise.drawn, frame);
        return frame;
    }
    HistoryMasses evaluate(const ActualNoise& noise, const DecisionValues& values,
            const DecisionValues& recovery_values, const NamedRows& known, const std::vector<int>& outcomes,
            const Bits& sector = {}, SamplingNetworkSession* session = nullptr,
            const std::vector<Bits>* numerical = nullptr, const ReferenceRows* sampling_rows = nullptr,
            const Frame* supplied_frame = nullptr, const Frame* target_frame = nullptr,
            const Scalar* sampled_mass = nullptr) {
        auto engine = plans.get(source.stage("recovery"));
        Frame local;
        if(!supplied_frame) {
            local = actual_frame(*engine, engine->plan->compiled.plan, noise, values, recovery_values, "recovery");
            supplied_frame = &local;
        }
        const auto& frame = *supplied_frame;
        HistoryMasses result;
        auto* completed_session = session;
        ReferenceRows converted;
        if(needs_history() && !sampling_rows) {
            converted = history_coordinates(*engine, "sampling", known, numerical);
            sampling_rows = &converted;
        }
        if(sampled_mass) result.history = *sampled_mass;
        else if(session || proposal) {
            std::optional<SamplingNetworkSession> local;
            if(!session) {
                local.emplace(*history, *engine->plan, sector);
                session = &*local;
            }
            context.check(limits);
            context.queries++;
            auto tick = std::chrono::steady_clock::now();
            result.history =
                    local ? session->mass_coordinates(frame, *sampling_rows, outcomes) : session->mass_committed(frame);
            if(local) {
                context.reference_network_resource_constructions += local->resource_constructions;
                context.reference_network_coordinate_constructions += local->coordinate_constructions;
            }
            context.reference_network_evaluations[proposal ? "sampling_capped" : "sampling"]++;
            context.kernel_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - tick).count();
        } else
            result.history = needs_history()
                    ? reference_total_coordinates(*engine, "sampling", frame, *sampling_rows, outcomes, sector)
                    : Scalar(1);
        session = completed_session;
        if(session) {
            session->committed.synchronize(frame);
            session->update_common(frame);
        }
        auto accepted_rows = !session ? history_coordinates(*engine, "acceptance", known, numerical) : ReferenceRows{};
        result.acceptance = session
                ? completed_total(*engine, "acceptance", frame, *session)
                : reference_total_coordinates(*engine, "acceptance", frame, accepted_rows, outcomes, sector);
        if(!task.boundaries.observables.empty()) {
            auto events = numerical ? record_names(*numerical) : known;
            events.insert(events.end(), task.boundaries.selection.begin(), task.boundaries.selection.end());
            auto rhs = outcomes;
            rhs.insert(rhs.end(), task.outcomes.begin(), task.outcomes.end());
            for(const auto& [x, z] : task.boundaries.observables)
                result.moments.push_back(total(*engine, {}, events, rhs, x, z, &frame));
        }
        if(source.target) {
            // All entry points consume the same completed protocol assignment:
            // batch draws, causal draws, enumeration and explicit assignments.
            // The target suffix receives no original-location faults.
            Frame continued;
            auto target = plans.get("target");
            if(!target_frame && target->plan->compiled.plan.continuation) {
                continued = target->plan->compiled.plan.continuation->apply(frame);
                target_frame = &continued;
                context.target_frame_reuses++;
            }
            evaluate_fidelity(
                    result, noise, values, recovery_values, known, outcomes, sector, numerical, target_frame, session);
        }
        return result;
    }

private:
    // One physical draw's existing local state. No cache or network owner lives here.
    struct CausalSample {
        CompiledExperiment& run;
        const ExecutionHooks& hooks;
        bool supplied;
        HistoryResult result;
        NoiseActions candidates;
        // Per inserted observation row: its reported bit and its reduced value
        // in the template's GF(2) basis.
        std::vector<int> observed, reduced;
        DecisionValues values, recovery_values;
        ReferenceAssignment protocol_assignment;
        std::shared_ptr<FileBoundary> engine;
        Bits sector;
        Scalar sampled_mass{1};
        std::optional<SamplingNetworkSession> session;

        CausalSample(CompiledExperiment& run, const ExecutionHooks& hooks, bool supplied)
            : run(run), hooks(hooks), supplied(supplied) {}

        void initialize() {
            // Supplied draws are candidates, as in exact categorical enumeration.
            // Future guarded candidates are absent until their reported guard is known.
            if(supplied)
                for(const auto& id : run.source.guarded_ids) {
                    auto found = result.noise.actions.find(id);
                    if(found != result.noise.actions.end()) {
                        candidates.emplace(id, std::move(found->second));
                        result.noise.actions.erase(found);
                    }
                }
            engine = run.plans.get(run.source.stage("recovery"));
            protocol_assignment.apply(engine->feedback_constant);
            protocol_assignment.append(engine->plan->compiled.plan,
                    engine->pack(run.source.original_faults(result.noise), result.noise.flips).faults);
            engine->transport_noise(run.noise.actions, result.noise.drawn, protocol_assignment.frame);
            if(run.proposal) result.proposal_denominator = Integer(1);
            prepare_sector_draw();
            prepare_session();
        }
        void prepare_sector_draw() {
            if(!run.sector_rows.empty()) {
                size_t dimension = run.sector_support.columns.size();
                Bits coordinates;
                sampled_mass.power = -int64_t(dimension);
                if(result.proposal_denominator)
                    *result.proposal_denominator = result.proposal_denominator->shift(dimension);
                if(dimension) {
                    Ket uniform;
                    uniform.width = dimension;
                    uniform.power = -2 * int64_t(dimension);
                    for(size_t j = 0; j < dimension; j++) uniform.columns.push_back(Bits::unit(j));
                    uniform.linear.resize(dimension);
                    uniform.upper.resize(dimension);
                    StabilizerTerms distribution{{Scalar(1), std::move(uniform)}};
                    coordinates = hooks.branch ? hooks.branch(distribution, dimension)
                                               : sample_boundary(distribution, dimension, hooks.seed());
                }
                sector = run.sector_support.base ^ xors(run.sector_support.columns, coordinates);
            }
        }
        void prepare_session() {
            if(run.history) {
                auto tick = std::chrono::steady_clock::now();
                session.emplace(*run.history, *engine->plan, sector, run.observation.layout.get());
                observed.reserve(run.observation.rows.size());
                reduced.reserve(run.observation.rows.size());
                run.context.support_seconds +=
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - tick).count();
            }
        }
        // Reported values of one prepared request; offsets are the expressions'
        // decision-dependent constants.
        std::vector<int> inputs(const ObservationRequest& request, const std::vector<int>& offsets) {
            Bits word;
            const size_t pending = request.pending;
            if(pending) {
                const auto& frame = protocol_assignment.frame;
                session->committed.synchronize(frame);
                if(request.affine) word = affine_word(request, sector, frame);
                else if(run.proposal) {
                    run.context.check(run.limits);
                    auto tick = std::chrono::steady_clock::now();
                    auto support = session->support_committed(frame, request.rows);
                    run.context.support_updates++;
                    run.context.support_seconds +=
                            std::chrono::duration<double>(std::chrono::steady_clock::now() - tick).count();
                    // A previous affine envelope can admit an interference-zero
                    // prefix. Complete its remaining decisions deterministically;
                    // its final zero weight remains in the average, without redraw.
                    if(!support.zero) {
                        auto dimension = support.columns.size();
                        *result.proposal_denominator = result.proposal_denominator->shift(dimension);
                        support.linear.resize(dimension);
                        support.upper.resize(dimension);
                        support.power = -2 * int64_t(dimension);
                        StabilizerTerms uniform{{Scalar(1), std::move(support)}};
                        word = hooks.branch ? hooks.branch(uniform, pending)
                                            : sample_boundary(uniform, pending, hooks.seed());
                    }
                } else {
                    run.context.check(run.limits);
                    run.context.queries++;
                    auto tick = std::chrono::steady_clock::now();
                    auto marginal = session->query_committed(frame, request.rows);
                    run.context.reference_network_evaluations["sampling"]++;
                    run.context.kernel_seconds +=
                            std::chrono::duration<double>(std::chrono::steady_clock::now() - tick).count();
                    if(hooks.branch) {
                        word = hooks.branch(marginal.terms, pending);
                        sampled_mass = ProbabilityMarginal{marginal.terms, pending}.value(word);
                    } else word = sample_boundary(marginal.terms, pending, hooks.seed(), &sampled_mass);
                }
            }
            if(word.size() > pending) throw std::invalid_argument("sampled word outside requested labels");
            // Dependent observations still receive their consistency check.
            for(size_t j = 0; j < pending; j++) {
                int bit = word.bit(j), value = bit;
                for(auto row = request.chains.begin(j); row != request.chains.end(j); ++row) value ^= reduced[*row];
                if(request.inserted[j] != ReferenceHistoryLayout::absent) {
                    reduced.push_back(value);
                    observed.push_back(bit);
                    session->committed.append(bit);
                } else if(value) throw std::runtime_error("sampled parities violate an existing linear relation");
            }
            std::vector<int> result;
            result.reserve(offsets.size());
            for(size_t e = 0; e < offsets.size(); e++) {
                int bit = offsets[e];
                for(auto row = request.known.begin(e); row != request.known.end(e); ++row) bit ^= reduced[*row];
                if(request.outputs[e]) bit ^= word.bit(*request.outputs[e]);
                result.push_back(bit);
            }
            return result;
        }
        std::vector<int> invoke(const std::string& stage, size_t index, const std::vector<int>& bits, size_t count) {
            auto tick = std::chrono::steady_clock::now();
            auto out = hooks.controller(stage, index, bits, count);
            run.context.decoder_seconds +=
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - tick).count();
            if(out.size() != count || std::any_of(out.begin(), out.end(), [](int b) { return b != 0 && b != 1; }))
                throw std::invalid_argument("controller must return the declared binary outputs");
            return out;
        }
        void apply_conditional_noise(const ConditionalNoiseCall& call) {
            std::vector<FeedbackFault> added;
            if(supplied) {
                auto found = candidates.find(call.id);
                if(found != candidates.end()) {
                    result.noise.actions.emplace(found->first, found->second);
                    for(const auto& [pauli, wire] : found->second) added.push_back({found->first, wire, pauli});
                }
            } else {
                std::vector<int64_t> drawn;
                auto site = run.noise.conditional_sites.find(call.instruction);
                if(site != run.noise.conditional_sites.end())
                    for(auto category : site->second) {
                        std::mt19937_64 rng(hooks.seed());
                        auto choice = run.noise.categories[category]->draw(rng);
                        if(choice >= 0) drawn.push_back(int64_t(run.noise.offsets[category] + size_t(choice)));
                    }
                engine->transport_noise(run.noise.actions, drawn, protocol_assignment.frame);
                result.noise.drawn.insert(result.noise.drawn.end(), drawn.begin(), drawn.end());
            }
            if(!added.empty()) protocol_assignment.append(engine->plan->compiled.plan, engine->pack(added, {}).faults);
        }
        void controllers(const FileControlMetadata& metadata, const std::vector<ObservationRequest>& requests,
                DecisionValues& state, const std::string& stage) {
            auto offset = [&](const ControllerInput& input) {
                int bit = input.constant;
                for(auto index : input.decisions) bit ^= state.at(metadata.decision_order.at(index));
                return bit;
            };
            for(const auto& request : requests) {
                if(request.guard) {
                    const auto& call = metadata.conditional_noise[request.index];
                    if(!inputs(request, {offset(call.guard)}).at(0)) continue;
                    apply_conditional_noise(call);
                } else {
                    size_t j = request.index;
                    const auto& call = metadata.callbacks[j];
                    std::vector<int> offsets;
                    offsets.reserve(call.inputs.size());
                    for(const auto& input : call.inputs) offsets.push_back(offset(input));
                    auto bits = inputs(request, offsets), out = invoke(stage, j, bits, call.outputs.size());
                    if(stage == "protocol") result.decoder_calls++;
                    else result.recovery_decoder_calls++;
                    for(size_t k = 0; k < out.size(); k++) {
                        auto index = call.outputs[k];
                        state[metadata.decision_order.at(index)] = out[k];
                        if(out[k]) {
                            bool protocol = stage == "protocol";
                            protocol_assignment.apply(
                                    (protocol ? engine->protocol_frames : engine->recovery_frames).at(index));
                        }
                    }
                }
            }
        }
        HistoryResult finish() {
            const auto& schedule = run.observation;
            controllers(run.source.controls, schedule.protocol, values, "protocol");
            if(schedule.samples)
                result.samples = inputs(*schedule.samples, std::vector<int>(run.task.boundaries.sample_records.size()));
            if(run.source.recovery)
                controllers(*run.source.recovery_controls, schedule.recovery, recovery_values, "recovery");
            result.masses = run.evaluate(result.noise, values, recovery_values, {}, observed, sector,
                    session ? &*session : nullptr, &schedule.rows, &schedule.coordinates, &protocol_assignment.frame,
                    nullptr, run.proposal ? nullptr : &sampled_mass);
            if(session) {
                run.context.support_geometry_updates += session->geometry_updates;
                run.context.support_geometry_reuses += session->geometry_reuses;
                run.context.support_term_visits += session->term_visits;
                run.context.support_tensor_expansions += session->tensor_expansions;
                run.context.reference_network_resource_constructions += session->resource_constructions;
                run.context.reference_network_coordinate_constructions += session->coordinate_constructions;
            }
            result.sampled_parities = observed.size();
            return std::move(result);
        }
    };

public:
    HistoryResult sample(const ExecutionHooks& hooks, std::optional<ActualNoise> supplied = {}) {
        if(!complete) prepare();
        CausalSample draw_state(*this, hooks, bool(supplied));
        draw_state.result.noise = supplied ? std::move(*supplied) : draw(hooks);
        draw_state.initialize();
        return draw_state.finish();
    }
};
} // namespace magic_native
