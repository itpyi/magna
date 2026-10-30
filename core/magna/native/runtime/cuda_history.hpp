// Causal CUDA schedules and exact completion at the existing decoder boundary.
#pragma once
#include "cuda_plan.hpp"
namespace magic_native {

// RNG domains are independent of chunk size and of earlier sites' consumption.
inline uint64_t cuda_history_seed(uint64_t seed, uint64_t domain) {
    uint64_t x = seed + UINT64_C(0x9e3779b97f4a7c15) * domain;
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}
inline int cuda_affine_value(const CudaQueryAffine& a, const Bits& bits) {
    return a.constant ^ a.row.parity_and(bits);
}
inline CudaQueryAffine cuda_affine_sum(const std::vector<CudaQueryAffine>& values, const Bits& mask) {
    CudaQueryAffine result;
    mask.each([&](size_t j) { result = cuda_query_xor(std::move(result), values.at(j)); });
    return result;
}
// A canonical affine coordinate order makes the CUDA stream independent of
// the elimination pivot heuristic used to find the same support space.
inline Ket cuda_history_canonical(Ket support) {
    if(support.zero) return support;
    std::map<size_t, Bits> basis;
    for(auto column : support.columns) {
        for(auto it = basis.rbegin(); it != basis.rend(); ++it)
            if(column.bit(it->first)) column ^= it->second;
        if(column.empty()) continue;
        auto pivot = column.size() - 1;
        for(auto& [j, row] : basis) if(row.bit(pivot)) row ^= column;
        basis.emplace(pivot, std::move(column));
    }
    for(auto it = basis.rbegin(); it != basis.rend(); ++it)
        if(support.base.bit(it->first)) support.base ^= it->second;
    support.columns.clear();
    for(auto& [j, row] : basis) support.columns.push_back(std::move(row));
    return support;
}
struct CudaHistorySite {
    std::string stage;
    size_t index = 0, observation_first = 0, history_count = 0;
    std::vector<CudaQueryAffine> inputs;
    std::vector<CudaQueryAffine> offsets, outcomes;
    std::vector<Bits> keep, events;
    std::vector<Bits> pending, labels;
    std::vector<uint32_t> observed, input_for_pending, outputs;
    std::vector<uint64_t> columns;
    std::vector<CudaQueryPlan> prefixes;
};
struct CudaHistoryEntry {
    Bits shift, base;
    std::vector<Bits> change, columns;
    Quadratic reference;
};
struct CudaHistoryPlan {
    size_t auxiliary_bits = 0, auxiliary_words = 0, sector_bits = 0, sector_rank = 0;
    size_t observations = 0, controls = 0, open_bits = 0;
    ScientificValue normalization{.5, 1};
    std::vector<CudaHistorySite> sites;
    std::vector<CudaQueryAffine> closed, open_base;
    std::vector<Bits> observed_labels;
    std::vector<size_t> observed_slots;
    Bits sector_base;
    std::vector<Bits> sector_columns;
    CudaQueryPlan mother;
    std::vector<CudaHistoryEntry> entries;
    std::vector<std::shared_ptr<const CudaQueryPlan>> query_bases;
    std::vector<CudaQueryPlan> terminal;
};

// Echelon coefficients are affine words, so a single schedule resolves every
// draw without treating decoder outputs as known compile-time constants.
struct CudaHistoryKnowledge {
    std::map<size_t, std::pair<Bits, CudaQueryAffine>> basis;
    std::pair<Bits, CudaQueryAffine> reduce(Bits row, CudaQueryAffine value = {}) const {
        while(!row.empty()) {
            auto found = basis.find(row.size() - 1);
            if(found == basis.end()) break;
            row ^= found->second.first;
            value = cuda_query_xor(std::move(value), found->second.second);
        }
        return {std::move(row), std::move(value)};
    }
    bool insert(const Bits& row, const CudaQueryAffine& value) {
        auto reduced = reduce(row, value);
        if(reduced.first.empty()) return false;
        basis.emplace(reduced.first.size() - 1, std::move(reduced));
        return true;
    }
};

inline CudaHistoryPlan build_cuda_history(CompiledExperiment& run, const CudaExperimentPlan& experiment) {
    if(!run.history) throw std::invalid_argument("CUDA history plan requires a live decoder");
    CudaHistoryPlan out;
    const auto& parent = *run.history->parent;
    const auto& network = *parent.reference;
    const auto& records = experiment.engine->reference_records.at("sampling");
    out.sector_bits = parent.closed.size();
    out.open_bits = network.boundary.keep.size();
    out.sector_rank = run.sector_support.columns.size();
    out.sector_base = run.sector_support.base;
    out.sector_columns = run.sector_support.columns;
    out.normalization = scaled_real_ratio(run.target_normalization, Scalar(1));
    if(!run.history->certificate)
        throw std::invalid_argument("CUDA history requires a fixed sampling-parent certificate");
    int largest_likelihood = 0;
    for(auto value : experiment.likelihood_values) {
        int exponent;
        std::frexp(value, &exponent);
        largest_likelihood = std::max(largest_likelihood, exponent);
    }
    size_t rank_bound = run.proposal ? out.open_bits + out.sector_rank : 0;
    if(rank_bound + size_t(std::max(0, largest_likelihood)) >= 1020)
        throw std::invalid_argument("CUDA history result-range bound exceeds finite float64 output");
    size_t control_offset = 0;
    // The first pass reserves observation and control coordinates; these are
    // stable for the whole run, including sites with no new record parity.
    CausalKnowledge reservation;
    auto reserve = [&](const FileControlMetadata& metadata, size_t offset) {
        for(const auto& call : metadata.callbacks) {
            std::vector<std::pair<Bits, int>> expressions;
            for(const auto& input : call.inputs) expressions.emplace_back(input.records << offset, 0);
            auto request = reservation.request(expressions);
            out.observations += request.pending.size();
            reservation.observe(request, {});
        }
        out.controls += metadata.decision_order.size();
    };
    reserve(run.source.controls, 0);
    if(run.source.recovery_controls) reserve(*run.source.recovery_controls, run.source.protocol.num_records);
    out.auxiliary_bits = out.sector_bits + out.observations + out.controls;
    out.auxiliary_words = (out.auxiliary_bits + 63) / 64;
    const size_t origin = 128 * experiment.words;
    auto auxiliary = [&](size_t j) { return CudaQueryAffine{Bits::unit(origin + j), 0}; };
    for(size_t j = 0; j < out.sector_bits; j++) {
        auto value = auxiliary(j);
        value.row ^= parent.closed[j];
        out.closed.push_back(std::move(value));
    }
    CudaHistoryKnowledge known;
    size_t observation = 0;
    auto schedule = [&](const FileControlMetadata& metadata, const std::string& stage, size_t offset,
                            const std::vector<Frame>& frames) {
        for(size_t index = 0; index < metadata.callbacks.size(); index++) {
            const auto& call = metadata.callbacks[index];
            CudaHistorySite site;
            site.stage = stage;
            site.index = index;
            site.observation_first = observation;
            site.history_count = out.observed_labels.size();
            std::map<Bits, size_t, BitsLess> pending;
            for(size_t i = 0; i < call.inputs.size(); i++) {
                const auto& input = call.inputs[i];
                CudaQueryAffine value{{}, input.constant};
                for(auto decision : input.decisions)
                    value = cuda_query_xor(std::move(value), auxiliary(out.sector_bits + out.observations + control_offset + decision));
                auto reduced = known.reduce(input.records << offset, std::move(value));
                if(!reduced.first.empty()) {
                    auto found = pending.find(reduced.first);
                    size_t position;
                    if(found == pending.end()) {
                        position = site.pending.size();
                        pending.emplace(reduced.first, position);
                        site.pending.push_back(reduced.first);
                        site.input_for_pending.push_back(uint32_t(i));
                    } else position = found->second;
                    reduced.second = cuda_query_xor(std::move(reduced.second), auxiliary(out.sector_bits + observation + position));
                }
                site.inputs.push_back(std::move(reduced.second));
            }
            site.labels = records(site.pending).labels;
            for(size_t j = 0; j < site.pending.size(); j++) {
                if(known.insert(site.pending[j], auxiliary(out.sector_bits + observation + j))) {
                    site.observed.push_back(uint32_t(j));
                    out.observed_labels.push_back(site.labels[j]);
                    out.observed_slots.push_back(observation + j);
                }
            }
            for(auto index : call.outputs) {
                site.outputs.push_back(uint32_t(out.sector_bits + out.observations + control_offset + index));
                auto first = site.columns.size();
                site.columns.resize(first + 2 * experiment.words);
                cuda_pack_frame(frames.at(index), experiment.words, site.columns.data() + first);
            }
            observation += site.pending.size();
            out.sites.push_back(std::move(site));
        }
        control_offset += metadata.decision_order.size();
    };
    schedule(run.source.controls, "protocol", 0, experiment.engine->protocol_frames);
    if(run.source.recovery_controls)
        schedule(*run.source.recovery_controls, "recovery", run.source.protocol.num_records, experiment.engine->recovery_frames);
    for(auto& site : out.sites) {
        for(const auto& label : site.labels) {
            site.keep.push_back(label >> out.sector_bits);
            site.offsets.push_back(cuda_affine_sum(out.closed, label.low(out.sector_bits)));
        }
        for(size_t j = 0; j < site.history_count; j++) {
            const auto& label = out.observed_labels[j];
            if(label.size() <= out.sector_bits) continue;
            site.events.push_back(label >> out.sector_bits);
            site.outcomes.push_back(cuda_query_xor(auxiliary(out.sector_bits + out.observed_slots[j]),
                    cuda_affine_sum(out.closed, label.low(out.sector_bits))));
        }
    }

    // Certify static plans against both feedback and all variable cap inputs.
    std::vector<Bits> changes;
    for(const auto& site : out.sites)
        for(size_t first = 0; first < site.columns.size(); first += 2 * experiment.words) {
            Bits row;
            row.w.assign(site.columns.begin() + first, site.columns.begin() + first + 2 * experiment.words);
            row.trim();
            changes.push_back(std::move(row));
        }
    for(size_t j = 0; j < out.auxiliary_bits; j++) changes.push_back(Bits::unit(origin + j));
    std::vector<CudaQueryAffine> mother_caps;
    for(auto bit : parent.selection) mother_caps.push_back({{}, bit});
    mother_caps.insert(mother_caps.end(), out.closed.begin(), out.closed.end());
    out.mother = build_cuda_query(run, experiment, *experiment.engine, "sampling", false, &network,
            &mother_caps, out.auxiliary_words, &changes, run.history->contraction, true);
    // Output offsets use the same event solve as the scalar query producer.
    std::vector<CudaQueryAffine> rhs;
    for(size_t j = 0; j < network.boundary.events.size(); j++) {
        auto value = mother_caps[j];
        value.row ^= network.boundary.events[j];
        value.constant ^= network.traced.clifford.boundary[out.open_bits + j].second;
        rhs.push_back(std::move(value));
    }
    std::vector<CudaQueryAffine> shift(network.traced.clifford.n);
    for(const auto& [j, pair] : network.event_support->factor.pivots)
        shift[j] = cuda_query_xor(cuda_affine_sum(shift, pair.first), cuda_affine_sum(rhs, pair.second));
    for(size_t j = 0; j < out.open_bits; j++) {
        auto base = cuda_affine_sum(shift, network.traced.clifford.boundary[j].first);
        base.row ^= network.boundary.keep[j];
        base.constant ^= network.restricted.clifford.boundary[j].second;
        out.open_base.push_back(std::move(base));
    }
    const auto& resource = run.history->certificate->reference;
    if(run.history->contraction) {
        CudaHistoryEntry entry;
        entry.columns = run.history->contraction->columns;
        entry.reference = run.history->contraction->pulled_reference;
        out.entries.push_back(std::move(entry));
    } else {
        const auto& terms = resource_entries(resource);
        auto columns = transpose(network.maps->restricted_forms, out.mother.n);
        for(const auto& term : terms) {
            std::vector<Bits> constraints;
            Bits target;
            for(size_t j = 0; j < term.constraints.size(); j++) {
                constraints.push_back(xors(resource.quotient.axes, term.constraints[j].first));
                if(term.constraints[j].second) target.set(j);
            }
            Factor support(constraints, out.mother.n);
            if(!support.consistent(target) || term.coefficient.zero()) continue;
            CudaHistoryEntry entry;
            entry.shift = support.solve(target);
            auto projected = support.kernel;
            std::vector<Bits> original;
            for(const auto& axis : projected) original.push_back(xors(columns, axis));
            auto reduced = column_change(original);
            for(const auto& axis : reduced) {
                entry.change.push_back(xors(projected, axis));
                entry.columns.push_back(xors(original, axis));
            }
            entry.base = xors(columns, entry.shift);
            auto l = resource.quotient.linear;
            auto u = resource.quotient.upper;
            PhaseTerms updates;
            for(const auto& [mask, coefficient] : term.parities)
                updates.emplace_back(xors(resource.quotient.axes, mask), 2 * mod4(coefficient));
            phase_add_even(l, u, updates);
            entry.reference = restrict_q(l, u, entry.shift, entry.change);
            out.entries.push_back(std::move(entry));
        }
    }
    if(out.entries.empty()) throw std::invalid_argument("CUDA sampling parent has no prepared affine entries");
    // Every prefix owns only its active-cap count. The full matrix, features
    // and magic masks are shared once, including for a large exact sampler.
    auto forms = run.history->contraction ? transpose(run.history->contraction->columns, out.open_bits)
                                         : network.maps->restricted_forms;
    std::vector<Bits> equations;
    std::vector<CudaQueryAffine> cap_values;
    auto prefix = [&](size_t active) {
        CudaQueryPlan result;
        result.mode = 6; result.frame_words = out.mother.frame_words;
        result.disabled_caps = active; // Converted to the inactive suffix below.
        return result;
    };
    CudaHistoryKnowledge committed;
    std::vector<CudaQueryAffine> contradictions;
    for(auto& site : out.sites) {
        if(!run.proposal && !site.pending.empty()) site.prefixes.push_back(prefix(equations.size()));
        for(size_t j = 0; j < site.pending.size(); j++) {
            auto value = auxiliary(out.sector_bits + site.observation_first + j);
            if(site.labels[j].size() > out.sector_bits) {
                auto open = site.labels[j] >> out.sector_bits;
                equations.push_back(xors(forms, open));
                cap_values.push_back(cuda_query_xor(value, cuda_query_xor(
                        cuda_affine_sum(out.closed, site.labels[j].low(out.sector_bits)), cuda_affine_sum(out.open_base, open))));
                if(!run.proposal) site.prefixes.push_back(prefix(equations.size()));
            } else if(!run.proposal) site.prefixes.emplace_back();
            auto row = site.labels[j] >> out.sector_bits;
            auto rhs = cuda_query_xor(value, cuda_affine_sum(out.closed, site.labels[j].low(out.sector_bits)));
            if(!row.empty()) {
                auto reduced = committed.reduce(row, rhs);
                if(reduced.first.empty()) contradictions.push_back(reduced.second);
                else committed.insert(row, rhs);
            }
        }
    }
    auto base = run.history->contraction ? out.mother : cuda_query_entries(out.mother, resource);
    auto cap = [&](CudaQueryPlan query) {
        auto result = cuda_query_cap(query, equations, cap_values);
        if(!equations.empty()) result.mode = 4;
        return result;
    };
    if(base.mode == 5) for(auto& component : base.components) component = cap(std::move(component));
    else base = cap(std::move(base));
    base = cuda_query_scale(std::move(base), run.target_normalization);
    out.query_bases.push_back(std::make_shared<const CudaQueryPlan>(std::move(base)));
    auto bind_prefix = [&](CudaQueryPlan& query) {
        if(query.mode != 6) return;
        query.shared_base = out.query_bases[0]; query.n = query.shared_base->n;
        query.disabled_caps = equations.size() - query.disabled_caps;
    };
    for(auto& site : out.sites) for(auto& query : site.prefixes) bind_prefix(query);
    out.terminal.push_back(prefix(equations.size()));
    bind_prefix(out.terminal.back());
    auto terminal = [&](const FileBoundary& engine, const char* purpose, bool target) {
        const auto& terminal_parent = *engine.reference_networks.at(purpose);
        std::vector<CudaQueryAffine> caps;
        for(auto value : terminal_parent.selection) caps.push_back({{}, value});
        for(const auto& coordinates : run.completed_caps.at(purpose)) {
            auto value = cuda_affine_sum(out.closed, coordinates.low(out.sector_bits));
            auto reduced = committed.reduce(coordinates >> out.sector_bits, std::move(value));
            if(!reduced.first.empty()) throw std::logic_error("CUDA terminal cap is outside the causal schedule");
            caps.push_back(std::move(reduced.second));
        }
        auto result = build_cuda_query(run, experiment, engine, purpose, target, terminal_parent.reference.get(),
                &caps, out.auxiliary_words, &changes);
        if(result.mode == 5) {
            for(auto& component : result.components)
                for(const auto& contradiction : contradictions) component.support.push_back(component.feature(contradiction));
        } else for(const auto& contradiction : contradictions) result.support.push_back(result.feature(contradiction));
        return result;
    };
    out.terminal.push_back(terminal(*experiment.engine, "acceptance", false));
    if(experiment.target) out.terminal.push_back(terminal(*experiment.target, "fidelity", true));
    out.terminal[1] = cuda_query_scale(std::move(out.terminal[1]), run.target_normalization);
    return out;
}

inline Ket cuda_history_support(const CudaHistoryPlan& plan, size_t site_index, const uint64_t* packed) {
    const auto& site = plan.sites.at(site_index);
    auto features = cuda_query_feature_values(plan.mother, packed);
    Ket empty;
    empty.zero = true; empty.width = site.pending.size();
    for(auto event : plan.mother.support) if(features[event]) return empty;
    std::vector<int> linear;
    std::vector<Bits> upper;
    cuda_query_actual(plan.mother, features, linear, upper);
    Bits bits;
    bits.w.assign(packed, packed + plan.mother.frame_words);
    Bits base;
    for(size_t j = 0; j < plan.open_base.size(); j++)
        if(cuda_affine_value(plan.open_base[j], bits)) base.set(j);
    std::vector<int> values;
    for(const auto& value : site.outcomes) values.push_back(cuda_affine_value(value, bits));
    AffineProposalEnvelope envelope(site.pending.size());
    for(const auto& entry : plan.entries) {
        Quadratic q;
        if(entry.change.empty()) q = {0, linear, upper};
        else {
            auto delta_l = linear;
            auto delta_u = upper;
            size_t words = (plan.mother.n + 63) / 64;
            for(size_t j = 0; j < linear.size(); j++) {
                delta_l[j] = mod4(delta_l[j] - plan.mother.linear[j]);
                Bits row;
                row.w.assign(plan.mother.upper.begin() + j * words, plan.mother.upper.begin() + (j + 1) * words);
                row.trim();
                delta_u[j] ^= row;
            }
            q = restrict_q(delta_l, delta_u, entry.shift, entry.change);
            for(size_t j = 0; j < q.linear.size(); j++) {
                q.linear[j] = mod4(q.linear[j] + entry.reference.linear[j]);
                q.upper[j] ^= entry.reference.upper[j];
            }
        }
        CliffordContraction contraction(entry.columns, q);
        auto support = contraction.support(plan.open_bits, base ^ entry.base, q.linear);
        envelope.include(project_support(support, site.keep, site.events, values));
    }
    for(size_t j = 0; j < site.offsets.size(); j++)
        if(cuda_affine_value(site.offsets[j], bits)) envelope.support.base.flip(j);
    return cuda_history_canonical(std::move(envelope.support));
}

// Unbounded exact Bernoulli continuation uses the same addressed Philox bits
// as the device prefix, including after the device comparison is inconclusive.
inline bool cuda_history_bernoulli(const Scalar& numerator, const Scalar& denominator, CudaPhilox& rng) {
    const Scalar minus(-1);
    if(real_sign(denominator) <= 0 || real_sign(numerator) < 0 || real_sign(denominator + numerator * minus) < 0)
        throw std::invalid_argument("invalid CUDA history marginal");
    if(numerator.zero()) return false;
    if((denominator + numerator * minus).zero()) return true;
    Integer prefix;
    for(size_t depth = 1;; depth++) {
        prefix = prefix.shift(1) + Integer(int64_t(rng() & 1));
        auto lower = file_integer_scalar(prefix), upper = file_integer_scalar(prefix + Integer(1));
        lower.power -= int64_t(depth);
        upper.power -= int64_t(depth);
        if(real_sign(numerator + denominator * upper * minus) >= 0) return true;
        if(real_sign(numerator + denominator * lower * minus) <= 0) return false;
    }
}
struct CudaHistoryReplay {
    CompiledExperiment& run;
    const CudaExperimentPlan& experiment;
    const CudaHistoryPlan& plan;
    uint64_t seed, shot;
    Frame frame;
    Bits sector, packed;
    CausalKnowledge known;
    ReferenceRows known_coordinates;
    std::unique_ptr<SamplingNetworkSession> session;
    HistoryResult result;
    size_t proposal_rank = 0;
    CudaHistoryReplay(CompiledExperiment& run, const CudaExperimentPlan& experiment, const CudaHistoryPlan& plan,
            uint64_t seed, uint64_t shot) : run(run), experiment(experiment), plan(plan), seed(seed), shot(shot) {
        std::vector<uint64_t> raw(2 * experiment.words);
        std::vector<uint32_t> actions;
        auto count = cuda_reference_draw(run.noise, experiment, seed, shot, raw.data(), &actions);
        FileNoiseDraw selected;
        for(auto action : actions) selected.selected.push_back(action);
        std::sort(selected.selected.begin(), selected.selected.end());
        if(experiment.has_importance_weights) {
            selected.weight = run.noise.count_weights.at(0).at(count);
            selected.scientific_weight = run.noise.count_scientific.at(count);
        }
        result.noise = noise_actions(run.noise, selected);
        frame.x.w.assign(raw.begin(), raw.begin() + experiment.words);
        frame.z.w.assign(raw.begin() + experiment.words, raw.end());
        frame.x.trim(); frame.z.trim();
        sector = plan.sector_base;
        CudaPhilox rng(shot, cuda_history_seed(seed, 1));
        for(const auto& column : plan.sector_columns) if(rng() & 1) sector ^= column;
        proposal_rank = plan.sector_rank;
        packed.w.resize(2 * experiment.words + plan.auxiliary_words);
        for(auto bit : sector.indices()) packed.w[2 * experiment.words + bit / 64] |= uint64_t(1) << (bit % 64);
        session = std::make_unique<SamplingNetworkSession>(*run.history, *experiment.engine->plan, sector);
        synchronize();
    }
    void synchronize() {
        packed.w.resize(2 * experiment.words + plan.auxiliary_words);
        std::fill(packed.w.begin(), packed.w.begin() + 2 * experiment.words, uint64_t(0));
        std::copy(frame.x.w.begin(), frame.x.w.end(), packed.w.begin());
        std::copy(frame.z.w.begin(), frame.z.w.end(), packed.w.begin() + experiment.words);
        session->committed.synchronize(frame);
    }
    std::vector<int> inputs(size_t index, const std::vector<int>* supplied = nullptr) {
        const auto& site = plan.sites.at(index);
        synchronize();
        Bits word;
        ReferenceRows labels{site.labels};
        auto support = cuda_history_canonical(session->support_committed(frame, labels));
        if(run.proposal && !support.zero) proposal_rank += support.columns.size();
        if(supplied) {
            if(supplied->size() != site.inputs.size()) throw std::invalid_argument("CUDA replay input width mismatch");
            for(size_t j = 0; j < site.pending.size(); j++) {
                auto value = site.inputs[site.input_for_pending[j]];
                value.row.flip(128 * experiment.words + plan.sector_bits + site.observation_first + j);
                if(supplied->at(site.input_for_pending[j]) ^ cuda_affine_value(value, packed)) word.set(j);
            }
        } else if(!site.pending.empty()) {
            CudaPhilox rng(shot, cuda_history_seed(seed, index + 2));
            auto affine = experiment.engine->reference_records.at("sampling").affine_values(
                    site.pending, labels, plan.sector_bits, frame, sector);
            if(affine) word = *affine;
            else if(run.proposal) {
                if(!support.zero) {
                    word = support.base;
                    for(const auto& column : support.columns) if(rng() & 1) word ^= column;
                }
            } else {
                auto marginal = session->query_committed(frame, labels);
                Workspace ws;
                auto mass = boundary_sum(marginal.terms, {}, {}, ws);
                if(real_sign(mass) > 0) {
                    std::vector<Bits> events;
                    std::vector<int> values;
                    for(size_t j = 0; j < site.pending.size(); j++) {
                        events.push_back(Bits::unit(j)); values.push_back(0);
                        auto zero = boundary_sum(marginal.terms, events, values, ws);
                        int bit = !cuda_history_bernoulli(zero, mass, rng);
                        mass = bit ? mass + zero * Scalar(-1) : zero;
                        values.back() = bit;
                        if(bit) word.set(j);
                    }
                }
            }
        }
        for(size_t j = 0; j < site.pending.size(); j++) {
            auto bit = 128 * experiment.words + plan.sector_bits + site.observation_first + j;
            auto mask = uint64_t(1) << (bit % 64);
            packed.w[bit / 64] = (packed.w[bit / 64] & ~mask) | (word.bit(j) ? mask : 0);
        }
        for(auto j : site.observed) {
            known.insert(site.pending[j], word.bit(j));
            known.rows.push_back(site.pending[j]); known.values.push_back(word.bit(j));
            known_coordinates.labels.push_back(site.labels[j]);
            Bits physical;
            if(site.labels[j].size() <= plan.sector_bits)
                site.pending[j].each([&](size_t bit) { physical.set(experiment.engine->reference_records.at("sampling").physical.at(bit)); });
            session->committed.append(site.labels[j], word.bit(j), site.labels[j].size() <= plan.sector_bits ? &physical : nullptr);
        }
        std::vector<int> result;
        for(const auto& input : site.inputs) result.push_back(cuda_affine_value(input, packed));
        return result;
    }
    void outputs(size_t index, const std::vector<int>& values) {
        const auto& site = plan.sites.at(index);
        if(values.size() != site.outputs.size()) throw std::invalid_argument("CUDA replay output width mismatch");
        for(size_t j = 0; j < values.size(); j++) {
            auto bit = 128 * experiment.words + site.outputs[j];
            auto mask = uint64_t(1) << (bit % 64);
            packed.w[bit / 64] = (packed.w[bit / 64] & ~mask) | (values[j] ? mask : 0);
            if(values[j]) {
                const auto* column = site.columns.data() + j * 2 * experiment.words;
                frame.x.w.resize(experiment.words); frame.z.w.resize(experiment.words);
                for(size_t w = 0; w < experiment.words; w++) { frame.x.w[w] ^= column[w]; frame.z.w[w] ^= column[experiment.words + w]; }
                frame.x.trim(); frame.z.trim();
            }
        }
        synchronize();
    }
    HistoryResult finish() {
        result.masses = run.evaluate(result.noise, {}, {}, {}, known.values, sector, session.get(),
                &known.rows, &known_coordinates, &frame);
        if(run.proposal) result.proposal_denominator = Integer(1).shift(proposal_rank);
        return std::move(result);
    }
};
struct CudaHistoryCompletion {
    std::vector<std::vector<int>> inputs;
    std::vector<uint64_t> frame;
    size_t proposal_rank = 0;
    std::optional<HistoryResult> result;
};
inline CudaHistoryCompletion cuda_replay_history(CompiledExperiment& run, const CudaExperimentPlan& experiment,
        const CudaHistoryPlan& plan, uint64_t seed, uint64_t shot, size_t stop,
        const std::vector<std::vector<int>>& inputs, const std::vector<std::vector<int>>& outputs) {
    if(stop > plan.sites.size() || outputs.size() != stop || (!inputs.empty() && inputs.size() != stop))
        throw std::invalid_argument("CUDA replay history length mismatch");
    CudaHistoryReplay replay(run, experiment, plan, seed, shot);
    CudaHistoryCompletion result;
    for(size_t j = 0; j < stop; j++) {
        result.inputs.push_back(replay.inputs(j, inputs.empty() ? nullptr : &inputs[j]));
        replay.outputs(j, outputs[j]);
    }
    if(stop < plan.sites.size()) result.inputs.push_back(replay.inputs(stop));
    else result.result = replay.finish();
    result.frame = replay.packed.w;
    result.proposal_rank = replay.proposal_rank;
    return result;
}
} // namespace magic_native
