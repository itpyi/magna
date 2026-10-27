// Affine proposal envelopes transported from the prepared probability parent.
// Exclude only affine-certified zeros; residual interference is evaluated once
// the complete history caps the parent. No record-support enumeration.
#pragma once
#include "../evaluate/reference_network.hpp"
#include "marginal.hpp"
namespace magic_native {
inline Ket project_support(const Ket& support, const std::vector<Bits>& keep, const std::vector<Bits>& events,
        const std::vector<int>& values) {
    Ket out;
    out.width = keep.size();
    if(support.zero) {
        out.zero = true;
        return out;
    }
    auto forms = transpose(support.columns, support.width);
    std::vector<Bits> equations;
    Bits rhs;
    for(size_t j = 0; j < events.size(); j++) {
        equations.push_back(xors(forms, events[j]));
        if(values[j] ^ events[j].parity_and(support.base)) rhs.set(j);
    }
    Factor allowed(equations, support.columns.size());
    if(!allowed.consistent(rhs)) {
        out.zero = true;
        return out;
    }
    auto shift = allowed.solve(rhs);
    std::vector<Bits> projected_rows;
    for(size_t j = 0; j < keep.size(); j++) {
        projected_rows.push_back(xors(forms, keep[j]));
        if(keep[j].parity_and(support.base) ^ projected_rows.back().parity_and(shift)) out.base.set(j);
    }
    auto projected = transpose(projected_rows, support.columns.size());
    CausalKnowledge basis;
    for(const auto& axis : allowed.kernel) {
        auto col = xors(projected, axis);
        if(basis.insert(col).first) out.columns.push_back(std::move(col));
    }
    return out;
}
struct AffineProposalEnvelope {
    Ket support;
    CausalKnowledge basis;
    explicit AffineProposalEnvelope(size_t width) {
        support.width = width;
        support.zero = true;
    }
    void include(const Ket& affine) {
        if(affine.zero) return;
        auto add = [&](const Bits& column) {
            if(basis.insert(column).first) support.columns.push_back(column);
        };
        if(support.zero) {
            support.base = affine.base;
            support.zero = false;
        } else add(affine.base ^ support.base);
        for(const auto& column : affine.columns) add(column);
    }
};
// Private sampling-parent transport shared by exact and proposal queries.
// The parent is unselected; acceptance and fidelity retain separate owners.
struct SamplingNetworkPlan {
    std::shared_ptr<const ReferenceNetwork> parent;
    const ResourceCertificate* certificate = nullptr;
    const PhaseContraction* contraction = nullptr;
    std::vector<Bits> sector_signs, bra_phase, ket_phase;
    explicit SamplingNetworkPlan(std::shared_ptr<const ReferenceNetwork> reference) : parent(std::move(reference)) {
        const auto& n = *parent->reference;
        certificate = n.fixed_certificate();
        if(certificate) {
            if(certificate->reference.quotient.axes.empty() && certificate->term_bound() == 1)
                contraction = certificate->reference.contraction.get();
            if(contraction) {
                auto dual = transpose(contraction->change, contraction->dimension);
                for(const auto& axis : contraction->change) sector_signs.push_back(xors(n.maps->event_polar, axis));
                for(size_t j = 0; j < n.magic.size(); j++) {
                    bra_phase.push_back(xors(dual, n.maps->bra_reduced[j]));
                    ket_phase.push_back(xors(dual, n.maps->ket_reduced[j]));
                }
            } else {
                sector_signs = n.maps->event_polar;
                bra_phase = n.maps->bra_reduced;
                ket_phase = n.maps->ket_reduced;
            }
        }
    }
    size_t payload_bytes() const {
        return NetworkPayload::rows(sector_signs) + NetworkPayload::rows(bra_phase) + NetworkPayload::rows(ket_phase);
    }
};
struct SamplingNetworkSession {
    const SamplingNetworkPlan& prepared;
    const BoundaryTracePlan& plan;
    ReferenceHistoryBoundary committed;
    ReferenceBoundary query_scratch;
    Bits base, bra_after, ket_after;
    Frame previous, common_previous;
    // Common symbolic rotation update before bra/ket trace cancellation.
    // Its coordinate order is compiler provenance, shared with continuations.
    ReferenceTransform common;
    Quadratic quadratic;
    std::optional<PhaseResource> resource;
    PhaseTensor tensor;
    int constant = 0;
    const CliffordContraction* elimination = nullptr;
    std::optional<CliffordContraction> changed_elimination;
    std::optional<StabilizerTerms> transported_terms;
    size_t geometry_updates = 0, geometry_reuses = 0, term_visits = 0, tensor_expansions = 0;
    size_t resource_constructions = 0, coordinate_constructions = 0;
    bool zero = false;
    SamplingNetworkSession(const SamplingNetworkPlan& prepared, const BoundaryTracePlan& plan, const Bits& sector,
            const PhaseResource* local_resource = nullptr)
        : prepared(prepared), plan(plan), committed(*prepared.parent, sector) {
        common.exponents = plan.compiled.plan.magic;
        const auto& n = *prepared.parent->reference;
        Bits rhs;
        for(size_t j = 0; j < sector.size(); j++)
            if(sector.bit(j)) rhs.set(j);
        for(size_t j = 0; j < n.reference_outcomes.size(); j++)
            if(n.reference_outcomes[j]) rhs.flip(j);
        if(!n.event_support->factor.consistent(rhs)) {
            zero = true;
            return;
        }
        auto shift = n.event_support->factor.solve(rhs);
        if(!prepared.certificate) return;
        if(prepared.contraction) quadratic = prepared.contraction->pulled_reference;
        else {
            resource = local_resource ? *local_resource : prepared.certificate->reference;
            quadratic = {0, resource->quotient.linear, resource->quotient.upper};
            tensor = n.restricted.clifford;
            tensor.boundary.resize(n.boundary.keep.size());
        }
        for(size_t j = 0; j < quadratic.linear.size(); j++)
            quadratic.linear[j] = mod4(quadratic.linear[j] + 2 * prepared.sector_signs[j].parity_and(shift));
        constant = mod8(n.restricted.clifford.constant + n.phase_constant +
                2 * qvalue(shift, n.traced.clifford.linear, n.traced.clifford.upper));
        for(size_t j = 0; j < n.boundary.keep.size(); j++)
            if(n.restricted.clifford.boundary[j].second ^ n.traced.clifford.boundary[j].first.parity_and(shift))
                base.set(j);
        for(size_t j = 0; j < n.magic.size(); j++) {
            int bra = n.bra_offsets[j] ^ n.maps->bra_magic[j].parity_and(shift);
            int ket = n.ket_offsets[j] ^ n.maps->ket_magic[j].parity_and(shift);
            if(bra) bra_after.set(j);
            if(ket) ket_after.set(j);
            phase_update(j, -n.reference_exponents[j] * ((bra ? -1 : 1) - (n.bra_offsets[j] ? -1 : 1)),
                    n.reference_exponents[j] * ((ket ? -1 : 1) - (n.ket_offsets[j] ? -1 : 1)));
            constant = mod8(constant + n.reference_exponents[j] * (ket - n.ket_offsets[j] - bra + n.bra_offsets[j]));
        }
        if(prepared.contraction) elimination = &prepared.contraction->clifford;
    }
    void phase_update(size_t j, int bra, int ket) {
        auto add = [&](const Bits& row, int coefficient) {
            if(mod8(coefficient)) phase_add_even(quadratic.linear, quadratic.upper, {{row, mod8(coefficient)}});
        };
        if(prepared.bra_phase[j] == prepared.ket_phase[j]) add(prepared.bra_phase[j], bra + ket);
        else {
            add(prepared.bra_phase[j], bra);
            add(prepared.ket_phase[j], ket);
        }
    }
    void update_common(const Frame& frame) {
        if(zero) return;
        const size_t physical = plan.compiled.plan.physical;
        auto changed = (frame.x ^ common_previous.x) | (frame.z ^ common_previous.z);
        changed.each([&](size_t bit) {
            if(bit >= physical && bit < physical + common.exponents.size()) {
                size_t j = bit - physical;
                common.exponents[j] =
                        mod8((frame.x.bit(bit) ? -plan.compiled.plan.magic[j] : plan.compiled.plan.magic[j]) +
                                4 * frame.z.bit(bit));
            }
        });
        common_previous = frame;
    }
    void update(const Frame& frame) {
        update_common(frame);
        if(!prepared.certificate || zero) return;
        const auto& n = *prepared.parent->reference;
        auto changed = (frame.x ^ previous.x) | (frame.z ^ previous.z);
        for(auto bit : changed.indices())
            if(bit >= n.physical && bit < n.physical + n.magic.size()) {
                size_t j = bit - n.physical;
                auto exponent = [&](const Frame& f) {
                    return mod8((f.x.bit(bit) ? -plan.compiled.plan.magic[j] : plan.compiled.plan.magic[j]) +
                            4 * f.z.bit(bit));
                };
                int delta = common.exponents[j] - exponent(previous);
                phase_update(j, -delta * (bra_after.bit(j) ? -1 : 1), delta * (ket_after.bit(j) ? -1 : 1));
                constant = mod8(constant + delta * (int(ket_after.bit(j)) - int(bra_after.bit(j))));
            }
        auto flip = frame.x ^ previous.x;
        for(size_t j = 0; j < n.boundary.keep.size(); j++)
            if(n.boundary.keep[j].parity_and(flip)) base.flip(j);
        previous = frame;
        if(!elimination) return;
        if(!elimination->matches(quadratic)) {
            changed_elimination.emplace(prepared.contraction->columns, quadratic);
            elimination = &*changed_elimination;
            geometry_updates++;
        } else geometry_reuses++;
    }
    const Bits& actual_closed(const Frame& frame) {
        committed.synchronize(frame);
        return committed.actual_closed;
    }
    const StabilizerTerms& expanded(const Frame& frame) {
        update_common(frame);
        // One current transported mother per draw, shared by all boundary
        // contractions until the actual assignment changes. No prefix states.
        if(transported_terms && frame.x == previous.x && frame.z == previous.z) return *transported_terms;
        Workspace ws;
        tensor_expansions++;
        if(prepared.certificate) {
            update(frame);
            if(zero) {
                transported_terms.emplace();
                return *transported_terms;
            }
            resource->quotient.linear = quadratic.linear;
            resource->quotient.upper = quadratic.upper;
            tensor.constant = constant;
            for(size_t j = 0; j < tensor.boundary.size(); j++) tensor.boundary[j].second = base.bit(j);
            transported_terms = phase_states(tensor, *resource, ws, plan.limits);
        } else {
            // A changed algebraic coordinate domain uses its established
            // certificate route; it is not selected by quotient size or purpose.
            std::vector<int> caps;
            const auto& actual = actual_closed(frame);
            for(size_t j = 0; j < prepared.parent->closed.size(); j++) caps.push_back(actual.bit(j));
            auto result = prepared.parent->reference->evaluate(
                    frame, plan.compiled.plan.magic, caps, false, ws, plan.limits, &plan.database, true, &common);
            resource_constructions += result.diagnostics.resource_constructions;
            coordinate_constructions += result.diagnostics.phase_coordinate_constructions;
            transported_terms = std::move(result.terms);
        }
        previous = frame;
        return *transported_terms;
    }
    Ket support(const Frame& frame, const std::vector<Bits>& keep, const std::vector<Bits>& events,
            const std::vector<int>& values) {
        return support_coordinates(
                frame, {prepared.parent->coordinates(keep)}, {prepared.parent->coordinates(events)}, values);
    }
    Ket support_coordinates(const Frame& frame, const ReferenceRows& keep, const ReferenceRows& events,
            const std::vector<int>& values) {
        const auto& parent = *prepared.parent;
        auto boundary = parent.project_coordinates(keep, events, values, actual_closed(frame));
        return support_projected(frame, keep.labels.size(), boundary.keep, boundary.events, boundary.outcomes,
                boundary.offset, boundary.zero);
    }
    Ket support_projected(const Frame& frame, size_t width, const std::vector<Bits>& keep,
            const std::vector<Bits>& events, const std::vector<int>& values, const Bits& offset, bool empty) {
        const auto& parent = *prepared.parent;
        if(empty) {
            Ket support;
            support.width = width;
            support.zero = true;
            return support;
        }
        if(prepared.contraction) {
            update(frame);
            Ket support;
            if(zero) support.zero = true;
            else support = elimination->support(parent.reference->boundary.keep.size(), base, quadratic.linear);
            auto result = project_support(support, keep, events, values);
            result.base ^= offset;
            return result;
        }
        // Each prepared stabilizer term has an affine support represented by
        // a base and columns. Condition/project those spaces and span their
        // union, never their individual words or cross-term cancellations.
        // Work is polynomial per prepared mother term, irrespective of 2^dim.
        AffineProposalEnvelope envelope(width);
        for(const auto& [coefficient, k] : expanded(frame)) {
            term_visits++;
            auto affine = project_support(k, keep, events, values);
            envelope.include(affine);
        }
        envelope.support.base ^= offset;
        return std::move(envelope.support);
    }
    const ReferenceBoundary& query_boundary(const Frame& frame, const ReferenceRows& keep) {
        // Only query-specific keep/trace scratch is materialized. Committed
        // event rows remain in the session and are passed by const reference.
        // The caller consumes this view before the next query. No reference
        // into keep survives its clear/reserve, or an observation append.
        auto& out = query_scratch;
        out.keep.clear();
        out.offset = parities(keep.labels, actual_closed(frame));
        out.keep.reserve(keep.labels.size());
        for(const auto& row : keep.labels) out.keep.push_back(row >> prepared.parent->closed.size());
        return out;
    }
    Ket support_committed(const Frame& frame, const ReferenceRows& keep) {
        const auto& query = query_boundary(frame, keep);
        const auto& b = committed.boundary;
        return support_projected(frame, keep.labels.size(), query.keep, b.events, b.outcomes, query.offset, b.zero);
    }
    StabilizerTerms weighted(const Frame& frame) {
        update(frame);
        if(zero) return {};
        auto k = elimination->evaluate(prepared.parent->reference->boundary.keep.size(), base, quadratic.linear,
                prepared.parent->reference->restricted.clifford.power, constant);
        StabilizerTerms terms;
        if(!k.zero) terms.emplace_back(resource_entries(prepared.certificate->reference)[0].coefficient, std::move(k));
        return terms;
    }
    BoundaryResult query_committed(const Frame& frame, const ReferenceRows& keep) {
        const auto& query = query_boundary(frame, keep);
        const auto& b = committed.boundary;
        BoundaryResult result;
        result.tensor.output_count = keep.labels.size();
        if(!b.zero) {
            Workspace ws;
            if(prepared.contraction) {
                auto terms = weighted(frame);
                result.terms = contract_probability_terms(terms, query.keep, b.events, b.outcomes, ws);
            } else result.terms = contract_probability_terms(expanded(frame), query.keep, b.events, b.outcomes, ws);
            for(auto& [coefficient, k] : result.terms) k.base ^= query.offset;
        }
        result.tensor.zero = result.terms.empty();
        return result;
    }
    Scalar mass_committed(const Frame& frame) {
        committed.synchronize(frame);
        const auto& b = committed.boundary;
        if(b.zero) return {};
        Workspace ws;
        if(prepared.contraction) {
            auto terms = weighted(frame);
            return boundary_sum(terms, b.events, b.outcomes, ws);
        }
        return boundary_sum(expanded(frame), b.events, b.outcomes, ws);
    }
    Scalar mass_coordinates(const Frame& frame, const ReferenceRows& events, const std::vector<int>& values) {
        const auto& parent = *prepared.parent;
        StabilizerTerms terms;
        auto boundary = parent.project_coordinates({}, events, values, actual_closed(frame));
        if(boundary.zero) return {};
        if(prepared.contraction) terms = weighted(frame);
        Workspace ws;
        return boundary_sum(prepared.contraction ? terms : expanded(frame), boundary.events, boundary.outcomes, ws);
    }
};
} // namespace magic_native
