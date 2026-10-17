// A prepared probability tensor owns the resource; record queries only cap it.
#pragma once
#include "boundary_trace.hpp"
#include "../foundation/causal_knowledge.hpp"
namespace magic_native {
inline std::vector<std::pair<Scalar, Ket>> contract_probability_terms(const std::vector<std::pair<Scalar, Ket>>& source,
        const std::vector<Bits>& keep, const std::vector<Bits>& events, const std::vector<int>& values, Workspace& ws) {
    std::vector<std::pair<Scalar, Ket>> terms;
    terms.reserve(source.size());
    for(const auto& [coefficient, k] : source) {
        auto forms = transpose(k.columns, k.width);
        std::vector<Bits> equations;
        Bits rhs;
        for(size_t j = 0; j < events.size(); j++) {
            equations.push_back(xors(forms, events[j]));
            if(values[j] ^ events[j].parity_and(k.base)) rhs.set(j);
        }
        Factor support(equations, k.columns.size());
        if(!support.consistent(rhs)) continue;
        auto shift = support.solve(rhs);
        auto q = restrict_q(k.linear, k.upper, shift, support.kernel);
        auto base = k.base ^ xors(k.columns, shift);
        Ket out;
        out.width = keep.size();
        out.base = parities(keep, base);
        for(const auto& axis : support.kernel) {
            Bits col;
            auto image = xors(k.columns, axis);
            col = parities(keep, image);
            out.columns.push_back(std::move(col));
        }
        out.linear = std::move(q.linear);
        out.upper = std::move(q.upper);
        out.phase = mod8(k.phase + 2 * q.constant);
        out.power = k.power;
        out = contract(std::move(out), ws);
        if(!out.zero) terms.push_back({coefficient, std::move(out)});
    }
    return terms;
}
inline BoundaryResult contract_probability_labels(BoundaryResult parent, const std::vector<Bits>& keep,
        const std::vector<Bits>& events, const std::vector<int>& values, Workspace& ws) {
    parent.terms = contract_probability_terms(parent.terms, keep, events, values, ws);
    parent.tensor.output_count = keep.size();
    parent.tensor.zero = parent.terms.empty();
    parent.affine = false;
    return parent;
}
// Invocation-local coordinates on the open labels. Closed labels contribute
// only affine offsets; they never need to be appended to every mother term.
struct ReferenceBoundary {
    std::vector<Bits> keep, events;
    std::vector<int> outcomes;
    Bits offset;
    bool zero = false;
};
// Coordinates certified at preparation, or linear combinations of certified
// generators. This internal type never accepts unchecked public query rows.
struct ReferenceRows {
    std::vector<Bits> labels;
};
// Fixed linear inverse on the label span. Resolve the triangular solve once,
// then store its images on physical unit rows, not on histories or parities.
// Reconstructing a row from its coordinates checks span membership without
// retaining the factor's physical-width family of consistency equations.
struct ReferenceCoordinates {
    std::vector<Bits> basis, images;
    ReferenceCoordinates() = default;
    ReferenceCoordinates(const std::vector<Bits>& rows, size_t width) : basis(rows) {
        Factor factor(transpose(basis, width), basis.size());
        std::vector<Bits> forms(basis.size());
        for(const auto& [j, rt] : factor.pivots) forms[j] = rt.second ^ xors(forms, rt.first);
        images = transpose(forms, width);
    }
    std::optional<Bits> operator()(const Bits& row) const {
        if(row.size() > images.size()) return {};
        auto coordinates = xors(images, row);
        if(xors(basis, coordinates) != row) return {};
        return coordinates;
    }
};
struct ReferenceNetwork {
    std::shared_ptr<const CompiledNetwork> reference;
    ReferenceCoordinates labels;
    std::vector<int> selection;
    std::vector<Bits> closed, closed_images;
    size_t capped_history = 0;
    ReferenceNetwork(BoundaryTracePlan& plan, const std::vector<Bits>& kept, const std::vector<Bits>& events,
            const std::vector<int>& outcomes, const std::vector<Bits>& commuting = {}, size_t terminal_history = 0)
        : selection(outcomes), closed(commuting) {
        auto basis = closed;
        CausalKnowledge independent;
        for(const auto& row : basis) independent.insert(row);
        std::vector<Bits> retained;
        for(size_t j = 0; j < kept.size(); j++)
            if(independent.insert(kept[j]).first) {
                basis.push_back(kept[j]);
                retained.push_back(kept[j]);
                if(j < terminal_history) capped_history++;
            }
        labels = ReferenceCoordinates(basis, plan.compiled.plan.physical);
        closed_images = transpose(closed, plan.compiled.plan.physical);
        auto caps = events;
        caps.insert(caps.end(), closed.begin(), closed.end());
        // The compiler chooses compatible affine reference outcomes for these
        // caps. They are transport coordinates, never a physical early draw.
        caps.insert(caps.end(), retained.begin(), retained.begin() + capped_history);
        retained.erase(retained.begin(), retained.begin() + capped_history);
        reference = plan.compile({retained, caps, {}, {}}, true, true);
    }
    std::vector<Bits> coordinates(const std::vector<Bits>& rows) const {
        std::vector<Bits> out;
        for(const auto& row : rows) {
            auto value = labels(row);
            if(!value) throw std::invalid_argument("query is outside the compiled reference network labels");
            out.push_back(std::move(*value));
        }
        return out;
    }
    Bits closed_values(const Frame& frame, const Bits& virtual_sector) const {
        auto out = virtual_sector.low(closed.size());
        xor_rows_prefix(out, closed_images, frame.x, closed_images.size());
        return out;
    }
    ReferenceBoundary project(const std::vector<Bits>& keep, const std::vector<Bits>& events,
            const std::vector<int>& outcomes, const Bits& actual_closed) const {
        return project_coordinates({coordinates(keep)}, {coordinates(events)}, outcomes, actual_closed);
    }
    ReferenceBoundary project_coordinates(const ReferenceRows& keep, const ReferenceRows& events,
            const std::vector<int>& outcomes, const Bits& actual_closed) const {
        ReferenceBoundary out;
        const auto& fixed = events.labels;
        out.keep.reserve(keep.labels.size());
        out.offset = parities(keep.labels, actual_closed);
        for(const auto& row : keep.labels) { out.keep.push_back(row >> closed.size()); }
        for(size_t j = 0; j < fixed.size(); j++) {
            int bit = outcomes[j] ^ fixed[j].parity_and(actual_closed);
            auto row = fixed[j] >> closed.size();
            if(row.empty()) {
                if(bit) {
                    out.zero = true;
                    return out;
                }
            } else {
                out.events.push_back(std::move(row));
                out.outcomes.push_back(bit);
            }
        }
        return out;
    }
    BoundaryResult query(const BoundaryTracePlan& plan, const Frame& frame, const std::vector<Bits>& keep,
            const std::vector<Bits>& events, const std::vector<int>& outcomes, Workspace& ws,
            const Bits& virtual_sector = {}) const {
        return query_coordinates(plan, frame, {coordinates(keep)}, {coordinates(events)}, outcomes, ws, virtual_sector);
    }
    std::optional<Bits> affine_coordinates(
            const ReferenceRows& rows, const Frame& frame, const Bits& virtual_sector) const {
        // Membership in the closed subspace is an actual mathematical branch.
        for(const auto& row : rows.labels)
            if(row.size() > closed.size()) return {};
        return parities(rows.labels, closed_values(frame, virtual_sector));
    }
    BoundaryResult query_coordinates(const BoundaryTracePlan& plan, const Frame& frame, const ReferenceRows& keep,
            const ReferenceRows& events, const std::vector<int>& outcomes, Workspace& ws,
            const Bits& virtual_sector = {}, const PhaseResource* local_resource = nullptr) const {
        auto caps = selection;
        auto actual = closed_values(frame, virtual_sector);
        for(size_t j = 0; j < closed.size(); j++) caps.push_back(actual.bit(j));
        auto boundary = project_coordinates(keep, events, outcomes, actual);
        if(boundary.zero) {
            BoundaryResult result;
            result.tensor.zero = true;
            result.tensor.output_count = keep.labels.size();
            return result;
        }
        if(capped_history) {
            std::vector<Bits> equations;
            Bits rhs;
            for(size_t j = 0; j < boundary.events.size(); j++)
                if(boundary.events[j].size() <= capped_history) {
                    if(boundary.outcomes[j]) rhs.set(equations.size());
                    equations.push_back(boundary.events[j]);
                }
            Factor completed(equations, capped_history);
            if(!completed.consistent(rhs)) {
                BoundaryResult result;
                result.tensor.zero = true;
                return result;
            }
            if(!completed.kernel.empty())
                throw std::invalid_argument("terminal reference requires the complete causal history");
            auto values = completed.solve(rhs);
            for(size_t j = 0; j < capped_history; j++) caps.push_back(values.bit(j));
            for(size_t j = 0; j < boundary.events.size(); j++) {
                boundary.outcomes[j] ^= boundary.events[j].parity_and(values);
                boundary.events[j] = boundary.events[j] >> capped_history;
            }
            for(size_t j = 0; j < boundary.keep.size(); j++) {
                if(boundary.keep[j].parity_and(values)) boundary.offset.flip(j);
                boundary.keep[j] = boundary.keep[j] >> capped_history;
            }
        }
        auto parent = reference->evaluate(
                frame, plan.compiled.plan.magic, caps, false, ws, plan.limits, &plan.database, true, nullptr,
                local_resource);
        auto result =
                contract_probability_labels(std::move(parent), boundary.keep, boundary.events, boundary.outcomes, ws);
        for(auto& [coefficient, ket] : result.terms) ket.base ^= boundary.offset;
        return result;
    }
};
// One current observation boundary, expressed in the parent's open coordinates.
// Echelon rows retain their closed-coordinate pullback, so subsequent assignment
// changes update RHSs without replaying observations or losing traced directions.
// Incidence stores indices, never pointers into growable row vectors.
struct ReferenceHistoryBoundary {
    const ReferenceNetwork& parent;
    ReferenceBoundary boundary;
    Bits actual_closed, previous_x;
    std::vector<Bits> closed_parts;
    std::vector<size_t> pivots, open_slots;
    std::vector<int> rhs;
    std::vector<std::vector<size_t>> closed_incidence, physical_incidence;
    size_t contradictions = 0, appended_rows = 0, rhs_updates = 0;
    static constexpr size_t absent = std::numeric_limits<size_t>::max();
    ReferenceHistoryBoundary(const ReferenceNetwork& parent, const Bits& sector)
        : parent(parent), actual_closed(sector.low(parent.closed.size())), closed_incidence(parent.closed.size()),
          physical_incidence(parent.closed_images.size()) {}
    void toggle(size_t index) {
        auto slot = open_slots[index];
        if(slot == absent) {
            if(rhs[index]) contradictions--;
            else contradictions++;
        } else boundary.outcomes[slot] ^= 1;
        rhs[index] ^= 1;
        rhs_updates++;
    }
    void synchronize(const Frame& frame) {
        if(frame.x == previous_x) return;
        auto delta = frame.x ^ previous_x;
        Bits closed_delta;
        xor_rows_prefix(closed_delta, parent.closed_images, delta, parent.closed_images.size());
        actual_closed ^= closed_delta;
        closed_delta.each([&](size_t bit) {
            for(auto index : closed_incidence[bit]) toggle(index);
        });
        delta.each([&](size_t bit) {
            if(bit < physical_incidence.size())
                for(auto index : physical_incidence[bit]) toggle(index);
        });
        previous_x = frame.x;
        boundary.zero = contradictions != 0;
    }
    // Resolve through the echelon rows maintained during causal observation.
    // No new factorization, history matrix or resource is constructed here.
    int value(const Bits& coordinates) const {
        int result = coordinates.parity_and(actual_closed);
        auto row = coordinates >> parent.closed.size();
        while(!row.empty()) {
            size_t pivot = row.size() - 1;
            if(pivot >= pivots.size() || pivots[pivot] == absent)
                throw std::invalid_argument("terminal reference requires the complete causal history");
            size_t slot = pivots[pivot];
            row ^= boundary.events[slot];
            result ^= boundary.outcomes[slot];
        }
        return result;
    }
    void append(const Bits& coordinates, int outcome, const Bits* physical = nullptr) {
        appended_rows++;
        auto row = coordinates >> parent.closed.size();
        int value = outcome ^ coordinates.parity_and(actual_closed);
        // Only wholly closed certified rows have this physical-frame pullback.
        const bool direct = row.empty() && physical;
        auto closed = direct ? Bits{} : coordinates.low(parent.closed.size());
        while(!row.empty()) {
            auto pivot = row.size() - 1;
            if(pivot >= pivots.size() || pivots[pivot] == absent) break;
            auto slot = pivots[pivot];
            row ^= boundary.events[slot];
            closed ^= closed_parts[slot];
            value ^= boundary.outcomes[slot];
        }
        size_t index = rhs.size();
        rhs.push_back(value);
        if(row.empty()) {
            open_slots.push_back(absent);
            contradictions += value;
        } else {
            auto pivot = row.size() - 1;
            if(pivot >= pivots.size()) pivots.resize(pivot + 1, absent);
            auto slot = boundary.events.size();
            pivots[pivot] = slot;
            open_slots.push_back(slot);
            boundary.events.push_back(std::move(row));
            boundary.outcomes.push_back(value);
            closed_parts.push_back(closed);
        }
        if(direct) physical->each([&](size_t bit) { physical_incidence.at(bit).push_back(index); });
        else closed.each([&](size_t bit) { closed_incidence[bit].push_back(index); });
        boundary.zero = contradictions != 0;
    }
};
// Compose record identity -> physical row -> label coordinates once. Only the
// validated observation-generator span may use this extension of the map.
struct ReferenceRecordMap {
    // Copies and moves retain the immutable coordinate owner, never its rows.
    std::shared_ptr<const ReferenceNetwork> parent;
    std::vector<size_t> physical;
    ReferenceRecordMap(std::shared_ptr<const ReferenceNetwork> reference, std::vector<size_t> records,
            const std::vector<Bits>& generators)
        : parent(std::move(reference)), physical(std::move(records)) {
        parent->coordinates(generators);
        for(auto index : physical) (void)parent->labels.images.at(index);
    }
    ReferenceRows operator()(const std::vector<Bits>& masks) const {
        ReferenceRows out;
        out.labels.reserve(masks.size());
        const auto& images = parent->labels.images;
        for(const auto& mask : masks) {
            Bits value;
            mask.each([&](size_t index) {
                if(index >= physical.size()) throw std::invalid_argument("mask outside vector");
                const auto& row = images[physical[index]];
                if(value.w.size() < row.w.size()) value.w.resize(row.w.size());
                for(size_t j = 0; j < row.w.size(); j++) value.w[j] ^= row.w[j];
            });
            value.trim();
            out.labels.push_back(std::move(value));
        }
        return out;
    }
    std::optional<Bits> affine_values(const std::vector<Bits>& masks, const ReferenceRows& rows, size_t closed,
            const Frame& frame, const Bits& sector) const {
        for(const auto& row : rows.labels)
            if(row.size() > closed) return {};
        Bits out;
        for(size_t j = 0; j < masks.size(); j++) {
            int value = rows.labels[j].parity_and(sector);
            masks[j].each([&](size_t bit) { value ^= frame.x.bit(physical.at(bit)); });
            if(value) out.set(j);
        }
        return out;
    }
};
// Only changes since the preceding causal query are propagated. Global ket
// phase cancels in these paired networks; all relative Pauli signs survive.
struct ReferenceAssignment {
    Frame frame;
    void apply(const Frame& change) {
        frame.x ^= change.x;
        frame.z ^= change.z;
    }
    void append(const Plan& plan, std::vector<Fault> delta) { apply(plan.fault_frame(std::move(delta), false)); }
};
} // namespace magic_native
