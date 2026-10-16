// Boundary-specific preparation and resource ownership, separate from updates.
#pragma once
#include "trace_query.hpp"
namespace magic_native {
struct BoundarySpec {
    std::vector<Bits> keep, events;
    Bits x, z;
    bool operator<(const BoundarySpec& b) const {
        auto rows_less = [](const std::vector<Bits>& a, const std::vector<Bits>& b) {
            return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end(), BitsLess{});
        };
        if(keep != b.keep) return rows_less(keep, b.keep);
        if(events != b.events) return rows_less(events, b.events);
        if(x != b.x) return BitsLess{}(x, b.x);
        return BitsLess{}(z, b.z);
    }
};
// Homogeneous maps are owned once per exact traced coordinate structure.
// Pauli effects change affine offsets and even linear coefficients; those
// remain in each CompiledNetwork. This index exists only during preparation.
struct NetworkMaps {
    size_t dimension = 0, kept = 0;
    std::vector<Bits> forms, bra, ket, upper, event_polar;
    std::vector<Bits> restricted_forms, restricted_bra, restricted_ket;
    std::vector<int> linear;
    std::shared_ptr<const TraceFactor> events;
    Quadratic restricted;
    mutable std::vector<ResourceCertificate> algebra;
    std::vector<Bits> bra_magic, ket_magic, bra_reduced, ket_reduced;
    static std::vector<Bits> masks(const PhaseTerms& terms) {
        std::vector<Bits> rows;
        for(const auto& term : terms) rows.push_back(term.first);
        return rows;
    }
    bool matches(const TracePairingKernel& traced, size_t keep) const {
        const auto& t = traced.clifford;
        if(dimension != t.n || kept != keep || upper != t.upper) return false;
        for(size_t j = 0; j < t.n; j++)
            if((linear[j] ^ t.linear[j]) & 1) return false;
        auto same = [](const std::vector<Bits>& rows, const PhaseTerms& terms) {
            if(rows.size() != terms.size()) return false;
            for(size_t j = 0; j < rows.size(); j++)
                if(rows[j] != terms[j].first) return false;
            return true;
        };
        return same(forms, t.boundary) && same(bra, traced.bra) && same(ket, traced.ket);
    }
    NetworkMaps(const TracePairingKernel& traced, const PhaseTerms& magic, size_t keep)
        : dimension(traced.clifford.n), kept(keep), forms(masks(traced.clifford.boundary)), bra(masks(traced.bra)),
          ket(masks(traced.ket)), upper(traced.clifford.upper), linear(traced.clifford.linear) {
        std::vector<Bits> equations(forms.begin() + keep, forms.end());
        events = std::make_shared<TraceFactor>(equations, dimension);
        auto restrict = [&](const std::vector<Bits>& rows, std::vector<Bits>& mapped) {
            for(const auto& row : rows) mapped.push_back(xors(events->dual, row));
        };
        restrict(std::vector<Bits>(forms.begin(), forms.begin() + keep), restricted_forms);
        restrict(bra, restricted_bra);
        restrict(ket, restricted_ket);
        restricted = restrict_q(linear, upper, {}, events->factor.kernel);
        auto polar = upper;
        for(size_t j = 0; j < dimension; j++) {
            if(linear[j] & 1) polar[j].set(j);
            for(auto k : upper[j].indices()) polar[k].set(j);
        }
        for(const auto& axis : events->factor.kernel) event_polar.push_back(xors(polar, axis));
        auto map = [&](const std::vector<Bits>& coords, std::vector<Bits>& original, std::vector<Bits>& reduced) {
            for(const auto& [mask, offset] : magic) {
                auto row = xors(coords, mask);
                original.push_back(row);
                reduced.push_back(xors(events->dual, row));
            }
        };
        map(bra, bra_magic, bra_reduced);
        map(ket, ket_magic, ket_reduced);
    }
};
// Preparation-only output-open Clifford dilation. Its phase coordinates are
// retained symbolically, so all original-location Pauli updates survive trace.
struct TracePreparation {
    DeferredTraceSide side;
    mutable std::optional<TracePairingGeometry> geometry;
    PhaseTerms magic;
    size_t physical = 0;
    mutable std::vector<std::shared_ptr<const NetworkMaps>> maps;
    mutable bool resource_prepared = false;
    mutable std::shared_ptr<const ResourceCertificate> phase_resource;
    mutable std::string resource_limit;
    // Consume only the temporary dilation; the Plan keeps its transport and
    // magic exponents for all later assignment queries.
    TracePreparation(Ket skeleton, size_t physical, const std::vector<int>& exponents) : physical(physical) {
        auto forms = transpose(skeleton.columns, skeleton.width);
        std::vector<Bits> masks(forms.begin() + physical, forms.begin() + physical + exponents.size());
        auto axes = phase_image_basis(masks);
        auto dual = transpose(phase_right_inverse(axes, skeleton.columns.size()), skeleton.columns.size());
        PhaseCounter phase;
        for(size_t j = 0; j < masks.size(); j++) {
            auto row = xors(dual, masks[j]);
            int bit = skeleton.base.bit(physical + j);
            magic.push_back({row, bit});
            phase[row] = mod8(phase[row] + exponents[j] * (bit ? -1 : 1));
        }
        side.phase = phase_terms(phase);
        side.phase_axes = axes.size();
        side.trace_width = physical;
        std::vector<Bits> rows(forms.begin(), forms.begin() + physical);
        rows.insert(rows.end(), axes.begin(), axes.end());
        auto columns = transpose(rows, skeleton.columns.size());
        side.clifford = std::move(skeleton);
        side.clifford.width = rows.size();
        side.clifford.columns = std::move(columns);
        for(auto j : side.clifford.base.indices())
            if(j >= physical) side.clifford.base.flip(j);
        Workspace ws;
        side.clifford = contract(std::move(side.clifford), ws, false);
    }
};
// Symbolic phase data in compiler rotation-slot coordinates, before parent
// trace/selection cancellation. Terminal maps pull these signed coefficients
// into their own domains and append only their changed-cap signs and constants.
struct ReferenceTransform {
    std::vector<int> exponents;
};
struct CompiledNetwork {
    BoundarySpec boundary;
    size_t input_dimension = 0, physical = 0, phase_dimension = 0;
    std::shared_ptr<const NetworkMaps> maps;
    bool shared_maps = false;
    std::shared_ptr<const TraceFactor> event_support;
    std::vector<int> reference_outcomes;
    PhaseTerms magic, fixed_phase;
    std::vector<int> bra_offsets, ket_offsets;
    PhaseTensor fixed_source;
    std::vector<int> reference_exponents;
    int phase_constant = 0;
    TracePairingKernel traced, restricted;
    std::optional<ResourceCertificate> joint;
    std::optional<ResourceCertificate> stable_pairing;
    std::shared_ptr<const ResourceCertificate> inherited;
    std::vector<ResourceCertificate> reduction_algebra;
    TraceDiagnostics reference_diagnostics;
    size_t reference_rank = 0, reference_terms = 0;
    bool reference_zero = false, resources_ready = false, stable_requested = false, joint_fixed_domain = false;
    double reduction_seconds = 0, resource_seconds = 0;
    std::string joint_limit, inherited_limit;

    size_t uniform_term_bound() const {
        if(!inherited) return 0;
        auto count = inherited->term_bound();
        if(count && count > std::numeric_limits<size_t>::max() / count)
            throw std::length_error("resource term bound capacity");
        return count * count;
    }
    TraceReduction reduce(const Frame& frame, const std::vector<int>& exponents, const std::vector<int>& outcomes,
            Workspace& ws, std::vector<ResourceCertificate>* prepare_algebra = nullptr, bool reference_only = false,
            bool absorb = true, const ReferenceTransform* common = nullptr) const {
        TraceReduction reduced;
        auto& d = reduced.result.diagnostics;
        auto& b = reduced.result.tensor;
        d.prepared_query = true;
        d.input_certificate_covered = true;
        d.output_trace_reused = true;
        d.pairing_hit = true;
        d.support_hit = true;
        d.input_bits = input_dimension;
        d.event_rows = boundary.events.size();
        d.phase_axes = phase_dimension;
        auto tick = std::chrono::steady_clock::now();
        auto elapsed = [&]() {
            auto now = std::chrono::steady_clock::now();
            double dt = std::chrono::duration<double>(now - tick).count();
            tick = now;
            return dt;
        };
        b.output_count = boundary.keep.size();
        b.boundary.resize(boundary.keep.size());
        if(traced.clifford.zero) {
            b.zero = true;
            return reduced;
        }
        auto& gamma = reduced.pairing;
        gamma = restricted;
        auto& t = gamma.clifford;
        Bits rhs;
        for(size_t j = 0; j < outcomes.size(); j++)
            if(outcomes[j] ^ traced.clifford.boundary[boundary.keep.size() + j].second ^
                    boundary.events[j].parity_and(frame.x))
                rhs.set(j);
        d.event_rank = event_support->factor.pivots.size();
        d.support_bits = event_support->factor.kernel.size();
        if(!event_support->factor.consistent(rhs)) {
            b.zero = true;
            d.support_seconds = elapsed();
            return reduced;
        }
        auto shift = event_support->factor.solve(rhs);
        for(size_t j = 0; j < t.n; j++) t.linear[j] = mod4(t.linear[j] + 2 * maps->event_polar[j].parity_and(shift));
        t.constant = mod8(t.constant + 2 * qvalue(shift, traced.clifford.linear, traced.clifford.upper) +
                4 * (boundary.x.parity_and(frame.z) ^ boundary.z.parity_and(frame.x)));
        auto offset = [&](PhaseTerms& rows, const PhaseTerms& original) {
            for(size_t j = 0; j < rows.size(); j++) rows[j].second ^= original[j].first.parity_and(shift);
        };
        offset(t.boundary, traced.clifford.boundary);
        offset(gamma.bra, traced.bra);
        offset(gamma.ket, traced.ket);
        for(size_t j = 0; j < t.boundary.size(); j++) t.boundary[j].second ^= boundary.keep[j].parity_and(frame.x);
        d.support_seconds = elapsed();
        auto& source = reduced.source;
        source = fixed_source;
        reduced.has_source = true;
        b = t;
        b.terms = fixed_phase;
        b.constant = mod8(b.constant + phase_constant);
        PhaseTerms increments, source_increments;
        for(size_t j = 0; j < magic.size(); j++) {
            int reference = reference_exponents[j];
            int actual = common && j < common->exponents.size()
                    ? common->exponents[j]
                    : mod8((frame.x.bit(physical + j) ? -exponents[j] : exponents[j]) + 4 * frame.z.bit(physical + j));
            int delta = mod8((actual - reference) * (magic[j].second ? -1 : 1));
            if(delta) source_increments.push_back({magic[j].first, delta});
            auto update = [&](const std::vector<Bits>& original, const std::vector<int>& offsets, int sign) {
                int before = offsets[j], after = before ^ original[j].parity_and(shift);
                int signed_actual = actual * (after ? -1 : 1);
                int difference = mod8(sign * (signed_actual - reference * (before ? -1 : 1)));
                b.constant = mod8(b.constant + sign * (actual * after - reference * before));
                return difference;
            };
            int bra = update(maps->bra_magic, bra_offsets, -1), ket = update(maps->ket_magic, ket_offsets, 1);
            // Equal fixed directions can cancel before expanding the Clifford
            // update. The constant phases above still enter exactly.
            if(maps->bra_reduced[j] == maps->ket_reduced[j]) {
                bra = mod8(bra + ket);
                ket = 0;
            }
            if(bra) increments.push_back({maps->bra_reduced[j], bra});
            if(ket) increments.push_back({maps->ket_reduced[j], ket});
        }
        // Pauli propagation changes only Clifford data on the prepared phase
        // coordinates. Do not rebuild the parity polynomial or its quotient.
        phase_add_even(b.linear, b.upper, increments);
        phase_add_even(source.linear, source.upper, source_increments);
        d.actual_seconds = elapsed();
        if(!absorb) return reduced;
        auto algebra = [&](const PhaseTensor& actual, size_t stage) {
            if(prepare_algebra) {
                if(stage < maps->algebra.size())
                    if(auto transported = maps->algebra[stage].transport(actual)) {
                        prepare_algebra->emplace_back(actual, std::move(*transported));
                        return prepare_algebra->back().reference.quotient;
                    }
                PhaseResource resource;
                resource.quotient = phase_quotient(actual);
                prepare_algebra->emplace_back(actual, std::move(resource));
                if(stage == maps->algebra.size()) maps->algebra.push_back(prepare_algebra->back());
                return prepare_algebra->back().reference.quotient;
            }
            if(stage < reduction_algebra.size()) {
                auto transported = reduction_algebra[stage].transport(actual, nullptr, reference_only);
                if(transported) return std::move(transported->quotient);
            }
            if(std::all_of(
                       actual.terms.begin(), actual.terms.end(), [](const auto& term) { return !(term.second & 1); })) {
                PolynomialQuotient q;
                q.linear = actual.linear;
                q.upper = actual.upper;
                phase_add_even(q.linear, q.upper, actual.terms);
                return q;
            }
            // A changed contraction support can introduce a different coordinate
            // domain. Only that actual-data branch needs a new coordinate map.
            // The prepared reference certifies every transported query domain.
            if(reference_only)
                throw std::logic_error("reference network does not cover this reduced phase coordinate domain");
            d.phase_coordinate_constructions++;
            return phase_quotient(actual);
        };
        b = absorb_phase(std::move(b), ws, false, algebra);
        d.reduce_seconds = elapsed();
        return reduced;
    }
    const ResourceCertificate* fixed_certificate() const {
        if(stable_pairing) return &*stable_pairing;
        return joint_fixed_domain && joint ? &*joint : nullptr;
    }
    BoundaryResult evaluate_fixed(const ResourceCertificate& certificate, const Frame& frame,
            const std::vector<int>& exponents, const std::vector<int>& outcomes, Workspace& ws,
            const PhaseLimits& limits, const ReferenceTransform* common,
            const PhaseResource* local_resource = nullptr) const {
        BoundaryResult result;
        auto& tensor = result.tensor;
        auto& diagnostics = result.diagnostics;
        diagnostics.prepared_query = true;
        diagnostics.input_certificate_covered = true;
        diagnostics.output_trace_reused = true;
        diagnostics.reference_entries_reused = true;
        diagnostics.pairing_hit = true;
        diagnostics.support_hit = true;
        diagnostics.input_bits = input_dimension;
        diagnostics.event_rows = boundary.events.size();
        diagnostics.phase_axes = phase_dimension;
        tensor = restricted.clifford;
        Bits rhs;
        for(size_t j = 0; j < outcomes.size(); j++)
            if(outcomes[j] ^ reference_outcomes[j] ^ boundary.events[j].parity_and(frame.x)) rhs.set(j);
        if(!event_support->factor.consistent(rhs)) {
            tensor.zero = true;
            return result;
        }
        auto shift = event_support->factor.solve(rhs);
        result.resource = local_resource ? *local_resource : certificate.reference;
        auto& q = result.resource.quotient;
        for(size_t j = 0; j < tensor.n; j++)
            q.linear[j] = mod4(q.linear[j] + 2 * maps->event_polar[j].parity_and(shift));
        tensor.constant = mod8(tensor.constant + phase_constant +
                2 * qvalue(shift, traced.clifford.linear, traced.clifford.upper) +
                4 * (boundary.x.parity_and(frame.z) ^ boundary.z.parity_and(frame.x)));
        for(size_t j = 0; j < tensor.boundary.size(); j++)
            tensor.boundary[j].second ^=
                    traced.clifford.boundary[j].first.parity_and(shift) ^ boundary.keep[j].parity_and(frame.x);
        PhaseTerms increments;
        for(size_t j = 0; j < magic.size(); j++) {
            int actual = common && j < common->exponents.size()
                    ? common->exponents[j]
                    : mod8((frame.x.bit(physical + j) ? -exponents[j] : exponents[j]) + 4 * frame.z.bit(physical + j));
            auto coefficient = [&](int before, const Bits& row, int sign) {
                int after = before ^ row.parity_and(shift);
                tensor.constant = mod8(tensor.constant + sign * (actual * after - reference_exponents[j] * before));
                return mod8(sign * (actual * (after ? -1 : 1) - reference_exponents[j] * (before ? -1 : 1)));
            };
            int bra = coefficient(bra_offsets[j], maps->bra_magic[j], -1);
            int ket = coefficient(ket_offsets[j], maps->ket_magic[j], 1);
            if(maps->bra_reduced[j] == maps->ket_reduced[j]) {
                bra = mod8(bra + ket);
                ket = 0;
            }
            if(bra) increments.push_back({maps->bra_reduced[j], bra});
            if(ket) increments.push_back({maps->ket_reduced[j], ket});
        }
        phase_add_even(q.linear, q.upper, increments);
        result.terms = phase_states(tensor, result.resource, ws, limits);
        result.resource.constructive_terms = result.terms.size();
        diagnostics.expanded_entries = resource_entries(result.resource).size();
        return result;
    }
    BoundaryResult evaluate(const Frame& frame, const std::vector<int>& exponents, const std::vector<int>& outcomes,
            bool affine_only, Workspace& ws, const PhaseLimits& limits, const PhaseDatabase* database,
            bool reference_only = false, const ReferenceTransform* common = nullptr,
            const PhaseResource* local_resource = nullptr) const {
        if(reference_only && !affine_only)
            if(auto certificate = fixed_certificate())
                return evaluate_fixed(*certificate, frame, exponents, outcomes, ws, limits, common, local_resource);

        return expand_trace_query(reduce(frame, exponents, outcomes, ws, nullptr, reference_only, true, common),
                affine_only, ws, limits, database, joint ? &*joint : nullptr, inherited ? &*inherited : nullptr,
                reference_only);
    }
};
struct QueryResourceSummary {
    size_t queries = 0, reference_coordinates = 0, reference_terms = 0, uniform_terms = 0;
    size_t max_decomposition_entries = 0, sum_decomposition_entries = 0;
    bool all_uniform = true;
    void include(const CompiledNetwork& network) {
        queries++;
        reference_coordinates = std::max(reference_coordinates, network.reference_rank);
        reference_terms = std::max(reference_terms, network.reference_terms);
        size_t entries = network.stable_pairing ? network.stable_pairing->term_bound()
                : network.joint                 ? network.joint->term_bound()
                                                : network.uniform_term_bound();
        if(!network.stable_pairing && network.inherited && network.uniform_term_bound() < entries)
            entries = network.uniform_term_bound();
        max_decomposition_entries = std::max(max_decomposition_entries, entries);
        sum_decomposition_entries += entries;
        all_uniform &= bool(network.inherited);
        if(network.inherited) uniform_terms = std::max(uniform_terms, network.uniform_term_bound());
    }
};
// Logical arithmetic payload, excluding container/allocator overhead. Shared
// maps and phase entries are counted once across the complete parent family.
struct NetworkPayload {
    std::set<const void*> shared;
    static size_t bits(const Bits& row) { return 8 * row.w.size(); }
    static size_t rows(const std::vector<Bits>& values) {
        size_t n = 8 * values.size();
        for(const auto& row : values) n += bits(row);
        return n;
    }
    static size_t phase(const PhaseTerms& values) {
        size_t n = 12 * values.size();
        for(const auto& item : values) n += bits(item.first);
        return n;
    }
    static size_t tensor(const PhaseTensor& t) {
        return 32 + 4 * t.linear.size() + rows(t.upper) + phase(t.terms) + phase(t.boundary);
    }
    static size_t factor(const Factor& f) {
        size_t n = rows(f.zeros) + rows(f.kernel);
        for(const auto& [i, p] : f.pivots) n += 8 + bits(p.first) + bits(p.second);
        return n;
    }
    size_t trace(const std::shared_ptr<const TraceFactor>& p) {
        if(!p || !shared.insert(p.get()).second) return 0;
        return 8 + rows(p->rows) + rows(p->dual) + factor(p->factor);
    }
    static size_t quadratic(const Quadratic& q) { return 4 * q.linear.size() + rows(q.upper) + 4; }
    size_t contraction(const std::shared_ptr<const PhaseContraction>& plan) {
        if(!plan || !shared.insert(plan.get()).second) return 0;
        const auto& c = plan->clifford;
        size_t n = rows(plan->rows) + rows(plan->change) + rows(plan->columns) + rows(plan->axes) +
                quadratic(plan->reference) + quadratic(plan->pulled_reference) + 4 * c.parity.size() +
                rows(c.input_upper) + rows(c.columns) + rows(c.upper) + 8 * c.keep.size();
        for(const auto& step : c.steps) n += 17 + bits(step.support) + bits(step.neighbors) + bits(step.column);
        return n;
    }
    size_t resource(const PhaseResource& r) {
        const auto& q = r.quotient;
        size_t n = rows(q.axes) + phase(q.core) +
                4 * q.linear.size() + rows(q.upper) + rows(q.lifts) + 4 * r.core_phase_mod8.size();
        if(!r.prepared_entries || shared.insert(r.prepared_entries.get()).second)
            for(const auto& term : resource_entries(r)) {
                n += phase(term.constraints) + phase(term.parities) + 8;
                for(const auto& value : term.coefficient.c) n += 1 + 4 * value.d.size();
            }
        if(r.entry_plans && shared.insert(r.entry_plans.get()).second) {
            n += 8 + rows(r.entry_plans->rows) + quadratic(r.entry_plans->reference);
            for(const auto& change : r.entry_plans->changes) n += rows(change);
            for(const auto& p : r.entry_plans->entries) {
                n += bits(p.shift) + bits(p.base) + rows(p.columns) + quadratic(p.reference) + 13;
                const auto& e = p.clifford;
                n += 4 * e.parity.size() + rows(e.input_upper) + rows(e.columns) + rows(e.upper) + 8 * e.keep.size();
                for(const auto& step : e.steps) n += 17 + bits(step.support) + bits(step.neighbors) + bits(step.column);
            }
        }
        return n + contraction(r.contraction);
    }
    size_t certificate(const ResourceCertificate& c) {
        return 8 + phase(c.terms) + 4 * c.linear.size() + rows(c.upper) + resource(c.reference);
    }
    size_t network(const CompiledNetwork& n) {
        if(!shared.insert(&n).second) return 0;
        auto pairing = [](const TracePairingKernel& p) { return tensor(p.clifford) + phase(p.bra) + phase(p.ket); };
        size_t bytes = rows(n.boundary.keep) + rows(n.boundary.events) + bits(n.boundary.x) + bits(n.boundary.z) +
                phase(n.magic) + phase(n.fixed_phase) + tensor(n.fixed_source) + pairing(n.traced) +
                pairing(n.restricted) +
                4 *
                        (n.reference_outcomes.size() + n.bra_offsets.size() + n.ket_offsets.size() +
                                n.reference_exponents.size()) +
                trace(n.event_support);
        if(n.joint) bytes += certificate(*n.joint);
        if(n.stable_pairing) bytes += certificate(*n.stable_pairing);
        if(n.inherited && shared.insert(n.inherited.get()).second) bytes += certificate(*n.inherited);
        for(const auto& c : n.reduction_algebra) bytes += certificate(c);
        if(n.maps && shared.insert(n.maps.get()).second) {
            const auto& m = *n.maps;
            for(const auto* v : {&m.forms, &m.bra, &m.ket, &m.upper, &m.event_polar, &m.restricted_forms,
                        &m.restricted_bra, &m.restricted_ket, &m.bra_magic, &m.ket_magic, &m.bra_reduced,
                        &m.ket_reduced, &m.restricted.upper})
                bytes += rows(*v);
            bytes += 4 * (m.linear.size() + m.restricted.linear.size()) + trace(m.events);
            for(const auto& c : m.algebra) bytes += certificate(c);
        }
        return bytes;
    }
};
inline CompiledNetwork compile_network(const TracePreparation& preparation, const std::vector<int>& exponents,
        const BoundarySpec& boundary, Workspace& ws, const TraceFactorCache& pairing, PathBudget max_doubled,
        const PhaseLimits& limits, const PhaseDatabase* database, bool prepare_resources = true,
        bool stable_reference = false) {
    CompiledNetwork network;
    network.boundary = boundary;
    network.physical = preparation.physical;
    network.stable_requested = stable_reference;
    network.magic = preparation.magic;
    network.phase_dimension = preparation.side.phase_axes;
    validate_trace_boundary(network.physical, boundary.keep, boundary.events, boundary.x, boundary.z);
    auto tick = std::chrono::steady_clock::now();
    auto rows = boundary.keep;
    rows.insert(rows.end(), boundary.events.begin(), boundary.events.end());
    auto joined = pair_trace_kernel(preparation.side, rows, boundary.x, boundary.z, pairing, max_doubled,
            network.reference_diagnostics, network.traced, ws, &preparation.geometry);
    // A zero Clifford pairing is zero for every phase and every Pauli offset.
    if(joined.zero) network.traced.clifford = std::move(joined);
    auto& tensor = network.traced.clifford;
    network.input_dimension = tensor.n;
    for(size_t j = 0; j < boundary.events.size(); j++)
        network.reference_outcomes.push_back(tensor.boundary[boundary.keep.size() + j].second);
    network.restricted = network.traced;
    if(!tensor.zero) {
        for(const auto& maps : preparation.maps)
            if(maps->matches(network.traced, boundary.keep.size())) {
                network.maps = maps;
                network.shared_maps = true;
                break;
            }
        if(!network.maps) {
            network.maps = std::make_shared<const NetworkMaps>(network.traced, network.magic, boundary.keep.size());
            preparation.maps.push_back(network.maps);
        }
        network.event_support = network.maps->events;
        const auto& maps = *network.maps;
        auto& r = network.restricted;
        r.clifford.n = maps.events->factor.kernel.size();
        r.clifford.linear = maps.restricted.linear;
        r.clifford.upper = maps.restricted.upper;
        Bits delta;
        for(size_t j = 0; j < tensor.n; j++)
            if(mod4(tensor.linear[j] - maps.linear[j])) delta.set(j);
        auto mapped = xors(maps.events->dual, delta);
        for(auto j : mapped.indices()) r.clifford.linear[j] = mod4(r.clifford.linear[j] + 2);
        r.clifford.boundary.resize(boundary.keep.size());
        r.clifford.output_count = boundary.keep.size();
        auto restrict = [](PhaseTerms& rows, const std::vector<Bits>& mapped) {
            for(size_t j = 0; j < rows.size(); j++) rows[j].first = mapped[j];
        };
        restrict(r.clifford.boundary, maps.restricted_forms);
        restrict(r.bra, maps.restricted_bra);
        restrict(r.ket, maps.restricted_ket);
    }
    if(tensor.zero)
        network.event_support = std::make_shared<TraceFactor>(std::vector<Bits>(boundary.events.size()), tensor.n);
    network.reference_exponents = exponents;
    network.fixed_source.n = network.phase_dimension;
    network.fixed_source.linear.resize(network.phase_dimension);
    network.fixed_source.upper.resize(network.phase_dimension);
    network.fixed_source.terms = preparation.side.phase;
    if(!tensor.zero) {
        PhaseCounter phases;
        auto prepare_phase = [&](const PhaseTerms& coords, const std::vector<Bits>& mapped, std::vector<int>& offsets,
                                     int sign) {
            Bits base;
            for(size_t j = 0; j < coords.size(); j++)
                if(coords[j].second) base.set(j);
            for(size_t j = 0; j < network.magic.size(); j++) {
                int bit = network.magic[j].second ^ (network.magic[j].first & base).parity();
                offsets.push_back(bit);
                const auto& row = mapped[j];
                phases[row] = mod8(phases[row] + sign * exponents[j] * (bit ? -1 : 1));
                network.phase_constant = mod8(network.phase_constant + sign * exponents[j] * bit);
            }
        };
        prepare_phase(network.traced.bra, network.maps->bra_reduced, network.bra_offsets, -1);
        prepare_phase(network.traced.ket, network.maps->ket_reduced, network.ket_offsets, 1);
        network.fixed_phase = phase_terms(phases);
    }
    std::vector<ResourceCertificate> algebra;
    auto reduced = network.reduce(Frame(), exponents, network.reference_outcomes, ws, &algebra);
    network.reduction_algebra = std::move(algebra);
    auto now = std::chrono::steady_clock::now();
    network.reduction_seconds = std::chrono::duration<double>(now - tick).count();
    tick = now;
    const auto& b = reduced.result.tensor;
    network.reference_zero = b.zero;
    if(!prepare_resources) return network;
    network.resources_ready = true;
    std::optional<PhaseBudgetError> joint_failure;
    if(!b.zero) {
        auto quotient = phase_quotient(b);
        network.reference_rank = quotient.axes.size();
        try {
            network.joint.emplace(b, phase_resource(b, std::move(quotient), limits, database));
        } catch(const PhaseBudgetError& e) {
            network.joint_limit = e.what();
            joint_failure = e;
        }
    }
    // This is a factorization of the saved, fully traced B/C tensor. No
    // output-open quantum tensor is needed to transport its phase entries.
    if(reduced.has_source && (!network.joint || network.joint->term_bound() > 1)) {
        if(!preparation.resource_prepared) {
            try {
                preparation.phase_resource = std::make_shared<const ResourceCertificate>(
                        reduced.source, phase_resource(reduced.source, limits, database));
            } catch(const PhaseBudgetError& e) { preparation.resource_limit = e.what(); }
            preparation.resource_prepared = true;
        }
        network.inherited = preparation.phase_resource;
        network.inherited_limit = preparation.resource_limit;
    }
    if(!b.zero && !network.joint && !network.inherited)
        throw PhaseBudgetError(*joint_failure,
                "; reference coordinates " + std::to_string(network.reference_rank) + ", retained labels " +
                        std::to_string(boundary.keep.size()));
    auto result = expand_trace_query(std::move(reduced), false, ws, limits, database,
            network.joint ? &*network.joint : nullptr, network.inherited ? &*network.inherited : nullptr);
    network.reference_terms = result.terms.size();
    network.reference_diagnostics = result.diagnostics;
    // Optional stable coordinates of this already paired probability tensor.
    // No single-side/output-open ancestor or per-prefix resource is retained.
    if(stable_reference && !network.traced.clifford.zero && !network.reduction_algebra.empty()) {
        auto fixed = network.reduce(Frame(), exponents, network.reference_outcomes, ws, nullptr, false, false);
        auto& tensor = fixed.result.tensor;
        network.joint_fixed_domain = network.joint && tensor.n == network.joint->dimension &&
                tensor.linear == network.joint->linear && tensor.upper == network.joint->upper &&
                tensor.terms == network.joint->terms &&
                ((network.joint->reference.contraction && network.joint->reference.contraction->matches(tensor)) ||
                        (network.joint->reference.entry_plans &&
                                network.joint->reference.entry_plans->matches(tensor)));
        if(!tensor.zero && !network.joint_fixed_domain) {
            const auto& quotient = network.reduction_algebra.front().reference.quotient;
            try {
                network.stable_pairing.emplace(tensor, phase_resource(tensor, quotient, limits, database));
            } catch(const PhaseBudgetError&) {} // The reduced/inherited reference remains available.
        }
    }
    network.resource_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - tick).count();
    return network;
}
} // namespace magic_native
