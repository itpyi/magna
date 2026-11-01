// Single-side symbolic reduction with a transported, deferred trace interface.
#pragma once
#include "phase.hpp"
#include <chrono>
#include <memory>
#include <mutex>

namespace magic_native {
struct TraceDiagnostics {
    bool wide = false, support_hit = false, pairing_hit = false, inherited_product = false, side_hit = false;
    size_t input_bits = 0, event_rows = 0, event_rank = 0, support_bits = 0;
    size_t phase_axes = 0, clifford_bits = 0, difference_bits = 0, pairing_bits = 0;
    size_t support_factorizations = 0, pairing_factorizations = 0;
    size_t expanded_entries = 0, inherited_side_entries = 0, resource_constructions = 0,
           phase_coordinate_constructions = 0;
    bool reference_entries_reused = false;
    bool prepared_query = false, input_certificate_covered = false, output_trace_reused = false;
    double actual_seconds = 0, support_seconds = 0, single_seconds = 0;
    double pairing_seconds = 0, reduce_seconds = 0, expand_seconds = 0;
};

// Cache only homogeneous linear algebra, never assignments or probabilities.
// The owned plan fixes the gate table/input/coordinate version. Actual rows and
// dimension complete the key; offsets (including faults and effects) are RHSs.
struct TraceFactor {
    size_t n, bytes = 0;
    std::vector<Bits> rows, dual;
    Factor factor;
    // Optional column form for event supports, indexed by row: the zero
    // combinations and the solution bits containing each row. Both maps are
    // linear, so an RHS is tested and solved in time proportional to its
    // support with the factor's results.
    IndexLists zero_columns;
    std::vector<Bits> solve_columns;
    TraceFactor(const std::vector<Bits>& r, size_t dim, bool columns = false) : n(dim), rows(r), factor(r, dim) {
        dual = transpose(factor.kernel, n);
        if(columns) {
            zero_columns = transpose_lists(factor.zeros, rows.size());
            // Factor::solve sets pivot p from its provenance and earlier pivots.
            std::map<size_t, Bits> forms;
            solve_columns.resize(rows.size());
            for(const auto& [p, rt] : factor.pivots) {
                auto form = rt.second;
                for(const auto& [k, earlier] : forms)
                    if(rt.first.bit(k)) form ^= earlier;
                for(auto j : form.indices()) solve_columns[j].set(p);
                forms.emplace(p, std::move(form));
            }
        }
        auto count = [&](const Bits& b) { bytes += sizeof(Bits) + b.w.capacity() * sizeof(uint64_t); };
        for(auto& b : rows) count(b);
        for(auto& b : dual) count(b);
        for(auto& b : factor.zeros) count(b);
        for(auto& b : factor.kernel) count(b);
        for(auto& b : solve_columns) count(b);
        bytes += (zero_columns.offsets.size() + zero_columns.items.size()) * sizeof(size_t);
        for(auto& [p, rt] : factor.pivots) {
            bytes += sizeof(p) + 4 * sizeof(void*);
            count(rt.first);
            count(rt.second);
        }
        bytes += sizeof(TraceFactor);
    }
    bool consistent(const Bits& rhs) const {
        if(solve_columns.size() != rows.size()) throw std::logic_error("event factor requires its column form");
        Bits residual;
        xor_lists(residual, zero_columns, rhs, factor.zeros.size());
        return residual.empty();
    }
    Bits solve(const Bits& rhs) const {
        if(solve_columns.size() != rows.size()) throw std::logic_error("event factor requires its column form");
        Bits shift;
        rhs.each([&](size_t j) {
            if(j < rows.size()) shift ^= solve_columns[j];
        });
        return shift;
    }
};
struct TraceFactorCache {
    static constexpr size_t max_entries = 8;
    mutable std::mutex mutex;
    mutable BoundedCache<std::shared_ptr<const TraceFactor>> entries;
    std::shared_ptr<const TraceFactor> get(const std::vector<Bits>& rows, size_t n, bool& hit) const {
        std::lock_guard<std::mutex> lock(mutex);
        auto* found = entries.find([&](const auto& item) { return item->n == n && item->rows == rows; });
        hit = found != nullptr;
        if(found) return *found;
        auto item = std::make_shared<TraceFactor>(rows, n);
        entries.insert(item, max_entries, item->bytes, default_cache_bytes);
        return item;
    }
};

// psi(z) = sum_u C(z,u) omega^F(u). C retains every physical trace slot;
// F is a sparse parity polynomial, with no resource-table/term expansion.
// Coordinate changes and Gaussian sums occur in C and therefore transport its
// trace interface and the phase labels together. No trace leg is summed alone.
struct DeferredTraceSide {
    Ket clifford;
    PhaseTerms phase;
    size_t trace_width = 0, phase_axes = 0;
};
struct TracePairingKernel {
    PhaseTensor clifford;
    PhaseTerms bra, ket;
};
// Owned by one immutable deferred side during preparation. No effect, event
// RHS or contraction result is stored here.
struct TracePairingGeometry {
    std::vector<Bits> forms, trace, ket_basis, ket_dual;
    std::shared_ptr<const TraceFactor> pairing;
    TracePairingGeometry(const DeferredTraceSide& side, const TraceFactorCache& cache, bool& hit) {
        const auto& k = side.clifford;
        size_t m = k.columns.size();
        forms = transpose(k.columns, k.width);
        trace.assign(forms.begin(), forms.begin() + side.trace_width);
        pairing = cache.get(trace, m, hit);
        for(size_t j = 0; j < m; j++) ket_basis.push_back(Bits::unit(j));
        ket_basis.insert(ket_basis.end(), pairing->factor.kernel.begin(), pairing->factor.kernel.end());
        ket_dual = transpose(ket_basis, m);
    }
};
// The structural choice depends only on dimension and normalized odd masks.
// Actual signs, Clifford updates, scalar and boundary remain per query.
inline PolynomialQuotient single_phase_template(size_t n, const PhaseTerms& odd) {
    std::vector<Bits> masks;
    for(auto& [a, c] : odd) masks.push_back(a);
    PolynomialQuotient q;
    q.axes = phase_image_basis(masks);
    auto form = phase_trilinear_rows(n, odd);
    std::vector<Bits> rows;
    for(auto& [ij, row] : form) rows.push_back(row);
    if(phase_image_basis(rows).size() < q.axes.size()) return polynomial_quotient(n, odd);
    q.lifts = phase_right_inverse(q.axes, n);
    auto dual = transpose(q.lifts, n);
    for(auto& a : masks) q.core.push_back({xors(dual, a), 1});
    q.linear.resize(n);
    q.upper.resize(n);
    return q;
}
inline DeferredTraceSide single_trace_side(PhaseTensor t, Workspace& ws) {
    DeferredTraceSide side;
    side.trace_width = t.boundary.size();
    PhaseCounter combined;
    for(auto& [a, c] : t.terms) combined[a] = mod8(combined[a] + c);
    PhaseTerms odd, even;
    for(auto& [a, c] : phase_terms(combined)) {
        if(a.size() > t.n) throw std::invalid_argument("phase mask outside dimension");
        if(c & 1) odd.push_back({a, 1});
        if(c - (c & 1)) even.push_back({a, c - (c & 1)});
    }
    static thread_local PhasePolynomialCache cache(single_phase_template);
    auto q = cache.get(t.n, odd);
    auto axes = std::move(q.axes);
    side.phase = std::move(q.core);
    for(size_t j = 0; j < t.n; j++) {
        t.linear[j] = mod4(t.linear[j] + q.linear[j]);
        t.upper[j] ^= q.upper[j];
    }
    phase_add_even(t.linear, t.upper, even);
    side.phase_axes = axes.size();
    std::vector<Bits> rows;
    Bits base;
    for(size_t j = 0; j < t.boundary.size(); j++) {
        rows.push_back(t.boundary[j].first);
        if(t.boundary[j].second) base.set(j);
    }
    rows.insert(rows.end(), axes.begin(), axes.end());
    Ket k;
    k.width = rows.size();
    k.base = base;
    k.columns = transpose(rows, t.n);
    k.linear = std::move(t.linear);
    k.upper = std::move(t.upper);
    k.phase = 0;
    k.power = t.power;
    // The next operation pairs this tensor with its conjugate. Only the
    // common phase may be omitted; its norm and relative phases are retained.
    side.clifford = contract(std::move(k), ws, false);
    return side;
}

inline PhaseTensor restrict_trace_support(PhaseTensor t, const TraceFactor& support, const Bits& rhs) {
    auto shift = support.factor.solve(rhs);
    auto q = restrict_q(t.linear, t.upper, shift, support.factor.kernel, false);
    t.n = support.factor.kernel.size();
    t.linear = std::move(q.linear);
    t.upper = std::move(q.upper);
    t.constant = 0;
    PhaseCounter terms;
    for(auto& [a, c] : t.terms) {
        int bit = (a & shift).parity();
        auto row = xors(support.dual, a);
        terms[row] = mod8(terms[row] + (bit ? -c : c));
    }
    t.terms = phase_terms(terms);
    for(size_t j = 0; j < t.boundary.size(); j++) {
        auto& [a, b] = t.boundary[j];
        b ^= (a & shift).parity();
        a = xors(support.dual, a);
    }
    return t;
}

// Implicit Gamma_E from one C and its conjugate, using bra y, ket y+d.
// The only interface factorization is B d = x, with B the reduced single-side
// trace map. There is no wide [B B] system and no enumeration of u,v.
inline PhaseTensor pair_trace_kernel(const DeferredTraceSide& side, const std::vector<Bits>& keep, const Bits& x,
        const Bits& z, const TraceFactorCache& cache, PathBudget max_paths, TraceDiagnostics& diag,
        TracePairingKernel& saved, Workspace& ws, std::optional<TracePairingGeometry>* prepared = nullptr) {
    const auto& k = side.clifford;
    size_t m = k.columns.size(), width = side.trace_width;
    PhaseTensor b;
    b.output_count = keep.size();
    b.boundary.resize(keep.size());
    if(k.zero) {
        b.zero = true;
        return b;
    }
    std::optional<TracePairingGeometry> local;
    auto& geometry = prepared ? *prepared : local;
    if(geometry) diag.pairing_hit = true;
    else geometry.emplace(side, cache, diag.pairing_hit);
    const auto& forms = geometry->forms;
    const auto& trace = geometry->trace;
    const auto& pairing = geometry->pairing;
    diag.pairing_factorizations = !diag.pairing_hit;
    diag.difference_bits = pairing->factor.kernel.size();
    b.n = m + diag.difference_bits;
    diag.pairing_bits = b.n;
    if(b.n > max_paths.bits)
        throw PhaseBudgetError("doubled network exceeds the path-variable budget", max_paths.option,
                double(max_paths.bits), double(b.n));
    if(!pairing->factor.consistent(x)) {
        b.zero = true;
        b.n = 0;
        return b;
    }
    auto delta = pairing->factor.solve(x);
    auto q = restrict_q(k.linear, k.upper, delta, geometry->ket_basis);
    b.linear = std::move(q.linear);
    b.upper = std::move(q.upper);
    for(size_t j = 0; j < m; j++) {
        b.linear[j] = mod4(b.linear[j] - k.linear[j]);
        b.upper[j] ^= k.upper[j];
    }
    b.power = 2 * k.power;
    b.constant = mod8(2 * int((x & z).count()) + 2 * q.constant);
    const auto& ket_dual = geometry->ket_dual;
    // Z acts on the ket, including the Hermitian Pauli i^(x.z) convention.
    Bits zr = xors(trace, z);
    int zb = (z & k.base).parity() ^ (zr & delta).parity();
    b.constant = mod8(b.constant + 4 * zb);
    phase_add_even(b.linear, b.upper, {{xors(ket_dual, zr), 4}});
    // Contract the Clifford pairing before choosing ANY resource expansion.
    // The retained coordinates are the requested labels and the symbolic u,v.
    // Thus Gamma is an affine/quadratic tensor, never a matrix of amplitudes.
    std::vector<Bits> gram_rows;
    Bits gram_base;
    auto append = [&](Bits row, int bit) {
        if(bit) gram_base.set(gram_rows.size());
        gram_rows.push_back(std::move(row));
    };
    for(auto& row : keep) append(xors(trace, row), (row & k.base).parity());
    for(size_t j = 0; j < side.phase_axes; j++) append(forms[width + j], k.base.bit(width + j));
    for(size_t j = 0; j < side.phase_axes; j++)
        append(xors(ket_dual, forms[width + j]), k.base.bit(width + j) ^ (forms[width + j] & delta).parity());
    Ket gram;
    gram.width = gram_rows.size();
    gram.base = gram_base;
    gram.columns = transpose(gram_rows, b.n);
    gram.linear = std::move(b.linear);
    gram.upper = std::move(b.upper);
    gram.power = b.power;
    gram.phase = b.constant;
    gram = contract(std::move(gram), ws);
    if(gram.zero) {
        b.zero = true;
        b.n = 0;
        return b;
    }
    auto mapped = transpose(gram.columns, gram.width);
    b.n = gram.columns.size();
    b.linear = std::move(gram.linear);
    b.upper = std::move(gram.upper);
    b.power = gram.power;
    b.constant = gram.phase;
    for(size_t j = 0; j < keep.size(); j++) b.boundary[j] = {mapped[j], int(gram.base.bit(j))};
    saved.clifford = b;
    for(size_t j = 0; j < side.phase_axes; j++) {
        saved.bra.push_back({mapped[keep.size() + j], int(gram.base.bit(keep.size() + j))});
        saved.ket.push_back(
                {mapped[keep.size() + side.phase_axes + j], int(gram.base.bit(keep.size() + side.phase_axes + j))});
    }
    return b;
}

// General one-off queries need the pulled phase polynomial. Persistent query
// compilation instead builds it on the event-restricted coordinates, so it
// uses only pair_trace_kernel and does not construct a discarded polynomial.
inline PhaseTensor pair_trace_side(const DeferredTraceSide& side, const std::vector<Bits>& keep, const Bits& x,
        const Bits& z, const TraceFactorCache& cache, PathBudget max_paths, TraceDiagnostics& diag,
        TracePairingKernel& saved, Workspace& ws) {
    auto b = pair_trace_kernel(side, keep, x, z, cache, max_paths, diag, saved, ws);
    if(b.zero) return b;
    PhaseCounter phases;
    auto pull = [&](const PhaseTerms& coords, int sign) {
        for(const auto& [a, c] : side.phase) {
            Bits row;
            int bit = 0;
            for(auto j : a.indices()) {
                row ^= coords[j].first;
                bit ^= coords[j].second;
            }
            int coefficient = sign * c * (bit ? -1 : 1);
            b.constant = mod8(b.constant + sign * c * bit);
            phases[row] = mod8(phases[row] + coefficient);
        }
    };
    pull(saved.bra, -1);
    pull(saved.ket, 1);
    b.terms = phase_terms(phases);
    return b;
}

// Transport an r-term phase decomposition through the contracted Gamma.
// Every conjugate pair contracts to at most one stabilizer tensor, preserving
// the r^2 construction instead of forcing a fresh Fourier expansion. Resource
// entries are first built after Gamma, so discarded ancillas need no table.
inline std::vector<std::pair<Scalar, Ket>> inherited_trace_states(const TracePairingKernel& gamma,
        const PhaseResource& source, const PhaseLimits& limits, Workspace& ws, TraceDiagnostics& diag) {
    auto t = gamma.clifford;
    auto transport = [&](const PhaseTerms& coords, int sign) {
        std::vector<Bits> rows;
        Bits base;
        for(size_t j = 0; j < coords.size(); j++) {
            rows.push_back(coords[j].first);
            if(coords[j].second) base.set(j);
        }
        auto q = restrict_q(source.quotient.linear, source.quotient.upper, base, transpose(rows, t.n));
        t.constant = mod8(t.constant + sign * 2 * q.constant);
        for(size_t j = 0; j < t.n; j++) {
            t.linear[j] = mod4(t.linear[j] + sign * q.linear[j]);
            t.upper[j] ^= q.upper[j];
        }
        PhaseTerms axes;
        for(auto& a : source.quotient.axes) axes.push_back({xors(rows, a), int((base & a).parity())});
        return axes;
    };
    auto bra = transport(gamma.bra, -1), ket = transport(gamma.ket, 1);
    std::vector<Bits> rows;
    Bits base;
    for(size_t j = 0; j < t.boundary.size(); j++) {
        rows.push_back(t.boundary[j].first);
        if(t.boundary[j].second) base.set(j);
    }
    auto columns = transpose(rows, t.n);
    std::vector<std::pair<Scalar, Ket>> out;
    for(auto& left : resource_entries(source))
        for(auto& right : resource_entries(source)) {
            std::vector<Bits> equations;
            Bits rhs;
            PhaseTerms updates;
            int constant = t.constant;
            auto add = [&](const PhaseEntryTerm& entry, const PhaseTerms& axes, int sign) {
                auto form = [&](const Bits& a) {
                    Bits row;
                    int bit = 0;
                    for(auto j : a.indices()) {
                        row ^= axes[j].first;
                        bit ^= axes[j].second;
                    }
                    return std::make_pair(row, bit);
                };
                for(auto& [a, v] : entry.constraints) {
                    auto [row, bit] = form(a);
                    if(v ^ bit) rhs.set(equations.size());
                    equations.push_back(row);
                }
                for(auto& [a, c] : entry.parities) {
                    auto [row, bit] = form(a);
                    constant = mod8(constant + sign * 2 * c * bit);
                    updates.push_back({row, sign * 2 * c * (bit ? -1 : 1)});
                }
            };
            add(left, bra, -1);
            add(right, ket, 1);
            auto linear = t.linear;
            auto upper = t.upper;
            phase_add_even(linear, upper, updates);
            Factor factor(equations, t.n);
            if(!factor.consistent(rhs)) continue;
            auto shift = factor.solve(rhs);
            auto q = restrict_q(linear, upper, shift, factor.kernel);
            Ket k;
            k.width = rows.size();
            k.base = base ^ xors(columns, shift);
            k.power = t.power;
            k.phase = mod8(constant + 2 * q.constant);
            k.linear = std::move(q.linear);
            k.upper = std::move(q.upper);
            for(auto& a : factor.kernel) k.columns.push_back(xors(columns, a));
            k = contract(std::move(k), ws);
            if(!k.zero) {
                out.push_back({left.coefficient.conjugate() * right.coefficient, std::move(k)});
                phase_check_terms(out.size(), limits);
            }
        }
    diag.expanded_entries = resource_entries(source).size() * resource_entries(source).size();
    diag.inherited_side_entries = resource_entries(source).size();
    diag.inherited_product = true;
    return out;
}

} // namespace magic_native
