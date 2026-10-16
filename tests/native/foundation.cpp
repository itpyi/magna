// Independent finite sums and analytical expectations; no CPython headers/runtime.
#include "runtime/noise.hpp"
#include "runtime/sampling.hpp"
#include "runtime/low_weight.hpp"
#include "evaluate/record_trace.hpp"
#include "evaluate/boundary_totals.hpp"
#include "runtime/marginal.hpp"
#include "compile/frontend.hpp"
#include "runtime/noise_model.hpp"
#include "compile/file_lowering.hpp"
#include "compile/deferred.hpp"
#include "compile/program.hpp"
#include "evaluate/resource_certificate.hpp"
#include "compile/terminal.hpp"
#include "runtime/causal.hpp"
#include "compile/file_program.hpp"
#include "runtime/file_execution.hpp"
#include "runtime/history_efficiency.hpp"
#include "runtime/file_enumeration.hpp"
#include "compile/instrument.hpp"
#include "runtime/branch.hpp"
#include "runtime/proposal.hpp"
#include "runtime/certificate_cache.hpp"
#include "runtime/estimate.hpp"
#include "runtime/boundary_service.hpp"
#include "runtime/operation_boundary.hpp"
#include "runtime/operation_transport.hpp"
#include "runtime/full_record.hpp"
#include "compile/task.hpp"
#include <iostream>
#include <complex>
#include <cmath>
using namespace magic_native;

void require(bool condition, const char* message) {
    if(!condition) throw std::runtime_error(message);
}
std::complex<double> scalar_value(const Scalar& value) {
    std::complex<double> out;
    for(int j = 0; j < 4; j++) {
        double coefficient = 0;
        for(size_t k = value.c[j].d.size(); k-- > 0;) coefficient = std::ldexp(coefficient, 32) + value.c[j].d[k];
        out += (value.c[j].neg ? -coefficient : coefficient) * std::polar(1., j * std::acos(-1.) / 4);
    }
    return out * std::exp2(double(value.power));
}
std::vector<std::complex<double>> dense(const Ket& k) {
    std::vector<std::complex<double>> result(size_t(1) << k.width);
    if(k.zero) return result;
    for(size_t u = 0; u < (size_t(1) << k.columns.size()); u++) {
        size_t word = k.base.w.empty() ? 0 : k.base.w[0];
        int phase = k.phase;
        for(size_t j = 0; j < k.columns.size(); j++)
            if((u >> j) & 1) {
                word ^= k.columns[j].w.empty() ? 0 : k.columns[j].w[0];
                phase += 2 * k.linear[j];
                for(size_t l = j + 1; l < k.columns.size(); l++)
                    if(((u >> l) & 1) && k.upper[j].bit(l)) phase += 4;
            }
        result[word] += std::polar(std::exp2(k.power / 2.), phase * std::acos(-1.) / 4);
    }
    return result;
}
// Directly enumerate phase variables, without a quotient or stabilizer expansion.
std::vector<std::complex<double>> finite_phase_sum(const PhaseTensor& tensor) {
    std::vector<std::complex<double>> result(size_t(1) << tensor.boundary.size());
    if(tensor.zero) return result;
    for(size_t word = 0; word < (size_t(1) << tensor.n); word++) {
        Bits variables(word);
        int exponent = tensor.constant;
        size_t output = 0;
        for(size_t j = 0; j < tensor.n; j++)
            if(variables.bit(j)) {
                exponent += 2 * tensor.linear[j];
                exponent += 4 * (tensor.upper[j] & variables).count();
            }
        for(const auto& [mask, coefficient] : tensor.terms) exponent += coefficient * (mask & variables).parity();
        for(size_t j = 0; j < tensor.boundary.size(); j++)
            output |= size_t((tensor.boundary[j].first & variables).parity() ^ tensor.boundary[j].second) << j;
        result[output] += std::polar(std::exp2(tensor.power / 2.), mod8(exponent) * std::acos(-1.) / 4);
    }
    return result;
}
void check_transported_phase(const PhaseTensor& tensor, const ResourceCertificate& certificate) {
    auto transported = certificate.transport(tensor);
    require(bool(transported), "Clifford-equivalent phase transports");
    require(transported->prepared_entries == certificate.reference.prepared_entries,
            "transport shares compiled resource entries");
    Workspace workspace;
    PhaseLimits limits;
    auto expected = finite_phase_sum(tensor);
    std::vector<std::complex<double>> actual(expected.size());
    auto accelerated = phase_states(tensor, *transported, workspace, limits);
    auto ordinary = *transported;
    ordinary.contraction.reset();
    auto unprepared = phase_states(tensor, ordinary, workspace, limits);
    require(accelerated.size() == unprepared.size(), "prepared contraction preserves zero terms");
    for(size_t j = 0; j < accelerated.size(); j++) {
        const auto& [a, k] = accelerated[j];
        const auto& [b, h] = unprepared[j];
        require(a.c == b.c && a.power == b.power && k.width == h.width && k.base == h.base && k.columns == h.columns &&
                        k.linear == h.linear && k.upper == h.upper && k.phase == h.phase && k.power == h.power &&
                        k.zero == h.zero,
                "shared contraction exactly matches ordinary per-entry contraction");
    }
    for(const auto& [coefficient, state] : accelerated) {
        auto values = dense(state);
        for(size_t j = 0; j < values.size(); j++) actual[j] += scalar_value(coefficient) * values[j];
    }
    for(size_t j = 0; j < actual.size(); j++)
        require(std::abs(actual[j] - expected[j]) < 1e-11,
                "incremental quotient and boundary transport against independent finite sum");
}
// Uncompressed seven-parity expansion and all-pairs check: an independent
// analysis schedule against which to test the compiler's span/prefix reuse.
std::pair<bool, std::vector<Bits>> full_axis_sector(const CompilerIR& ir, const std::vector<Bits>& rows) {
    auto stripped = ir;
    std::vector<std::vector<Fault>> slots;
    for(auto& op : stripped.operations) {
        if((op.kind == "T" || op.kind == "S" || op.kind == "PHASE") && (op.exponent & 1)) {
            std::vector<Fault> faults;
            for(int q : op.qubits) faults.push_back({size_t(op.location), q, 2});
            slots.push_back(faults);
            op.kind = "I";
            op.qubits.clear();
        } else if(op.kind == "CCZ") {
            for(size_t mask = 1; mask < 8; mask++) {
                std::vector<Fault> faults;
                for(size_t j = 0; j < 3; j++)
                    if((mask >> j) & 1) faults.push_back({size_t(op.location), op.qubits[j], 2});
                slots.push_back(faults);
            }
            op.kind = "I";
            op.qubits.clear();
        }
    }
    auto reference = prepare_compiler(stripped);
    std::vector<Frame> axes;
    std::vector<Bits> anti;
    for(const auto& faults : slots) axes.push_back(reference.plan.fault_frame(faults));
    for(const auto& a : axes)
        for(const auto& b : axes)
            if((a.x & b.z).parity() ^ (a.z & b.x).parity()) return {false, {}};
    for(const auto& a : axes) {
        Bits row;
        for(size_t j = 0; j < rows.size(); j++)
            if((a.x & rows[j]).parity()) row.set(j);
        anti.push_back(row);
    }
    return {true, Factor(anti, rows.size()).kernel};
}
void check_axis_prefix() {
    auto append = [](CompilerIR& ir, const std::string& kind, std::vector<int> qubits, int exponent = 0) {
        CompilerOp op;
        op.kind = kind;
        op.qubits = std::move(qubits);
        op.exponent = exponent;
        op.location = int(ir.operations.size());
        ir.operations.push_back(std::move(op));
    };
    CompilerIR ir;
    ir.inputs = {0, 1, 2};
    ir.outputs = {0, 1, 2};
    ir.initial.width = 3;
    append(ir, "H", {0});
    append(ir, "CCZ", {0, 1, 2});
    append(ir, "CCZ", {0, 1, 2});
    append(ir, "PHASE", {0, 2}, 3);
    const std::vector<Bits> rows = {Bits(1), Bits(2), Bits(4), Bits(3), Bits(0)};
    for(bool noncommuting : {false, true}) {
        auto physical = ir;
        if(noncommuting) {
            append(physical, "H", {0});
            append(physical, "T", {0}, 1);
        }
        BoundaryTracePlan base;
        base.ir = physical;
        base.prepare();
        auto sector = base.commuting_rows(rows);
        auto expected = full_axis_sector(physical, rows);
        require(base.common_axes == expected.first && sector == expected.second,
                "compressed axes match full expansion");
        require(base.axes.size() < base.axis_slots, "dependent rotation axes are removed");
        for(int variation = 0; variation < 3; variation++) {
            auto attached = physical;
            // Deferral can rename every record wire when target ancillas are added.
            for(auto& q : attached.inputs) q += 10;
            for(auto& op : attached.operations)
                for(auto& q : op.qubits) q += 10;
            attached.outputs = {12, 10, 11, 70};
            append(attached, "PREP_PLUS", {70});
            append(attached, "CX", {70, 11});
            append(attached, "RESET_PLUS", {10});
            append(attached, "H", {11});
            if(variation == 1) append(attached, "CCZ", {70, 11, 12});
            if(variation == 2) attached.operations[0].kind = "X";
            BoundaryTracePlan shared, fresh;
            shared.ir = attached;
            fresh.ir = attached;
            shared.prepare();
            fresh.prepare();
            auto result = shared.commuting_rows(rows, &base);
            auto independent = fresh.commuting_rows(rows);
            auto full = full_axis_sector(attached, rows);
            require(result == independent && result == full.second && shared.common_axes == fresh.common_axes &&
                            shared.common_axes == full.first,
                    "transported prefix matches uncached full axis analysis");
            require(shared.shared_axis_generators == (variation == 2 ? 0 : base.axes.size()),
                    "exact prefix reuse and mismatch");
        }
    }
}
void check_reference_coordinates() {
    uint64_t seed = 928271;
    auto next = [&]() {
        seed = seed * 6364136223846793005ULL + 1;
        return seed >> 32;
    };
    // Enumerate span membership independently, including dependent/zero basis
    // rows and arbitrary parity requests; also preserve the old free-zero solve.
    for(size_t width = 0; width <= 7; width++)
        for(size_t count = 0; count <= 7; count++) {
            std::vector<Bits> basis;
            std::vector<uint64_t> words;
            for(size_t j = 0; j < count; j++) {
                auto word = next() & ((uint64_t(1) << width) - 1);
                words.push_back(word);
                basis.emplace_back(word);
            }
            ReferenceCoordinates map(basis, width);
            Factor old(transpose(basis, width), count);
            std::set<uint64_t> span;
            for(size_t choice = 0; choice < (size_t(1) << count); choice++) {
                uint64_t word = 0;
                for(size_t j = 0; j < count; j++)
                    if((choice >> j) & 1) word ^= words[j];
                span.insert(word);
            }
            for(uint64_t word = 0; word < (uint64_t(1) << width); word++) {
                auto value = map(Bits(word));
                require(bool(value) == bool(span.count(word)),
                        "prepared coordinates match independently enumerated span");
                require(bool(value) == old.consistent(Bits(word)), "prepared coordinates retain consistency");
                if(value)
                    require(*value == old.solve(Bits(word)), "prepared coordinates retain exact original solution");
            }
            require(!map(Bits::unit(width)), "coordinates reject rows outside the physical width");
        }
    // Exercise packed-word boundaries and dense general parities in a proper
    // subspace: 67 independent columns embedded in 137 physical positions.
    std::vector<Bits> basis;
    for(size_t j = 0; j < 67; j++) basis.push_back(Bits::unit(j) ^ Bits::unit(70 + j));
    basis.push_back(basis[1] ^ basis[65]);
    basis.emplace_back();
    ReferenceCoordinates map(basis, 137);
    Factor old(transpose(basis, 137), basis.size());
    for(size_t trial = 0; trial < 128; trial++) {
        Bits choice;
        for(size_t j = 0; j < basis.size(); j++)
            if(next() & 1) choice.set(j);
        auto row = xors(basis, choice);
        auto value = map(row);
        require(value && *value == old.solve(row), "multiword parity coordinates match original solve");
        require(!map(row ^ Bits::unit(68)), "multiword parity outside label span is rejected");
    }
}
void check_packed_clifford_updates() {
    for(size_t width : {size_t(0), size_t(1), size_t(63), size_t(64), size_t(65), size_t(137)}) {
        std::vector<Bits> rows(70);
        for(size_t j = 0; j < rows.size(); j++)
            for(size_t k = 0; k < width; k++)
                if((j * 17 + k * 13) % 7 < 3) rows[j].set(k);
        for(size_t trial = 0; trial < 32; trial++) {
            Bits mask, expected;
            for(size_t j = 0; j < rows.size(); j++)
                if((j * 11 + trial * 3) % 13 < 7) mask.set(j);
            for(size_t k = 0; k < width; k++) {
                bool bit = false;
                for(size_t j = 0; j < rows.size(); j++)
                    if(mask.bit(j)) bit ^= rows[j].bit(k);
                if(bit) expected.set(k);
            }
            require(xors(rows, mask) == expected, "single-trim XOR matches independent parity oracle");
        }
        bool rejected = false;
        try {
            xors(rows, Bits::unit(70));
        } catch(const std::invalid_argument&) { rejected = true; }
        require(rejected, "packed XOR keeps index safety");
    }
    for(size_t n : {size_t(0), size_t(1), size_t(7), size_t(63), size_t(64), size_t(65), size_t(137)}) {
        Bits mask;
        for(size_t j = 0; j < n; j++)
            if(j % 3 != 1) mask.set(j);
        for(size_t cut : {size_t(0), size_t(1), size_t(63), size_t(64), size_t(65), size_t(140)}) {
            Bits expected = Bits::ones(n), actual = expected;
            for(size_t j = cut + 1; j < n; j++)
                if(mask.bit(j)) expected.flip(j);
            actual.xor_above(mask, cut);
            require(actual == expected, "packed upper XOR matches independent bit oracle");
        }
        for(int coefficient : {-6, -4, -2, 0, 2, 4, 6, 8, 10}) {
            std::vector<int> linear(n), expected(n);
            std::vector<Bits> upper(n), expected_upper(n);
            for(size_t j = 0; j < n; j++) {
                linear[j] = expected[j] = int(j % 4);
                if(j + 2 < n) upper[j] = expected_upper[j] = Bits::unit(j + 2);
            }
            int c = mod8(coefficient) / 2;
            for(size_t j = 0; j < n; j++)
                if(mask.bit(j)) {
                    expected[j] = mod4(expected[j] + c);
                    for(size_t k = j + 1; k < n; k++)
                        if((c & 1) && mask.bit(k)) expected_upper[j].flip(k);
                }
            phase_add_even(linear, upper, {{mask, coefficient}});
            require(linear == expected && upper == expected_upper,
                    "packed Clifford updates preserve every linear and quadratic coefficient");
        }
    }
}
void check_quadratic_batches() {
    std::mt19937_64 random(20261004);
    // Independent scalar bit oracle covers both sparse and packed transposes,
    // rectangular shapes, final partial words, zero rows and invalid indices.
    for(size_t n : {size_t(0), size_t(1), size_t(63), size_t(64), size_t(65), size_t(137)})
        for(size_t m : {size_t(0), size_t(7), size_t(64), size_t(131)})
            for(size_t density : {size_t(2), size_t(31)}) {
                std::vector<Bits> columns(m), expected(n);
                for(size_t j = 0; j < m; j++)
                    for(size_t k = 0; k < n; k++)
                        if(random() % density == 0) {
                            columns[j].set(k);
                            expected[k].set(j);
                        }
                require(transpose(columns, n) == expected, "packed rectangular transpose matches scalar oracle");
            }
    bool rejected = false;
    try {
        transpose({Bits::unit(65)}, 65);
    } catch(const std::invalid_argument&) { rejected = true; }
    require(rejected, "transpose validates final-word bounds");
    for(size_t n : {size_t(0), size_t(1), size_t(8), size_t(63), size_t(64), size_t(65), size_t(137), size_t(513),
                size_t(1025)})
        for(size_t count : {size_t(1), size_t(7), size_t(8), size_t(32)})
            for(size_t density : {size_t(2), size_t(31)}) {
                std::vector<int> linear(n);
                std::vector<Bits> upper(n);
                PhaseTerms terms;
                for(size_t j = 0; j < n; j++) {
                    linear[j] = random() % 4;
                    for(size_t k = j + 1; k < n; k++)
                        if(random() % 31 == 0) upper[j].set(k);
                }
                for(size_t t = 0; t < count; t++) {
                    Bits mask;
                    for(size_t j = 0; j < n; j++)
                        if(random() % density == 0) mask.set(j);
                    terms.push_back({mask, 2 * int(random() % 9) - 8});
                }
                auto updated_l = linear;
                auto updated_u = upper;
                phase_add_even(updated_l, updated_u, terms);
                // Evaluate the original parity expression, without expanding its
                // products. Large cases cross the bounded scratch tile boundary.
                for(size_t trial = 0; trial < (n <= 8 ? size_t(1) << n : 32); trial++) {
                    Bits word;
                    if(n <= 8) word = Bits(trial);
                    else
                        for(size_t j = 0; j < n; j++)
                            if(random() & 1) word.set(j);
                    auto scalar = [&](const std::vector<int>& l, const std::vector<Bits>& u) {
                        int value = 0;
                        for(size_t j = 0; j < n; j++)
                            if(word.bit(j)) {
                                value += 2 * l[j];
                                for(size_t k = j + 1; k < n; k++)
                                    if(word.bit(k) && u[j].bit(k)) value += 4;
                            }
                        return mod8(value);
                    };
                    int expected = scalar(linear, upper);
                    for(const auto& term : terms) {
                        bool parity = false;
                        for(size_t j = 0; j < n; j++) parity ^= word.bit(j) && term.first.bit(j);
                        expected += term.second * parity;
                    }
                    require(scalar(updated_l, updated_u) == mod8(expected),
                            "batched quadratic update matches original parity expression");
                }
            }
    // Affine substitution of a Z4 quadratic, checked by direct evaluation;
    // includes dependent/zero columns and nonzero affine constants.
    for(size_t n : {size_t(8), size_t(65), size_t(137)}) {
        std::vector<int> linear(n);
        std::vector<Bits> upper(n), basis(n + 3);
        Bits shift;
        for(size_t j = 0; j < n; j++) {
            linear[j] = random() % 4;
            if(random() & 1) shift.set(j);
            for(size_t k = j + 1; k < n; k++)
                if(random() % 3 == 0) upper[j].set(k);
            for(size_t k = 0; k < n; k++)
                if(random() & 1) basis[j].set(k);
        }
        auto result = restrict_q(linear, upper, shift, basis);
        auto without_constant = restrict_q(linear, upper, shift, basis, false);
        require(without_constant.constant == 0 && result.linear == without_constant.linear &&
                        result.upper == without_constant.upper,
                "optional global constant preserves pulled coefficients");
        for(size_t trial = 0; trial < 64; trial++) {
            Bits word, original = shift;
            for(size_t j = 0; j < basis.size(); j++)
                if(random() & 1) {
                    word.set(j);
                    original ^= basis[j];
                }
            require(mod4(result.constant + qvalue(word, result.linear, result.upper)) ==
                            qvalue(original, linear, upper),
                    "quadratic affine substitution preserves phase");
        }
    }
}
void check_composed_feedback_frames() {
    auto protocol = parse_circuit(
            "H 0\nM 0\nDETECTOR rec[-1]\nCONTROL[bits] det[-1] -> ctrl[0] ctrl[1]\n"
            "CONTROL[XOR] ctrl[0] ctrl[1] -> ctrl[2]\nCX ctrl[0] 1\nCS ctrl[2] 1\nCY ctrl[1] 2\n"
            "DETECTOR rec[-1]\nCONTROL[bits] det[-1] -> ctrl[0] ctrl[1]\nCZ ctrl[0] 1\nCONTROL[NOT] ctrl[1] -> ctrl[3]\nPOSTSELECT ctrl[3]\n",
            "protocol", {"bits"}, 100);
    auto recovery = parse_circuit("H 2\nM 2\nDETECTOR rec[-1]\nCONTROL[bits] det[-1] -> ctrl[0] ctrl[1]\n"
                                  "CY ctrl[0] 0\nCH ctrl[1] 1\n",
            "recovery", {"bits"}, 100);
    HistoryTask task;
    CompiledExperiment run(
            FileSource(protocol, {1, 2}, recovery, {}), task, std::make_shared<const FileNoisePlan>(), {}, {});
    run.prepare();
    auto engine = run.plans.get("recovery");
    const auto& p = run.source.controls;
    const auto& r = *run.source.recovery_controls;
    require(p.decision_order.size() == 4 && r.decision_order.size() == 2, "versioned protocol and recovery decisions");
    for(size_t choice = 0; choice < 729; choice++) {
        size_t digits = choice;
        DecisionValues pv, rv;
        ReferenceAssignment incremental;
        incremental.apply(engine->feedback_constant);
        for(size_t j = 0; j < 6; j++) {
            int state = digits % 3;
            digits /= 3;
            const auto& key = j < 4 ? p.decision_order[j] : r.decision_order[j - 4];
            if(state) (j < 4 ? pv : rv)[key] = state - 1;
            if(state == 2) incremental.apply(j < 4 ? engine->protocol_frames[j] : engine->recovery_frames[j - 4]);
            auto full = run.assignment(*engine, {}, pv, rv, "recovery");
            auto direct = engine->plan->compiled.plan.fault_frame(full.faults, false);
            require(incremental.frame.x == direct.x && incremental.frame.z == direct.z,
                    "each reached decision delta equals complete ordered original-location transport");
        }
    }
}
void check_reference_boundary() {
    // Fixed |1> tensor T|+> measured in X: the first coordinate is closed,
    // and the second has the independent biased-T probability law.
    CompilerIR ir;
    ir.inputs = {0, 1};
    ir.outputs = {0, 1};
    ir.initial.width = 2;
    ir.initial.base = Bits(1);
    for(const auto& kind : {"H", "T", "H"}) {
        CompilerOp op;
        op.kind = kind;
        op.qubits = {1};
        op.exponent = std::string(kind) == "T" ? 1 : 0;
        op.location = int(ir.operations.size());
        ir.operations.push_back(std::move(op));
    }
    BoundaryTracePlan plan;
    plan.ir = ir;
    plan.prepare();
    auto owner = std::make_shared<const ReferenceNetwork>(plan, std::vector<Bits>{Bits(1), Bits(2)},
            std::vector<Bits>{}, std::vector<int>{}, std::vector<Bits>{Bits(1)});
    const auto& parent = *owner;
    // Independent reference for closed-subspace values: express each physical row in the
    // closed generators directly, instead of through the prepared label coordinates.
    const ReferenceCoordinates closed_labels(parent.closed, plan.compiled.plan.physical);
    auto affine_values = [&](const std::vector<Bits>& rows, const Frame& frame,
                                 const Bits& sector) -> std::optional<Bits> {
        Bits out;
        for(size_t j = 0; j < rows.size(); j++) {
            auto value = closed_labels(rows[j]);
            if(!value) return {};
            if(value->parity_and(sector) ^ rows[j].parity_and(frame.x)) out.set(j);
        }
        return out;
    };
    std::optional<ReferenceRecordMap> stored;
    stored.emplace(owner, std::vector<size_t>{1, 0}, std::vector<Bits>{Bits(3), Bits(2)});
    const auto& records = *stored;
    for(size_t a = 0; a < 4; a++)
        for(size_t b = 0; b < 4; b++) {
            auto physical = [](size_t mask) { return Bits(((mask & 1) << 1) | ((mask >> 1) & 1)); };
            auto mapped = records({Bits(a), Bits(b), Bits(a ^ b)});
            require(mapped.labels == parent.coordinates({physical(a), physical(b), physical(a ^ b)}),
                    "composed record map preserves arbitrary generator combinations and record ordering");
            require((mapped.labels[0] ^ mapped.labels[1]) == mapped.labels[2], "record map is linear");
        }
    {
        std::vector<size_t> repeated(70);
        for(size_t j = 0; j < repeated.size(); j++) repeated[j] = j % 2;
        ReferenceRecordMap wide(owner, std::move(repeated), {Bits(3), Bits(2)});
        auto mapped = wide({Bits::unit(0) ^ Bits::unit(64), Bits::unit(1) ^ Bits::unit(65) ^ Bits::unit(69)});
        require(mapped.labels == parent.coordinates({Bits(), Bits(2)}),
                "record view preserves cross-word XOR cancellation and canonical trimming");
        bool rejected = false;
        try {
            records({Bits::unit(70)});
        } catch(const std::invalid_argument&) { rejected = true; }
        require(rejected, "record view rejects masks outside its record index span");
        rejected = false;
        try {
            ReferenceRecordMap invalid(owner, {2}, {Bits(3)});
        } catch(const std::out_of_range&) { rejected = true; }
        require(rejected, "record view validates physical indices before borrowing rows");
    }
    const std::vector<Bits> kept = {Bits(3), Bits(1)};
    for(size_t sector = 0; sector < 2; sector++)
        for(size_t flip = 0; flip < 4; flip++) {
            Frame frame;
            frame.x = Bits(flip);
            auto value = affine_values({Bits(1), Bits(0), Bits(1)}, frame, Bits(sector));
            require(value && *value == Bits(((sector ^ flip) & 1) ? 5 : 0),
                    "closed affine values use actual sector and frame");
            require(!affine_values({Bits(3)}, frame, Bits(sector)), "nonclosed parity retains ordinary query path");
            for(size_t mask = 0; mask < 4; mask++) {
                auto physical = Bits(((mask & 1) << 1) | ((mask >> 1) & 1));
                require(parent.affine_coordinates(records({Bits(mask)}), frame, Bits(sector)) ==
                                affine_values({physical}, frame, Bits(sector)),
                        "certified closed membership retains actual mathematical branch and offsets");
            }
            auto boundary = parent.project(
                    {Bits(3), Bits(3), Bits(0)}, {Bits(1), Bits(1)}, {0, 1}, parent.closed_values(frame, Bits(sector)));
            require(boundary.zero, "contradictory closed boundaries remain zero");
        }
    // A committed equation keeps its pullback onto closed coordinates even
    // after open-coordinate elimination. Assignment changes must also revive
    // a formerly contradictory dependent row when its RHS changes back.
    const std::vector<Bits> observed = {Bits(3), Bits(2), Bits(1), Bits(0), Bits(3)};
    auto coordinates = parent.coordinates(observed);
    for(size_t sector = 0; sector < 2; sector++)
        for(size_t outcomes = 0; outcomes < 32; outcomes++)
            for(bool direct_closed : {false, true}) {
                ReferenceHistoryBoundary committed(parent, Bits(sector));
                std::vector<int> values;
                Frame frame;
                frame.x = Bits(3);
                committed.synchronize(frame);
                for(size_t j = 0; j < observed.size(); j++) {
                    values.push_back((outcomes >> j) & 1);
                    committed.append(coordinates[j], values.back(), direct_closed ? &observed[j] : nullptr);
                }
                for(size_t flip : {size_t(0), size_t(1), size_t(3), size_t(2), size_t(0), size_t(0), size_t(3)}) {
                    frame.x = Bits(flip);
                    committed.synchronize(frame);
                    auto fresh = parent.project({}, observed, values, parent.closed_values(frame, Bits(sector)));
                    require(committed.actual_closed == parent.closed_values(frame, Bits(sector)),
                            "incremental closed values retain initial assignment and cancelling deltas");
                    for(size_t word = 0; word < 2; word++) {
                        auto permits = [&](const ReferenceBoundary& boundary) {
                            if(boundary.zero) return false;
                            for(size_t j = 0; j < boundary.events.size(); j++)
                                if(boundary.events[j].parity_and(Bits(word)) != boundary.outcomes[j]) return false;
                            return true;
                        };
                        require(permits(committed.boundary) == permits(fresh),
                                "committed dependent and mixed rows retain absolute event under changed closed RHS");
                    }
                    auto updates = committed.rhs_updates;
                    committed.synchronize(frame);
                    require(committed.rhs_updates == updates && committed.appended_rows == observed.size(),
                            "unchanged assignment processes neither old observations nor RHS updates");
                }
            }
    for(size_t sector = 0; sector < 2; sector++)
        for(size_t flip = 0; flip < 4; flip++) {
            Frame frame;
            frame.x = Bits(flip);
            for(size_t event = 0; event < 5; event++)
                for(int bit = 0; bit < 2; bit++) {
                    std::vector<Bits> rows;
                    std::vector<int> values;
                    if(event < 4) {
                        rows.push_back(Bits(event));
                        values.push_back(bit);
                    }
                    Workspace ws;
                    auto result = parent.query(plan, frame, kept, rows, values, ws, Bits(sector));
                    std::vector<std::complex<double>> actual(4);
                    for(const auto& [coefficient, term] : result.terms) {
                        auto v = dense(term);
                        for(size_t j = 0; j < v.size(); j++) actual[j] += scalar_value(coefficient) * v[j];
                    }
                    std::vector<double> expected(4);
                    if(sector == 1)
                        for(size_t second = 0; second < 2; second++) {
                            size_t word = (1 | (second << 1)) ^ flip;
                            if(event < 4 && Bits(event).parity_and(Bits(word)) != bit) continue;
                            size_t label = ((word & 1) ^ ((word >> 1) & 1)) | ((word & 1) << 1);
                            expected[label] += (1 + (second ? -1 : 1) / std::sqrt(2.)) / 2;
                        }
                    for(size_t j = 0; j < 4; j++)
                        require(std::abs(actual[j] - expected[j]) < 1e-12,
                                "closed-label offsets and mixed queries preserve physical probabilities");
                }
        }
    const std::vector<Bits> masks = {Bits(0), Bits(1), Bits(2), Bits(3)};
    auto expected = records(masks).labels;
    std::weak_ptr<const ReferenceNetwork> lifetime = owner;
    {
        auto copied = records;
        auto moved = std::move(copied);
        owner.reset();
        stored.reset();
        require(lifetime.use_count() == 1, "moved record map owns its coordinate parent after original owners reset");
        require(moved(masks).labels == expected,
                "copied and moved record view remains valid after original owners reset");
    }
    require(lifetime.expired(), "record map releases its shared coordinate owner without a reference cycle");
}
void check_incremental_history_session() {
    auto probability = [](const BoundaryResult& result, size_t width) {
        std::vector<std::complex<double>> out(size_t(1) << width);
        for(const auto& [coefficient, term] : result.terms) {
            auto values = dense(term);
            for(size_t j = 0; j < out.size(); j++) out[j] += scalar_value(coefficient) * values[j];
        }
        return out;
    };
    auto same_boundary = [](const ReferenceHistoryBoundary& a, const ReferenceHistoryBoundary& b) {
        return a.boundary.keep == b.boundary.keep && a.boundary.events == b.boundary.events &&
                a.boundary.outcomes == b.boundary.outcomes && a.boundary.offset == b.boundary.offset &&
                a.boundary.zero == b.boundary.zero && a.actual_closed == b.actual_closed &&
                a.previous_x == b.previous_x && a.closed_parts == b.closed_parts && a.pivots == b.pivots &&
                a.open_slots == b.open_slots && a.rhs == b.rhs && a.closed_incidence == b.closed_incidence &&
                a.physical_incidence == b.physical_incidence && a.contradictions == b.contradictions &&
                a.appended_rows == b.appended_rows && a.rhs_updates == b.rhs_updates;
    };
    auto support_words = [](const Ket& support) {
        std::vector<bool> out(size_t(1) << support.width);
        if(!support.zero)
            for(size_t word = 0; word < (size_t(1) << support.columns.size()); word++) {
                auto value = support.base ^ xors(support.columns, Bits(word));
                out[value.empty() ? 0 : value.w[0]] = true;
            }
        return out;
    };
    for(bool magic : {false, true}) {
        CompilerIR ir;
        ir.inputs = {0, 1, 2};
        ir.outputs = {0, 1, 2};
        ir.initial.width = 3;
        auto gate = [&](std::string kind, std::vector<int> qubits, int exponent = 0) {
            CompilerOp op;
            op.kind = std::move(kind);
            op.qubits = std::move(qubits);
            op.exponent = exponent;
            op.location = int(ir.operations.size());
            ir.operations.push_back(std::move(op));
        };
        gate("H", {0});
        gate("H", {1});
        gate("H", {2});
        gate(magic ? "T" : "S", {2}, magic ? 1 : 2);
        gate("H", {2});
        // Three parity phases implement CS(0,1). Closing q0 leaves a
        // Clifford conditional law; an X before the phases changes its
        // support while a later X can restore the committed q0 boundary.
        size_t before = ir.operations.size();
        gate("I", {0});
        gate("PHASE", {0}, 1);
        gate("PHASE", {1}, 1);
        gate("PHASE", {0, 1}, -1);
        gate("H", {1});
        size_t after = ir.operations.size();
        gate("I", {0, 1});
        BoundaryTracePlan plan;
        plan.ir = ir;
        plan.prepare();
        auto parent = std::make_shared<const ReferenceNetwork>(plan, std::vector<Bits>{Bits(1), Bits(2), Bits(4)},
                std::vector<Bits>{}, std::vector<int>{}, std::vector<Bits>{Bits(1)});
        SamplingNetworkPlan prepared(parent);
        require(bool(prepared.contraction) == !magic, "fixture exercises certified one-term and general history paths");
        for(size_t sector = 0; sector < 2; sector++) {
            SamplingNetworkSession current(prepared, plan, Bits(sector));
            std::vector<Bits> observed;
            std::vector<int> outcomes;
            Frame frame;
            auto append = [&](Bits row, int bit) {
                current.committed.synchronize(frame);
                auto coordinates = parent->coordinates({row});
                current.committed.append(coordinates[0], bit, &row);
                observed.push_back(std::move(row));
                outcomes.push_back(bit);
            };
            auto check = [&](const std::vector<Bits>& keep) {
                current.committed.synchronize(frame);
                auto saved = current.committed;
                ReferenceRows coordinates{parent->coordinates(keep)};
                auto actual = current.query_committed(frame, coordinates);
                require(same_boundary(saved, current.committed),
                        "disposable marginal leaves every committed boundary field unchanged");
                Workspace workspace;
                auto expected = parent->query(plan, frame, keep, observed, outcomes, workspace, Bits(sector));
                auto a = probability(actual, keep.size()), b = probability(expected, keep.size());
                ProbabilityMarginal exact_actual{actual.terms, keep.size()},
                        exact_expected{expected.terms, keep.size()};
                for(size_t j = 0; j < a.size(); j++) {
                    require((exact_actual.value(Bits(j)) + Scalar(-1) * exact_expected.value(Bits(j))).zero(),
                            "persistent query preserves every absolute probability exactly in the scalar ring");
                    require(std::abs(a[j] - b[j]) < 1e-12,
                            "persistent session equals original parent query in absolute probability, including traced labels");
                }
                SamplingNetworkSession uncached(prepared, plan, Bits(sector));
                auto fresh_support = uncached.support(frame, keep, observed, outcomes);
                auto support = current.support_committed(frame, coordinates);
                require(support_words(support) == support_words(fresh_support),
                        "committed proposal conditions individual transported supports before forming its envelope");
                require(same_boundary(saved, current.committed),
                        "support query never commits its keep or trace branch");
                auto mass = current.mass_committed(frame);
                auto direct = parent->query(plan, frame, {}, observed, outcomes, workspace, Bits(sector));
                auto expected_mass = boundary_sum(direct.terms, {}, {}, workspace);
                require((mass + Scalar(-1) * expected_mass).zero(),
                        "committed history retains full scalar mass rather than normalized support");
                require(same_boundary(saved, current.committed),
                        "completed mass query leaves committed algebra unchanged");
                return a;
            };
            append(Bits(1), int(sector));
            check({Bits(2)}); // q2 is traced only in this temporary query.
            auto second = check({Bits(4)});
            double bias = magic ? 1 / std::sqrt(2.) : 0;
            require(std::abs(second[0] - (1 + bias) / 4.) < 1e-12 && std::abs(second[1] - (1 - bias) / 4.) < 1e-12,
                    "later query can recover a formerly traced label with the complete sector mass");
            check({Bits(3), Bits(4)});
            append(Bits(3), int(sector));
            append(Bits(2), 0); // Mixed plus dependent observation.
            check({Bits(4), Bits(2)});
            frame = plan.compiled.plan.fault_frame({{before, 0, 1}}, false);
            auto zero = check({Bits(2), Bits(4)});
            require(std::all_of(zero.begin(), zero.end(), [](auto value) { return std::abs(value) < 1e-12; }),
                    "changed closed RHS retains zero without redrawing or forgetting the observation");
            // The preceding support query returned before phase transport.
            // Restore q0 with a later physical X, retaining earlier phase signs.
            auto restore = plan.compiled.plan.fault_frame({{after, 0, 1}}, false);
            frame.x ^= restore.x;
            frame.z ^= restore.z;
            check({Bits(2)});
            check({Bits(4), Bits(3)});
            auto report_flip = plan.compiled.plan.fault_frame({{after, 1, 1}}, false);
            frame.x ^= report_flip.x;
            frame.z ^= report_flip.z;
            check({Bits(4), Bits(2)});
            frame = {};
            check({Bits(2), Bits(4)});
            require(current.committed.appended_rows == observed.size(),
                    "only newly committed observations enter the persistent boundary");
            if(!magic)
                require(current.geometry_updates > 0,
                        "actual feedback fault updates the certified Clifford geometry in this fixture");
        }
    }
}
int main() {
    {
        // A rejected row must not alter statistics or contaminate the reusable
        // scratch storage used for the following draw's exponent changes.
        using Row = std::vector<std::optional<ScientificValue>>;
        ScientificAccumulator actual, expected;
        auto same = [](const ScientificSummary& left, const ScientificSummary& right) {
            require(left.count == right.count, "rejected scientific row preserves count");
            require(left.estimates.size() == right.estimates.size() &&
                            left.standard_errors.size() == right.standard_errors.size(),
                    "rejected scientific row preserves column count");
            for(size_t j = 0; j < left.estimates.size(); j++) {
                require(left.estimates[j].scientific == right.estimates[j].scientific,
                        "rejected scientific row preserves estimates");
                require(left.standard_errors[j].scientific == right.standard_errors[j].scientific,
                        "rejected scientific row preserves standard errors");
            }
        };
        for(int64_t exponent : {-520, -500, 510, 520, 520}) {
            Row valid{ScientificValue{.5, exponent}, ScientificValue{.125, exponent},
                    ScientificValue{-.25, exponent}};
            actual.add(valid);
            expected.add(valid);
            same(actual.summary(), expected.summary());
            std::vector<Row> rejected{{}, {ScientificValue{.5, 1000}},
                    {std::nullopt, ScientificValue{.25, 1000}, ScientificValue{.5, 1000}},
                    {ScientificValue{.5, 1000}, std::nullopt, ScientificValue{.5, 1000}},
                    {ScientificValue{.5, 1000}, ScientificValue{.25, 1000}, std::nullopt},
                    {ScientificValue{std::numeric_limits<double>::quiet_NaN(), 1000},
                            ScientificValue{.25, 1000}, ScientificValue{.5, 1000}}};
            for(const auto& row : rejected) {
                bool failed = false;
                try {
                    actual.add(row);
                } catch(const std::invalid_argument&) {
                    failed = true;
                }
                require(failed, "invalid scientific row is rejected");
                same(actual.summary(), expected.summary());
            }
        }
    }
    {
        // Embed a small independently enumerable quadratic form across word
        // boundaries. Other coordinates have explicit one-variable Gauss sums.
        // Duplicate visible columns exercise non-injective output boundaries.
        std::mt19937_64 random(2026100307);
        Workspace ws;
        for(size_t n : {size_t(63), size_t(64), size_t(65), size_t(129), size_t(257)})
            for(int trial = 0; trial < 24; trial++) {
                Ket wide, small;
                wide.width = small.width = 3;
                wide.base = small.base = Bits(random() % 8);
                wide.phase = small.phase = int(random() % 8);
                wide.power = -int64_t(n);
                small.power = -8;
                wide.columns.resize(n);
                wide.linear.resize(n);
                wide.upper.resize(n);
                std::vector<size_t> locations;
                for(size_t j = 0; j < 8; j++) locations.push_back(j * (n - 1) / 7);
                std::complex<double> factor = 1;
                for(size_t j = 0; j < n; j++)
                    if(std::find(locations.begin(), locations.end(), j) == locations.end()) {
                        wide.linear[j] = (random() & 1) ? 1 : 3;
                        factor *= (1. + std::polar(1., wide.linear[j] * std::acos(-1.) / 2)) / std::sqrt(2.);
                    }
                for(size_t j = 0; j < 8; j++) {
                    auto c = Bits(j % 3 == 0 ? 0 : random() % 8);
                    int l = int(random() % 4);
                    Bits u;
                    for(size_t k = j + 1; k < 8; k++)
                        if(random() & 1) {
                            u.set(k);
                            wide.upper[locations[j]].set(locations[k]);
                        }
                    small.columns.push_back(c);
                    small.linear.push_back(l);
                    small.upper.push_back(u);
                    wide.columns[locations[j]] = c;
                    wide.linear[locations[j]] = l;
                }
                auto expected = dense(small);
                for(auto& a : expected) a *= factor;
                CliffordContraction prepared(wide.columns, {0, wide.linear, wide.upper});
                auto direct = contract_reduced(wide, ws),
                     actual = prepared.evaluate(wide.width, wide.base, wide.linear, wide.power, wide.phase);
                for(const auto& result : {direct, actual}) {
                    auto amplitudes = dense(result);
                    for(size_t j = 0; j < expected.size(); j++)
                        require(std::abs(amplitudes[j] - expected[j]) < 1e-10,
                                "cross-word Clifford elimination preserves non-injective amplitudes, phases and normalization");
                }
                auto unphased = contract_reduced(wide, ws, false);
                require(unphased.zero == direct.zero, "phase omission retains exact cancellation");
                auto a = dense(unphased), b = dense(direct);
                for(size_t j = 0; j < a.size(); j++)
                    require(std::abs(std::norm(a[j]) - std::norm(b[j])) < 1e-10,
                            "trace-only elimination preserves squared amplitudes");
            }
    }
    {
        // Independent bit-by-bit shifts across word boundaries, empty masks,
        // large offsets, and dense/sparse non-machine-word-sized masks.
        std::mt19937_64 random(20261003);
        for(size_t width : {size_t(0), size_t(1), size_t(63), size_t(64), size_t(65), size_t(130), size_t(1025)}) {
            Bits mask;
            for(size_t j = 0; j < width; j++)
                if(random() & 1) mask.set(j);
            for(size_t offset : {size_t(0), size_t(1), size_t(63), size_t(64), size_t(65), size_t(127), size_t(1024),
                        size_t(2049)}) {
                Bits left, right;
                for(size_t j = 0; j < width; j++)
                    if(mask.bit(j)) {
                        left.set(j + offset);
                        if(j >= offset) right.set(j - offset);
                    }
                require((mask << offset) == left && (mask >> offset) == right,
                        "wordwise shifts match independent individual bits");
                require(((mask << offset) >> offset) == mask, "wordwise shift roundtrip and canonical trimming");
            }
        }
    }
    {
        HistoryEfficiency efficiency;
        efficiency.add({.5, 1}, ScientificValue{0, 0}, {.5, 1}, {.5, 1});
        efficiency.add({.5, 1}, ScientificValue{.5, 1}, {.5, 1}, {.5, 1});
        require(efficiency.kappa() == std::make_optional(ScientificValue{.5, 1}), "unit weights have kappa one");
        require(efficiency.variance(false) == std::make_optional(ScientificValue{.5, -1}),
                "Bernoulli variance coefficient");
        require(sampler_decision(4, 2, 4, 10).gamma == 2, "native pilot Gamma");
    }
    check_axis_prefix();
    check_reference_coordinates();
    check_packed_clifford_updates();
    check_quadratic_batches();
    check_composed_feedback_frames();
    check_reference_boundary();
    check_incremental_history_session();
    {
        // Prepared elimination against independent finite sums, including every
        // even-linear assignment on each small randomized Clifford structure.
        uint64_t seed = 71819;
        auto next = [&]() {
            seed = seed * 6364136223846793005ULL + 1;
            return seed >> 32;
        };
        Workspace workspace;
        for(size_t n = 0; n <= 7; n++)
            for(size_t width = 0; width <= 3; width++)
                for(int trial = 0; trial < 16; trial++) {
                    Ket input;
                    input.width = width;
                    input.power = -int64_t(n);
                    input.phase = int(next() % 8);
                    input.base = Bits(next() & ((uint64_t(1) << width) - 1));
                    Quadratic q;
                    q.constant = 0;
                    for(size_t j = 0; j < n; j++) {
                        input.columns.push_back(Bits(next() & ((uint64_t(1) << width) - 1)));
                        q.linear.push_back(int(next() % 4));
                        q.upper.push_back(Bits(next() & ((uint64_t(1) << n) - 1)).above(j));
                    }
                    CliffordContraction plan(input.columns, q);
                    for(size_t assignment = 0; assignment < (size_t(1) << n); assignment++) {
                        input.linear = q.linear;
                        input.upper = q.upper;
                        for(size_t j = 0; j < n; j++)
                            input.linear[j] = mod4(input.linear[j] + 2 * ((assignment >> j) & 1));
                        require(plan.matches({0, input.linear, input.upper}),
                                "even signs reuse compiled Clifford pivots");
                        auto expected = contract_reduced(input, workspace);
                        auto actual = plan.evaluate(width, input.base, input.linear, input.power, input.phase);
                        require(actual.zero == expected.zero, "compiled Clifford cancellation");
                        if(!actual.zero)
                            require(actual.base == expected.base && actual.columns == expected.columns &&
                                            actual.linear == expected.linear && actual.upper == expected.upper &&
                                            actual.phase == expected.phase && actual.power == expected.power,
                                    "compiled Clifford exact phase and affine map");
                        auto exact = dense(input), result = dense(actual);
                        for(size_t j = 0; j < exact.size(); j++)
                            require(std::abs(exact[j] - result[j]) < 1e-10,
                                    "prepared Clifford against independent finite sum");
                    }
                    PhaseTensor tensor;
                    tensor.n = n;
                    tensor.linear = q.linear;
                    tensor.upper = q.upper;
                    tensor.power = input.power;
                    tensor.boundary.resize(width);
                    auto rows = transpose(input.columns, width);
                    for(size_t j = 0; j < width; j++) tensor.boundary[j] = {rows[j], int(input.base.bit(j))};
                    PhaseLimits limits;
                    ResourceCertificate certificate(tensor, phase_resource(tensor, limits));
                    require(bool(certificate.reference.contraction),
                            "rank-zero reference compiles its Clifford contraction");
                    check_transported_phase(tensor, certificate);
                }
    }
    {
        CertificateCache certificates(7, 2);
        certificates.put({"a", "one", Scalar(1), 0, 1.});
        certificates.put({"b", "two", Scalar(1), 0, 2.});
        require(!certificates.get("a") && certificates.get("b") && certificates.bytes == 4 &&
                        certificates.evictions == 1,
                "native certificate byte eviction");
        certificates.clear();
        require(certificates.entries.empty() && certificates.bytes == 0, "native cache clear");
        {
            auto repeated =
                    parse_circuit("DEPOLARIZE1(1/8) 0 1\nX_ERROR(1/8) 0\nDEPOLARIZE1(1/8) 2\n", "shared-law", {}, 100);
            auto plan = prepare_file_noise(repeated);
            require(plan.categories[0] == plan.categories[1] && plan.categories[0] == plan.categories[3],
                    "repeated physical sites share one immutable categorical law");
            require(plan.categories[0] != plan.categories[2], "different channel alternatives retain distinct laws");
            require(plan.actions[plan.offsets[0]].gates != plan.actions[plan.offsets[1]].gates,
                    "shared laws retain each original location's actions");
        }
        {
            FileNoiseActions actions;
            using T = FileNoiseActions::Template;
            actions.push_back({"manual", "", {{'Z', 2}, {'X', 2}, {'Z', 2}}});
            actions.add(T::Two, "pair", 7, 3);
            actions.push_back({"empty", "", {}});
            actions.add(T::One, "single", 9);
            auto saved = actions;
            actions = FileNoiseActions{};
            size_t index = 1;
            for(char a : std::string("IXYZ"))
                for(char b : std::string("IXYZ"))
                    if(a != 'I' || b != 'I') {
                        std::vector<std::pair<char, size_t>> expected;
                        if(a != 'I') expected.emplace_back(a, 7);
                        if(b != 'I') expected.emplace_back(b, 3);
                        auto actual = saved.at(index++);
                        require(actual.anchor == "pair" && actual.gates == expected,
                                "fixed pair template preserves IXYZ candidate and gate ordering");
                    }
            require(saved.at(0).gates == std::vector<std::pair<char, size_t>>{{'Z', 2}, {'X', 2}, {'Z', 2}},
                    "manual repeated noncommuting actions retain order");
            require(saved.at(16).gates.empty() && saved.at(19).gates[0] == std::make_pair('Z', size_t(9)),
                    "mixed explicit and template offsets survive copies");
            bool rejected = false;
            try {
                saved.at(20);
            } catch(const std::out_of_range&) { rejected = true; }
            require(rejected, "virtual action index bounds");
            require(saved.storage_counts()[2] == 2, "one stored site per fixed channel, not per alternative");
        }
        std::vector<Integer> bounds;
        auto proposal =
                independent_proposal({"a", "b"}, {{"a", Rational(Integer(2), Integer(7))}}, [&](const Integer& bound) {
                    bounds.push_back(bound);
                    return Integer(0);
                });
        require(bounds == std::vector<Integer>({Integer(7), Integer(2)}) && proposal.true_word == Bits(3) &&
                        proposal.probability == Rational(Integer(1), Integer(7)),
                "native exact independent proposal and RNG order");
        ProposalEstimate estimator;
        estimator.add(DecimalValue(Integer(1)), DecimalValue(Integer(1)), DecimalValue(Integer(0)));
        estimator.add(DecimalValue(Integer(1)), DecimalValue(Integer(1)), DecimalValue(Integer(1)));
        auto estimates = estimator.result();
        require(estimates.at("conditional_diagnostic")->rational() == Rational(Integer(1), Integer(2)) &&
                        estimates.at("conditional_standard_error")->rational() == Rational(Integer(1), Integer(2)),
                "native decimal paired ratio uncertainty");
        BoundaryCoordinates coordinates({"a", "b", "c"}, {{"a", "b"}, {"b", "a"}}, {0, 1}, {"c"});
        require(!coordinates.compatible && coordinates.selection == 1 && coordinates.feedback == 1 &&
                        coordinates.transform.size() == 3,
                "native boundary coordinates retain inconsistent redundant RHS");
        LiteralProgramLayout literal({{"x", "MZ", "__inline_1"}, {"x", "MX", {}}}, {0, 1});
        require(literal.operations[1].id == "__literal_operation_1" && literal.operations[1].record == "__inline_1_" &&
                        literal.truth[1].second == 1,
                "literal compiler IDs and inline records avoid original-name collisions");
    }
    InstrumentLayout branch_layout(
            {0}, {0}, 1, 0, 0, {{"m", "MZ", "m"}, {"c", "CONTROL", {}}, {"n", "MX", "n"}}, {{"c", {{"m"}, "v1"}}});
    BranchSchedule branch_schedule(branch_layout, {{"c", std::nullopt}});
    std::vector<OriginalFault> branch_faults{{"c", 0, 0}, {"m", 0, 2}, {"c", 0, 3}};
    auto branch_assignment = branch_layout.validate({{"m", 1}, {"n", 0}}, {"m"}, branch_faults);
    size_t control_calls = 0;
    auto branch = branch_schedule.collect(branch_assignment, branch_faults, [&](size_t index, const auto& parents) {
        control_calls++;
        require(index == 0 && parents == std::vector<std::pair<std::string, int>>{{"m", 0}},
                "controller gets only declared reported parents");
        return std::vector<BranchGate>{branch_gate("X", {0})};
    });
    require(control_calls == 1 && branch.decisions.size() == 1 && branch.reported_word.empty(),
            "native causal branch decisions");
    require(branch.ordered_faults == std::vector<size_t>({1, 0, 2}),
            "original operation order with stable duplicate faults");
    require(branch.steps.size() == 5 && branch.steps[3].id == "c/fault/1" &&
                    branch.steps[1].replacement->operation.value.pauli == "Z",
            "literal branch preserves identity-fault positions");
    {
        Ket initial;
        initial.width = 1;
        initial.columns = {Bits(1)};
        initial.linear = {0};
        initial.upper = {Bits()};
        initial.power = -1;
        auto m = operation("MZ", {0});
        m.id = "m";
        m.record = "m";
        auto c = operation("CONTROL");
        c.id = "c";
        auto n = operation("MZ", {0});
        n.id = "n";
        n.record = "n";
        auto family = feedback_family({0}, {}, {});
        DeclaredGate gate;
        gate.kind = "X";
        gate.qubits = {0};
        gate.guard = record_expression("m");
        BranchSchedule schedule(branch_layout, {{"c", std::vector<DeclaredGate>{gate}}});
        auto input = compile_input(initial, {0}, {0},
                {program_operation(m), program_operation(c), program_operation(n)}, {{"c", family}}, 4096);
        GlobalPlan global;
        global.ir = input.ir;
        prepare_global(global, 1 << 20);
        OperationTransport transport(input, global, schedule, {{"c", family}}, 1 << 20);
        auto values = branch_layout.validate({{"m", 1}, {"n", 0}}, {}, {});
        auto packed = transport.pack(values, {});
        apply_native_controls(global.controls, values.true_word ^ values.flip_word, global.plan().magic, packed);
        Workspace workspace;
        require(global.controls.size() == 1 &&
                        std::abs(scalar_value(global.evaluate(packed, workspace)).real() - .5) < 1e-12,
                "native original assignment transport and declared control execution");
    }
    {
        auto m = operation("MZ", {0});
        m.id = "m";
        m.record = "m";
        auto c = operation("CONTROL");
        c.id = "c";
        auto n = operation("MZ", {0});
        n.id = "n";
        n.record = "n";
        Ket initial;
        initial.width = 1;
        OperationBoundaryTask task;
        task.selection = {{"n"}};
        task.outcomes = {1};
        task.observables = {{{}, Bits(1)}};
        OperationBoundarySampler sampler(branch_layout, initial, {0}, {0}, {m, c, n}, {{"c", std::nullopt}},
                {{"c", feedback_family({0}, {}, {})}}, task, {}, {});
        size_t calls = 0;
        auto draw = sampler.sample(
                {}, {"m"}, [](size_t) { return Bits(); },
                [&](const auto&, const auto& parents) {
                    calls++;
                    require(parents == std::vector<std::pair<std::string, int>>{{"m", 1}},
                            "native operation decoder only sees reported input");
                    return std::vector<Operation>{operation("X", {0})};
                });
        require(calls == 1 && draw.mass.c == Scalar(1).c && draw.acceptance.c == Scalar(1).c && !draw.error &&
                        draw.moments[0].c == Scalar(-1).c,
                "native full causal operation sampling and terminal masses");
    }
    InstrumentLayout instrument({0}, {0}, 1, 0, 0, {{"m", "MZ", "a"}, {"c", "CONTROL", {}}}, {{"c", {{"a"}, "1"}}});
    auto assigned = instrument.validate({{"a", 1}}, {"a", "a"}, {{"c", 0, 1}, {"m", 0, 2}, {"c", 0, 3}});
    require(assigned.true_word == Bits(1) && assigned.flip_word.empty(), "true records and duplicate report flips");
    require(assigned.fault_groups[0].second == std::vector<size_t>({0, 2}), "ordered coincident original faults");
    auto ideal_file = parse_circuit("H 0\nT 0\n", "target", {}, 100);
    IdealTarget ideal(ideal_file, 1);
    auto projection = ideal.attach({5}, 7, 8);
    require(projection.second == std::vector<std::string>{"m7"}, "target record namespace");
    require(projection.first[0].kind == "PHASE" && projection.first[0].exponent == -1 &&
                    projection.first[0].qubits == std::vector<size_t>{5},
            "target inverse order and mapping");
    auto recovery_file = parse_circuit("H 1\nM 1\nCX rec[-1] 0\n", "recovery", {}, 100);
    validate_recovery(recovery_file);
    auto recovery = attach_recovery(recovery_file, 1, {5}, 7, 8, {});
    require(recovery.operations.front().kind == "PREP_ZERO" &&
                    recovery.operations.front().qubits == std::vector<size_t>{8},
            "fresh recovery ancilla");
    require(recovery.controls.front().second.guard.records == std::vector<size_t>{7} &&
                    recovery.controls.front().second.gates.front().qubits == std::vector<size_t>{5},
            "recovery record and physical output mapping");
    auto prepared_file = parse_circuit("H 0\nT 0\n", "physical", {}, 100);
    FeedbackTemplate prepared_feedback(prepared_file);
    FileProgram full(prepared_file, {0}, prepared_feedback, nullptr, nullptr, &ideal, "target");
    auto prepared_program = full.compile(4096);
    BoundaryTracePlan prepared_target;
    prepared_target.ir = std::move(prepared_program.second.ir);
    prepared_target.prepare();
    Workspace target_workspace;
    auto target_projection = prepared_target.query({}, {}, {Bits(2)}, {0}, {}, {}, false, target_workspace);
    require((boundary_sum(target_projection.terms, {}, {}, target_workspace) + Scalar(-1)).zero(),
            "native file target fidelity");
    auto runtime_file =
            parse_circuit("H 0\nM(1/3) 0\nDETECTOR rec[-1]\nCONTROL[echo] det[-1] -> ctrl[0]\nCX ctrl[0] 1\n",
                    "runtime", {"echo"}, 100);
    HistoryTask runtime_task;
    runtime_task.boundaries.sample_records = {0};
    runtime_task.boundaries.observables = {{{}, Bits(1)}};
    runtime_task.boundaries.has_target = true;
    runtime_task.boundaries.target_records = {"m1"};
    IdealTarget zero_target(parse_circuit("", "target", {}, 100), 1);
    {
        auto rejects_task = [&](HistoryTask task) {
            bool rejected = false;
            try {
                CompiledExperiment invalid(FileSource(runtime_file, {1}, {}, {}), std::move(task),
                        std::make_shared<const FileNoisePlan>(), {}, {});
            } catch(const std::invalid_argument&) { rejected = true; }
            require(rejected, "malformed history task is rejected at construction before indexed runtime access");
        };
        HistoryTask selected;
        selected.boundaries.selection = {{"m0"}};
        rejects_task(selected);
        selected.outcomes = {2};
        rejects_task(selected);
    }
    auto shared_noise = std::make_shared<const FileNoisePlan>(prepare_file_noise(runtime_file));
    CompiledExperiment runtime(FileSource(runtime_file, {1}, {}, zero_target), runtime_task, shared_noise, {}, {});
    require(&runtime.noise == shared_noise.get(), "runtime shares the original immutable noise plan");
    shared_noise.reset();
    runtime.prepare();
    require(runtime.context.queries == 0 && runtime.complete, "complete native compilation before sampling");
    size_t callbacks = 0, seeds = 0;
    ExecutionHooks hooks;
    hooks.seed = [&]() { return uint64_t(seeds++); };
    hooks.controller = [&](const std::string& stage, size_t index, const std::vector<int>& values, size_t count) {
        require(stage == "protocol" && index == 0 && count == 1, "native external callback interface");
        callbacks++;
        return values;
    };
    for(size_t j = 0; j < 12; j++) {
        auto draw = runtime.sample(hooks);
        int report = draw.samples.at(0);
        require(std::abs(scalar_value(draw.masses.history).real() - .5) < 1e-12, "native causal history normalization");
        require(std::abs(scalar_value(*draw.masses.error).real() - .5 * report) < 1e-12,
                "native target error follows reported physical feedback");
        require(std::abs(scalar_value(draw.masses.moments[0]).real() - .5 * (1 - 2 * report)) < 1e-12,
                "native terminal observable");
        require(draw.decoder_calls == 1 && draw.sampled_parities == 1, "native repeated parity reuse");
    }
    require(callbacks == 12 && seeds == 24, "native caller RNG and decoder counts");
    auto enum_task = runtime_task;
    enum_task.boundaries.sample_records.clear();
    auto enum_run = std::make_shared<CompiledExperiment>(FileSource(runtime_file, {1}, {}, zero_target), enum_task,
            std::make_shared<const FileNoisePlan>(prepare_file_noise(runtime_file)), RuntimeLimits(), PhaseDatabase());
    enum_run->prepare();
    FileEnumeration enumeration(enum_run, 1, 4, 4, 8);
    while(enumeration.next(hooks)) {}
    auto enum_total = enumeration.aggregate();
    require(enum_total.covered == enumeration.denominator && enumeration.next_index == 2 && enumeration.branches == 4,
            "native complete categorical/history enumeration");
    require(std::abs(scaled_real_ratio(enum_total.sums[2], file_integer_scalar(enumeration.denominator)).first - .5) <
                    1e-12,
            "native exact error ratio mantissa");
    require(seeds == 24, "exact history traversal consumes no random values");
    CausalKnowledge causal;
    auto pending = causal.request({{Bits(3), 1}, {Bits(3), 0}, {Bits(1), 0}});
    require(pending.pending == std::vector<Bits>({Bits(3), Bits(1)}), "duplicate decoder parity reuse");
    require(causal.observe(pending, Bits(1)) == std::vector<int>({0, 1, 0}), "causal parity outcomes");
    auto known = causal.request({{Bits(2), 0}});
    require(known.pending.empty() && causal.observe(known, {}) == std::vector<int>{1}, "known parity consequence");
    auto circuit = parse_circuit(
            "M(1/30) 0\nCONTROL[XOR](1) rec[-1] rec[-1] -> ctrl[0]\nCONTROL[decoder] ctrl[0] -> ctrl[1]\nPOSTSELECT ctrl[1]\n",
            "native.magna", {"decoder"}, 100);
    require(circuit.num_records == 1 && circuit.instructions[0].args[0] == Rational(Integer(1), Integer(30)),
            "native exact file parser");
    require(circuit.expression(circuit.instructions[1].outputs[0]) == AffineExpression{1, {}},
            "exact static cancellation");
    require(circuit.external == std::vector<size_t>{2} && circuit.format_version() == "magna-0.1",
            "native control metadata");
    require(parse_circuit("REPEAT 99999999999999999999999999 {\n}\n", "empty", {}, 1).instructions.empty(),
            "empty repeat range");
    auto law = prepare_file_noise(
            parse_circuit("DEPOLARIZE2(1/10) 0 1\nCORRELATED_ERROR(1/3) X0\nELSE_CORRELATED_ERROR(1/2) Y0\nM(1/5) 0\n",
                    "noise", {}, 100),
            Rational(Integer(1), Integer(2)));
    require(law.categories.size() == 3 && law.actions.size() == 18 && law.tilted == 3,
            "native original noise construction");
    require(law.categories[1]->probabilities[0].n == Integer(1) && law.categories[1]->probabilities[0].d == Integer(6),
            "exclusive correlated survival under a common rate scale");
    {
        auto empty = parse_circuit("I 0\n", "empty-noise", {}, 100);
        for(int scale : {-1, 0}) {
            bool rejected_file = false, rejected_model = false;
            try {
                prepare_file_noise(empty, Rational(Integer(scale)));
            } catch(const std::invalid_argument&) { rejected_file = true; }
            try {
                draw_noise_model({}, [](const Integer&) { return Integer(); }, {}, Rational(Integer(scale)));
            } catch(const std::invalid_argument&) { rejected_model = true; }
            require(rejected_file && rejected_model, "native entries validate scales even without noise sites");
        }
        auto mixed = parse_circuit("X_ERROR(1/10) 0\nZ_ERROR(1/5) 1\n", "mixed-noise", {}, 100);
        for(int scale : {5, 6}) {
            bool rejected = false;
            try {
                prepare_file_noise(mixed, Rational(Integer(scale)));
            } catch(const std::invalid_argument&) { rejected = true; }
            require(rejected, "scaled total must retain the physical no-error outcome");
        }
    }
    {
        struct Channel {
            int numerator, denominator;
            bool guarded;
        };
        std::vector<Channel> channels{
                {1, 1000, false}, {1, 1000, false}, {1, 1000, true}, {0, 6, false}, {1, 6, false}};
        for(int k = 2; k <= 36; k++) channels.push_back({k, 1000, false});
        channels.push_back({1, 1000, false}); // The first law has crossed the 32-entry retention cap.
        std::string source = "CONTROL[XOR](1) -> ctrl[0]\n";
        for(size_t j = 0; j < channels.size(); j++) {
            const auto& c = channels[j];
            source += "PAULI_CHANNEL_1(";
            for(int multiple = 1; multiple <= 3; multiple++) {
                if(multiple > 1) source += ",";
                source += std::to_string(multiple * c.numerator) + "/" + std::to_string(c.denominator);
            }
            source += ") " + std::string(c.guarded ? "ctrl[0] " : "") + std::to_string(j % 3) + "\n";
        }
        auto parsed = parse_circuit(source, "repeated-noise", {}, 1000);
        for(const std::optional<Rational>& scale : std::vector<std::optional<Rational>>{
                    {}, Rational(Integer(1), Integer(2)), Rational(Integer(1)), Rational(Integer(2))}) {
            auto cached = prepare_file_noise(parsed, scale);
            FileNoisePlan independent;
            independent.has_actions = true;
            size_t expected_tilted = 0;
            require(cached.categories.size() == channels.size(), "every original noise location remains present");
            require(cached.guarded == 1 && cached.deterministic == 2,
                    "guard and deterministic channel metadata survive law reuse");
            for(size_t j = 0; j < channels.size(); j++) {
                const auto& c = channels[j];
                Rational total(Integer(6 * c.numerator), Integer(c.denominator));
                bool eligible = scale && !c.guarded && c.numerator > 0 && 6 * c.numerator < c.denominator;
                Rational proposal_total = eligible ? total * (*scale) : total;
                std::vector<FileFraction> expected;
                for(int multiple = 1; multiple <= 3; multiple++) {
                    // Each active channel has the fixed independent ratio 1:2:3.
                    Rational p(Integer(multiple * c.numerator), Integer(c.denominator));
                    if(eligible) p = p * (*scale);
                    expected.emplace_back(p.n, p.d);
                    const auto& observed = cached.categories[j]->probabilities[multiple - 1];
                    require(observed.n * p.d == p.n * observed.d,
                            "cached exact law matches independent channel probabilities");
                    const auto& action = cached.actions[cached.offsets[j] + multiple - 1];
                    require(action.anchor == parsed.instructions[j + 1].op_id && action.report.empty() &&
                                    action.gates == std::vector<std::pair<char, size_t>>{{"XYZ"[multiple - 1], j % 3}},
                            "reused law preserves each original anchor and ordered Pauli action");
                }
                independent.offsets.push_back(3 * j);
                independent.categories.push_back(std::make_shared<const FileNoiseCategory>(expected));
                if(scale) {
                    Rational absent(Integer(1)), present(Integer(1));
                    if(eligible) {
                        absent = (Rational(Integer(1)) - total) / (Rational(Integer(1)) - proposal_total);
                        present = total / proposal_total;
                    }
                    independent.factors.push_back(
                            {FileFraction(absent.n, absent.d), FileFraction(present.n, present.d)});
                    for(size_t choice = 0; choice < 2; choice++) {
                        const auto& actual = cached.factors[j][choice];
                        const auto& expected = independent.factors[j][choice];
                        require(actual.n * expected.d == expected.n * actual.d,
                                "cached exact proposal likelihood ratio");
                    }
                }
                expected_tilted += eligible && proposal_total != total;
            }
            independent.prepare_weights();
            require(cached.tilted == expected_tilted, "proposal treatment distinguishes guarded channels");
            for(uint64_t seed = 0; seed < 32; seed++) {
                std::mt19937_64 reused_rng(seed), independent_rng(seed);
                for(size_t j = 0; j < channels.size(); j++)
                    require(cached.categories[j]->draw(reused_rng) == independent.categories[j]->draw(independent_rng),
                            "cached and freshly constructed noise laws consume the same seeded choices");
                require(reused_rng == independent_rng, "noise law reuse preserves final RNG state");
                for(bool sparse : {false, true}) {
                    auto actual = draw_noise(cached, seed, sparse), expected = draw_noise(independent, seed, sparse);
                    require(actual.selected == expected.selected &&
                                    actual.weight.n * expected.weight.d == expected.weight.n * actual.weight.d,
                            "law reuse preserves seeded full and sparse draws and exact importance weights");
                }
            }
        }
    }
    std::vector<Integer> bounds;
    auto choose = [&](const Integer& n) {
        bounds.push_back(n);
        return Integer();
    };
    require(weighted_choice({Rational(), Rational(Integer(1), Integer(3)), Rational(Integer(2), Integer(3))}, choose) ==
                    1,
            "native exact categorical choice");
    require(bounds == std::vector<Integer>{Integer(3)}, "zero channel consumes no caller randomness");
    auto physical = parse_circuit(
            "M 0\nDETECTOR rec[-1]\nCONTROL[decoder] det[-1] -> ctrl[0]\nCH ctrl[0] 1\nCONTROL[NOT] ctrl[0] -> ctrl[1]\nPOSTSELECT ctrl[1]\n",
            "feedback", {"decoder"}, 100);
    FeedbackTemplate feedback(physical);
    require(feedback.wire_count == 4 && feedback.virtual_controls.size() == 1 && feedback.selections.size() == 1,
            "complete fixed virtual template");
    auto lowered = lower_file(physical, {0, 1}, {}, {}, &feedback);
    require(std::count_if(lowered.operations.begin(), lowered.operations.end(),
                    [](const auto& op) {
                        return op.kind == "PHASE" && op.exponent && (*op.exponent == 1 || *op.exponent == -1);
                    }) == 2,
            "fixed controlled-H resource gates");
    auto offset = feedback.faults({});
    require(offset.size() == 1 && offset[0].anchor.find("selection") != std::string::npos,
            "selection constant retained as assignment");
    auto affine_file = parse_circuit("H 0\nM 0\nCH rec[-1] 1\n", "affine", {}, 100);
    auto affine_lowered = lower_file(affine_file, {0, 1}, {}, {});
    auto deferred = defer_instrument(
            {0, 1}, {0, 1}, affine_lowered.operations, file_declarations(affine_lowered), {}, true, {});
    require(deferred.records.size() == 1 && deferred.report_anchors.size() == 1,
            "native file-to-coherent-control pipeline");
    require(std::none_of(deferred.operations.begin(), deferred.operations.end(),
                    [](const auto& op) { return op.value.kind == "CONTROL"; }),
            "coherent controls fully lowered");
    std::vector<ProgramOperation> program;
    for(const auto& op : deferred.operations) program.push_back(program_operation(op.value));
    Ket fixed_input;
    fixed_input.width = 2;
    std::vector<int> outputs{0, 1};
    for(const auto& [name, wire] : deferred.records) outputs.push_back(int(wire));
    auto input = compile_input(fixed_input, {0, 1}, outputs, program, {}, 4096);
    {
        // Trace compilation omits amplitude-only objects, while preserving
        // the complete fixed-input dilation and original-location transport.
        auto amplitude = prepare_compiler(input.ir, false);
        auto trace = prepare_compiler(input.ir, false, nullptr, true, CompilerPurpose::Trace);
        require(trace.plan.basis.empty() && trace.plan.forms.empty() && trace.plan.polar.empty() &&
                        trace.plan.records.kernel.empty() && trace.linear.empty() && trace.upper.empty() &&
                        trace.magic_terms.empty(),
                "trace does not construct amplitude projection objects");
        const auto& a = amplitude.plan.skeleton;
        const auto& b = trace.plan.skeleton;
        require(a.width == b.width && a.base == b.base && a.columns == b.columns && a.linear == b.linear &&
                        a.upper == b.upper && a.power == b.power && a.phase == b.phase && a.zero == b.zero,
                "trace purpose preserves the full dilation");
        require(amplitude.plan.basis.size() == b.columns.size(),
                "deferred amplitude dimension needs no identity basis");
        require(amplitude.plan.magic == trace.plan.magic &&
                        amplitude.rotation_axes.size() == trace.rotation_axes.size(),
                "trace purpose preserves rotation metadata");
        for(size_t j = 0; j < amplitude.rotation_axes.size(); j++) {
            const auto& x = amplitude.rotation_axes[j];
            const auto& y = trace.rotation_axes[j];
            require(x.x == y.x && x.z == y.z && x.p == y.p, "trace purpose preserves endpoint axes");
        }
        for(size_t location = 0; location < trace.plan.locations.size(); location++)
            for(int wire = 0; wire < 2; wire++)
                for(int pauli = 1; pauli < 4; pauli++) {
                    auto x = amplitude.plan.fault_frame({{location, wire, pauli}});
                    auto y = trace.plan.fault_frame({{location, wire, pauli}});
                    require(x.x == y.x && x.z == y.z && x.p == y.p, "trace purpose preserves fault transport");
                }
    }
    BoundaryTracePlan prepared;
    prepared.ir = input.ir;
    prepared.prepare();
    Workspace compiler_workspace;
    auto sampled_record =
            sample_boundary_rows(prepared, Assignment(), {Bits::unit(2)}, {}, {}, [](size_t) { return Bits(); });
    require(sampled_record.word.empty() && sampled_record.mass.c == Scalar(1, -1).c && sampled_record.mass.power == -1,
            "native operation boundary marginal sampling");
    auto result = prepared.query(Assignment(), {}, {}, {}, Bits(), Bits::unit(1), false, compiler_workspace);
    require(std::abs(scalar_value(boundary_sum(result.terms, {}, {}, compiler_workspace)).real() - .5) < 1e-12,
            "native parse-lower-compile-evaluate pipeline");
    BoundarySpec event_boundary{{}, {}, Bits(), Bits(3)};
    auto network = prepared.compile(event_boundary);
    require(network == prepared.compile(event_boundary) && (network->joint || network->inherited),
            "compiler owns boundary resources before assignments");
    const auto& [report_name, report_anchor, report_wire] = deferred.report_anchors.front();
    Assignment flipped_report;
    flipped_report.faults.push_back({input.locations.at(report_anchor), int(report_wire), 1});
    auto shifted = prepared.query(flipped_report, {}, {}, {}, Bits(), Bits(3), false, compiler_workspace);
    require(std::abs(scalar_value(boundary_sum(shifted.terms, {}, {}, compiler_workspace)).real() + .5) < 1e-12,
            "prepared report update preserves physical correlation");
    require(shifted.diagnostics.reference_entries_reused && shifted.diagnostics.resource_constructions == 0,
            "certified evaluation constructs no resource entries");
    require(!prepared.preparation && prepared.compiled.plan.skeleton.columns.empty() &&
                    prepared.compiled.plan.forms.empty(),
            "runtime retains no output-open amplitude network");
    require(shifted.diagnostics.output_trace_reused && shifted.diagnostics.pairing_seconds == 0 &&
                    shifted.diagnostics.phase_coordinate_constructions == 0,
            "runtime propagates updates through prepared trace and phase maps");
    PhaseTensor reference;
    reference.n = 2;
    reference.linear = {0, 0};
    reference.upper.resize(2);
    reference.terms = {{Bits(1), 1}, {Bits(2), 1}};
    reference.boundary = {{Bits(1), 0}, {Bits(2), 0}};
    reference.power = -2;
    PhaseLimits certificate_limits;
    ResourceCertificate certificate(reference, phase_resource(reference, certificate_limits));
    require(bool(certificate.reference.contraction), "character geometry is compiled with the resource");
    {
        // Different widths and noninjective maps exercise hidden Gaussian sums.
        // Each actual quadratic update is also checked by direct finite sums.
        for(size_t n = 2; n <= 6; n++)
            for(size_t width : {size_t(0), size_t(1), size_t(3)}) {
                PhaseTensor base;
                base.n = n;
                base.linear.resize(n);
                base.upper.resize(n);
                base.terms = {{Bits(1), 1}, {Bits::unit(n - 1), 3}};
                base.power = -int64_t(n);
                for(size_t j = 0; j < width; j++) base.boundary.push_back({Bits((j + 1) * 3).low(n), 0});
                ResourceCertificate prepared(base, phase_resource(base, certificate_limits));
                require(bool(prepared.reference.contraction), "generic character expansion prepares shared geometry");
                for(size_t sample = 0; sample < 16; sample++) {
                    auto changed = base;
                    for(size_t j = 0; j < n; j++) {
                        changed.linear[j] = int((sample + 3 * j) % 4);
                        changed.upper[j] = Bits(sample * (j + 1)).low(n).above(j);
                    }
                    changed.constant = int(sample % 8);
                    changed.power -= int64_t(sample % 3);
                    changed.terms.push_back({Bits::ones(n), int(2 * (sample % 4))});
                    for(size_t j = 0; j < width; j++) changed.boundary[j].second = (sample >> j) & 1;
                    auto assigned = prepared.transport(changed);
                    require(assigned->contraction == prepared.reference.contraction,
                            "actual assignments share one immutable contraction map");
                    check_transported_phase(changed, prepared);
                }
                auto changed_map = base;
                changed_map.boundary.push_back({Bits::unit(n - 1), 1});
                check_transported_phase(changed_map, prepared);
            }
        auto budgeted = phase_resource(reference, certificate_limits);
        PhaseLimits tight = certificate_limits;
        tight.max_table_bytes = budgeted.core_phase_mod8.size() * sizeof(int) + phase_entries_bytes(budgeted.entries);
        budgeted.contraction.reset();
        prepare_phase_contraction(reference, budgeted, budgeted.core_phase_mod8.size() * sizeof(int), tight);
        require(!budgeted.contraction, "optional geometry respects remaining resource-table budget");
        auto constrained = budgeted;
        constrained.entries.front().constraints.push_back({Bits(1), 0});
        prepare_phase_contraction(reference, constrained, 0, certificate_limits);
        require(!constrained.contraction, "entry-dependent supports retain ordinary contraction");
        auto clifford_entries = budgeted;
        clifford_entries.entries.front().parities.push_back({Bits(1), 1});
        prepare_phase_contraction(reference, clifford_entries, 0, certificate_limits);
        require(!clifford_entries.contraction, "entry-dependent quadratic phases retain ordinary contraction");
    }
    auto actual = reference;
    actual.linear = {1, 2};
    actual.upper[0] = Bits(2);
    actual.constant = 3;
    actual.power = -4;
    actual.boundary = {{Bits(3), 1}, {Bits(2), 0}};
    CliffordResourceUpdate update;
    auto transported = certificate.transport(actual, &update);
    require(transported && transported->prepared_entries == certificate.reference.prepared_entries,
            "reference entries share immutable ownership");
    require(update.linear == std::vector<int>({1, 2}) && update.upper[0] == Bits(2) && update.constant == 3,
            "explicit Clifford update map");
    auto states = phase_states(actual, *transported, compiler_workspace, certificate_limits);
    std::vector<std::complex<double>> amplitudes(4);
    for(const auto& [coefficient, state] : states) {
        auto values = dense(state);
        for(size_t j = 0; j < 4; j++) amplitudes[j] += scalar_value(coefficient) * values[j];
    }
    for(size_t word = 0; word < 4; word++) {
        size_t x = word & 1, y = word >> 1, destination = (x ^ y ^ 1) | (y << 1);
        int exponent = int(x + y + 2 * x + 4 * y + 4 * x * y + 3);
        require(std::abs(amplitudes[destination] - std::polar(.25, exponent * std::acos(-1.) / 4)) < 1e-12,
                "transported exact resource against independent finite sum");
    }
    actual.terms.push_back({Bits(3), 1});
    require(!certificate.transport(actual), "uncovered phase family stays explicit");
    {
        PhaseTensor base;
        base.n = 4;
        base.linear = {1, 3, 2, 0};
        base.upper.resize(4);
        base.upper[0] = Bits(6);
        base.terms = {{Bits(1), 1}, {Bits(3), 7}, {Bits(7), 1}};
        base.constant = 5;
        base.power = -4;
        base.boundary = {{Bits(1), 0}, {Bits(2), 0}, {Bits(4), 0}, {Bits(8), 0}};
        ResourceCertificate compiled(base, phase_resource(base, certificate_limits));
        for(int increment : {0, 2, 4, 6})
            for(int offset = 0; offset < 2; offset++) {
                auto changed = base;
                // Change an existing odd coefficient, add an even parity and change
                // explicit Clifford coefficients; include a noninjective boundary.
                changed.terms[0].second += increment;
                changed.terms.push_back({Bits(14), increment});
                changed.linear = {3, 0, 1, 2};
                changed.upper[1] ^= Bits(12);
                changed.constant = offset;
                changed.boundary = {{Bits(3), offset}, {Bits(3), 1}, {Bits(12), offset}};
                changed.power = -6;
                check_transported_phase(changed, compiled);
            }
        auto alternative = base;
        // Every nonzero assignment satisfies exactly eight of these 15 parities,
        // so this changes the representation by the zero function modulo eight.
        for(size_t mask = 1; mask < 16; mask++) alternative.terms.push_back({Bits(mask), 1});
        alternative.linear[2] = 1;
        alternative.upper[0] ^= Bits(8);
        require(!compiled.transport(alternative, nullptr, true),
                "strict reference network transport never reconstructs a quotient for a different parity representation");
        require(bool(compiled.transport(base, nullptr, true)),
                "strict reference network transport reuses the prepared parity representation");
        check_transported_phase(alternative, compiled);
        check_transported_phase(base, compiled);
    }
    {
        PhaseTensor scalar_reference;
        scalar_reference.constant = 3;
        scalar_reference.power = -2;
        ResourceCertificate clifford(scalar_reference, phase_resource(scalar_reference, certificate_limits));
        require(clifford.reference.quotient.axes.empty(), "scalar reference has no non-Clifford coordinates");
        for(size_t dimension : {size_t(2), size_t(3)}) {
            PhaseTensor expanded;
            expanded.n = dimension;
            expanded.linear = {1, 3};
            expanded.upper = {Bits(2), Bits()};
            if(dimension == 3) {
                expanded.linear.push_back(2);
                expanded.upper.push_back(Bits());
                expanded.upper[0] ^= Bits(4);
            }
            expanded.constant = 7;
            expanded.power = -4;
            expanded.boundary = {{Bits(3), 1}, {Bits::unit(dimension - 1), 0}};
            check_transported_phase(expanded, clifford);
            expanded.terms = {{Bits(3), 2}, {Bits::unit(dimension - 1), 6}};
            check_transported_phase(expanded, clifford);
            CliffordResourceUpdate update;
            auto transported = clifford.transport(expanded, &update);
            require(transported && update.linear == transported->quotient.linear &&
                            update.upper == transported->quotient.upper && update.constant == 4,
                    "dimension-changing Clifford transport exposes the actual quadratic update");
            expanded.terms.push_back({Bits(1), 1});
            require(!clifford.transport(expanded), "Clifford scalar entries cannot cover a new odd phase coordinate");
        }
        auto changed_dimension = reference;
        changed_dimension.n = 3;
        changed_dimension.linear.push_back(0);
        changed_dimension.upper.push_back(Bits());
        require(!certificate.transport(changed_dimension),
                "non-Clifford reference entries are not reused across different dimensions");
    }
    // Every one-variable Clifford phase, including hidden cancellation and i phases.
    Workspace ws;
    {
        std::mt19937_64 random(2026100503);
        for(size_t n = 0; n <= 7; n++)
            for(size_t trial = 0; trial < 40; trial++) {
                Ket input;
                input.width = 3;
                input.base = Bits(random() % 8);
                input.phase = random() % 8;
                input.power = -int64_t(n);
                for(size_t j = 0; j < n; j++) {
                    input.columns.push_back(Bits(random() % 8));
                    input.linear.push_back(random() % 4);
                    Bits upper;
                    for(size_t k = j + 1; k < n; k++)
                        if(random() & 1) upper.set(k);
                    input.upper.push_back(upper);
                }
                auto expected = dense(input);
                auto copied = contract(input, ws);
                auto moved = contract(std::move(input), ws);
                auto actual = dense(moved), snapshot = dense(copied);
                for(size_t j = 0; j < expected.size(); j++)
                    require(std::abs(actual[j] - expected[j]) < 1e-10 && std::abs(snapshot[j] - expected[j]) < 1e-10,
                            "consuming contraction matches finite sums across reused workspace");
            }
    }
    for(int phase = 0; phase < 8; phase++)
        for(int linear = 0; linear < 4; linear++)
            for(int visible = 0; visible < 2; visible++) {
                Ket k;
                k.width = 1;
                k.columns = {Bits(visible)};
                k.linear = {linear};
                k.upper = {Bits()};
                k.phase = phase;
                k.power = -1;
                auto expected = dense(k), actual = dense(contract(k, ws));
                for(size_t j = 0; j < actual.size(); j++)
                    require(std::abs(actual[j] - expected[j]) < 1e-12, "finite-sum contraction");
            }
    Ket plus;
    plus.width = 1;
    plus.columns = {Bits(1)};
    plus.linear = {0};
    plus.upper = {Bits()};
    plus.power = -1;
    RecordTracePlan records(plus, plus, {Bits(1)}, {}, {});
    auto marginal = records.evaluate(plus, plus, ws);
    auto values = dense(marginal);
    require(std::abs(values[0] - .5) < 1e-12 && std::abs(values[1] - .5) < 1e-12, "completed-pair marginal");
    std::vector<std::pair<Scalar, Ket>> terms{{Scalar(1), marginal}};
    auto words = enumerate_boundary(terms, 1, 2);
    require(words.size() == 2 && words[0].empty() && words[1] == Bits(1), "exact branch order");
    bool exhausted = false;
    try {
        enumerate_boundary(terms, 1, 1);
    } catch(const ResourceLimitError& e) { exhausted = e.option == "max_branches" && e.limit == 1 && e.needed; }
    require(exhausted, "branch limit before partial result");
    size_t ones = 0;
    for(uint64_t seed = 0; seed < 1000; seed++) ones += sample_boundary(terms, 1, seed).bit(0);
    require(ones > 400 && ones < 600, "native Born distribution");
    FileNoisePlan noise;
    noise.categories.push_back(std::make_shared<const FileNoiseCategory>(
            std::vector<FileFraction>{{Integer(1), Integer(4)}, {Integer(1), Integer(4)}}));
    noise.has_actions = true;
    noise.offsets = {0};
    for(uint64_t seed = 0; seed < 100; seed++) {
        auto full = draw_noise(noise, seed, false), sparse = draw_noise(noise, seed, true);
        require((full.selected[0] < 0 && sparse.selected.empty()) ||
                        (full.selected[0] >= 0 && sparse.selected == full.selected),
                "sparse/full seeded stream");
    }
    auto patterns = enumerate_noise({{Integer(3), Integer(1)}, {Integer(3), Integer(1)}}, 1, 3);
    require(patterns.size() == 3 && patterns[0].numerator == Integer(9), "exact weight sectors");
    std::vector<WeightedExactRow> contributions;
    for(const auto& row : patterns)
        contributions.push_back({row.numerator, {Scalar(1), Scalar(1), Scalar(row.weight != 0)}});
    auto aggregate = aggregate_exact(contributions, Integer(16), {false, false, false});
    require(aggregate.covered == Integer(15), "exact omitted mass");
    require(std::abs(scalar_value(aggregate.sums[2]).real() - 6) < 1e-12, "exact weighted error");
    FileStatistics stats;
    stats.add({1, true, 0, {1}});
    stats.add({1, true, 1, {-1}});
    require(stats.ratio() == .5L && stats.ratio_error() == .5L, "paired ratio covariance");
    auto weights = summarize_weights({{.5, 20000}, {.5, 20000}});
    require(weights.ess == 2 && weights.mantissa == .5 && weights.exponent == 20000, "range-safe weight summary");
    // Arbitrary width arithmetic stays exact even when output floats cannot represent it.
    auto huge = file_integer_scalar(Integer(1).shift(10000));
    require(scaled_real_ratio(huge, huge) == std::make_pair(.5, int64_t(1)), "wide exact scalar");
    auto solution = solve_low_pivots({Bits::unit(0) ^ Bits::unit(193), Bits::unit(1)}, {1, 0}, 194);
    require(solution && solution->particular == Bits(1) && solution->kernel.size() == 192, "wide low-pivot witness");
    size_t calls = 0;
    require(exact_bernoulli(Scalar(1, -2048), Scalar(1),
                    [&]() {
                        calls++;
                        return 0;
                    }),
            "tiny exact probability");
    require(calls == 2048, "caller-owned RNG consumption");
    // Signed pieces on the same support must be grouped before rejecting a mixture.
    ProbabilityMarginal grouped{{{Scalar(-1), plus}, {Scalar(2), plus}}, 1};
    auto mixture = grouped.mixture();
    require(mixture && mixture->components.size() == 1, "signed affine grouping");
    Ket zero;
    zero.width = 1;
    StabilizerTerms expansion{{Scalar(1), plus}, {omega(2, 0), zero}};
    auto expected_plus = dense(plus), expected_zero = dense(zero);
    double expected_norm = 0;
    for(size_t j = 0; j < 2; j++)
        expected_norm += std::norm(expected_plus[j] + std::complex<double>(0, 1) * expected_zero[j]);
    require(std::abs(scalar_value(phase_norm(expansion, ws)).real() - expected_norm) < 1e-12,
            "coherent expansion norm");
    auto target = contract_target(plus, zero, 1, ws);
    require(std::abs(dense(target)[0] - expected_plus[0]) < 1e-12, "target amplitude contraction");
    {
        Ket state;
        state.width = 1;
        state.columns = {Bits(1)};
        state.linear = {0};
        state.upper = {Bits()};
        state.power = -1;
        InstrumentLayout layout({-2}, {-2}, 1, -1, 1, {{"m", "MZ", "r"}, {"c", "CONTROL", {}}, {"n", "MX", "s"}},
                {{"c", {{"r"}, "v1"}}});
        BranchSchedule schedule(layout, {{"c", std::nullopt}});
        std::vector<ProgramOperation> operations = {{"m", branch_gate("MZ", {-2}).operation.value, "r"},
                {"c", branch_gate("CONTROL").operation.value, {}}, {"n", branch_gate("MX", {-2}).operation.value, "s"}};
        FullRecordExecution execution(state, {-2}, {-2}, operations, {0, 0, 0}, schedule, {{"c", std::nullopt}}, {}, {},
                2048, 1024 * 1024, false, true);
        require(!execution.global, "undeclared callback chooses native literal compiler");
        FullRecordBatch batch(execution, true);
        FullRecordInput row;
        row.values = layout.validate({{"r", 1}, {"s", 0}}, {}, {});
        size_t callbacks = 0;
        batch.add(row, [&](size_t, const auto& parents) {
            callbacks++;
            require(parents[0].second == 1, "reported-only callback input");
            return std::vector<BranchGate>{branch_gate("H", {-2})};
        });
        auto results = batch.finish();
        require(callbacks == 1 && results.size() == 1 && results[0].probability.zero(),
                "literal callback executes once and preserves signed wires");
        require(execution.literal_compilations == 1 && results[0].literal_compiler.has_value(),
                "native literal certificate ownership");
    }
    {
        // Two affine pieces with exponentially many words: work uses columns,
        // not words or coefficient cancellation. Their hull has 81 free bits.
        Ket piece;
        piece.width = 82;
        for(size_t j = 0; j < 80; j++) piece.columns.push_back(Bits::unit(j));
        std::vector<Bits> rows;
        for(size_t j = 0; j < 82; j++) rows.push_back(Bits::unit(j));
        AffineProposalEnvelope envelope(82);
        envelope.include(project_support(piece, rows, {}, {}));
        piece.base = Bits::unit(80) ^ Bits::unit(81);
        envelope.include(project_support(piece, rows, {}, {}));
        require(!envelope.support.zero && envelope.support.columns.size() == 81, "large affine proposal hull");
        Factor image(transpose(envelope.support.columns, 82), 81);
        require(image.consistent(Bits::unit(80) ^ Bits::unit(81)), "affine base difference retained");
        require(!image.consistent(Bits::unit(80)), "affine correlation excludes zero");
        auto impossible = project_support(envelope.support, rows, {Bits::unit(80), Bits::unit(81)}, {0, 1});
        require(impossible.zero, "affine empty prefix detected");
    }
    {
        // Explicit finite affine sets independently check conditioning and
        // projection, including dependent columns and inconsistent events.
        std::mt19937_64 random(202609301);
        for(size_t n = 0; n <= 7; n++)
            for(size_t trial = 0; trial < 40; trial++) {
                Ket support;
                support.width = 1 + random() % 7;
                support.base = Bits(random() & ((1ULL << support.width) - 1));
                for(size_t j = 0; j < n; j++) support.columns.push_back(Bits(random() & ((1ULL << support.width) - 1)));
                std::vector<Bits> keep, events;
                std::vector<int> values;
                size_t width = 1 + random() % 5;
                for(size_t j = 0; j < width; j++) keep.push_back(Bits(random() & ((1ULL << support.width) - 1)));
                size_t count = random() % 5;
                for(size_t j = 0; j < count; j++) {
                    events.push_back(Bits(random() & ((1ULL << support.width) - 1)));
                    values.push_back(random() & 1);
                }
                std::set<size_t> expected, actual;
                for(size_t u = 0; u < (1ULL << n); u++) {
                    auto point = support.base ^ xors(support.columns, Bits(u));
                    bool allowed = true;
                    for(size_t j = 0; j < count; j++) allowed &= events[j].parity_and(point) == values[j];
                    if(!allowed) continue;
                    size_t word = 0;
                    for(size_t j = 0; j < width; j++) word |= size_t(keep[j].parity_and(point)) << j;
                    expected.insert(word);
                }
                auto projected = project_support(support, keep, events, values);
                if(!projected.zero)
                    for(size_t u = 0; u < (1ULL << projected.columns.size()); u++) {
                        auto word = projected.base ^ xors(projected.columns, Bits(u));
                        actual.insert(word.empty() ? 0 : word.w[0]);
                    }
                require(actual == expected, "affine projection matches independent finite-set image");
            }
    }
    // Independent complex finite sums validate every predicted nonzero word.
    // Includes odd Z4 coefficients, noninjective maps and nonzero offsets.
    {
        std::mt19937_64 random(20260930);
        for(size_t n = 0; n <= 7; n++)
            for(size_t trial = 0; trial < 80; trial++) {
                PhaseTensor tensor;
                tensor.n = n;
                PolynomialQuotient phase;
                size_t width = 1 + random() % 4;
                for(size_t j = 0; j < n; j++) {
                    phase.linear.push_back(random() % 4);
                    Bits row;
                    for(size_t k = j + 1; k < n; k++)
                        if(random() & 1) row.set(k);
                    phase.upper.push_back(row);
                }
                for(size_t j = 0; j < width; j++)
                    tensor.boundary.push_back({Bits(random() & ((1ULL << n) - 1)), int(random() & 1)});
                auto rows = NetworkMaps::masks(tensor.boundary);
                auto columns = transpose(rows, n);
                auto change = column_change(columns);
                auto original = columns;
                for(size_t j = 0; j < n; j++) columns[j] = xors(original, change[j]);
                auto pulled = restrict_q(phase.linear, phase.upper, {}, change);
                CliffordContraction prepared(columns, pulled);
                Bits base;
                for(size_t j = 0; j < width; j++)
                    if(tensor.boundary[j].second) base.set(j);
                auto support = prepared.support(width, base, pulled.linear);
                std::vector<std::complex<double>> sums(1ULL << width);
                for(size_t x = 0; x < (1ULL << n); x++) {
                    Bits word;
                    for(size_t j = 0; j < width; j++)
                        if(tensor.boundary[j].first.parity_and(Bits(x)) ^ tensor.boundary[j].second) word.set(j);
                    sums[word.w.empty() ? 0 : word.w[0]] +=
                            std::polar(1., std::acos(-1.) / 2 * qvalue(Bits(x), phase.linear, phase.upper));
                }
                Factor image(transpose(support.columns, width), support.columns.size());
                for(size_t word = 0; word < sums.size(); word++)
                    require((!support.zero && image.consistent(Bits(word) ^ support.base)) ==
                                    (std::abs(sums[word]) > 1e-8),
                            "quadratic radical nonzero support");
            }
    }
    std::cout << "native foundation checks passed\n";
}
