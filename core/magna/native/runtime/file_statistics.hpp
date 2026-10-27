// Conditional contributions and range-safe summaries of completed histories.
#pragma once
#include "sampling.hpp"
#include "noise.hpp"
#include <optional>
namespace magic_native {
using ScientificValue = std::pair<double, int64_t>;
struct RangedValue {
    std::optional<ScientificValue> scientific;
    std::optional<double> ordinary;
    bool underflow = false, overflow = false;
};
inline RangedValue ranged_value(std::optional<ScientificValue> value) {
    RangedValue out;
    out.scientific = value;
    if(!value) return out;
    auto [mantissa, exponent] = *value;
    if(!std::isfinite(mantissa)) throw std::invalid_argument("finite scientific mantissa required");
    double number = exponent > std::numeric_limits<int>::max()
            ? std::copysign(std::numeric_limits<double>::infinity(), mantissa)
            : exponent < std::numeric_limits<int>::min() ? std::copysign(0., mantissa)
                                                         : std::ldexp(mantissa, int(exponent));
    out.overflow = !std::isfinite(number);
    out.underflow = mantissa != 0 && number == 0;
    if(!out.overflow && !out.underflow) out.ordinary = number;
    return out;
}
struct HistoryContribution {
    std::vector<RangedValue> values;
    ScientificValue weight, noise_weight;
};
inline HistoryContribution history_contribution(const Scalar& mass,
        const std::vector<std::optional<Scalar>>& numerators, const FileFraction& weight,
        std::optional<ScientificValue> prepared_weight = {}) {
    if(weight.n.neg || weight.d.neg || weight.d.zero())
        throw std::invalid_argument("invalid nonnegative importance weight");
    std::optional<Scalar> n, d;
    auto prepare = [&]() {
        if(!n) {
            n = file_integer_scalar(weight.n);
            d = file_integer_scalar(weight.d);
        }
    };
    HistoryContribution result;
    if(prepared_weight) result.weight = *prepared_weight;
    else {
        prepare();
        result.weight = scaled_real_ratio(*n, *d);
    }
    result.noise_weight = result.weight;
    for(const auto& value : numerators) {
        std::optional<ScientificValue> ratio;
        if(value) {
            if(value->zero()) ratio = ScientificValue{0, 0};
            else if(value->power == mass.power && value->c == mass.c) ratio = result.weight;
            else {
                prepare();
                ratio = scaled_real_ratio(*value * (*n), mass * (*d));
            }
        }
        result.values.push_back(ranged_value(ratio));
    }
    return result;
}
struct ScientificSummary {
    // acceptance, error, conditional error, then requested moments.
    std::vector<RangedValue> estimates, standard_errors;
    size_t count = 0;
};
// Streaming range-safe statistics of completed contributions. A column whose
// largest nonzero binary exponent lies outside [-500, 500] is accumulated at that
// exponent; when the largest exponent moves, the running sums are rescaled by the
// exact power of two. Without a scale change this is the same arithmetic, in the
// same order, as accumulating the retained draws at the end.
struct ScientificAccumulator {
    size_t count = 0;
    std::vector<std::optional<int64_t>> top;
    std::vector<int64_t> scales;
    FileStatistics stats;
    // Reuse preparation storage without changing committed statistics until the
    // entire next row has been checked.
    std::vector<std::optional<int64_t>> next_top;
    std::vector<int64_t> next_scales;
    FileContribution next_value;
    static long double rescale(long double value, int64_t shift) {
        return shift == 0 ? value : shift < std::numeric_limits<int>::min() ? 0 : std::ldexp(value, int(shift));
    }
    void add(const std::vector<std::optional<ScientificValue>>& row) {
        if(count == 0) {
            if(row.size() < 2) throw std::invalid_argument("acceptance and error columns required");
        } else if(row.size() != top.size()) throw std::invalid_argument("inconsistent scientific columns");
        if(!row[0]) throw std::invalid_argument("acceptance contribution required");
        for(size_t j = 2; j < row.size(); j++)
            if(!row[j]) throw std::invalid_argument("moment contribution required");
        if(count) next_top = top;
        else next_top.assign(row.size(), std::nullopt);
        next_scales.assign(row.size(), 0);
        for(size_t j = 0; j < row.size(); j++) {
            if(row[j] && row[j]->first)
                next_top[j] = next_top[j] ? std::max(*next_top[j], row[j]->second) : row[j]->second;
            if(next_top[j] && (*next_top[j] > 500 || *next_top[j] < -500)) next_scales[j] = *next_top[j];
        }
        auto convert = [&](size_t j) {
            if(!row[j]) return 0.;
            auto delta = row[j]->second - next_scales[j];
            return delta < std::numeric_limits<int>::min() ? 0. : std::ldexp(row[j]->first, int(delta));
        };
        auto& value = next_value;
        value.acceptance = convert(0);
        value.has_error = bool(row[1]);
        value.error = convert(1);
        value.moments.resize(row.size() - 2);
        for(size_t j = 2; j < row.size(); j++) value.moments[j - 2] = convert(j);
        // Reject a bad row before rescaling, so a failed draw leaves the partial statistics intact.
        bool finite = std::isfinite(value.acceptance) && std::isfinite(value.error);
        for(auto moment : value.moments) finite = finite && std::isfinite(moment);
        if(!finite) throw std::invalid_argument("finite draw value required");
        if(count && value.has_error != stats.has_error)
            throw std::invalid_argument("inconsistent error metric presence");
        if(count) {
            auto shift = [&](size_t j) { return scales[j] - next_scales[j]; };
            auto& a = stats.acceptance;
            a.mean = rescale(a.mean, shift(0));
            a.m2 = rescale(a.m2, 2 * shift(0));
            stats.error.mean = rescale(stats.error.mean, shift(1));
            stats.error.m2 = rescale(stats.error.m2, 2 * shift(1));
            stats.cross = rescale(stats.cross, shift(0) + shift(1));
            for(size_t k = 0; k < stats.moments.size(); k++) {
                stats.moments[k].mean = rescale(stats.moments[k].mean, shift(k + 2));
                stats.moments[k].m2 = rescale(stats.moments[k].m2, 2 * shift(k + 2));
            }
        }
        stats.add(value);
        top.swap(next_top);
        scales.swap(next_scales);
        count++;
    }
    ScientificSummary summary() const {
        ScientificSummary result;
        result.count = count;
        if(!count) return result;
        auto append = [&](std::vector<RangedValue>& values, bool exists, long double value, int64_t scale) {
            if(!exists) {
                values.push_back({});
                return;
            }
            double number = double(value);
            if(!std::isfinite(number)) throw std::invalid_argument("summary exceeds finite floating-point range");
            values.push_back(ranged_value(ScientificValue{number, scale}));
        };
        bool se = count > 1;
        append(result.estimates, true, stats.acceptance.mean, scales[0]);
        append(result.estimates, stats.has_error, stats.error.mean, scales[1]);
        append(result.estimates, stats.has_ratio(), stats.ratio(), scales[1] - scales[0]);
        append(result.standard_errors, se, se ? stats.acceptance.standard_error() : 0, scales[0]);
        append(result.standard_errors, se && stats.has_error, se ? stats.error.standard_error() : 0, scales[1]);
        append(result.standard_errors, se && stats.has_ratio(), stats.ratio_error(), scales[1] - scales[0]);
        for(size_t j = 0; j < stats.moments.size(); j++) {
            append(result.estimates, true, stats.moments[j].mean, scales[j + 2]);
            append(result.standard_errors, se, se ? stats.moments[j].standard_error() : 0, scales[j + 2]);
        }
        return result;
    }
};
inline ScientificSummary summarize_scientific(const std::vector<std::vector<std::optional<ScientificValue>>>& input) {
    ScientificAccumulator accumulator;
    for(const auto& row : input) accumulator.add(row);
    return accumulator.summary();
}
} // namespace magic_native
