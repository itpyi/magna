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
// Centered covariance factors preserve residual errors before final conversion.
struct ScientificMoments {
    size_t count = 0;
    std::array<double, 2> mean{};
    std::array<double, 3> root{};
    std::array<int64_t, 2> scales{};
    static double rescale(double value, int64_t shift) {
        return value == 0 || shift < std::numeric_limits<int>::min() ? 0 : std::ldexp(value, int(shift));
    }
    void add_row(double x, double y) {
        if(x == 0) { root[2] = std::hypot(root[2], y); return; }
        double radius = std::hypot(root[0], x), c = root[0] / radius, s = x / radius;
        double xy = s * root[1], yy = s * y;
        double residual = std::fma(c, y, -xy) - std::fma(s, root[1], -xy);
        root[1] = std::fma(c, root[1], yy) + std::fma(s, y, -yy);
        root[0] = radius;
        root[2] = std::hypot(root[2], residual);
    }
    void merge(ScientificMoments other) {
        if(!other.count) return;
        if(!count) { *this = other; return; }
        for(size_t j = 0; j < 2; j++) {
            auto scale = mean[j] == 0 ? other.scales[j] : other.mean[j] == 0 ? scales[j]
                                                                 : std::max(scales[j], other.scales[j]);
            auto left = scales[j] - scale, right = other.scales[j] - scale;
            mean[j] = rescale(mean[j], left); other.mean[j] = rescale(other.mean[j], right);
            for(size_t k = j ? 1 : 0; k <= (j ? 2 : 0); k++) {
                root[k] = rescale(root[k], left); other.root[k] = rescale(other.root[k], right);
            }
            scales[j] = scale;
        }
        auto total = count + other.count;
        double da = other.mean[0] - mean[0], de = other.mean[1] - mean[1];
        double weight = std::sqrt(double(count) * double(other.count) / double(total));
        add_row(other.root[0], other.root[1]); add_row(0, other.root[2]); add_row(weight * da, weight * de);
        mean[0] += da * (double(other.count) / double(total));
        mean[1] += de * (double(other.count) / double(total));
        count = total;
    }
    ScientificSummary summary(bool has_error) const {
        ScientificSummary result;
        result.count = count;
        auto append = [](std::vector<RangedValue>& out, bool exists, double value, int64_t scale) {
            out.push_back(ranged_value(exists ? std::optional<ScientificValue>{{value, scale}} : std::nullopt));
        };
        bool ratio_exists = count && has_error && mean[0] != 0;
        double ratio = ratio_exists ? mean[1] / mean[0] : 0;
        append(result.estimates, count, mean[0], scales[0]);
        append(result.estimates, count && has_error, mean[1], scales[1]);
        append(result.estimates, ratio_exists, ratio, scales[1] - scales[0]);
        double divisor = count > 1 ? std::sqrt(double(count) * double(count - 1)) : 1;
        append(result.standard_errors, count > 1, root[0] / divisor, scales[0]);
        append(result.standard_errors, count > 1 && has_error, std::hypot(root[1], root[2]) / divisor, scales[1]);
        double residual = ratio_exists ? std::hypot(std::fma(-ratio, root[0], root[1]), root[2]) / divisor / std::abs(mean[0]) : 0;
        append(result.standard_errors, count > 1 && ratio_exists, residual, scales[1] - scales[0]);
        return result;
    }
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
    static long double rescale(long double value, int64_t shift) {
        return shift < std::numeric_limits<int>::min() ? 0 : std::ldexp(value, int(shift));
    }
    void add(const std::vector<std::optional<ScientificValue>>& row) {
        if(count == 0) {
            if(row.size() < 2) throw std::invalid_argument("acceptance and error columns required");
        } else if(row.size() != top.size()) throw std::invalid_argument("inconsistent scientific columns");
        if(!row[0]) throw std::invalid_argument("acceptance contribution required");
        for(size_t j = 2; j < row.size(); j++)
            if(!row[j]) throw std::invalid_argument("moment contribution required");
        auto next_top = count ? top : std::vector<std::optional<int64_t>>(row.size());
        std::vector<int64_t> next_scales(row.size(), 0);
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
        FileContribution value{convert(0), bool(row[1]), convert(1), {}};
        for(size_t j = 2; j < row.size(); j++) value.moments.push_back(convert(j));
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
        top = std::move(next_top);
        scales = std::move(next_scales);
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
