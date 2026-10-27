// Existing proposal estimator with precision-80 decimal operations and paired errors.
#pragma once
#include "../foundation/decimal.hpp"
#include <map>
#include <optional>
namespace magic_native {
struct ProposalEstimate {
    using D = DecimalValue;
    size_t n = 0, zeros = 0, failures = 0;
    D sw, sw2;
    std::array<D, 5> sums;
    void failure() {
        n++;
        failures++;
    }
    void add(const D& w, const D& a, const D& d) {
        Rational zero, one(Integer(1));
        auto ar = a.rational(), dr = d.rational();
        if(ar < zero || one < ar || dr < zero || one < dr) throw std::invalid_argument("scores must lie in [0,1]");
        n++;
        zeros += w.zero();
        sw = D::add(sw, w);
        sw2 = D::add(sw2, D::mul(w, w));
        auto x = D::mul(w, a), y = D::mul(x, d);
        std::array<D, 5> values{x, y, D::mul(x, x), D::mul(y, y), D::mul(x, y)};
        for(size_t j = 0; j < 5; j++) sums[j] = D::add(sums[j], values[j]);
    }
    std::map<std::string, std::optional<D>> result() const {
        if(!n) throw std::invalid_argument("at least one proposal required");
        if(failures) return {};
        auto [sx, sy, xx, yy, xy] = sums;
        D count{Integer(int64_t(n))}, denominator{Integer(int64_t(n)) * Integer(int64_t(n - 1))};
        auto positive = [](D x) { return x.coefficient.neg ? D() : x; };
        auto se = [&](const D& sum, const D& square) -> std::optional<D> {
            if(n < 2) return {};
            return D::sqrt(positive(D::div(D::sub(square, D::div(D::mul(sum, sum), count)), denominator)));
        };
        D mean = D::div(sx, count);
        std::optional<D> ratio, error;
        if(!sx.zero()) {
            ratio = D::div(sy, sx);
            if(n > 1) {
                auto variance = D::add(
                        D::sub(yy, D::mul(D::mul(D(Integer(2)), *ratio), xy)), D::mul(D::mul(*ratio, *ratio), xx));
                error = D::div(D::sqrt(D::div(positive(variance), denominator)), mean);
            }
        }
        return {{"acceptance", mean}, {"acceptance_standard_error", se(sx, xx)},
                {"joint_diagnostic", D::div(sy, count)}, {"joint_standard_error", se(sy, yy)},
                {"conditional_diagnostic", ratio}, {"conditional_standard_error", error},
                {"mean_weight", D::div(sw, count)},
                {"effective_sample_size", sw2.zero() ? D() : D::div(D::mul(sw, sw), sw2)}};
    }
};
inline double sample_weight_log2(const Scalar& probability, const Rational& weight) {
    if(weight.n.zero()) return -std::numeric_limits<double>::infinity();
    Scalar numerator;
    numerator.c[0] = weight.n;
    Scalar denominator;
    denominator.c[0] = weight.d;
    return scalar_log2(probability) + scalar_log2(numerator) - scalar_log2(denominator);
}
} // namespace magic_native
