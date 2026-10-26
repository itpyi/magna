// Normalized exact rational input arithmetic. Likelihood products may stay unreduced.
#pragma once
#include "scalar.hpp"
#include "numeric.hpp"
#include <regex>
#include <string>
namespace magic_native {
inline std::pair<Integer, Integer> integer_divmod(Integer a, Integer b) {
    if(a.neg || b.neg || b.zero()) throw std::invalid_argument("positive integer division required");
    Integer q, r;
    for(size_t j = integer_bits(a); j-- > 0;) {
        r = r.shift(1);
        if((a.d[j / 32] >> (j % 32)) & 1) r = r + Integer(1);
        if(Integer::cmp(r, b) >= 0) {
            r = r - b;
            if(q.d.size() <= j / 32) q.d.resize(j / 32 + 1);
            q.d[j / 32] |= uint32_t(1) << (j % 32);
        }
    }
    q.trim();
    return {q, r};
}
inline Integer integer_gcd(Integer a, Integer b) {
    a.neg = b.neg = false;
    while(!b.zero()) {
        auto r = integer_divmod(a, b).second;
        a = std::move(b);
        b = std::move(r);
    }
    return a;
}
inline Integer integer_decimal(std::string text) {
    bool negative = false;
    size_t start = 0;
    if(!text.empty() && (text[0] == '+' || text[0] == '-')) {
        negative = text[0] == '-';
        start = 1;
    }
    if(start == text.size()) throw std::invalid_argument("integer required");
    Integer value;
    for(size_t j = start; j < text.size(); j++) {
        if(text[j] < '0' || text[j] > '9') throw std::invalid_argument("integer required");
        value = value * Integer(10) + Integer(text[j] - '0');
    }
    value.neg = negative && !value.zero();
    return value;
}
struct Rational {
    Integer n, d;
    Rational(Integer numerator = Integer(), Integer denominator = Integer(1))
        : n(std::move(numerator)), d(std::move(denominator)) {
        if(d.zero()) throw std::invalid_argument("zero rational denominator");
        if(d.neg) {
            n = -n;
            d = -d;
        }
        auto common = integer_gcd(n, d);
        bool sign = n.neg;
        n.neg = false;
        n = integer_divmod(n, common).first;
        n.neg = sign && !n.zero();
        d = integer_divmod(d, common).first;
    }
    friend Rational operator+(const Rational& a, const Rational& b) { return {a.n * b.d + b.n * a.d, a.d * b.d}; }
    friend Rational operator-(const Rational& a, const Rational& b) { return {a.n * b.d - b.n * a.d, a.d * b.d}; }
    friend Rational operator*(const Rational& a, const Rational& b) { return {a.n * b.n, a.d * b.d}; }
    friend Rational operator/(const Rational& a, const Rational& b) { return {a.n * b.d, a.d * b.n}; }
    friend bool operator==(const Rational& a, const Rational& b) { return a.n == b.n && a.d == b.d; }
    friend bool operator!=(const Rational& a, const Rational& b) { return !(a == b); }
    friend bool operator<(const Rational& a, const Rational& b) {
        auto diff = a.n * b.d - b.n * a.d;
        return diff.neg;
    }
    friend bool operator>(const Rational& a, const Rational& b) { return b < a; }
    friend bool operator<=(const Rational& a, const Rational& b) { return !(b < a); }
};
inline std::string trim(std::string text) {
    auto first = text.find_first_not_of(" \t\r\n\f\v");
    if(first == std::string::npos) return {};
    return text.substr(first, text.find_last_not_of(" \t\r\n\f\v") - first + 1);
}
inline Rational parse_rational(std::string text) {
    text = trim(text);
    std::smatch m;
    static const std::regex fraction(R"(^([+-]?[0-9]+(?:_[0-9]+)*)/([+-]?[0-9]+(?:_[0-9]+)*)$)");
    auto digits = [](std::string s) {
        s.erase(std::remove(s.begin(), s.end(), '_'), s.end());
        return s;
    };
    if(std::regex_match(text, m, fraction)) return {integer_decimal(digits(m[1])), integer_decimal(digits(m[2]))};
    static const std::regex decimal(
            R"(^([+-]?)([0-9]+(?:_[0-9]+)*)?(?:\.([0-9]*(?:_[0-9]+)*))?(?:[eE]([+-]?[0-9]+(?:_[0-9]+)*))?$)");
    if(!std::regex_match(text, m, decimal) || (!m[2].matched && !m[3].matched) ||
            (digits(m[2]).empty() && digits(m[3]).empty()))
        throw std::invalid_argument("finite rational required");
    auto whole = digits(m[2]), part = digits(m[3]);
    int64_t exponent = m[4].matched ? std::stoll(digits(m[4])) : 0;
    if(exponent < -1000000 || exponent > 1000000) throw std::length_error("rational exponent budget");
    exponent -= int64_t(part.size());
    auto n = integer_decimal(whole + part);
    if(m[1] == "-") n = -n;
    Integer d(1);
    for(int64_t j = 0; j < std::abs(exponent); j++)
        if(exponent < 0) d = d * Integer(10);
        else n = n * Integer(10);
    return {n, d};
}
} // namespace magic_native
