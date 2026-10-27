// Exact arbitrary-width integers and scalars in Z[omega, 1/2], omega = exp(i pi/4); no silent overflow.
#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace magic_native {
using std::size_t;
inline int mod4(int x) { return x & 3; }
inline int mod8(int x) { return x & 7; }

// Signed magnitude, base 2^32. No narrowing fast path or silent overflow.
struct Integer {
    bool neg = false;
    std::vector<uint32_t> d;
    Integer(int64_t x = 0) {
        neg = x < 0;
        uint64_t a = neg ? uint64_t(-(x + 1)) + 1 : uint64_t(x);
        if(a) {
            d.push_back(uint32_t(a));
            if(a >> 32) d.push_back(uint32_t(a >> 32));
        }
    }
    void trim() {
        while(!d.empty() && !d.back()) d.pop_back();
        if(d.empty()) neg = false;
    }
    bool zero() const { return d.empty(); }
    static int cmp(const Integer& a, const Integer& b) {
        if(a.d.size() != b.d.size()) return a.d.size() < b.d.size() ? -1 : 1;
        for(size_t i = a.d.size(); i-- > 0;)
            if(a.d[i] != b.d[i]) return a.d[i] < b.d[i] ? -1 : 1;
        return 0;
    }
    Integer operator-() const {
        Integer a = *this;
        if(!a.zero()) a.neg = !a.neg;
        return a;
    }
    friend bool operator==(const Integer& a, const Integer& b) { return a.neg == b.neg && a.d == b.d; }
    friend Integer operator+(const Integer& a, const Integer& b) {
        Integer r;
        if(a.neg == b.neg) {
            r.neg = a.neg;
            r.d.resize(std::max(a.d.size(), b.d.size()));
            uint64_t c = 0;
            for(size_t i = 0; i < r.d.size(); i++) {
                c += (i < a.d.size() ? a.d[i] : 0ULL) + (i < b.d.size() ? b.d[i] : 0ULL);
                r.d[i] = uint32_t(c);
                c >>= 32;
            }
            if(c) r.d.push_back(uint32_t(c));
        } else {
            const Integer *hi = &a, *lo = &b;
            if(cmp(a, b) < 0) std::swap(hi, lo);
            r.neg = hi->neg;
            r.d.resize(hi->d.size());
            uint64_t borrow = 0;
            for(size_t i = 0; i < r.d.size(); i++) {
                uint64_t h = hi->d[i], l = (i < lo->d.size() ? lo->d[i] : 0ULL) + borrow;
                r.d[i] = uint32_t(h - l);
                borrow = h < l;
            }
        }
        r.trim();
        return r;
    }
    friend Integer operator-(const Integer& a, const Integer& b) { return a + (-b); }
    friend Integer operator*(const Integer& a, const Integer& b) {
        Integer r;
        if(a.zero() || b.zero()) return r;
        r.neg = a.neg != b.neg;
        r.d.assign(a.d.size() + b.d.size(), 0);
        for(size_t i = 0; i < a.d.size(); i++) {
            uint64_t c = 0;
            for(size_t j = 0; j < b.d.size(); j++) {
                uint64_t t = uint64_t(a.d[i]) * b.d[j] + r.d[i + j] + c;
                r.d[i + j] = uint32_t(t);
                c = t >> 32;
            }
            r.d[i + b.d.size()] = uint32_t(c);
        }
        r.trim();
        return r;
    }
    Integer shift(size_t n) const {
        Integer r;
        r.neg = neg;
        if(zero()) return r;
        r.d.assign(n / 32, 0);
        uint64_t carry = 0;
        for(auto x : d) {
            uint64_t v = (uint64_t(x) << (n % 32)) | carry;
            r.d.push_back(uint32_t(v));
            carry = v >> 32;
        }
        if(carry) r.d.push_back(uint32_t(carry));
        return r;
    }
    bool even() const { return zero() || !(d[0] & 1); }
    void half() {
        uint32_t c = 0;
        for(size_t i = d.size(); i-- > 0;) {
            auto next = d[i] & 1;
            d[i] = (d[i] >> 1) | (c << 31);
            c = next;
        }
        trim();
    }
};
struct Scalar {
    std::array<Integer, 4> c{};
    int64_t power = 0;
    Scalar() = default;
    Scalar(int64_t v, int64_t p = 0) {
        c[0] = Integer(v);
        power = p;
        normalize();
    }
    bool zero() const {
        for(auto& v : c)
            if(!v.zero()) return false;
        return true;
    }
    void normalize() {
        size_t shift = std::numeric_limits<size_t>::max();
        for(const auto& v : c)
            if(!v.zero()) {
                size_t j = 0;
                while(!v.d[j]) j++;
                shift = std::min(shift, 32 * j + size_t(__builtin_ctz(v.d[j])));
            }
        if(shift == std::numeric_limits<size_t>::max()) {
            power = 0;
            return;
        }
        if(!shift) return;
        for(auto& v : c)
            if(!v.zero()) {
                size_t words = shift / 32, bits = shift % 32, count = v.d.size() - words;
                for(size_t j = 0; j < count; j++) {
                    uint32_t word = v.d[j + words] >> bits;
                    if(bits && j + words + 1 < v.d.size()) word |= v.d[j + words + 1] << (32 - bits);
                    v.d[j] = word;
                }
                v.d.resize(count);
                v.trim();
            }
        power += int64_t(shift);
    }
    Scalar conjugate() const {
        Scalar r;
        r.c = {c[0], -c[3], -c[2], -c[1]};
        r.power = power;
        return r;
    }
    friend Scalar operator+(const Scalar& a, const Scalar& b) {
        Scalar r;
        r.power = std::min(a.power, b.power);
        for(int j = 0; j < 4; j++) r.c[j] = a.c[j].shift(a.power - r.power) + b.c[j].shift(b.power - r.power);
        r.normalize();
        return r;
    }
    friend Scalar operator*(const Scalar& a, const Scalar& b) {
        Scalar r;
        r.power = a.power + b.power;
        for(int i = 0; i < 4; i++)
            for(int j = 0; j < 4; j++) {
                Integer t = a.c[i] * b.c[j];
                r.c[(i + j) % 4] = r.c[(i + j) % 4] + (i + j >= 4 ? -t : t);
            }
        r.normalize();
        return r;
    }
};
inline Scalar omega(int phase, int64_t power_sqrt2) {
    Scalar r;
    phase = mod8(phase);
    r.c[phase % 4] = Integer(phase < 4 ? 1 : -1);
    r.power = (power_sqrt2 - (power_sqrt2 & 1)) / 2;
    if(power_sqrt2 & 1) {
        Scalar s;
        s.c[1] = Integer(1);
        s.c[3] = Integer(-1);
        r = r * s;
    }
    return r;
}
inline int real_sign(const Scalar& v) {
    if(!v.c[2].zero() || !(v.c[1] == -v.c[3])) throw std::runtime_error("probability is not exactly real");
    auto& a = v.c[0];
    auto& b = v.c[1];
    if(a.zero()) return b.zero() ? 0 : (b.neg ? -1 : 1);
    if(b.zero()) return a.neg ? -1 : 1;
    if(a.neg == b.neg) return a.neg ? -1 : 1;
    int cmp = Integer::cmp(a * a, (b * b).shift(1));
    return cmp == 0 ? 0 : (cmp > 0 ? (a.neg ? -1 : 1) : (b.neg ? -1 : 1));
}
} // namespace magic_native
