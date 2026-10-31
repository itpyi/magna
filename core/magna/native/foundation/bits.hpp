// Packed arbitrary-width GF(2) vectors and their row operations.
#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace magic_native {
using std::size_t;
struct Bits {
    std::vector<uint64_t> w;
    Bits(uint64_t v = 0) {
        if(v) w.push_back(v);
    }
    void trim() {
        while(!w.empty() && !w.back()) w.pop_back();
    }
    bool empty() const { return w.empty(); }
    bool bit(size_t j) const { return j / 64 < w.size() && ((w[j / 64] >> (j % 64)) & 1); }
    void flip(size_t j) {
        if(j / 64 >= w.size()) w.resize(j / 64 + 1);
        w[j / 64] ^= uint64_t(1) << (j % 64);
        trim();
    }
    void set(size_t j) {
        if(!bit(j)) flip(j);
    }
    size_t size() const { return w.empty() ? 0 : 64 * (w.size() - 1) + 64 - __builtin_clzll(w.back()); }
    size_t count() const {
        size_t n = 0;
        for(auto v : w) n += __builtin_popcountll(v);
        return n;
    }
    int parity() const { return int(count() & 1); }
    int parity_and(const Bits& b) const {
        uint64_t combined = 0;
        for(size_t j = 0; j < std::min(w.size(), b.w.size()); j++) combined ^= w[j] & b.w[j];
        return __builtin_popcountll(combined) & 1;
    }
    std::vector<size_t> indices() const {
        std::vector<size_t> out;
        for(size_t i = 0; i < w.size(); i++) {
            auto v = w[i];
            while(v) {
                out.push_back(i * 64 + __builtin_ctzll(v));
                v &= v - 1;
            }
        }
        return out;
    }
    template <class F> void each(F&& f) const {
        for(size_t i = 0; i < w.size(); i++) {
            auto v = w[i];
            while(v) {
                f(i * 64 + __builtin_ctzll(v));
                v &= v - 1;
            }
        }
    }
    Bits& operator^=(const Bits& b) {
        w.resize(std::max(w.size(), b.w.size()));
        for(size_t i = 0; i < b.w.size(); i++) w[i] ^= b.w[i];
        trim();
        return *this;
    }
    // XOR the part above j without allocating a temporary shifted/masked row.
    void xor_above(const Bits& b, size_t j) {
        size_t first = j / 64;
        if(first >= b.w.size()) return;
        w.resize(std::max(w.size(), b.w.size()));
        if(j % 64 != 63) w[first] ^= b.w[first] & (~uint64_t(0) << (j % 64 + 1));
        for(size_t k = first + 1; k < b.w.size(); k++) w[k] ^= b.w[k];
        trim();
    }
    Bits& operator|=(const Bits& b) {
        w.resize(std::max(w.size(), b.w.size()));
        for(size_t i = 0; i < b.w.size(); i++) w[i] |= b.w[i];
        return *this;
    }
    friend Bits operator^(Bits a, const Bits& b) { return a ^= b; }
    friend Bits operator|(Bits a, const Bits& b) { return a |= b; }
    friend Bits operator&(Bits a, const Bits& b) {
        a.w.resize(std::min(a.w.size(), b.w.size()));
        for(size_t i = 0; i < a.w.size(); i++) a.w[i] &= b.w[i];
        a.trim();
        return a;
    }
    friend bool operator==(const Bits& a, const Bits& b) { return a.w == b.w; }
    friend bool operator!=(const Bits& a, const Bits& b) { return !(a == b); }
    Bits above(size_t j) const {
        Bits a = *this;
        for(size_t i = 0; i < a.w.size() && i < j / 64; i++) a.w[i] = 0;
        if(j / 64 < a.w.size()) a.w[j / 64] &= (j % 64 == 63 ? 0 : ~uint64_t(0) << (j % 64 + 1));
        a.trim();
        return a;
    }
    Bits low(size_t n) const {
        const size_t words = n / 64 + bool(n % 64);
        Bits a;
        a.w.assign(w.begin(), w.begin() + std::min(w.size(), words));
        if(n % 64 && a.w.size() == words) a.w.back() &= (uint64_t(1) << (n % 64)) - 1;
        a.trim();
        return a;
    }
    Bits operator>>(size_t n) const {
        Bits a;
        size_t words = n / 64, shift = n % 64;
        if(words >= w.size()) return a;
        a.w.resize(w.size() - words);
        for(size_t j = 0; j < a.w.size(); j++) {
            a.w[j] = w[j + words] >> shift;
            if(shift && j + words + 1 < w.size()) a.w[j] |= w[j + words + 1] << (64 - shift);
        }
        a.trim();
        return a;
    }
    Bits operator<<(size_t n) const {
        Bits a;
        if(w.empty()) return a;
        size_t words = n / 64, shift = n % 64;
        if(words > a.w.max_size() - w.size() || (shift && words == a.w.max_size() - w.size()))
            throw std::length_error("shifted mask exceeds capacity");
        a.w.assign(words + w.size() + bool(shift), 0);
        for(size_t j = 0; j < w.size(); j++) {
            a.w[j + words] |= w[j] << shift;
            if(shift) a.w[j + words + 1] |= w[j] >> (64 - shift);
        }
        a.trim();
        return a;
    }
    static Bits unit(size_t j) {
        Bits a;
        a.set(j);
        return a;
    }
    static Bits ones(size_t n) {
        Bits a;
        a.w.assign((n + 63) / 64, ~uint64_t(0));
        return a.low(n);
    }
};
struct BitsLess {
    bool operator()(const Bits& a, const Bits& b) const {
        if(a.w.size() != b.w.size()) return a.w.size() < b.w.size();
        for(size_t i = a.w.size(); i-- > 0;)
            if(a.w[i] != b.w[i]) return a.w[i] < b.w[i];
        return false;
    }
};
// Flattened index lists in push order: list j is items[offsets[j], offsets[j + 1]).
struct IndexLists {
    std::vector<size_t> offsets{0}, items;
    template <class Range> void push(const Range& values) {
        items.insert(items.end(), values.begin(), values.end());
        offsets.push_back(items.size());
    }
    size_t size() const { return offsets.size() - 1; }
    const size_t* begin(size_t j) const { return items.data() + offsets[j]; }
    const size_t* end(size_t j) const { return items.data() + offsets[j + 1]; }
};
// Sparse transpose: list j holds the indices of the rows containing bit j.
inline IndexLists transpose_lists(const std::vector<Bits>& rows, size_t width) {
    std::vector<std::vector<size_t>> lists(width);
    for(size_t i = 0; i < rows.size(); i++) {
        if(rows[i].size() > width) throw std::invalid_argument("column outside width");
        rows[i].each([&](size_t j) { lists[j].push_back(i); });
    }
    IndexLists out;
    for(const auto& list : lists) out.push(list);
    return out;
}
// XOR the lists selected by mask into out, as xor_rows_prefix does for packed
// rows; mask bits beyond the lists are ignored. `width` bounds every index.
inline void xor_lists(Bits& out, const IndexLists& lists, const Bits& mask, size_t width) {
    out.w.resize(std::max(out.w.size(), (width + 63) / 64));
    mask.each([&](size_t j) {
        if(j < lists.size())
            for(auto i = lists.begin(j); i != lists.end(j); i++) out.w[*i / 64] ^= uint64_t(1) << (*i % 64);
    });
    out.trim();
}
inline void validate_masks(const std::vector<Bits>& masks, size_t width) {
    for(const auto& mask : masks)
        if(mask.size() > width) throw std::invalid_argument("mask outside dimension");
}
inline Bits xors(const std::vector<Bits>& v, const Bits& mask) {
    Bits r;
    for(size_t word = 0; word < mask.w.size(); word++) {
        auto bits = mask.w[word];
        while(bits) {
            size_t j = 64 * word + __builtin_ctzll(bits);
            bits &= bits - 1;
            if(j >= v.size()) throw std::invalid_argument("mask outside vector");
            const auto& row = v[j];
            if(r.w.size() < row.w.size()) r.w.resize(row.w.size());
            for(size_t k = 0; k < row.w.size(); k++) r.w[k] ^= row.w[k];
        }
    }
    r.trim();
    return r;
}
// Apply only the selected input prefix to an existing output. The input mask
// need not be copied or truncated, and the output is trimmed only once.
inline void xor_rows_prefix(Bits& out, const std::vector<Bits>& rows, const Bits& mask, size_t limit) {
    limit = std::min(limit, rows.size());
    for(size_t word = 0; word < std::min(mask.w.size(), (limit + 63) / 64); word++) {
        auto bits = mask.w[word];
        if(word == limit / 64 && limit % 64) bits &= (uint64_t(1) << (limit % 64)) - 1;
        while(bits) {
            const auto& row = rows[64 * word + __builtin_ctzll(bits)];
            bits &= bits - 1;
            if(out.w.size() < row.w.size()) out.w.resize(row.w.size());
            for(size_t k = 0; k < row.w.size(); k++) out.w[k] ^= row.w[k];
        }
    }
    out.trim();
}
// Pack a batch of dot products directly, avoiding one set/resize per result.
inline Bits parities(const std::vector<Bits>& rows, const Bits& value) {
    if(value.empty()) return {};
    Bits out;
    out.w.resize((rows.size() + 63) / 64);
    for(size_t first = 0; first < rows.size(); first += 64) {
        uint64_t word = 0;
        for(size_t j = first; j < std::min(first + 64, rows.size()); j++)
            word |= uint64_t(rows[j].parity_and(value)) << (j - first);
        out.w[first / 64] = word;
    }
    out.trim();
    return out;
}
inline std::vector<Bits> transpose(const std::vector<Bits>& cols, size_t width) {
    std::vector<Bits> rows(width);
    size_t entries = 0;
    for(const auto& col : cols) {
        if(col.size() > width) throw std::invalid_argument("column outside width");
        entries += col.count();
    }
    // Six wordwise exchanges transpose a dense 64-by-64 tile. Sparse matrices
    // retain the set-bit path; packing cost is selected by dimensions/density.
    if(width >= 64 && cols.size() >= 64 && entries / width >= 8) {
        for(size_t first = 0; first < cols.size(); first += 64)
            for(size_t w = 0; w < (width + 63) / 64; w++) {
                std::array<uint64_t, 64> tile{};
                for(size_t j = first; j < std::min(first + 64, cols.size()); j++)
                    if(w < cols[j].w.size()) tile[j - first] = cols[j].w[w];
                constexpr uint64_t masks[] = {0x5555555555555555ULL, 0x3333333333333333ULL, 0x0f0f0f0f0f0f0f0fULL,
                        0x00ff00ff00ff00ffULL, 0x0000ffff0000ffffULL, 0x00000000ffffffffULL};
                for(size_t stage = 0, shift = 1; stage < 6; stage++, shift *= 2) {
                    for(size_t base = 0; base < 64; base += 2 * shift)
                        for(size_t k = base; k < base + shift; k++) {
                            auto delta = ((tile[k] >> shift) ^ tile[k + shift]) & masks[stage];
                            tile[k] ^= delta << shift;
                            tile[k + shift] ^= delta;
                        }
                }
                for(size_t k = 0; k < 64 && 64 * w + k < width; k++)
                    if(tile[k]) {
                        auto& row = rows[64 * w + k];
                        row.w.resize(first / 64 + 1);
                        row.w.back() = tile[k];
                    }
            }
        return rows;
    }
    for(size_t j = 0; j < cols.size(); j++) cols[j].each([&](size_t i) { rows[i].set(j); });
    return rows;
}
} // namespace magic_native
