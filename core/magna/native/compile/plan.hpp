// Compiled assignment data and original-location ordered Pauli transport.
#pragma once
#include "../foundation/stabilizer.hpp"
#include <optional>
namespace magic_native {
struct Frame {
    Bits x, z;
    int p = 0;
};
inline Frame multiply(const Frame& a, const Frame& b) {
    return {a.x ^ b.x, a.z ^ b.z, mod4(a.p + b.p + 2 * (a.z & b.x).parity())};
}
// Immutable suffix values need not retain leading zero words. Widely separated
// nonzero words use index/value pairs only when that is smaller than the span.
// Working Clifford frames keep the ordinary dense Bits representation.
class StoredBits {
    static_assert(sizeof(size_t) <= sizeof(uint64_t), "stored word index capacity");
    size_t first_ = 0;
    std::vector<uint64_t> words_;

public:
    StoredBits() = default;
    StoredBits(const Bits& value) {
        size_t first = 0, last = value.w.size();
        while(first < last && !value.w[first]) first++;
        if(first == last) return;
        while(!value.w[last - 1]) last--;
        size_t nonzero = 0;
        for(size_t j = first; j < last; j++) nonzero += value.w[j] != 0;
        const size_t span = last - first;
        if(nonzero < span - nonzero) {
            first_ = std::numeric_limits<size_t>::max();
            words_.reserve(2 * nonzero);
            for(size_t j = first; j < last; j++)
                if(value.w[j]) {
                    words_.push_back(uint64_t(j));
                    words_.push_back(value.w[j]);
                }
        } else {
            first_ = first;
            words_.assign(value.w.begin() + first, value.w.begin() + last);
        }
    }
    bool sparse() const { return first_ == std::numeric_limits<size_t>::max(); }
    size_t word_limit() const {
        if(words_.empty()) return 0;
        // Both encodings are formed from an existing vector: these sums are
        // bounded by its size, and private storage cannot acquire invalid rows.
        return sparse() ? size_t(words_[words_.size() - 2]) + 1 : first_ + words_.size();
    }
    size_t payload_bytes() const { return sizeof(first_) + sizeof(uint64_t) * words_.size(); }
    void xor_into(Bits& out) const {
        if(words_.empty()) return;
        out.w.resize(std::max(out.w.size(), word_limit()));
        if(sparse())
            for(size_t j = 0; j < words_.size(); j += 2) out.w[size_t(words_[j])] ^= words_[j + 1];
        else
            for(size_t j = 0; j < words_.size(); j++) out.w[first_ + j] ^= words_[j];
        out.trim();
    }
    Bits materialize() const {
        Bits out;
        xor_into(out);
        return out;
    }
};
struct StoredFrame {
    StoredBits x, z;
    int p = 0;
    StoredFrame() = default;
    StoredFrame(const Frame& frame) : x(frame.x), z(frame.z), p(frame.p) {}
    Frame materialize() const { return {x.materialize(), z.materialize(), p}; }
    // Paired-network transport omits only the common global ket phase.
    void xor_into(Frame& frame) const {
        x.xor_into(frame.x);
        z.xor_into(frame.z);
    }
};
struct Location {
    int index, position;
};
struct Suffix {
    int position;
    StoredFrame x, z;
};
struct Fault {
    size_t location;
    int wire;
    int pauli;
};
struct Assignment {
    Bits record;
    std::vector<Fault> faults;
    bool feedback = false;
    std::vector<int> actual_magic;
    int feedback_phase = 0;
};
// Linear continuation of the already accumulated prefix Pauli frame.
// Captured phase signs are retained before any probability-network cancellation.
// This has one image per endpoint generator, never one image per fault location.
struct FrameContinuation {
    size_t source_physical = 0, source_magic = 0, target_physical = 0;
    std::vector<StoredFrame> x, z;
    Frame apply(const Frame& source) const {
        Frame out;
        source.x.each([&](size_t j) {
            if(j < x.size()) x[j].xor_into(out);
        });
        source.z.each([&](size_t j) {
            if(j < z.size()) z[j].xor_into(out);
        });
        for(size_t j = 0; j < source_magic; j++) {
            if(source.x.bit(source_physical + j)) out.x.flip(target_physical + j);
            if(source.z.bit(source_physical + j)) out.z.flip(target_physical + j);
        }
        return out;
    }
};
struct Plan {
    std::optional<FrameContinuation> continuation;
    Ket skeleton;
    size_t retained = 0, physical = 0, nrecords = 0;
    Factor records;
    std::vector<Bits> basis, polar, forms;
    std::vector<int> magic;
    std::vector<Location> locations;
    std::map<int, std::vector<std::pair<int, int>>> histories;
    std::map<int, std::vector<Suffix>> suffix;
    // Images of a Pauli placed on one wire after an original location.
    const Suffix& fault_image(size_t location, int wire) const {
        auto loc = locations.at(location);
        auto hit = histories.find(wire);
        if(hit == histories.end()) throw std::invalid_argument("fault outside wire lifetime");
        auto& hs = hit->second;
        auto h = std::upper_bound(hs.begin(), hs.end(), loc.index, [](int v, auto& a) { return v < a.first; });
        if(h == hs.begin() || (--h)->second < 0) throw std::invalid_argument("fault outside wire lifetime");
        auto& ss = suffix.at(h->second);
        auto it = std::lower_bound(ss.begin(), ss.end(), loc.position, [](auto& a, int v) { return a.position < v; });
        if(it == ss.end()) throw std::invalid_argument("fault suffix missing");
        return *it;
    }
    // Untracked-phase transport of one Pauli (1 X, 2 Z, 3 Y) with known images.
    static void xor_fault(const Suffix& images, int pauli, Frame& frame) {
        if(pauli == 1 || pauli == 3) images.x.xor_into(frame);
        if(pauli == 2 || pauli == 3) images.z.xor_into(frame);
    }
    Frame fault_frame(std::vector<Fault> faults, bool track_global_phase = true) const {
        std::stable_sort(faults.begin(), faults.end(),
                [&](auto& a, auto& b) { return locations.at(a.location).index < locations.at(b.location).index; });
        Frame frame;
        for(auto& f : faults) {
            if(!f.pauli) continue;
            const auto& images = fault_image(f.location, f.wire);
            if(!track_global_phase) {
                xor_fault(images, f.pauli, frame);
                continue;
            }
            Frame image;
            if(f.pauli == 3) {
                image = multiply(images.x.materialize(), images.z.materialize());
                image.p = mod4(image.p + 1);
            } else image = (f.pauli == 1 ? images.x : images.z).materialize();
            frame = multiply(image, frame);
        }
        return frame;
    }
};
} // namespace magic_native
