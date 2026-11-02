
typedef unsigned long long u64;
typedef unsigned int u32;
struct Philox {
    u64 shot, seed, block, saved;
    bool has_saved;
    __device__ Philox(u64 s, u64 k) : shot(s), seed(k), block(0), saved(0), has_saved(false) {}
    __device__ u64 next() {
        if (has_saved) { has_saved = false; return saved; }
        u32 a = (u32)shot, b = (u32)(shot >> 32);
        u32 c = (u32)block, d = (u32)(block >> 32);
        u32 k0 = (u32)seed, k1 = (u32)(seed >> 32);
        ++block;
        #pragma unroll
        for (unsigned r = 0; r < 10; ++r) {
            u32 hi0 = __umulhi(0xd2511f53U, a), lo0 = 0xd2511f53U * a;
            u32 hi1 = __umulhi(0xcd9e8d57U, c), lo1 = 0xcd9e8d57U * c;
            a = hi1 ^ b ^ k0; b = lo1;
            c = hi0 ^ d ^ k1; d = lo0;
            k0 += 0x9e3779b9U; k1 += 0xbb67ae85U;
        }
        saved = (u64)c | ((u64)d << 32);
        has_saved = true;
        return (u64)a | ((u64)b << 32);
    }
};
__device__ bool coin(Philox& rng, u64 threshold, unsigned char certain) {
    return certain || (threshold && rng.next() < threshold);
}
__device__ u32 choice(Philox& rng, u32 site, const u32* choice_offsets,
        const u32* choices, const unsigned char* uniform,
        const u64* conditional_thresholds, const unsigned char* conditional_certain) {
    u32 first = choice_offsets[site], end = choice_offsets[site + 1];
    if (uniform[site]) {
        u64 count = end - first;
        if (count == 1) return choices[first];
        u64 reject = (0ULL - count) % count, value;
        do { value = rng.next(); } while (value < reject);
        return choices[first + value % count];
    }
    for (u32 j = first; j + 1 < end; ++j)
        if (coin(rng, conditional_thresholds[j], conditional_certain[j])) return choices[j];
    return choices[end - 1];
}
extern "C" __global__ void fused_noise(
        const u64* columns, const u64* base, const u64* thresholds,
        const unsigned char* certain, const u32* block_offsets, const u32* block_sites,
        const u32* choice_offsets, const u32* choices, const unsigned char* uniform,
        const u64* conditional_thresholds, const unsigned char* conditional_certain,
        const unsigned char* weighted_sites, const double* likelihood_table,
        u64* frames, double* likelihoods,
        u64 shots, u64 seed, u64 offset, u32 width, u32 blocks, int weighted) {
    u64 shot = (u64)blockDim.x * blockIdx.x + threadIdx.x;
    if (shot >= shots) return;
    Philox rng(offset + shot, seed);
    u32 weight_count = 0;
    u64* frame = frames + shot * width;
    for (u32 word = 0; word < width; ++word) frame[word] = base[word];
    for (u32 block = 0; block < blocks; ++block) {
        u32 first = block_offsets[block], count = block_offsets[block + 1] - first;
        u64 selected = 0;
        if (count < 8) {
            // Preserve interleaving of occurrence/choice draws for small groups.
            for (u32 bit = 0; bit < count; ++bit) {
                if (!coin(rng, thresholds[block], certain[block])) continue;
                u32 site = block_sites[first + bit];
                u32 action = choice(rng, site, choice_offsets, choices, uniform,
                                    conditional_thresholds, conditional_certain);
                const u64* column = columns + (u64)action * width;
                for (u32 word = 0; word < width; ++word) frame[word] ^= column[word];
                if (weighted) weight_count += weighted_sites[site];
            }
            continue;
        }
        u64 pending = count == 64 ? ~0ULL : ((1ULL << count) - 1);
        if (certain[block]) selected = pending;
        else {
            u64 remaining = thresholds[block];
            while (pending && remaining) {
                u64 bits = rng.next();
                if (remaining & (1ULL << 63)) {
                    selected |= pending & ~bits;
                    pending &= bits;
                } else pending &= ~bits;
                remaining <<= 1;
            }
        }
        while (selected) {
            u32 bit = __ffsll((long long)selected) - 1;
            selected &= selected - 1;
            u32 site = block_sites[first + bit];
            u32 action = choice(rng, site, choice_offsets, choices, uniform,
                                conditional_thresholds, conditional_certain);
            const u64* column = columns + (u64)action * width;
            for (u32 word = 0; word < width; ++word) frame[word] ^= column[word];
            if (weighted) weight_count += weighted_sites[site];
        }
    }
    if (weighted) likelihoods[shot] = likelihood_table[weight_count];
}
