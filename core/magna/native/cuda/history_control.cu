typedef unsigned long long u64;
typedef long long i64;
typedef unsigned int u32;

__device__ u64 history_seed(u64 seed, u64 domain) {
    u64 x = seed + 0x9e3779b97f4a7c15ULL * domain;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
struct HistoryRng {
    u64 shot, seed, block, saved, has_saved;
    __device__ u64 next() {
        if (has_saved) { has_saved = 0; return saved; }
        u32 a = (u32)shot, b = (u32)(shot >> 32), c = (u32)block, d = (u32)(block >> 32);
        u32 k0 = (u32)seed, k1 = (u32)(seed >> 32);
        ++block;
        for (int round = 0; round < 10; ++round) {
            u64 p0 = 0xd2511f53ULL * a, p1 = 0xcd9e8d57ULL * c;
            a = (u32)(p1 >> 32) ^ b ^ k0; b = (u32)p1;
            c = (u32)(p0 >> 32) ^ d ^ k1; d = (u32)p0;
            k0 += 0x9e3779b9U; k1 += 0xbb67ae85U;
        }
        saved = (u64)c | ((u64)d << 32); has_saved = 1;
        return (u64)a | ((u64)b << 32);
    }
};
__device__ int affine(const u64* frame, const u32* offsets, const u32* words,
                      const u64* masks, const unsigned char* constants, int row) {
    u64 value = 0;
    for (u32 k = offsets[row]; k < offsets[row + 1]; ++k) value ^= frame[words[k]] & masks[k];
    return (__popcll(value) & 1) ^ constants[row];
}
__device__ void assign(u64* frame, int pauli_words, int bit, int value) {
    u64 mask = 1ULL << (bit % 64);
    u64& word = frame[pauli_words + bit / 64];
    word = (word & ~mask) | (value ? mask : 0ULL);
}
extern "C" __global__ void initialize_history(const u64* noise, const u64* sector_base,
        const u64* sector_columns, u64* frames, i64* ranks, u64 shots, u64 seed, u64 offset,
        int pauli_words, int stride, int sector_bits, int sector_rank, int proposal) {
    u64 shot = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (shot >= shots) return;
    u64* frame = frames + shot * stride;
    for (int w = 0; w < pauli_words; ++w) frame[w] = noise[shot * pauli_words + w];
    for (int w = pauli_words; w < stride; ++w) frame[w] = 0;
    int words = (sector_bits + 63) / 64;
    for (int w = 0; w < words; ++w) frame[pauli_words + w] = sector_base[w];
    HistoryRng rng{offset + shot, history_seed(seed, 1), 0, 0, 0};
    for (int j = 0; j < sector_rank; ++j)
        if (rng.next() & 1) for (int w = 0; w < words; ++w)
            frame[pauli_words + w] ^= sector_columns[(u64)j * words + w];
    if (proposal) ranks[shot] = sector_rank;
}
extern "C" __global__ void history_inputs(const u64* frames, const u32* offsets,
        const u32* words, const u64* masks, const unsigned char* constants,
        unsigned char* inputs, u64 shots, int stride, int count, int column_major) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= shots * count) return;
    u64 shot = column_major ? i % shots : i / count;
    int row = column_major ? i / shots : i % count;
    inputs[i] = affine(frames + shot * stride, offsets, words, masks, constants, row);
}
extern "C" __global__ void history_closed(u64* frames, const u32* offsets,
        const u32* words, const u64* masks, const unsigned char* constants,
        const unsigned char* closed, u64 shots, int stride, int pauli_words,
        int sector_bits, int observation_first, int count) {
    u64 shot = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (shot >= shots) return;
    u64* frame = frames + shot * stride;
    for (int j = 0; j < count; ++j) if (closed[j])
        assign(frame, pauli_words, sector_bits + observation_first + j,
               affine(frame, offsets, words, masks, constants, j));
}
extern "C" __global__ void history_feedback(u64* frames, const u32* offsets,
        const u32* words, const u64* masks, const unsigned char* constants,
        const u32* pending_inputs, const u32* outputs, const u64* columns,
        const unsigned char* inputs, const unsigned char* decisions, u64 shots,
        int stride, int pauli_words, int sector_bits, int observation_first,
        int pending_count, int input_count, int output_count) {
    u64 shot = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (shot >= shots) return;
    u64* frame = frames + shot * stride;
    for (int j = 0; j < pending_count; ++j) {
        int bit = sector_bits + observation_first + j, i = pending_inputs[j];
        int old = (frame[pauli_words + bit / 64] >> (bit % 64)) & 1;
        int known = affine(frame, offsets, words, masks, constants, i) ^ old;
        assign(frame, pauli_words, bit, inputs[shot * input_count + i] ^ known);
    }
    for (int j = 0; j < output_count; ++j) {
        int value = decisions[shot * output_count + j];
        assign(frame, pauli_words, outputs[j], value);
        if (value) for (int w = 0; w < pauli_words; ++w)
            frame[w] ^= columns[(u64)j * pauli_words + w];
    }
}
