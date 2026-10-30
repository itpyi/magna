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

__device__ u64 magnitude(i64 value) { return value < 0 ? (u64)(-(value + 1)) + 1 : (u64)value; }
__device__ int exact_sign(i64 a, i64 b) {
    if (!a) return b ? (b < 0 ? -1 : 1) : 0;
    if (!b || ((a < 0) == (b < 0))) return a < 0 ? -1 : 1;
    u64 x = magnitude(a), y = magnitude(b);
    u64 xl = x * x, xh = __umul64hi(x, x), yl = y * y, yh = __umul64hi(y, y);
    yh = (yh << 1) | (yl >> 63); yl <<= 1;
    int sign = xh == yh && xl == yl ? 0 : (xh > yh || (xh == yh && xl > yl) ? 1 : -1);
    return a < 0 ? -sign : sign;
}
__device__ bool scaled_difference(const i64* a, i64 pa, const i64* b, i64 pb,
                                  u64 multiplier, i64* output, i64& power) {
    if (!(a[0] || a[1])) pa = pb;
    if (!(b[0] || b[1]) || !multiplier) pb = pa;
    power = pa < pb ? pa : pb;
    i64 da = pa - power, db = pb - power;
    if (da > 61 || db > 61) return false;
    const u64 bound = (1ULL << 61) - 1;
    for (int j = 0; j < 2; ++j) {
        u64 av = magnitude(a[j]), bv = magnitude(b[j]);
        if (av > (bound >> da) || (multiplier && bv > (bound >> db) / multiplier)) return false;
        output[j] = a[j] * (i64)(1ULL << da) - b[j] * (i64)multiplier * (i64)(1ULL << db);
    }
    return true;
}
extern "C" __global__ void history_choose(u64* frames, i64* current, int* current_power,
        const int* current_status, const i64* zero_mass, const int* zero_power,
        const int* zero_status, u64* random_state, unsigned char* continuation,
        u64 shots, u64 seed, u64 offset, int site, int stride, int pauli_words, int auxiliary_bit) {
    u64 shot = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (shot >= shots || continuation[shot]) return;
    i64* d = current + 4 * shot;
    const i64* n = zero_mass + 4 * shot;
    if (current_status[shot] || zero_status[shot] || d[2] || n[2] ||
        d[3] == (-9223372036854775807LL - 1) || n[3] == (-9223372036854775807LL - 1) ||
        d[1] != -d[3] || n[1] != -n[3] || exact_sign(d[0], d[1]) <= 0 || exact_sign(n[0], n[1]) < 0) {
        continuation[shot] = 1; return;
    }
    i64 remaining[2], power;
    if (!scaled_difference(d, current_power[shot], n, zero_power[shot], 1, remaining, power) ||
        exact_sign(remaining[0], remaining[1]) < 0) { continuation[shot] = 1; return; }
    int bit = -1;
    if (!(n[0] || n[1])) bit = 1;
    else if (!(remaining[0] || remaining[1])) bit = 0;
    u64* state = random_state + 3 * shot;
    HistoryRng rng{offset + shot, history_seed(seed, (u64)site + 2), state[0], state[1], state[2]};
    u64 prefix = 0;
    for (int depth = 1; bit < 0 && depth <= 48; ++depth) {
        prefix = (prefix << 1) | (rng.next() & 1);
        i64 difference[2], ignored;
        if (!scaled_difference(n, zero_power[shot], d, (i64)current_power[shot] - depth,
                               prefix + 1, difference, ignored)) break;
        if (exact_sign(difference[0], difference[1]) >= 0) { bit = 0; break; }
        if (!scaled_difference(n, zero_power[shot], d, (i64)current_power[shot] - depth,
                               prefix, difference, ignored)) break;
        if (exact_sign(difference[0], difference[1]) <= 0) { bit = 1; break; }
    }
    state[0] = rng.block; state[1] = rng.saved; state[2] = rng.has_saved;
    if (bit < 0) { continuation[shot] = 1; return; }
    if (!bit) {
        for (int j = 0; j < 4; ++j) d[j] = n[j];
        current_power[shot] = zero_power[shot];
    } else {
        while ((remaining[0] || remaining[1]) && !((remaining[0] | remaining[1]) & 1)) {
            remaining[0] /= 2; remaining[1] /= 2; ++power;
        }
        d[0] = remaining[0]; d[1] = remaining[1]; d[2] = 0; d[3] = -remaining[1];
        current_power[shot] = (int)power;
    }
    assign(frames + shot * stride, pauli_words, auxiliary_bit, bit);
}
