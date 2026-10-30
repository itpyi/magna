typedef unsigned long long u64;
typedef long long i64;
__device__ u64 magnitude(i64 x) { return x < 0 ? (u64)(-(x+1))+1 : (u64)x; }
__device__ bool add(i64& target, i64 value) {
    const i64 bound = (1LL << 61)-1;
    if (value > 0 ? target > bound-value : target < -bound-value) return false;
    target += value;
    return true;
}
extern "C" __global__ void sum_queries(const i64* input, const int* powers,
        const int* flags, const i64* weights, const i64* weight_powers,
        i64* output, int* output_powers, int* output_flags,
        u64 shots, int terms, int queries, int query, int real_output) {
    u64 shot = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (shot >= shots) return;
    i64 total[4] = {0, 0, 0, 0}, power = 0;
    bool live = false, ok = true;
    const u64 bound = (1ULL << 61)-1;
    for (int term = 0; term < terms && ok; ++term) {
        u64 index = shot * terms + term;
        if (flags[index]) { ok = false; break; }
        const i64* c = input + 4 * index;
        const i64* w = weights + 4 * term;
        i64 value[4] = {0, 0, 0, 0};
        for (int j = 0; j < 4 && ok; ++j) for (int k = 0; k < 4 && ok; ++k) {
            if (!c[j] || !w[k]) continue;
            if (magnitude(c[j]) > bound/magnitude(w[k])) { ok = false; break; }
            i64 product = c[j] * w[k];
            ok = add(value[(j+k)&3], j+k < 4 ? product : -product);
        }
        if (!ok || !(value[0] || value[1] || value[2] || value[3])) continue;
        if (weight_powers[term] < -2147483647LL - powers[index] ||
            weight_powers[term] > 2147483647LL - powers[index]) { ok = false; break; }
        i64 exponent = (i64)powers[index] + weight_powers[term];
        while (!((value[0] | value[1] | value[2] | value[3]) & 1)) {
            for (int j = 0; j < 4; ++j) value[j] /= 2;
            ++exponent;
        }
        if (!live) { for (int j = 0; j < 4; ++j) total[j] = value[j]; power = exponent; live = true; continue; }
        i64 common = power < exponent ? power : exponent, a = power-common, b = exponent-common;
        if (a > 61 || b > 61) { ok = false; break; }
        for (int j = 0; j < 4 && ok; ++j) {
            if (magnitude(total[j]) > (bound >> a) || magnitude(value[j]) > (bound >> b)) { ok = false; break; }
            total[j] *= (i64)(1ULL << a);
            ok = add(total[j], value[j] * (i64)(1ULL << b));
        }
        power = common;
        live = total[0] || total[1] || total[2] || total[3];
        while (live && !((total[0] | total[1] | total[2] | total[3]) & 1)) {
            for (int j = 0; j < 4; ++j) total[j] /= 2;
            ++power;
        }
    }
    u64 index = shot * queries + query;
    if (power < -2147483647LL || power > 2147483647LL) ok = false;
    if (real_output && (total[2] || total[1] != -total[3])) ok = false;
    for (int j = 0; j < 4; ++j) output[4*index+j] = ok ? total[j] : 0;
    output_powers[index] = ok && live ? (int)power : 0;
    output_flags[index] = ok ? 0 : 1;
}
