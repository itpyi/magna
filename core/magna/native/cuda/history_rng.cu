typedef unsigned long long u64;
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
