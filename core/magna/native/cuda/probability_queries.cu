
#define N @N@
#define MODE @MODE@
#define STRIDE @STRIDE@
#define FRAME_BITS @FRAME_BITS@
#define PAULI_WORDS @PAULI_WORDS@
#define REAL_OUTPUT @REAL_OUTPUT@
#define FEATURES @FEATURES@
#define SUPPORT @SUPPORT@
#define GUARDS @GUARDS@
#define CONSTANT_LINEAR @CONSTANT_LINEAR@
#define CONSTANT_PAIRS @CONSTANT_PAIRS@
#define MAGIC @MAGIC@
#define STEPS @STEPS@
#define CONSTANT @CONSTANT@
#define POWER @POWER@
#define OUTPUT_POWER @OUTPUT_POWER@
#define WORDS ((N + 63) / 64)
#define F(k) features[(unsigned long long)(k) * shots + shot]

extern "C" __global__ void query_features(
    const unsigned long long* frames, const unsigned int* offsets,
    const unsigned int* words, const unsigned long long* masks,
    const unsigned char* constants, unsigned char* output, unsigned long long shots) {
    unsigned long long i = (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= shots * FEATURES) return;
    unsigned long long shot = i % shots;
    unsigned int feature = (unsigned int)(i / shots);
    unsigned long long v = 0;
    for (unsigned int k = offsets[feature]; k < offsets[feature + 1]; ++k)
        v ^= frames[shot * STRIDE + words[k]] & masks[k];
    output[i] = constants[feature] ^ (__popcll(v) & 1);
}

#if MODE != 4
extern "C" __global__ void exact_query(
    const unsigned long long* frames, const unsigned char* features,
    const unsigned int* support, const unsigned int* guards,
    const unsigned char* initial_linear, const unsigned long long* initial_upper,
    const unsigned int* linear_shift,
    const int* constant_linear, const int* constant_pairs,
    const int* magic, const unsigned long long* magic_masks,
    const unsigned char* phase_table,
    const int* steps, const unsigned int* step_offsets,
    const unsigned int* step_indices, const unsigned char* step_tags,
    long long* coefficients, int* powers, int* status,
    unsigned long long shots, int queries, int query) {
    unsigned long long shot = (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (shot >= shots) return;
    unsigned long long output = shot * queries + query;
    for (int k = 0; k < 4; ++k) coefficients[4 * output + k] = 0;
    powers[output] = 0;
    status[output] = 0;
#if (FRAME_BITS % 64) != 0 || FRAME_BITS == 0
    const unsigned long long valid = FRAME_BITS == 0 ? 0 : ((1ULL << (FRAME_BITS % 64)) - 1);
    if ((frames[shot * STRIDE + PAULI_WORDS / 2 - 1] |
         frames[shot * STRIDE + PAULI_WORDS - 1]) & ~valid) {
        status[output] = 2;
        return;
    }
#endif
#if MODE == 0
    return;
#else
    for (int j = 0; j < GUARDS; ++j) if (F(guards[j])) { status[output] = 1; return; }
    for (int j = 0; j < SUPPORT; ++j) if (F(support[j])) return;
    int linear[N ? N : 1];
#if MODE == 1 || MODE == 3
    unsigned long long adj[N ? N : 1];
#endif
    for (int j = 0; j < N; ++j) {
        linear[j] = (initial_linear[j] + 2 * F(linear_shift[j])) & 3;
#if MODE == 1 || MODE == 3
        adj[j] = initial_upper[j];
#endif
    }
    int phase = CONSTANT;
    for (int j = 0; j < CONSTANT_LINEAR; ++j)
        phase = (phase + constant_linear[2 * j + 1] * F(constant_linear[2 * j])) & 7;
    for (int j = 0; j < CONSTANT_PAIRS; ++j)
        phase = (phase + 4 * F(constant_pairs[2 * j]) * F(constant_pairs[2 * j + 1])) & 7;
    for (int j = 0; j < MAGIC; ++j) {
        const int* m = magic + 8 * j;
        int actual = ((F(m[4]) ? -m[0] : m[0]) + 4 * F(m[5])) & 7;
        for (int side = 0; side < 2; ++side) {
            int before = m[2 + side], after = before ^ F(m[6 + side]);
            int sign = side ? 1 : -1;
            phase = (phase + sign * (actual * after - m[1] * before)) & 7;
            int delta = (sign * (actual * (after ? -1 : 1) - m[1] * (before ? -1 : 1))) & 7;
            int c = delta / 2;
            if (!c) continue;
            for (int w = 0; w < WORDS; ++w) {
                unsigned long long mask = magic_masks[(2 * j + side) * WORDS + w];
                unsigned long long bits = mask;
                while (bits) {
                    int bit = __ffsll((long long)bits) - 1;
                    int k = 64 * w + bit;
                    bits &= bits - 1;
                    linear[k] = (linear[k] + c) & 3;
#if MODE == 1 || MODE == 3
                    if (c & 1) adj[k] ^= mask & ~(1ULL << bit);
#endif
                }
            }
        }
    }
    phase &= 7;
    int power = POWER;
    long long result[4] = {0, 0, 0, 0};
#if MODE == 3
    // At most 4096 exact unit roots; signed bins therefore fit in int64.
    for (unsigned int value = 0; value < (1U << N); ++value) {
        int exponent = phase + phase_table[value];
        unsigned long long bits = value;
        while (bits) {
            int j = __ffsll((long long)bits) - 1;
            bits &= bits - 1;
            exponent += 2 * linear[j] + 4 * (__popcll(adj[j] & bits) & 1);
        }
        exponent &= 7;
        result[exponent & 3] += exponent < 4 ? 1 : -1;
    }
#else
#if MODE == 1
    unsigned long long active = N == 64 ? ~0ULL : ((1ULL << N) - 1);
    while (active) {
        int h = __ffsll((long long)active) - 1;
        active &= ~(1ULL << h);
        unsigned long long v = adj[h] & active;
        int lh = linear[h];
        if (lh & 1) {
            int sign = lh == 1 ? -1 : 1;
            unsigned long long bits = v;
            while (bits) {
                int j = __ffsll((long long)bits) - 1;
                bits &= bits - 1;
                linear[j] = (linear[j] + sign) & 3;
                adj[j] ^= v & ~(1ULL << j);
            }
            ++power;
            phase = (phase + (lh == 1 ? 1 : -1)) & 7;
            continue;
        }
        power += 2;
        int rhs = lh / 2;
        if (!v) { if (rhs) return; continue; }
        int p = __ffsll((long long)v) - 1;
        active &= ~(1ULL << p);
        unsigned long long a = v & active, b = adj[p] & active, bits = a | b;
        int lp = linear[p];
        phase = (phase + 2 * rhs * lp) & 7;
        while (bits) {
            int j = __ffsll((long long)bits) - 1;
            bits &= bits - 1;
            int sj = (a >> j) & 1, nj = (b >> j) & 1;
            linear[j] = (linear[j] + lp * sj + 2 * sj * nj + 2 * rhs * (nj ^ ((lp & 1) * sj))) & 3;
            adj[j] ^= (((nj ^ (sj && (lp & 1))) ? a : 0) ^ (sj ? b : 0)) & ~(1ULL << j);
        }
    }
#else
    for (int s = 0; s < STEPS; ++s) {
        int h = steps[3 * s], p = steps[3 * s + 1], odd = steps[3 * s + 2], lh = linear[h];
        if (odd) {
            int sign = lh == 1 ? -1 : 1;
            for (unsigned int k = step_offsets[s]; k < step_offsets[s + 1]; ++k) {
                unsigned int j = step_indices[k];
                linear[j] = (linear[j] + sign) & 3;
            }
            ++power;
            phase = (phase + (lh == 1 ? 1 : -1)) & 7;
        } else {
            power += 2;
            int rhs = lh / 2;
            if (p == N) { if (rhs) return; continue; }
            int lp = linear[p];
            phase = (phase + 2 * rhs * lp) & 7;
            for (unsigned int k = step_offsets[s]; k < step_offsets[s + 1]; ++k) {
                int sj = step_tags[k] & 1, nj = (step_tags[k] >> 1) & 1;
                unsigned int j = step_indices[k];
                linear[j] = (linear[j] + lp * sj + 2 * sj * nj + 2 * rhs * (nj ^ ((lp & 1) * sj))) & 3;
            }
        }
    }
#endif
    phase &= 7;
    result[phase & 3] = phase < 4 ? 1 : -1;
#endif
    // omega(phase,power_sqrt2): factor out 2^floor(power/2), retain sqrt(2)
    // exactly as omega - omega^3 when power is odd.
    if (power & 1) {
        long long rotated[4] = {0, 0, 0, 0};
        for (int k = 0; k < 4; ++k) {
            int a = k + 1, b = k + 3;
            rotated[a & 3] += (a < 4 ? 1 : -1) * result[k];
            rotated[b & 3] -= (b < 4 ? 1 : -1) * result[k];
        }
        for (int k = 0; k < 4; ++k) result[k] = rotated[k];
    }
    int exponent = (power - (power & 1)) / 2 + OUTPUT_POWER;
    if (!(result[0] | result[1] | result[2] | result[3])) return;
    while (((result[0] | result[1] | result[2] | result[3]) & 1) == 0) {
        for (int k = 0; k < 4; ++k) result[k] /= 2;
        ++exponent;
    }
    if (REAL_OUTPUT && (result[2] || result[1] != -result[3])) status[output] = 3;
    for (int k = 0; k < 4; ++k) coefficients[4 * output + k] = result[k];
    powers[output] = exponent;
#endif
}
#else
// Blocks reuse their matrix storage across draws with changing quadratic forms.
extern "C" __global__ void exact_query_large(
    const unsigned long long* frames, const unsigned char* features,
    const unsigned int* support,
    const unsigned char* initial_linear, const unsigned long long* initial_upper,
    const unsigned int* linear_shift, const int* constant_linear, const int* constant_pairs,
    const int* magic, const unsigned long long* magic_masks,
    long long* coefficients, int* powers, int* status,
    unsigned long long shots, int queries, int query, int active_n, unsigned long long* scratch) {
    typedef unsigned long long u64;
    __shared__ int linear[N ? N : 1], phase, power, h, p, lh, lp, zero;
    __shared__ u64 active[WORDS ? WORDS : 1];
    unsigned tid=threadIdx.x;
    u64* adj=scratch+(u64)blockIdx.x*N*WORDS;
    for(u64 shot=blockIdx.x;shot<shots;shot+=gridDim.x) {
        u64 output=shot*queries+query;
        if(!tid) {
            zero=0;phase=CONSTANT;power=POWER+2*(N-active_n);
            for(int k=0;k<4;++k)coefficients[4*output+k]=0;
            powers[output]=0;status[output]=0;
#if (FRAME_BITS % 64) != 0 || FRAME_BITS == 0
            const u64 valid=FRAME_BITS==0?0:((1ULL<<(FRAME_BITS%64))-1);
            if((frames[shot*STRIDE+PAULI_WORDS/2-1]|
                frames[shot*STRIDE+PAULI_WORDS-1])&~valid) {zero=1;status[output]=2;}
#endif
            for(int j=0;j<SUPPORT;++j)if(F(support[j]))zero=1;
            for(int j=0;j<CONSTANT_LINEAR;++j)
                phase+=constant_linear[2*j+1]*F(constant_linear[2*j]);
            for(int j=0;j<CONSTANT_PAIRS;++j)
                phase+=4*F(constant_pairs[2*j])*F(constant_pairs[2*j+1]);
            for(int j=0;j<MAGIC;++j) {
                const int* m=magic+8*j;
                int actual=((F(m[4])?-m[0]:m[0])+4*F(m[5]))&7;
                for(int side=0;side<2;++side) {
                    int before=m[2+side],after=before^F(m[6+side]);
                    phase+=(side?1:-1)*(actual*after-m[1]*before);
                }
            }
            phase&=7;
        }
        __syncthreads();
        if(zero) {__syncthreads();continue;}
        for(int w=tid;w<WORDS;w+=blockDim.x) {
            int remaining=active_n-64*w;
            active[w]=remaining<=0?0ULL:remaining>=64?~0ULL:((1ULL<<remaining)-1);
        }
        for(int j=tid;j<active_n;j+=blockDim.x) {
            int value=(initial_linear[j]+2*F(linear_shift[j]))&3;
            for(int w=0;w<WORDS;++w)adj[(u64)j*WORDS+w]=initial_upper[(u64)j*WORDS+w];
            for(int k=0;k<MAGIC;++k) {
                const int* m=magic+8*k;
                int actual=((F(m[4])?-m[0]:m[0])+4*F(m[5]))&7;
                for(int side=0;side<2;++side) {
                    const u64* mask=magic_masks+(2*k+side)*WORDS;
                    if(!((mask[j/64]>>(j%64))&1))continue;
                    int before=m[2+side],after=before^F(m[6+side]),sign=side?1:-1;
                    int c=((sign*(actual*(after?-1:1)-m[1]*(before?-1:1)))&7)/2;
                    value=(value+c)&3;
                    if(c&1)for(int w=0;w<WORDS;++w)
                        adj[(u64)j*WORDS+w]^=mask[w]&~(w==j/64?(1ULL<<(j%64)):0ULL);
                }
            }
            linear[j]=value;
        }
        __syncthreads();
        while(true) {
            if(!tid) {
                h=-1;
                for(int w=0;w<WORDS;++w)if(active[w]) {h=64*w+__ffsll((long long)active[w])-1;break;}
                if(h>=0) {
                    active[h/64]&=~(1ULL<<(h%64));lh=linear[h];p=-1;
                    if(lh&1) {++power;phase=(phase+(lh==1?1:-1))&7;}
                    else {
                        power+=2;
                        for(int w=0;w<WORDS;++w) {
                            u64 v=adj[(u64)h*WORDS+w]&active[w];
                            if(v) {p=64*w+__ffsll((long long)v)-1;break;}
                        }
                        if(p<0) {if(lh/2)zero=1;}
                        else {active[p/64]&=~(1ULL<<(p%64));lp=linear[p];phase=(phase+2*(lh/2)*lp)&7;}
                    }
                }
            }
            __syncthreads();
            if(h<0 || zero)break;
            for(int j=tid;j<active_n;j+=blockDim.x) {
                if(!((active[j/64]>>(j%64))&1))continue;
                int sj=(adj[(u64)h*WORDS+j/64]>>(j%64))&1;
                if(lh&1) {
                    if(!sj)continue;
                    linear[j]=(linear[j]+(lh==1?-1:1))&3;
                    for(int w=0;w<WORDS;++w)
                        adj[(u64)j*WORDS+w]^=(adj[(u64)h*WORDS+w]&active[w])&
                            ~(w==j/64?(1ULL<<(j%64)):0ULL);
                } else if(p>=0) {
                    int nj=(adj[(u64)p*WORDS+j/64]>>(j%64))&1;
                    if(!(sj||nj))continue;
                    linear[j]=(linear[j]+lp*sj+2*sj*nj+2*(lh/2)*(nj^((lp&1)*sj)))&3;
                    for(int w=0;w<WORDS;++w) {
                        u64 a=adj[(u64)h*WORDS+w]&active[w],b=adj[(u64)p*WORDS+w]&active[w];
                        adj[(u64)j*WORDS+w]^=(((nj^(sj&&(lp&1)))?a:0ULL)^(sj?b:0ULL))&
                            ~(w==j/64?(1ULL<<(j%64)):0ULL);
                    }
                }
            }
            __syncthreads();
        }
        if(!tid && !zero) {
            long long value[4]={0,0,0,0};
            value[phase&3]=phase<4?1:-1;
            if(power&1) {
                long long rotated[4]={0,0,0,0};
                for(int k=0;k<4;++k) {
                    int a=k+1,b=k+3;
                    rotated[a&3]+=(a<4?1:-1)*value[k];
                    rotated[b&3]-=(b<4?1:-1)*value[k];
                }
                for(int k=0;k<4;++k)value[k]=rotated[k];
            }
            powers[output]=(power-(power&1))/2+OUTPUT_POWER;
            if(REAL_OUTPUT && (value[2] || value[1]!=-value[3]))status[output]=3;
            for(int k=0;k<4;++k)coefficients[4*output+k]=value[k];
        }
        __syncthreads();
    }
}
#endif
