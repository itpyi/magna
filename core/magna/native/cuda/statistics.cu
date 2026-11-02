
typedef unsigned long long u64;
typedef long long i64;
__device__ u64 magnitude(i64 value) {
    return value < 0 ? (u64)(-(value + 1)) + 1 : (u64)value;
}
__device__ bool zero4(const i64* c) { return !(c[0] || c[1] || c[2] || c[3]); }
// |a*a - 2*b*b| for int64 a and b, exact in 128 bits and then rounded to double.
// The sign is returned in sign.
__device__ double norm_difference(i64 a, i64 b, int& sign) {
    u64 x = magnitude(a), y = magnitude(b);
    u64 xl = x*x, xh = __umul64hi(x, x);
    u64 yl = y*y, yh = __umul64hi(y, y);
    yh = (yh << 1) | (yl >> 63); yl <<= 1;
    if (xh == yh && xl == yl) { sign = 0; return 0.; }
    sign = (xh > yh || (xh == yh && xl > yl)) ? 1 : -1;
    if (sign < 0) { u64 t = xh; xh = yh; yh = t; t = xl; xl = yl; yl = t; }
    u64 lo = xl - yl, hi = xh - yh - (xl < yl);
    return ldexp((double)hi, 64) + (double)lo;
}
__device__ void check_nonnegative_real(const i64* c, unsigned int& status) {
    const u64 safe = (1ULL << 61)-1;
    for (int j=0;j<4;++j) if (magnitude(c[j]) > safe) { status |= 2; return; }
    if (c[2] || c[1] != -c[3]) { status |= 1; return; }
    i64 a=c[0], b=c[1];
    if (a >= 0 && b >= 0) return;
    if (a <= 0 && b <= 0) { status |= 4; return; }
    int sign;
    norm_difference(a, b, sign);
    if ((a>0 && sign<0) || (a<0 && sign>0)) status |= 4;
}
__device__ double real_scalar(const i64* c, int power, unsigned int& status) {
    if (c[2] || c[1] != -c[3]) { status |= 1; return 0.; }
    i64 a = c[0], b = c[1];
    double value;
    if ((a < 0 && b > 0) || (a > 0 && b < 0)) {
        // a + b*sqrt(2) = (a*a - 2*b*b) / (a - b*sqrt(2)) avoids cancellation;
        // the denominator adds two numbers of the same sign.
        int sign;
        double numerator = norm_difference(a, b, sign);
        double denominator = (double)magnitude(a) + (double)magnitude(b) * 1.4142135623730950488;
        value = (a > 0 ? sign : -sign) * (numerator / denominator);
    } else value = (double)a + (double)b * 1.4142135623730950488;
    double result = ldexp(value, power);
    if (!isfinite(result) || (value != 0. && result == 0.)) status |= 8;
    return result;
}
// R is upper triangular and R^T R is the centered 2-by-2 covariance sum.
// Adding a row uses a QR rotation, without subtracting nearly equal variances.
__device__ void add_covariance_row(double& r00, double& r01, double& r11,
                                  double x, double y) {
    if (x == 0. && y == 0.) return;
    if (x == 0.) { r11 = hypot(r11, y); return; }
    double radius = hypot(r00, x);
    double cosine=r00/radius, sine=x/radius;
    double xy=sine*r01;
    double residual=fma(cosine,y,-xy)-fma(sine,r01,-xy);
    double yy=sine*y;
    r01=fma(cosine,r01,yy)+fma(sine,y,-yy);
    r00 = radius;
    r11 = hypot(r11, residual);
}
__device__ void merge_centered(u64& n, double* state, u64 other_n, const double* other) {
    if (!other_n) return;
    if (!n) {
        for (int j=0;j<5;++j) state[j]=other[j];
        n=other_n;
        return;
    }
    double da=other[0]-state[0], de=other[1]-state[1];
    double total=(double)(n+other_n), weight=sqrt((double)n*(double)other_n/total);
    add_covariance_row(state[2],state[3],state[4],other[2],other[3]);
    add_covariance_row(state[2],state[3],state[4],0.,other[4]);
    add_covariance_row(state[2],state[3],state[4],weight*da,weight*de);
    state[0] += da*((double)other_n/total);
    state[1] += de*((double)other_n/total);
    n += other_n;
}
extern "C" __global__ void scalar_moments(
        const i64* coefficients, const int* powers, const int* query_flags,
        const double* likelihoods, int weighted, u64 shots,
        int queries, int norm_power, double* partial, unsigned int* flags) {
    __shared__ double sums[5][128];
    __shared__ u64 counts[128];
    unsigned tid = threadIdx.x;
    u64 shot = (u64)blockIdx.x * blockDim.x + tid;
    double a = 0., e = 0.;
    unsigned int status = 0;
    if (shot < shots) {
        for (int q=0;q<queries;++q) if (query_flags[shot*queries+q]) status |= 16;
        const i64* ca = coefficients + shot * queries * 4;
        int pa = powers[shot * queries];
        a = real_scalar(ca, pa, status);
        if (queries == 2) {
            const i64* ct = ca + 4;
            check_nonnegative_real(ct, status);
            long long ptl = (long long)powers[shot * queries + 1] - norm_power;
            if (ptl < -2147483647LL || ptl > 2147483647LL) status |= 2;
            int pt = (int)ptl;
            if (zero4(ca)) pa = pt;
            if (zero4(ct)) pt = pa;
            int common = pa < pt ? pa : pt;
            long long da = (long long)pa-common, dt = (long long)pt-common;
            i64 ce[4] = {0, 0, 0, 0};
            const u64 align_bound = (1ULL << 61) - 1;
            if (da > 61 || dt > 61) status |= 2;
            else for (int j = 0; j < 4; ++j) {
                if (magnitude(ca[j]) > (align_bound >> da) ||
                    magnitude(ct[j]) > (align_bound >> dt)) { status |= 2; break; }
                ce[j] = ca[j] * (i64)(1ULL << da) - ct[j] * (i64)(1ULL << dt);
            }
            e = real_scalar(ce, common, status);
        }
        if (a < 0. || e < 0. || a > 1.+1e-13 || e > a+1e-13) status |= 4;
        if (weighted) {
            double w=likelihoods[shot], old_a=a, old_e=e;
            // Bounds weighted values so that their squares and centered sums stay finite.
            if (!isfinite(w) || w<0. || w>ldexp(1.,480)) status |= 32;
            a*=w; e*=w;
            if (!isfinite(a) || !isfinite(e) || !isfinite(a*a) || !isfinite(e*e) ||
                (w!=0. && ((old_a!=0. && a==0.) || (old_e!=0. && e==0.)))) status |= 8;
        }
        if ((a != 0. && a*a == 0.) || (e != 0. && e*e == 0.)) status |= 8;
    }
    if (status) atomicOr(flags, status);
    // Each draw starts as mean (a, e) with a zero covariance factor.
    sums[0][tid] = a; sums[1][tid] = e;
    sums[2][tid] = 0.; sums[3][tid] = 0.; sums[4][tid] = 0.;
    counts[tid] = shot < shots ? 1 : 0;
    __syncthreads();
    for (unsigned step = 64; step; step >>= 1) {
        if (tid < step) {
            double left[5],right[5];
            for(int k=0;k<5;++k) {left[k]=sums[k][tid];right[k]=sums[k][tid+step];}
            u64 count=counts[tid];
            merge_centered(count,left,counts[tid+step],right);
            counts[tid]=count;
            for(int k=0;k<5;++k) sums[k][tid]=left[k];
        }
        __syncthreads();
    }
    if (!tid) for (int k=0;k<5;++k) partial[(u64)blockIdx.x*5+k]=sums[k][0];
}
extern "C" __global__ void finish_moments(
        const double* partial, u64 blocks, u64 shots, const unsigned int* flags, double* output) {
    __shared__ double sums[5][128];
    __shared__ u64 counts[128];
    unsigned tid=threadIdx.x;
    double state[5]={0.,0.,0.,0.,0.};
    u64 count=0;
    for(u64 block=tid;block<blocks;block+=128) {
        u64 remaining=shots-block*128, block_count=remaining<128?remaining:128;
        merge_centered(count,state,block_count,partial+block*5);
    }
    for(int k=0;k<5;++k) sums[k][tid]=state[k];
    counts[tid]=count;
    __syncthreads();
    for (unsigned step=64;step;step>>=1) {
        if(tid<step) {
            double left[5],right[5];
            for(int k=0;k<5;++k) {left[k]=sums[k][tid];right[k]=sums[k][tid+step];}
            u64 n=counts[tid];
            merge_centered(n,left,counts[tid+step],right);
            counts[tid]=n;
            for(int k=0;k<5;++k) sums[k][tid]=left[k];
        }
        __syncthreads();
    }
    if(!tid) {
        for(int k=0;k<5;++k) output[k]=sums[k][0];
        output[5]=(double)(*flags);
    }
}
