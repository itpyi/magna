typedef unsigned long long u64;
typedef long long i64;

__device__ u64 absolute(i64 value) { return value < 0 ? (u64)(-(value+1))+1 : (u64)value; }
__device__ bool empty_scalar(const i64* c) { return !(c[0] || c[1] || c[2] || c[3]); }
__device__ double norm_delta(i64 a, i64 b, int& sign) {
    u64 x=absolute(a), y=absolute(b), lo=x*x, hi=__umul64hi(x,x);
    u64 yl=y*y, yh=__umul64hi(y,y);
    yh=(yh<<1)|(yl>>63); yl<<=1;
    if (hi==yh && lo==yl) { sign=0; return 0.; }
    sign=(hi>yh || (hi==yh && lo>yl)) ? 1 : -1;
    if (sign<0) { u64 t=hi; hi=yh; yh=t; t=lo; lo=yl; yl=t; }
    u64 upper=hi-yh-(lo<yl);
    return ldexp((double)upper,64)+(double)(lo-yl);
}
__device__ bool real_value(const i64* c, i64 power, double& mantissa, i64& exponent) {
    for(int j=0;j<4;++j) if(absolute(c[j])>((1ULL<<62)-1)) return false;
    if(c[2] || c[1]!=-c[3]) return false;
    i64 a=c[0],b=c[1];
    double value;
    if((a<0 && b>0)||(a>0 && b<0)) {
        int sign;
        double numerator=norm_delta(a,b,sign);
        value=(a>0?sign:-sign)*numerator/((double)absolute(a)+(double)absolute(b)*1.4142135623730950488);
    } else value=(double)a+(double)b*1.4142135623730950488;
    if(value<0.) return false;
    int shift=0;
    mantissa=frexp(value,&shift); exponent=mantissa ? power+shift : 0;
    return true;
}
__device__ bool difference(const i64* a, i64 pa, const i64* b, i64 pb, i64* out, i64& power) {
    if(empty_scalar(a)) pa=pb;
    if(empty_scalar(b)) pb=pa;
    power=pa<pb?pa:pb;
    i64 da=pa-power,db=pb-power;
    if(da>61 || db>61) return false;
    const u64 bound=(1ULL<<61)-1;
    for(int j=0;j<4;++j) {
        if(absolute(a[j])>(bound>>da) || absolute(b[j])>(bound>>db)) return false;
        out[j]=a[j]*(i64)(1ULL<<da)-b[j]*(i64)(1ULL<<db);
    }
    return true;
}
__device__ bool contribution(const i64* c,const int* p,int queries,i64 norm_power,
                             bool affine,i64 rank,double weight,double* value,i64* exponent) {
    if(queries<2 || queries>3 || rank<-(1LL<<50) || rank>(1LL<<50) ||
       norm_power<-(1LL<<50) || norm_power>(1LL<<50) || !isfinite(weight) || weight<0.) return false;
    double pm,am,em=0.,tm,unused; i64 pe,ae,ee=0,te,up;
    if(!real_value(c,p[0],pm,pe) || !real_value(c+4,p[1],am,ae)) return false;
    i64 delta[4],dp;
    if(!difference(c,p[0],c+4,p[1],delta,dp) || !real_value(delta,dp,unused,up)) return false;
    if(queries==3) {
        if(!real_value(c+8,(i64)p[2]-norm_power,tm,te) ||
           !difference(c+4,p[1],c+8,(i64)p[2]-norm_power,delta,dp) ||
           !real_value(delta,dp,em,ee)) return false;
    }
    if(!affine && pm==0.) return false;
    int we=0;
    double wm=frexp(weight,&we);
    double numerator[2]={am,em}; i64 powers[2]={ae,ee};
    for(int j=0;j<2;++j) {
        double v=numerator[j]*wm/(affine?1.:pm);
        int shift=0;
        value[j]=frexp(v,&shift);
        exponent[j]=v ? powers[j]+we+shift+(affine?rank:-pe) : 0;
    }
    return true;
}
__device__ void add_row(double* s,double x,double y) {
    if(x==0.) {s[4]=hypot(s[4],y);return;}
    double radius=hypot(s[2],x),cosine=s[2]/radius,sine=x/radius;
    double xy=sine*s[3],yy=sine*y;
    double residual=fma(cosine,y,-xy)-fma(sine,s[3],-xy);
    s[3]=fma(cosine,s[3],yy)+fma(sine,y,-yy);
    s[2]=radius;s[4]=hypot(s[4],residual);
}
__device__ double rescale(double value,i64 shift) {
    return value==0. || shift < -2147483648LL ? 0. : ldexp(value,(int)shift);
}
__device__ void merge(double* s,const double* other) {
    if(other[5]==0.) return;
    if(s[5]==0.) {for(int j=0;j<8;++j)s[j]=other[j];return;}
    double t[8];for(int j=0;j<8;++j)t[j]=other[j];
    for(int j=0;j<2;++j) {
        i64 scale=s[j]==0.?(i64)t[6+j]:t[j]==0.?(i64)s[6+j]:
                  ((i64)s[6+j]>(i64)t[6+j]?(i64)s[6+j]:(i64)t[6+j]);
        i64 ds=(i64)s[6+j]-scale,dt=(i64)t[6+j]-scale;
        s[j]=rescale(s[j],ds);t[j]=rescale(t[j],dt);
        int first=j==0?2:3,last=j==0?2:4;
        for(int k=first;k<=last;++k){s[k]=rescale(s[k],ds);t[k]=rescale(t[k],dt);}
        s[6+j]=(double)scale;
    }
    double da=t[0]-s[0],de=t[1]-s[1],total=s[5]+t[5],w=sqrt(s[5]*t[5]/total);
    add_row(s,t[2],t[3]);add_row(s,0.,t[4]);add_row(s,w*da,w*de);
    s[0]+=da*(t[5]/total);s[1]+=de*(t[5]/total);s[5]=total;
}
__device__ void reduce_block(double shared[8][128],double* state) {
    unsigned tid=threadIdx.x;
    for(int j=0;j<8;++j)shared[j][tid]=state[j];
    __syncthreads();
    for(unsigned step=64;step;step>>=1) {
        if(tid<step) {
            double other[8];
            for(int j=0;j<8;++j){state[j]=shared[j][tid];other[j]=shared[j][tid+step];}
            merge(state,other);
            for(int j=0;j<8;++j)shared[j][tid]=state[j];
        }
        __syncthreads();
    }
}
extern "C" __global__ void history_moments(const i64* coefficients,const int* powers,const int* status,
        const i64* ranks,const double* weights,int affine,int weighted,u64 shots,int queries,i64 norm_power,
        double* partial,unsigned char* continuation) {
    __shared__ double shared[8][128];
    u64 shot=(u64)blockIdx.x*128+threadIdx.x;
    double state[8]={0.,0.,0.,0.,0.,0.,0.,0.};
    if(shot<shots) {
        bool ok=true;for(int j=0;j<queries;++j)if(status[shot*queries+j])ok=false;
        i64 exponent[2]={0,0};
        if(ok)ok=contribution(coefficients+shot*queries*4,powers+shot*queries,queries,norm_power,
                               affine,affine?ranks[shot]:0,weighted?weights[shot]:1.,state,exponent);
        continuation[shot]=ok?0:1;
        if(ok){state[5]=1.;state[6]=(double)exponent[0];state[7]=(double)exponent[1];}
        else for(int j=0;j<8;++j)state[j]=0.;
    }
    reduce_block(shared,state);
    if(!threadIdx.x)for(int j=0;j<8;++j)partial[(u64)blockIdx.x*8+j]=shared[j][0];
}
extern "C" __global__ void history_finish(const double* partial,u64 blocks,double* output) {
    __shared__ double shared[8][128];
    double state[8]={0.,0.,0.,0.,0.,0.,0.,0.};
    for(u64 block=threadIdx.x;block<blocks;block+=128)merge(state,partial+block*8);
    reduce_block(shared,state);
    if(!threadIdx.x)for(int j=0;j<8;++j)output[j]=shared[j][0];
}
