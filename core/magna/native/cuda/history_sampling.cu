// Packed affine support transport. One CTA owns a reusable scratch slot and
// walks an addressed subsequence of the chunk; no history frames visit the CPU.
#define HISTORY_MAX_N @MAX_N@
#define HISTORY_MAX_OPEN @MAX_OPEN@
#define HISTORY_MAX_PENDING @MAX_PENDING@
#define HISTORY_MAX_WORDS ((HISTORY_MAX_N+63)/64)
#define HISTORY_OPEN_WORDS ((HISTORY_MAX_OPEN+63)/64)
#define HISTORY_PENDING_WORDS ((HISTORY_MAX_PENDING+63)/64)
typedef unsigned long long hu64;

__device__ int history_dot(const hu64* a,const hu64* b,int words) {
    hu64 v=0;for(int w=0;w<words;++w)v^=a[w]&b[w];return __popcll(v)&1;
}
__device__ int history_affine(const hu64* rows,const unsigned char* constants,int row,
                            const hu64* frame,int stride) {
    return constants[row]^history_dot(rows+(hu64)row*stride,frame,stride);
}
__device__ int history_first(const hu64* bits,int words) {
    for(int w=0;w<words;++w)if(bits[w])return 64*w+__ffsll((long long)bits[w])-1;
    return -1;
}
__device__ int history_delta_q(const hu64* x,const int* actual,const unsigned char* initial,
                             const hu64* adj,const hu64* original,int n,int words) {
    int value=0;
    for(int j=0;j<n;++j)if((x[j/64]>>(j%64))&1) {
        value+=actual[j]-initial[j];
        hu64 v=0;
        for(int w=j/64;w<words;++w) {
            hu64 above=w==j/64?(j%64==63?0ULL:(~0ULL<<(j%64+1))):~0ULL;
            v^=(adj[(hu64)j*words+w]^original[(hu64)j*words+w])&x[w]&above;
        }
        value+=2*(__popcll(v)&1);
    }
    return value&3;
}
__device__ int history_delta_polar(const hu64* a,const hu64* b,const int* actual,
                                 const unsigned char* initial,const hu64* adj,
                                 const hu64* original,int n,int words) {
    int result=0;
    for(int j=0;j<n;++j)if((a[j/64]>>(j%64))&1) {
        hu64 v=0;
        for(int w=0;w<words;++w)v^=(adj[(hu64)j*words+w]^original[(hu64)j*words+w])&b[w];
        result^=(__popcll(v)&1)^(((actual[j]-initial[j])&1)*((b[j/64]>>(j%64))&1));
    }
    return result;
}
// Highest-pivot RREF, indexed by pivot rather than insertion order.
__device__ void history_insert(hu64* basis,hu64* column,int width,int words,long long* rank) {
    for(int pivot=width-1;pivot>=0;--pivot)if((column[pivot/64]>>(pivot%64))&1) {
        hu64* row=basis+(hu64)pivot*words;
        if((row[pivot/64]>>(pivot%64))&1)
            for(int w=0;w<words;++w)column[w]^=row[w];
    }
    for(int pivot=width-1;pivot>=0;--pivot)if((column[pivot/64]>>(pivot%64))&1) {
        for(int j=0;j<width;++j) {
            hu64* old=basis+(hu64)j*words;
            if((old[pivot/64]>>(pivot%64))&1)
                for(int w=0;w<words;++w)old[w]^=column[w];
        }
        for(int w=0;w<words;++w)basis[(hu64)pivot*words+w]=column[w];
        ++*rank;return;
    }
}

extern "C" __global__ void history_affine_entry(
    const hu64* frames,const unsigned char* features,const unsigned int* support,
    const unsigned char* initial_linear,const hu64* initial_upper,const unsigned int* linear_shift,
    const int* magic,const hu64* magic_masks,
    const hu64* open_rows,const unsigned char* open_constants,
    const hu64* entry_shift,const hu64* entry_change,const hu64* entry_columns,const hu64* entry_base,
    const unsigned char* entry_linear,const hu64* entry_upper,
    const hu64* keep,const hu64* offset_rows,const unsigned char* offset_constants,
    const hu64* events,const hu64* outcome_rows,const unsigned char* outcome_constants,
    hu64* scratch,hu64* envelope_base,hu64* envelope_basis,unsigned char* envelope_valid,long long* envelope_rank,
    hu64 shots,int stride,int n,int m,int open_bits,int pending,int event_count,int support_count,int magic_count,
    int identity,int first_entry) {
    const int tid=threadIdx.x,nw=(n+63)/64,mw=(m+63)/64,ow=(open_bits+63)/64,pw=(pending+63)/64;
    const hu64 scratch_words=(hu64)n*nw+(hu64)m*mw+(hu64)m*ow+(hu64)m*pw;
    hu64* mother=scratch+(hu64)blockIdx.x*scratch_words;
    hu64* adj=mother+(hu64)n*nw;
    hu64* columns=adj+(hu64)m*mw;
    hu64* projected=columns+(hu64)m*ow;
    __shared__ int mother_linear[HISTORY_MAX_N?HISTORY_MAX_N:1];
    __shared__ int linear[HISTORY_MAX_N?HISTORY_MAX_N:1];
    __shared__ hu64 active[HISTORY_MAX_WORDS?HISTORY_MAX_WORDS:1],hidden[HISTORY_MAX_WORDS?HISTORY_MAX_WORDS:1];
    __shared__ hu64 base[HISTORY_OPEN_WORDS?HISTORY_OPEN_WORDS:1];
    __shared__ hu64 projected_base[HISTORY_PENDING_WORDS?HISTORY_PENDING_WORDS:1];
    __shared__ unsigned char incidence[HISTORY_MAX_N?HISTORY_MAX_N:1];
    __shared__ int h,p,lh,lp,zero,rhs;
    for(hu64 shot=blockIdx.x;shot<shots;shot+=gridDim.x) {
        const hu64* frame=frames+shot*stride;
        hu64* ebase=envelope_base+shot*pw;
        hu64* ebasis=envelope_basis+shot*pending*pw;
        if(first_entry) {
            for(int j=tid;j<pending*pw;j+=blockDim.x)ebasis[j]=0;
            for(int w=tid;w<pw;w+=blockDim.x)ebase[w]=0;
            if(!tid){envelope_valid[shot]=0;envelope_rank[shot]=0;}
        }
        if(!tid) {
            zero=0;
            for(int j=0;j<support_count;++j)if(features[(hu64)support[j]*shots+shot])zero=1;
        }
        __syncthreads();
        if(zero){__syncthreads();continue;}
        for(int j=tid;j<n;j+=blockDim.x) {
            int value=(initial_linear[j]+2*features[(hu64)linear_shift[j]*shots+shot])&3;
            for(int w=0;w<nw;++w)mother[(hu64)j*nw+w]=initial_upper[(hu64)j*nw+w];
            for(int k=0;k<magic_count;++k) {
                const int* q=magic+8*k;
                int actual=((features[(hu64)q[4]*shots+shot]?-q[0]:q[0])+4*features[(hu64)q[5]*shots+shot])&7;
                for(int side=0;side<2;++side) {
                    const hu64* mask=magic_masks+(2*k+side)*nw;
                    if(!((mask[j/64]>>(j%64))&1))continue;
                    int before=q[2+side],after=before^features[(hu64)q[6+side]*shots+shot];
                    int c=(((side?1:-1)*(actual*(after?-1:1)-q[1]*(before?-1:1)))&7)/2;
                    value=(value+c)&3;
                    if(c&1)for(int w=0;w<nw;++w)
                        mother[(hu64)j*nw+w]^=mask[w]&~(w==j/64?(1ULL<<(j%64)):0ULL);
                }
            }
            mother_linear[j]=value;
        }
        for(int w=tid;w<ow;w+=blockDim.x) {
            hu64 value=entry_base[w];
            for(int bit=0;bit<64 && 64*w+bit<open_bits;++bit)
                value^=(hu64)history_affine(open_rows,open_constants,64*w+bit,frame,stride)<<bit;
            base[w]=value;
        }
        __syncthreads();
        for(int j=tid;j<m;j+=blockDim.x) {
            if(identity) {
                linear[j]=mother_linear[j];
                for(int w=0;w<mw;++w)adj[(hu64)j*mw+w]=mother[(hu64)j*nw+w];
            } else {
                const hu64* axis=entry_change+(hu64)j*nw;
                linear[j]=(entry_linear[j]+history_delta_q(axis,mother_linear,initial_linear,mother,initial_upper,n,nw)+
                    2*history_delta_polar(axis,entry_shift,mother_linear,initial_linear,mother,initial_upper,n,nw))&3;
                for(int w=0;w<mw;++w) {
                    hu64 value=entry_upper[(hu64)j*mw+w];
                    for(int bit=0;bit<64 && 64*w+bit<m;++bit)if(64*w+bit!=j)
                        value^=(hu64)history_delta_polar(axis,entry_change+(hu64)(64*w+bit)*nw,
                            mother_linear,initial_linear,mother,initial_upper,n,nw)<<bit;
                    adj[(hu64)j*mw+w]=value;
                }
            }
            for(int w=0;w<ow;++w)columns[(hu64)j*ow+w]=entry_columns[(hu64)j*ow+w];
        }
        for(int w=tid;w<mw;w+=blockDim.x) {
            hu64 live=0,unseen=0;
            for(int bit=0;bit<64 && 64*w+bit<m;++bit) {
                live|=1ULL<<bit;bool empty=true;
                for(int k=0;k<ow;++k)if(entry_columns[(hu64)(64*w+bit)*ow+k])empty=false;
                if(empty)unseen|=1ULL<<bit;
            }
            active[w]=live;hidden[w]=unseen;
        }
        __syncthreads();
        while(true) {
            if(!tid) {
                h=history_first(hidden,mw);p=-1;
                if(h>=0) {
                    hidden[h/64]&=~(1ULL<<(h%64));active[h/64]&=~(1ULL<<(h%64));lh=linear[h];
                    if(!(lh&1)) {
                        for(int pass=0;pass<2 && p<0;++pass)
                            for(int w=0;w<mw;++w) {
                                hu64 v=adj[(hu64)h*mw+w]&active[w]&(pass?~0ULL:hidden[w]);
                                if(v){p=64*w+__ffsll((long long)v)-1;break;}
                            }
                        if(p<0){if(lh/2)zero=1;}
                        else {active[p/64]&=~(1ULL<<(p%64));hidden[p/64]&=~(1ULL<<(p%64));lp=linear[p];}
                    }
                }
            }
            __syncthreads();
            if(h<0 || zero)break;
            if(!(lh&1) && p>=0 && lh/2)
                for(int w=tid;w<ow;w+=blockDim.x)base[w]^=columns[(hu64)p*ow+w];
            for(int j=tid;j<m;j+=blockDim.x) {
                if(!((active[j/64]>>(j%64))&1))continue;
                int sj=(adj[(hu64)h*mw+j/64]>>(j%64))&1;
                if(lh&1) {
                    if(!sj)continue;
                    linear[j]=(linear[j]+(lh==1?-1:1))&3;
                    for(int w=0;w<mw;++w)
                        adj[(hu64)j*mw+w]^=(adj[(hu64)h*mw+w]&active[w])&~(w==j/64?(1ULL<<(j%64)):0ULL);
                } else if(p>=0) {
                    int nj=(adj[(hu64)p*mw+j/64]>>(j%64))&1;
                    if(!(sj||nj))continue;
                    linear[j]=(linear[j]+lp*sj+2*sj*nj+2*(lh/2)*(nj^((lp&1)*sj)))&3;
                    for(int w=0;w<mw;++w) {
                        hu64 a=adj[(hu64)h*mw+w]&active[w],b=adj[(hu64)p*mw+w]&active[w];
                        adj[(hu64)j*mw+w]^=(((nj^(sj&&(lp&1)))?a:0ULL)^(sj?b:0ULL))&~(w==j/64?(1ULL<<(j%64)):0ULL);
                    }
                    if(sj)for(int w=0;w<ow;++w)columns[(hu64)j*ow+w]^=columns[(hu64)p*ow+w];
                }
            }
            __syncthreads();
        }
        if(zero){__syncthreads();continue;}
        // Affine conditioning substitutes one parameter per independent cap.
        for(int event=0;event<event_count;++event) {
            const hu64* equation=events+(hu64)event*ow;
            for(int j=tid;j<m;j+=blockDim.x)
                incidence[j]=((active[j/64]>>(j%64))&1)?history_dot(equation,columns+(hu64)j*ow,ow):0;
            __syncthreads();
            if(!tid) {
                p=-1;for(int j=0;j<m;++j)if(incidence[j]){p=j;break;}
                rhs=history_affine(outcome_rows,outcome_constants,event,frame,stride)^history_dot(equation,base,ow);
                if(p<0){if(rhs)zero=1;}else active[p/64]&=~(1ULL<<(p%64));
            }
            __syncthreads();
            if(zero)break;
            if(p>=0) {
                if(rhs)for(int w=tid;w<ow;w+=blockDim.x)base[w]^=columns[(hu64)p*ow+w];
                for(int j=tid;j<m;j+=blockDim.x)
                    if(j!=p && incidence[j])for(int w=0;w<ow;++w)columns[(hu64)j*ow+w]^=columns[(hu64)p*ow+w];
            }
            __syncthreads();
        }
        if(zero){__syncthreads();continue;}
        for(int w=tid;w<pw;w+=blockDim.x) {
            hu64 value=0;
            for(int bit=0;bit<64 && 64*w+bit<pending;++bit) {
                int k=64*w+bit;
                value|=(hu64)(history_dot(keep+(hu64)k*ow,base,ow)^history_affine(offset_rows,offset_constants,k,frame,stride))<<bit;
            }
            projected_base[w]=value;
        }
        for(int j=tid;j<m;j+=blockDim.x)if((active[j/64]>>(j%64))&1)
            for(int w=0;w<pw;++w) {
                hu64 value=0;
                for(int bit=0;bit<64 && 64*w+bit<pending;++bit)
                    value|=(hu64)history_dot(keep+(hu64)(64*w+bit)*ow,columns+(hu64)j*ow,ow)<<bit;
                projected[(hu64)j*pw+w]=value;
            }
        __syncthreads();
        if(!tid) {
            hu64 temporary[HISTORY_PENDING_WORDS?HISTORY_PENDING_WORDS:1];
            if(!envelope_valid[shot]) {
                for(int w=0;w<pw;++w)ebase[w]=projected_base[w];
                envelope_valid[shot]=1;
            } else {
                for(int w=0;w<pw;++w)temporary[w]=projected_base[w]^ebase[w];
                history_insert(ebasis,temporary,pending,pw,envelope_rank+shot);
            }
            for(int j=0;j<m;++j)if((active[j/64]>>(j%64))&1) {
                for(int w=0;w<pw;++w)temporary[w]=projected[(hu64)j*pw+w];
                history_insert(ebasis,temporary,pending,pw,envelope_rank+shot);
            }
        }
        __syncthreads();
    }
}

__device__ hu64 history_domain(hu64 seed,hu64 domain) {
    hu64 x=seed+0x9e3779b97f4a7c15ULL*domain;
    x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);
}
struct HistoryPhilox {
    hu64 shot,seed,block,saved;bool have;
    __device__ HistoryPhilox(hu64 s,hu64 k):shot(s),seed(k),block(0),saved(0),have(false){}
    __device__ hu64 next() {
        if(have){have=false;return saved;}
        unsigned a=(unsigned)shot,b=(unsigned)(shot>>32),c=(unsigned)block,d=(unsigned)(block>>32);
        unsigned k0=(unsigned)seed,k1=(unsigned)(seed>>32);++block;
        for(int r=0;r<10;++r) {
            hu64 p0=0xd2511f53ULL*a,p1=0xcd9e8d57ULL*c;
            a=(unsigned)(p1>>32)^b^k0;b=(unsigned)p1;c=(unsigned)(p0>>32)^d^k1;d=(unsigned)p0;
            k0+=0x9e3779b9U;k1+=0xbb67ae85U;
        }
        saved=(hu64)c|((hu64)d<<32);have=true;return (hu64)a|((hu64)b<<32);
    }
};
extern "C" __global__ void history_affine_draw(
    hu64* frames,hu64* envelope_base,const hu64* envelope_basis,const unsigned char* envelope_valid,
    const long long* envelope_rank,long long* proposal_rank,hu64 shots,hu64 seed,hu64 offset,
    int site,int stride,int pauli_words,int sector_bits,int observation_first,int pending) {
    hu64 shot=(hu64)blockIdx.x*blockDim.x+threadIdx.x;if(shot>=shots)return;
    int pw=(pending+63)/64;hu64* base=envelope_base+shot*pw;
    const hu64* basis=envelope_basis+shot*pending*pw;
    if(envelope_valid[shot]) {
        for(int pivot=pending-1;pivot>=0;--pivot)
            if((base[pivot/64]>>(pivot%64))&1) {
                const hu64* row=basis+(hu64)pivot*pw;
                if((row[pivot/64]>>(pivot%64))&1)for(int w=0;w<pw;++w)base[w]^=row[w];
            }
        HistoryPhilox rng(offset+shot,history_domain(seed,(hu64)site+2));
        for(int pivot=0;pivot<pending;++pivot) {
            const hu64* row=basis+(hu64)pivot*pw;
            if(((row[pivot/64]>>(pivot%64))&1) && (rng.next()&1))
                for(int w=0;w<pw;++w)base[w]^=row[w];
        }
        proposal_rank[shot]+=envelope_rank[shot];
    }
    hu64* frame=frames+shot*stride+pauli_words;
    for(int j=0;j<pending;++j) {
        int bit=sector_bits+observation_first+j;
        hu64 mask=1ULL<<(bit%64);
        if(envelope_valid[shot] && ((base[j/64]>>(j%64))&1))frame[bit/64]|=mask;
        else frame[bit/64]&=~mask;
    }
}
