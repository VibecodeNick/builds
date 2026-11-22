// Every float in [-16, 16] through tanh_ff against float(tanh(double)); then the two timed.
#define R4D_TANH_FF_COUNT 1
#include "r4d_tanh_ff.h"
#include <cstdio>
#include <vector>
#include <algorithm>
#include <stdexcept>
static void hip(hipError_t e){if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e));}
__global__ void sweep(uint32_t base,uint32_t n,unsigned long long* mism,uint32_t* first,unsigned long long* seen){
    uint32_t off=blockIdx.x*blockDim.x+threadIdx.x; if(off>=n)return;
    uint32_t u=base+off; float x=__uint_as_float(u);
    if(!isfinite(x)||fabsf(x)>16.f)return;
    float a=tanh_ff(x), b=(float)tanh((double)x);
    if(threadIdx.x==0)atomicAdd(seen,(unsigned long long)blockDim.x);
    if(__float_as_uint(a)!=__float_as_uint(b)){unsigned long long k=atomicAdd(mism,1ull);if(k<16)first[k]=u;}
}
__global__ void bench_ff(const float* in,float* out,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=tanh_ff(in[i]);}
__global__ void bench_ref(const float* in,float* out,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=(float)tanh((double)in[i]);}
__global__ void bench_f(const float* in,float* out,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=tanhf(in[i]);}
int main(){try{
    unsigned long long *mism,*seen; uint32_t* first; hip(hipMalloc(&mism,8));hip(hipMalloc(&seen,8));hip(hipMalloc(&first,64));
    hip(hipMemset(mism,0,8));hip(hipMemset(seen,0,8));
    const uint32_t ranges[2][2]={{0u,0x41800001u},{0x80000000u,0xC1800001u}};
    for(auto& r:ranges){ for(uint32_t base=r[0];base<r[1];){ uint32_t n=std::min<uint32_t>(r[1]-base,1u<<28); sweep<<<(n+255)/256,256>>>(base,n,mism,first,seen); hip(hipDeviceSynchronize()); base+=n; } }
    unsigned long long hm,hs,fb; uint32_t hf[16]; hip(hipMemcpy(&hm,mism,8,hipMemcpyDeviceToHost));hip(hipMemcpy(&hs,seen,8,hipMemcpyDeviceToHost));hip(hipMemcpy(hf,first,64,hipMemcpyDeviceToHost));
    hip(hipMemcpyFromSymbol(&fb,HIP_SYMBOL(r4d_tanh_ff::g_tanh_ff_fallbacks),8));
    std::printf("sweep |x|<=16: evaluated~%llu mismatches=%llu fp64_fallbacks=%llu (%.4f%%)\n",hs,hm,fb,100.0*fb/(double)std::max(1ull,hs));
    for(unsigned k=0;k<std::min<unsigned long long>(hm,16);++k){float x=__builtin_bit_cast(float,hf[k]);std::printf("  mismatch x=%a (%.9g)\n",x,x);}
    const int n=1<<25; std::vector<float> h(n); uint32_t s=12345; for(auto& v:h){s^=s<<13;s^=s>>17;s^=s<<5;v=((int)(s%2000001)-1000000)/250000.f;} // uniform [-4,4]
    float *din,*dout; hip(hipMalloc(&din,n*4));hip(hipMalloc(&dout,n*4));hip(hipMemcpy(din,h.data(),n*4,hipMemcpyHostToDevice));
    hipEvent_t a,b;hip(hipEventCreate(&a));hip(hipEventCreate(&b));
    auto time=[&](const char* name,auto launch){std::vector<float> t;for(int i=0;i<5;++i){hip(hipEventRecord(a));launch();hip(hipEventRecord(b));hip(hipEventSynchronize(b));float ms;hip(hipEventElapsedTime(&ms,a,b));if(i)t.push_back(ms);}std::sort(t.begin(),t.end());std::printf("%-22s %d values: median %.3f ms = %.2f G/s\n",name,n,t[1],n/t[1]/1e6);};
    hip(hipMemset(mism,0,8)); hip(hipMemcpyToSymbol(HIP_SYMBOL(r4d_tanh_ff::g_tanh_ff_fallbacks),mism,8));
    time("tanh_ff (float-float)",[&]{bench_ff<<<n/256,256>>>(din,dout,n);});
    hip(hipMemcpyFromSymbol(&fb,HIP_SYMBOL(r4d_tanh_ff::g_tanh_ff_fallbacks),8)); std::printf("  fallbacks over 4 runs of %d uniform[-4,4]: %llu (%.4f%% per value)\n",n,fb,100.0*fb/(4.0*n));
    time("float(tanh(double))",[&]{bench_ref<<<n/256,256>>>(din,dout,n);});
    time("tanhf (not exact)",[&]{bench_f<<<n/256,256>>>(din,dout,n);});
    return 0;}catch(const std::exception&e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
