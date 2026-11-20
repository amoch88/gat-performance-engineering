// CPU-only, inference-only GAT primitives. No fast-math, no substituted attention.
// EXPERIMENTAL CHANGE (layouts): logical indices are unchanged; zoff/eoff honor
// NHF/HNF and EH/HE physical storage in EVERY compiled producer/consumer.
// EXPERIMENTAL CHANGE (scheduling): row lists/partitions change thread ownership;
// arithmetic and each row's neighbor order are identical across partition policies.
#include <omp.h>
#include <cstdint>
#include <cmath>
#include <limits>

static inline int64_t zoff(int64_t n,int h,int f,int64_t N,int H,int F,int layout) {
  return layout ? (int64_t(h)*N+n)*F+f : (n*H+h)*F+f;
}
static inline int64_t eoff(int64_t e,int h,int64_t E,int H,int layout) {
  return layout ? int64_t(h)*E+e : e*H+h;
}
extern "C" {
int axes_team_size(int threads) {
  omp_set_dynamic(0); int result=0;
  #pragma omp parallel num_threads(threads)
  { if (omp_get_thread_num()==0) result=omp_get_num_threads(); }
  return result;
}
void axes_node(const float* Z,const float* al,const float* ar,float* L,float* R,
               int64_t N,int H,int F,int layout,int threads) {
  #pragma omp parallel for schedule(static) num_threads(threads)
  for (int64_t n=0;n<N;n++) for(int h=0;h<H;h++) {
    float l=0.f,r=0.f;
    for(int f=0;f<F;f++) {
      const float z=Z[zoff(n,h,f,N,H,F,layout)];
      l+=z*al[h*F+f];r+=z*ar[h*F+f];
    }
    L[n*H+h]=l;R[n*H+h]=r;
  }
}
void axes_add(const float* L,const float* R,const int64_t* src,const int64_t* dst,
              float* Q,int64_t E,int H,int layout,int threads) {
  // Actual additive GAT contributions, NOT Z_u dot Z_v / dot-product SDDMM.
  #pragma omp parallel for schedule(static) num_threads(threads)
  for(int64_t e=0;e<E;e++) for(int h=0;h<H;h++)
    Q[eoff(e,h,E,H,layout)]=L[src[e]*H+h]+R[dst[e]*H+h];
}
void axes_leaky(const float* Q,float* S,int64_t count,float slope,int threads) {
  // Separate pass, as in the reference: do not confound layout with score fusion.
  #pragma omp parallel for schedule(static) num_threads(threads)
  for(int64_t i=0;i<count;i++) S[i]=Q[i]>=0.f ? Q[i] : Q[i]*slope;
}
void axes_softmax(const int64_t* ptr,const int64_t* eid,const float* S,float* A,
                 int64_t N,int64_t E,int H,int layout,const int64_t* order,
                 const int64_t* parts,int threads,double* service_times) {
  #pragma omp parallel num_threads(threads)
  {
    int t=omp_get_thread_num();double start=service_times?omp_get_wtime():0.;
    for(int64_t k=parts[t];k<parts[t+1];k++) {
      const int64_t v=order[k],begin=ptr[v],end=ptr[v+1];
      for(int h=0;h<H;h++) {
        float maximum=-std::numeric_limits<float>::infinity(),sum=0.f;
        for(int64_t j=begin;j<end;j++) maximum=std::fmax(maximum,S[eoff(eid[j],h,E,H,layout)]);
        for(int64_t j=begin;j<end;j++) {
          int64_t pos=eoff(eid[j],h,E,H,layout);
          float value=std::exp(S[pos]-maximum);A[pos]=value;sum+=value;
        }
        for(int64_t j=begin;j<end;j++) A[eoff(eid[j],h,E,H,layout)]/=sum;
      }
    }
    if(service_times) service_times[t]=omp_get_wtime()-start;
  }
}
void axes_aggregate(const int64_t* ptr,const int64_t* src,const int64_t* eid,
                    const float* Z,const float* A,float* Y,int64_t N,int64_t E,
                    int H,int F,int zlayout,int elayout,const int64_t* order,
                    const int64_t* parts,int threads,double* service_times) {
  #pragma omp parallel num_threads(threads)
  {
    int t=omp_get_thread_num();double start=service_times?omp_get_wtime():0.;
    for(int64_t k=parts[t];k<parts[t+1];k++) {
      const int64_t v=order[k];
      for(int h=0;h<H;h++) for(int f=0;f<F;f++) Y[(v*H+h)*F+f]=0.f;
      // No atomics: exactly one thread owns every complete destination row.
      for(int64_t j=ptr[v];j<ptr[v+1];j++) for(int h=0;h<H;h++) {
        const float a=A[eoff(eid[j],h,E,H,elayout)];
        for(int f=0;f<F;f++) Y[(v*H+h)*F+f]+=a*Z[zoff(src[j],h,f,N,H,F,zlayout)];
      }
    }
    if(service_times) service_times[t]=omp_get_wtime()-start;
  }
}
}
