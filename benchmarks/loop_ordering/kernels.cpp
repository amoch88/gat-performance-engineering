// Serial FP32 GAT at the fixed Milestone 11 H=4, F=8. No packing, OMP or atomics.
// Each cell's floating-point reduction visits exactly the same elements in order.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

using index = std::int64_t;
constexpr int H = 4, F = 8;
constexpr float minus_inf = -std::numeric_limits<float>::infinity();
#define IN(T) const T* __restrict
#define OUT(T) T* __restrict

inline void terms_at(index n, int h, IN(float) z, IN(float) al, IN(float) ar,
                     OUT(float) left, OUT(float) right) {
    const index i = n*H+h;
    float l = 0.f, r = 0.f;
    for (int f=0; f<F; ++f) {
        const float x = z[i*F+f];
        l += x*al[h*F+f];
        r += x*ar[h*F+f];
    }
    left[i] = l;
    right[i] = r;
}

inline float activated(float left, float right, float slope) {
    const float q = left+right;
    return q >= 0.f ? q : q*slope;
}

inline void softmax_at(index begin, index end, int h, IN(index) eid, IN(float) s,
                       OUT(float) a, OUT(float) m, OUT(float) den) {
    m[h] = minus_inf;
    den[h] = 0.f;
    for (index j=begin; j<end; ++j) m[h] = std::fmax(m[h],s[eid[j]*H+h]);
    for (index j=begin; j<end; ++j) {
        const index i = eid[j]*H+h;
        const float x = std::exp(s[i]-m[h]);
        a[i] = x;
        den[h] += x;
    }
    for (index j=begin; j<end; ++j) a[eid[j]*H+h] /= den[h];
}

inline void accumulate_features(index v, index u, index e, int h, IN(float) z,
                                IN(float) a, OUT(float) y) {
    const float w = a[e*H+h];
    const index vi = (v*H+h)*F, ui = (u*H+h)*F;
    for (int f=0; f<F; ++f) y[vi+f] += w*z[ui+f];
}

extern "C" {

void loop_a1(IN(float) z, IN(float) al, IN(float) ar, OUT(float) l, OUT(float) r, index n) {
    for (index v=0; v<n; ++v) for (int h=0; h<H; ++h) terms_at(v,h,z,al,ar,l,r);
}
void loop_a2(IN(float) z, IN(float) al, IN(float) ar, OUT(float) l, OUT(float) r, index n) {
    for (int h=0; h<H; ++h) for (index v=0; v<n; ++v) terms_at(v,h,z,al,ar,l,r);
}

void loop_b1(IN(float) l, IN(float) r, IN(index) src, IN(index) dst, OUT(float) s,
             index edges, float slope) {
    for (index e=0; e<edges; ++e) {
        const index u = src[e]*H, v = dst[e]*H;
        for (int h=0; h<H; ++h) s[e*H+h] = activated(l[u+h],r[v+h],slope);
    }
}
void loop_b2(IN(float) l, IN(float) r, IN(index) src, IN(index) dst, OUT(float) s,
             index edges, float slope) {
    for (int h=0; h<H; ++h) for (index e=0; e<edges; ++e)
        s[e*H+h] = activated(l[src[e]*H+h],r[dst[e]*H+h],slope);
}

void loop_c1(IN(index) ptr, IN(index) eid, IN(float) s, OUT(float) a,
             OUT(float) scratch, index n) {
    for (index v=0; v<n; ++v) for (int h=0; h<H; ++h)
        softmax_at(ptr[v],ptr[v+1],h,eid,s,a,scratch,scratch+H);
}
void loop_c2(IN(index) ptr, IN(index) eid, IN(float) s, OUT(float) a,
             OUT(float) scratch, index n) {
    for (int h=0; h<H; ++h) for (index v=0; v<n; ++v)
        softmax_at(ptr[v],ptr[v+1],h,eid,s,a,scratch,scratch+H);
}
void loop_c3(IN(index) ptr, IN(index) eid, IN(float) s, OUT(float) a,
             OUT(float) scratch, index n) {
    float* m = scratch;
    float* den = scratch+H;
    for (index v=0; v<n; ++v) {
        std::fill(m,m+H,minus_inf);
        std::fill(den,den+H,0.f);
        for (index j=ptr[v]; j<ptr[v+1]; ++j) {
            const index i = eid[j]*H;
            for (int h=0; h<H; ++h) m[h] = std::fmax(m[h],s[i+h]);
        }
        for (index j=ptr[v]; j<ptr[v+1]; ++j) {
            const index i = eid[j]*H;
            for (int h=0; h<H; ++h) {
                const float x = std::exp(s[i+h]-m[h]);
                a[i+h] = x;
                den[h] += x;
            }
        }
        for (index j=ptr[v]; j<ptr[v+1]; ++j) {
            const index i = eid[j]*H;
            for (int h=0; h<H; ++h) a[i+h] /= den[h];
        }
    }
}

void loop_d1(IN(index) ptr, IN(index) src, IN(index) eid, IN(float) z, IN(float) a,
             OUT(float) y, index n) {
    std::fill(y,y+n*H*F,0.f);
    for (index v=0; v<n; ++v) for (index j=ptr[v]; j<ptr[v+1]; ++j) {
        const index u = src[j], e = eid[j];
        for (int h=0; h<H; ++h) accumulate_features(v,u,e,h,z,a,y);
    }
}
void loop_d2(IN(index) ptr, IN(index) src, IN(index) eid, IN(float) z, IN(float) a,
             OUT(float) y, index n) {
    std::fill(y,y+n*H*F,0.f);
    for (index v=0; v<n; ++v) for (int h=0; h<H; ++h)
        for (index j=ptr[v]; j<ptr[v+1]; ++j)
            accumulate_features(v,src[j],eid[j],h,z,a,y);
}
void loop_d3(IN(index) ptr, IN(index) src, IN(index) eid, IN(float) z, IN(float) a,
             OUT(float) y, index n) {
    std::fill(y,y+n*H*F,0.f);
    for (index v=0; v<n; ++v) for (int h=0; h<H; ++h) for (int f=0; f<F; ++f) {
        const index i = (v*H+h)*F+f;
        for (index j=ptr[v]; j<ptr[v+1]; ++j) y[i] += a[eid[j]*H+h]*z[(src[j]*H+h)*F+f];
    }
}
void loop_d4(IN(index) ptr, IN(index) src, IN(index) eid, IN(float) z, IN(float) a,
             OUT(float) y, index n) {
    std::fill(y,y+n*H*F,0.f);
    for (int h=0; h<H; ++h) for (index v=0; v<n; ++v)
        for (index j=ptr[v]; j<ptr[v+1]; ++j)
            accumulate_features(v,src[j],eid[j],h,z,a,y);
}

}
