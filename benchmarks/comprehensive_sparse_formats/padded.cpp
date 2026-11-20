// The padded layouts use the Milestone 6 arithmetic and canonical EID outputs.
// Traverse only occupied dense cells or valid sliced ELLPACK lanes.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

using I = std::int32_t;
using L = std::int64_t;
constexpr int H = 4, F = 8;

struct Layout {
    I size, groups, extras;
    const I *a, *b;
    const void* c;
    const I *d, *parallel_edges;
};

template<bool Sell, class Visit>
void visit_edges(const Layout& g, Visit visit) {
    if constexpr (Sell) {
        const auto* eids = static_cast<const I*>(g.c);
        for (I slice = 0; slice < g.groups; ++slice)
            for (I offset = g.a[slice]; offset < g.a[slice+1]; offset += g.size)
                for (I lane = 0; lane < g.size; ++lane) {
                    const I p = offset + lane;
                    if (eids[p] >= 0) visit(g.b[p], slice*g.size+lane, eids[p]);
                }
    } else {
        const auto* masks = static_cast<const std::uint64_t*>(g.c);
        for (I tile = 0; tile < g.groups; ++tile)
            for (I row = 0; row < g.size; ++row) {
                const L offset = L(tile)*g.size+row;
                for (std::uint64_t bits = masks[offset]; bits; bits &= bits-1) {
                    const I col = __builtin_ctzll(bits);
                    visit(g.a[tile]*g.size+row,g.b[tile]*g.size+col,g.d[offset*g.size+col]);
                }
            }
        for (I i = 0; i < g.extras; ++i) {
            const I* edge = g.parallel_edges+L(i)*3;
            visit(edge[0],edge[1],edge[2]);
        }
    }
}

template<bool Sell>
void run(I op, I n, const Layout& g, const float* x, const float* y,
         float slope, float* output, float* maxima, float* sums) {
    if (op == 0) {
        visit_edges<Sell>(g,[&](I u,I v,I e) {
            for (int h=0;h<H;++h) {
                const float z = x[L(u)*H+h]+y[L(v)*H+h];
                output[L(e)*H+h] = z >= 0 ? z : z*slope;
            }
        });
    } else if (op == 1) {
        std::fill_n(maxima,L(n)*H,-std::numeric_limits<float>::infinity());
        std::fill_n(sums,L(n)*H,0.f);
        visit_edges<Sell>(g,[&](I,I v,I e) {
            for (int h=0;h<H;++h) {
                auto& max = maxima[L(v)*H+h];
                max = std::max(max,x[L(e)*H+h]);
            }
        });
        visit_edges<Sell>(g,[&](I,I v,I e) {
            for (int h=0;h<H;++h) {
                const L k=L(e)*H+h,j=L(v)*H+h;
                output[k]=std::exp(x[k]-maxima[j]);
                sums[j]+=output[k];
            }
        });
        visit_edges<Sell>(g,[&](I,I v,I e) {
            for (int h=0;h<H;++h) output[L(e)*H+h]/=sums[L(v)*H+h];
        });
    } else {
        std::fill_n(output,L(n)*H*F,0.f);
        visit_edges<Sell>(g,[&](I u,I v,I e) {
            for (int h=0;h<H;++h) {
                const float a=y[L(e)*H+h];
                for (int f=0;f<F;++f)
                    output[(L(v)*H+h)*F+f]+=a*x[(L(u)*H+h)*F+f];
            }
        });
    }
}

extern "C" void padded_run(I sell,I op,I n,I size,I groups,I extras,
                           const I* a,const I* b,const void* c,const I* d,const I* sidecar,
                           const float* x,const float* y,float slope,
                           float* output,float* maxima,float* sums) {
    const Layout g{size,groups,extras,a,b,c,d,sidecar};
    if (sell) run<true>(op,n,g,x,y,slope,output,maxima,sums);
    else run<false>(op,n,g,x,y,slope,output,maxima,sums);
}
