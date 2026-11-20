// All variants share arithmetic and FP32 feature layout. Only traversal differs.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

using I = std::int32_t;
constexpr int H = 4, F = 8;

struct Graph {
    I n, e, block, tiles;
    const I *a, *b, *c, *d, *f, *eid;
};

template<int Kind, class Visit>
void edges(const Graph& g, Visit visit) {
    if constexpr (Kind == 0) {
        for (I i = 0; i < g.e; ++i) visit(g.a[i], g.b[i], i);
    } else if constexpr (Kind == 1 || Kind == 2) {
        for (I row = 0; row < g.n; ++row)
            for (I i = g.a[row]; i < g.a[row + 1]; ++i)
                if constexpr (Kind == 1) visit(row, g.b[i], g.eid[i]);
                else visit(g.b[i], row, g.eid[i]);
    } else if constexpr (Kind == 3 || Kind == 4) {
        const I blocks = g.n / g.block + (g.n % g.block != 0);
        for (I major = 0; major < blocks; ++major)
            for (I tile = g.a[major]; tile < g.a[major + 1]; ++tile)
                for (I row = 0; row < g.block; ++row) {
                    const I k = tile * g.block + row;
                    for (I i = g.d[k]; i < g.d[k + 1]; ++i) {
                        const I u = major * g.block + row;
                        const I v = g.b[tile] * g.block + g.f[i];
                        if constexpr (Kind == 3) visit(u, v, g.eid[i]);
                        else visit(v, u, g.eid[i]);
                    }
                }
    } else {
        for (I tile = 0; tile < g.tiles; ++tile) {
            const I u = g.b[tile] * g.block, v = g.c[tile] * g.block;
            for (I i = g.a[tile]; i < g.a[tile + 1]; ++i)
                visit(u + g.d[i], v + g.f[i], g.eid[i]);
        }
    }
}

template<int Kind>
void run(int op, const Graph& g, const float* x, const float* y,
         float slope, float* out, float* maxima, float* sums) {
    if (op == 0) {
        edges<Kind>(g, [&](I u, I v, I eid) {
            for (int h = 0; h < H; ++h) {
                const float s = x[int64_t(u)*H+h] + y[int64_t(v)*H+h];
                out[int64_t(eid)*H+h] = s >= 0 ? s : s*slope;
            }
        });
    } else if (op == 1) {
        // Global destination reductions across all tiles, never tile-local softmax.
        std::fill_n(maxima, int64_t(g.n)*H, -std::numeric_limits<float>::infinity());
        std::fill_n(sums, int64_t(g.n)*H, 0.f);
        edges<Kind>(g, [&](I, I v, I eid) {
            for (int h = 0; h < H; ++h) {
                auto& m = maxima[int64_t(v)*H+h];
                m = std::max(m, x[int64_t(eid)*H+h]);
            }
        });
        edges<Kind>(g, [&](I, I v, I eid) {
            for (int h = 0; h < H; ++h) {
                const auto k = int64_t(eid)*H+h, j = int64_t(v)*H+h;
                out[k] = std::exp(x[k] - maxima[j]);
                sums[j] += out[k];
            }
        });
        edges<Kind>(g, [&](I, I v, I eid) {
            for (int h = 0; h < H; ++h) out[int64_t(eid)*H+h] /= sums[int64_t(v)*H+h];
        });
    } else {
        std::fill_n(out, int64_t(g.n)*H*F, 0.f);
        edges<Kind>(g, [&](I u, I v, I eid) {
            for (int h = 0; h < H; ++h) {
                const float w = y[int64_t(eid)*H+h];
                for (int f = 0; f < F; ++f)
                    out[(int64_t(v)*H+h)*F+f] += w*x[(int64_t(u)*H+h)*F+f];
            }
        });
    }
}

extern "C" void sparse_run(int kind, int op, I n, I e, I block, I tiles,
                           const I* a, const I* b, const I* c, const I* d,
                           const I* f, const I* eid, const float* x,
                           const float* y, float slope, float* out,
                           float* maxima, float* sums) {
    const Graph g{n,e,block,tiles,a,b,c,d,f,eid};
    switch (kind) {
        case 0: run<0>(op,g,x,y,slope,out,maxima,sums); break;
        case 1: run<1>(op,g,x,y,slope,out,maxima,sums); break;
        case 2: run<2>(op,g,x,y,slope,out,maxima,sums); break;
        case 3: run<3>(op,g,x,y,slope,out,maxima,sums); break;
        case 4: run<4>(op,g,x,y,slope,out,maxima,sums); break;
        case 5: run<5>(op,g,x,y,slope,out,maxima,sums); break;
    }
}
