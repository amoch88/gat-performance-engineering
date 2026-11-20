// Standalone Memcheck fixture for the sparse-storage and loop-nest kernels.
// Deliberately avoids Python, Torch, and their unrelated allocator/runtime paths.
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

using I = std::int32_t;
using J = std::int64_t;
using U = std::uint64_t;
constexpr int N=4, E=5, H=4, F=8;
constexpr float slope=.2f;

extern "C" {
void sparse_run(int,int,I,I,I,I,const I*,const I*,const I*,const I*,const I*,const I*,
                const float*,const float*,float,float*,float*,float*);
void padded_run(I,I,I,I,I,I,const I*,const I*,const void*,const I*,const I*,
                const float*,const float*,float,float*,float*,float*);
}
#define A(name) extern "C" void name(const float*,const float*,const float*,float*,float*,J)
#define B(name) extern "C" void name(const float*,const float*,const J*,const J*,float*,J,float)
#define C(name) extern "C" void name(const J*,const J*,const float*,float*,float*,J)
#define D(name) extern "C" void name(const J*,const J*,const J*,const float*,const float*,float*,J)
A(loop_a1); A(loop_a2);
B(loop_b1); B(loop_b2);
C(loop_c1); C(loop_c2); C(loop_c3);
D(loop_d1); D(loop_d2); D(loop_d3); D(loop_d4);

template<std::size_t S>
void close(const std::array<float,S>& output,const std::array<float,S>& reference) {
    for (std::size_t k=0;k<S;++k)
        assert(std::isfinite(output[k]) && std::abs(output[k]-reference[k]) <=
               1e-5f + 1e-4f*std::abs(reference[k]));
}

int main() {
    // Parallel 3->1 records, self-loops, destination 3 has zero incoming edges.
    const I u[]={3,0,3,2,0}, v[]={1,1,1,2,0};
    const J u64[]={3,0,3,2,0}, v64[]={1,1,1,2,0};
    const I csc_ptr[]={0,1,4,5,5}, csc_src[]={0,3,0,3,2}, csc_eid[]={4,0,1,2,3};
    const J ptr64[]={0,1,4,5,5}, src64[]={0,3,0,3,2}, eid64[]={4,0,1,2,3};
    std::array<float,N*H*F> z{}, expected_y{}, output_y;
    std::array<float,H*F> al{}, ar{};
    std::array<float,N*H> l{},r{},left,right,maxima,sums;
    std::array<float,E*H> score{}, alpha{}, output_edge;
    std::array<float,2*H> scratch;
    for (int k=0;k<N*H*F;++k) z[k]=(k%13-6)*.25f;
    for (int k=0;k<H*F;++k) { al[k]=(k%7-3)*.2f; ar[k]=(k%3-1)*.6f; }
    for (int n=0;n<N;++n) for (int h=0;h<H;++h) {
        double a=0,b=0;
        for (int f=0;f<F;++f) { a+=double(z[(n*H+h)*F+f])*al[h*F+f]; b+=double(z[(n*H+h)*F+f])*ar[h*F+f]; }
        l[n*H+h]=float(a); r[n*H+h]=float(b);
    }
    for (int e=0;e<E;++e) for (int h=0;h<H;++h) {
        const float s=l[u[e]*H+h]+r[v[e]*H+h];
        score[e*H+h]=s>=0?s:s*slope;
    }
    for (int n=0;n<N;++n) for (int h=0;h<H;++h) {
        double m=-std::numeric_limits<double>::infinity(), denom=0;
        for (int e=0;e<E;++e) if (v[e]==n) m=std::max(m,double(score[e*H+h]));
        for (int e=0;e<E;++e) if (v[e]==n) denom+=std::exp(score[e*H+h]-m);
        for (int e=0;e<E;++e) if (v[e]==n) alpha[e*H+h]=float(std::exp(score[e*H+h]-m)/denom);
    }
    for (int e=0;e<E;++e) for (int h=0;h<H;++h) for (int f=0;f<F;++f)
        expected_y[(v[e]*H+h)*F+f]+=alpha[e*H+h]*z[(u[e]*H+h)*F+f];

    const float poison=std::numeric_limits<float>::quiet_NaN();
    for (auto fn : {loop_a1,loop_a2}) {
        left.fill(poison); right.fill(poison);
        fn(z.data(),al.data(),ar.data(),left.data(),right.data(),N);
        close(left,l); close(right,r);
    }
    for (auto fn : {loop_b1,loop_b2}) {
        output_edge.fill(poison);
        fn(l.data(),r.data(),u64,v64,output_edge.data(),E,slope);
        close(output_edge,score);
    }
    for (auto fn : {loop_c1,loop_c2,loop_c3}) {
        output_edge.fill(poison);
        fn(ptr64,eid64,score.data(),output_edge.data(),scratch.data(),N);
        close(output_edge,alpha);
    }
    for (auto fn : {loop_d1,loop_d2,loop_d3,loop_d4}) {
        output_y.fill(poison);
        fn(ptr64,src64,eid64,z.data(),alpha.data(),output_y.data(),N);
        close(output_y,expected_y);
    }

    auto sparse = [&](int kind,int block,int tiles,const I*a,const I*b,const I*c,const I*d,const I*f,const I*eid) {
        output_edge.fill(poison);
        sparse_run(kind,0,N,E,block,tiles,a,b,c,d,f,eid,l.data(),r.data(),slope,output_edge.data(),nullptr,nullptr);
        close(output_edge,score);
        output_edge.fill(poison);
        sparse_run(kind,1,N,E,block,tiles,a,b,c,d,f,eid,score.data(),nullptr,slope,output_edge.data(),maxima.data(),sums.data());
        close(output_edge,alpha);
        output_y.fill(poison);
        sparse_run(kind,2,N,E,block,tiles,a,b,c,d,f,eid,z.data(),alpha.data(),slope,output_y.data(),nullptr,nullptr);
        close(output_y,expected_y);
    };
    const I csr_ptr[]={0,2,2,3,5}, csr_dst[]={1,0,2,1,1}, csr_eid[]={1,4,3,0,2};
    const I bc_a[]={0,1,3}, bc_b[]={0,0,1}, bc_d[]={0,2,2,2,4,5,5};
    const I trans_a[]={0,2,3}, trans_b[]={0,1,1}, trans_d[]={0,1,2,2,4,5,5};
    const I trans_local[]={0,0,1,1,0}, trans_eid[]={4,1,0,2,3};
    const I tile_ptr[]={0,2,4,5}, tile_u[]={0,1,1}, tile_v[]={0,0,1};
    const I tile_rows[]={0,0,1,1,0}, tile_cols[]={1,0,1,1,0}, tile_eid[]={1,4,0,2,3};
    sparse(0,0,0,u,v,nullptr,nullptr,nullptr,nullptr);
    sparse(1,0,0,csr_ptr,csr_dst,nullptr,nullptr,nullptr,csr_eid);
    sparse(2,0,0,csc_ptr,csc_src,nullptr,nullptr,nullptr,csc_eid);
    sparse(3,2,3,bc_a,bc_b,nullptr,bc_d,tile_cols,tile_eid);
    sparse(4,2,3,trans_a,trans_b,nullptr,trans_d,trans_local,trans_eid);
    sparse(5,2,3,tile_ptr,tile_u,tile_v,tile_rows,tile_cols,tile_eid);

    auto padded = [&](int sell,int groups,int extras,const I*a,const I*b,const void*c,const I*d,const I*sidecar) {
        output_edge.fill(poison);
        padded_run(sell,0,N,2,groups,extras,a,b,c,d,sidecar,l.data(),r.data(),slope,output_edge.data(),nullptr,nullptr);
        close(output_edge,score);
        output_edge.fill(poison);
        padded_run(sell,1,N,2,groups,extras,a,b,c,d,sidecar,score.data(),nullptr,slope,output_edge.data(),maxima.data(),sums.data());
        close(output_edge,alpha);
        output_y.fill(poison);
        padded_run(sell,2,N,2,groups,extras,a,b,c,d,sidecar,z.data(),alpha.data(),slope,output_y.data(),nullptr,nullptr);
        close(output_y,expected_y);
    };
    const U masks[]={3,0,0,2,1,0};
    const I dense_eid[]={4,1,-1,-1,-1,-1,-1,0,3,-1,-1,-1}, duplicates[]={3,1,2};
    const I slice_ptr[]={0,6,8}, slots_src[]={0,3,-1,0,-1,3,2,-1}, slots_eid[]={4,0,-1,1,-1,2,3,-1};
    padded(0,3,1,tile_u,tile_v,masks,dense_eid,duplicates);
    padded(1,2,0,slice_ptr,slots_src,slots_eid,nullptr,nullptr);
    std::puts("Native fixture passed: 11 loop orders, 6 sparse traversals, 2 padded traversals");
}
