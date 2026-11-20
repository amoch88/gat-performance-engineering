# Graph Attention Network Performance Engineering

A reproducible framework for evaluating sparse storage formats, graph reordering strategies, and loop-nest optimizations in CPU Graph Attention Networks.

This technical analysis documents the experimental design, measured results, and limits of the evidence.
Measured results come from the preserved research snapshot
`9659684473a0531aa06e40f614e1fbfbbcfc5b76` of
[amoch88/dgl-performance-model](https://github.com/amoch88/dgl-performance-model).
See [Experimental Findings](EXPERIMENTAL_FINDINGS.md) for 25 exact contrasts, including the
numbers, controls, confidence intervals, correction families, interpretation,
and source selectors. The tables below are generated from the committed
analysis and raw JSON by [build_experiment_evidence.py](../tools/build_experiment_evidence.py).
Data from different acquisitions are deliberately not pooled.

**Research question.** Which storage formats, graph orders, and loop nests
improve CPU GAT, and do gains in individual kernels survive in a complete
forward? The storage evaluation and matched-backend controls, reproducible
PubMed edge-order result, and PubMed D2 isolated/in-forward/whole-layer gap
illustrate why both measurement boundaries matter. The evidence supports a
narrowed set of choices for future modeling.
It does not establish a completed automatic tensor cost model or a combined
storage × ordering × loop-order speedup.

## Background and motivation

### What a GAT layer computes

Graph Attention Networks assign data-dependent weights to neighbors and
then reduce weighted features. This study times a single homogeneous,
inference-only, multi-head layer, not a trained multi-layer classifier.
Let $N$ be the number of nodes, $E$ the number of **directed edge records**
after preprocessing, $I$ the input feature count, $H=4$ the attention-head
count, and $F=8$ the channels per head. For canonical edge ID $e$, write
$u_e=\mathrm{src}[e]$ and $v_e=\mathrm{dst}[e]$; a directed edge goes from
$u_e$ to $v_e$. The incoming group is
$\mathcal I(v)=\{e:v_e=v\}$. Parallel edges are distinct members of this
group, and a self-loop is just another record. There is no edge sampling.

1. **Projection.** $X\in\mathbb R^{N\times I}$ and, in mathematical orientation,
   $W\in\mathbb R^{I\times(HF)}$. Define
   $Z=\operatorname{reshape}_{N,H,F}(XW)$. PyTorch stores the linear
   weight transposed relative to this notation. Its output is viewed as
   $[N,H,F]$ without duplicating the projection.
2. **Factorized attention terms.** The logical parameter arrays
   $a_L,a_R\in\mathbb R^{H\times F}$ generate
   $L_{u,h}=\sum_f Z_{u,h,f}a_{L,h,f}$ and
   $R_{v,h}=\sum_f Z_{v,h,f}a_{R,h,f}$. Both terms have logical size
   $[N,H]$. A DGL boundary may expose a singleton trailing dimension.
   The split is equivalent to applying a linear functional to the
   concatenated source and destination feature vectors, with the
   nonlinearity applied afterward.
3. **Edge score.** $S_{e,h}=\operatorname{LeakyReLU}_{0.2}(L_{u_e,h}+R_{v_e,h})$,
   logically $[E,H]$. This is additive attention and **not** dot-product QKV
   attention. The stock DGL path uses built-in source-plus-destination
   edge computation; the custom sparse and loop implementations use the
   same activation either fused or separate as documented for each.
4. **Stable incoming softmax.** For nonempty $\mathcal I(v)$,
   $m_{v,h}=\max_{e\in\mathcal I(v)}S_{e,h}$,
   $d_{v,h}=\sum_{e\in\mathcal I(v)}\exp(S_{e,h}-m_{v,h})$, and
   $\alpha_{e,h}=\exp(S_{e,h}-m_{v_e,h})/d_{v_e,h}$.
   The maximum and denominator span **all** edges into that destination,
   even when storage divides them among tiles. Padding contributes to
   neither. Each active destination/head sums to one within tolerance.
5. **Aggregation and output.** $Y_{v,h,f}=\sum_{e\in\mathcal I(v)}
   \alpha_{e,h}Z_{u_e,h,f}$. The result is
   $O_{v,h,f}=Y_{v,h,f}+b_{h,f}$ with learned-parameter *shapes* preserved.
   Bias is broadcast across nodes. There is no residual connection or
   extra activation. Output shape is $[N,H,F]$, not $[N,H]$ or final
   class logits. Bias/output is its own timed region in the timing-validation, reordering, and loop-nest experiments, which is
   why those reports display six instrumented regions for five conceptual
   stages.

These are randomly initialized, reproducible seed-0 **model weights**,
not pretrained weights or accuracy results. Initialization is outside
timing and the exact tensors are fingerprinted. There is no optimization
of training loss or claim about predictive quality.

### Why the representation and schedule matter

Projection performs dense work on the order of $NIHF$ arithmetic
operations. The two attention terms scale as $NHF$, edge scoring and
softmax as $EH$, and aggregation as $EHF$. These are operation-count
scalings, not a calibrated cost law. Sparse kernels repeatedly read
endpoint IDs or row pointers, gather nonadjacent features, traverse
irregular incoming groups, and write edge- or node-indexed outputs.
Degree skew changes the lengths of reductions and the usefulness of a
fixed schedule. A hardware implementation can spend more time moving
or shuffling values than doing the mathematical add and multiply.

- **Storage format** changes which indices are explicit, which coordinates
  follow from position, how nonempty blocks are enumerated, and how much
  padding or scratch exists. DGL dispatch limits which native format is
  legal for each primitive. A layout useful for a reduction may be poor
  for producing EID-ordered scores.
- **Ordering** preserves the set of edge records and all feature values
  but relabels nodes or changes the edge sequence. This can change the
  physical order of feature reads, where EID-indexed rows land, and
  sparse traversal locality. In particular, stable destination grouping
  can change the CSC-to-current-EID map without changing the neighbors
  traversed in the CSC adjacency.
- **Loop order** changes repeated index visits, accumulator lifetime,
  stride, SIMD opportunities, and access to adjacent features and heads
  with all logical and physical representations held fixed. A compiler
  can produce different code from equivalent sources with the same
  flags. Loop-Nest Optimization checks the actual compiled definitions.

A single-stage gain is upper-bounded in its impact by the stage's share
of total runtime under a simplifying no-interaction model. If fraction
$f$ of the complete pass alone sped up by $s$, the conditional Amdahl
prediction would be $1/((1-f)+f/s)$. Here $f$ must come from a comparable
execution context; isolated stage medians cannot simply be summed or
divided into a total, and the assumptions are not always met. Allocator
behavior, input freshness, and upstream traversal can also change the
context of a downstream stage. Report the independently measured
complete-forward latency, then use stages as diagnostics.

### Original and delivered scope

The original research goal was to represent measured costs over workload,
kernel, and implementation-choice axes, learn or interpolate missing
costs, model cross-stage dependencies, and select a compatible
configuration. This project delivers **experiment design, exact
implementations, stock-backed validation, measured cost evidence, and
independent replay**, not a fitted tensor or an automatic dispatch system.
The repository retains an earlier exploratory joint DGL graph
node/edge/format experiment in [joint_tuning](../benchmarks/joint_tuning/REPORT.md).
That does not establish a tested combination of custom storage, the winning
edge order, and a custom loop nest. It is a separate exploratory study, not a
completed tuner.

## Related work and references

These references give algorithm and implementation context. Speedups
reported for other kernels or hardware are not evidence for these
experiments. Titles and publication details were checked against
publication or author records; the archived reports also cite the
implementation sources used in acquisition.

1. Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero,
   Pietro Liò, and Yoshua Bengio. 2018. **Graph Attention Networks.**
   ICLR. [OpenReview conference paper](https://openreview.net/forum?id=rJXMpikCZ).
   Define learned attention over neighbors and multi-head attention.
2. Minjie Wang, Da Zheng, Zihao Ye, Quan Gan, Mufei Li, Xiang Song,
   Jinjing Zhou, Chao Ma, Lingfan Yu, Yu Gai, Tianjun Xiao, Tong He,
   George Karypis, Jinyang Li, and Zheng Zhang. 2019 (v2 2020).
   **Deep Graph Library: A Graph-Centric, Highly-Performant Package for
   Graph Neural Networks.** arXiv:1909.01315.
   [Preprint](https://arxiv.org/abs/1909.01315).
   DGL's generalized sparse primitives motivate the score and reduction
   decomposition. For exact implementation behavior, cite the
   [v1.1.3 GATConv source](https://github.com/dmlc/dgl/blob/v1.1.3/python/dgl/nn/pytorch/conv/gatconv.py)
   and [v1.1.3 CPU dispatch](https://github.com/dmlc/dgl/blob/v1.1.3/src/array/kernel.cc).
   The locally preserved [annotated copy](../benchmarks/dgl_baseline/baseline.py)
   is checked against the installed class AST but is not the main timing
   oracle for the reported experiments.
3. Zhilin Yang, William Cohen, and Ruslan Salakhutdinov. 2016.
   **Revisiting Semi-Supervised Learning with Graph Embeddings.**
   ICML, PMLR 48, 40–48. [Proceedings](https://proceedings.mlr.press/v48/yanga16.html).
   Citation-graph benchmark context. Use recorded DGL loader counts
   as authoritative for the measured graph, not counts taken from a
   paper with another convention.
4. Aydin Buluç, Jeremy T. Fineman, Matteo Frigo, John R. Gilbert, and
   Charles E. Leiserson. 2009. **Parallel sparse matrix-vector and
   matrix-transpose-vector multiplication using compressed sparse
   blocks.** SPAA, 233–244.
   [DOI: 10.1145/1583991.1584053](https://doi.org/10.1145/1583991.1584053).
   Cite for compressed sparse blocks; this project's serial CSB
   GAT traversals are not the original parallel SpMV algorithm.
5. Moritz Kreutzer, Georg Hager, Gerhard Wellein, Holger Fehske, and
   Alan R. Bishop. 2014. **A Unified Sparse Matrix Data Format for
   Efficient General Sparse Matrix-Vector Multiplication on Modern
   Processors with Wide SIMD Units.** SIAM Journal on Scientific
   Computing 36(5), C401–C423.
   [DOI: 10.1137/130930352](https://doi.org/10.1137/130930352).
   Cite for sliced ELLPACK and padding/vectorization tradeoffs. Sparse Storage
   uses fixed original row order, not an independent SELL sorting-window
   experiment.
6. E. Cuthill and J. McKee. 1969. **Reducing the bandwidth of sparse
   symmetric matrices.** 24th ACM National Conference, 157–172.
   [DOI: 10.1145/800195.805928](https://doi.org/10.1145/800195.805928).
   The Graph Reordering RCMK policy uses SciPy's reverse Cuthill–McKee on a symmetric
   structural adjacency; it does not synthesize a new mathematical graph.
7. George Karypis and Vipin Kumar. 1998. **A Fast and High Quality
   Multilevel Scheme for Partitioning Irregular Graphs.** SIAM Journal
   on Scientific Computing 20(1), 359–392.
   [DOI: 10.1137/S1064827595287997](https://doi.org/10.1137/S1064827595287997).
   Cite for graph partitioning. The extension uses DGL's installed METIS
   API once per graph and replays saved assignments.
8. Michael E. Wolf and Monica S. Lam. 1991. **A data locality optimizing
   algorithm.** PLDI, 30–44.
   [DOI: 10.1145/113445.113449](https://doi.org/10.1145/113445.113449).
   General compiler locality and loop transformation background, not a
   claim that this sparse benchmark runs the published optimizer.
9. Samuel Williams, Andrew Waterman, and David Patterson. 2009.
   **Roofline: An Insightful Visual Performance Model for Multicore
   Architectures.** Communications of the ACM 52(4), 65–76.
   [DOI: 10.1145/1498765.1498785](https://doi.org/10.1145/1498765.1498785).
   Use as modeling context only. Cache traffic and operational intensity
   were not measured by the three principal acquisitions.
10. Tamara G. Kolda and Brett W. Bader. 2009. **Tensor Decompositions
    and Applications.** SIAM Review 51(3), 455–500.
    [DOI: 10.1137/07070111X](https://doi.org/10.1137/07070111X).
    Background for proposed low-rank cost modeling, not an implemented
    result in this study.
11. Sture Holm. 1979. **A Simple Sequentially Rejective Multiple Test
    Procedure.** Scandinavian Journal of Statistics 6(2), 65–70.
    [JSTOR 4615733](https://www.jstor.org/stable/4615733).
    Reference for familywise adjustment, which is separate from the
    pointwise confidence intervals.

## Experimental environment and analysis

### Recorded controls

- Hardware: **AMD Ryzen 7 7800X3D 8-Core Processor**, 16 logical CPUs
  visible in the recorded metadata. Every principal acquisition pins
  one worker to guest logical CPU **0** and uses one computational
  thread. There is no host physical-core/frequency isolation guarantee.
- Environment: CPU Docker on
  `Linux-6.18.58.1-microsoft-standard-WSL2-x86_64-with-glibc2.36`;
  Python **3.11.11**, PyTorch **2.2.2+cpu**, DGL **1.1.3**,
  NumPy **1.26.4**. The pinned lock uses SciPy **1.12.0** and
  uv **0.13.4**. The preserved Dockerfile pins the linux/amd64
  Python and uv images by digest. The compiler image adds GNU C++
  and binutils. These are benchmark environments, not workstation
  Windows Python. Verification smoke runs do not replace the
  archived performance measurements.
- PyTorch intra-op and inter-op threads both one. The workload setup
  also sets DGL threads to one. Recorded environment includes
  `OMP_NUM_THREADS=1`, `OMP_DYNAMIC=FALSE`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `OMP_WAIT_POLICY=PASSIVE`,
  `KMP_BLOCKTIME=0`; `OMP_PROC_BIND` and `OMP_PLACES` unset.
  Affinity is recorded and validated per process. Stock GATConv is
  evaluated with gradients off, dropout zero, bias enabled, no
  residual, and no output activation.
- Native arrays are canonical EID ordered. Sparse Storage DGL indices are int64;
  custom Sparse Storage coordinate/EID data are int32 (occupancy bits are
  uint64). Graph Reordering uses native DGL and enables/materializes COO, CSR, and
  CSC identically for every policy. Loop-Nest Optimization fixes int64 COO for scores and
  int64 incoming CSC for softmax/aggregation; all variants retain
  contiguous FP32 NHF, NH and EH. The layouts are controlled variables
  in Loop-Nest Optimization, not additional tunable axes.
- Sparse Storage native extensions were built with
  `g++ (Debian 12.2.0-14+deb12u1) 12.2.0` and
  `-std=c++17 -O3 -fPIC -shared -fopenmp -ffp-contract=off`.
  Graph Reordering changes graph data via DGL/PyTorch APIs, no new timed kernels.
  Loop-Nest Optimization uses the same GNU C++ version with
  `-std=c++17 -O3 -fPIC -shared -ffp-contract=off -fno-loop-interchange -fno-loop-unroll-and-jam`.
  No fast math and no linked OpenMP runtime in the Loop-Nest Optimization custom library.
  Recorded build hashes, symbol offsets, and normalized disassembly
  correspond to the actual loaded ELF. The dry-run harness below is
  a different diagnostic build and should not be cited as timed code.

### Acquisition protocol

<!-- evidence:protocol:start -->

| Acquisition | Fresh processes per graph | Warmup blocks | Measured blocks | Build observations | Order seed |
| --- | --- | --- | --- | --- | --- |
| Sparse Storage Format Evaluation | 8 | 10 | 30 | 5 | 2026111800 |
| Sparse Backend and Timing Validation | 8 | 10 | 30 | 5 | 2026111900 |
| Graph Reordering (matched allocation) | 8 | 10 | 30 | 5 | 2026112000 |
| Graph Reordering (extended sweeps) | 8 | 10 | 30 | 5 | 2026112100 |
| Loop-Nest Optimization | 8 | 10 | 30 | not swept | 2026112200 |

<!-- evidence:protocol:end -->

The process is the statistical replicate. A round starts a fresh
process with the same graph and model fingerprints. Timing blocks
interleave all alternatives and their reference in randomized order
within a sweep/phase; the schedule is saved and replayed. Retain raw
warmups and observations but omit warmups from statistics.
Loading, verification, graph/format construction, and metadata
serialization are outside the repeated forward. Runtime tensor
allocations in a forward are inside its timer. Sparse Storage separately randomizes
forward and isolated-stage phases, and five format-construction samples.
the reordering and loop-nest experiments distinguish direct execution from an instrumented copy of the
same forward with six stage timing regions. Loop-Nest Optimization also interleaves an
isolated target call and a bare allocator diagnostic using the same
allocation factories. These are separate measured calls, not
alternative definitions of total forward.

Latency is the median across eight within-process medians. For a
matched candidate $a$ and baseline $b$, let $r_i=\log(T_{b,i}/T_{a,i})$
for process $i$. The paired estimate is $\exp(\overline r)$.
The pointwise interval is $\exp(\overline r\;\pm\;
t_{0.975,7}s_r/\sqrt{8})$. Each two-sided test uses the process log
ratios, then the planned family receives Holm adjustment. The raw
ratio of two reported medians is a distinct descriptive statistic.
Pointwise 95% intervals are not simultaneous intervals; an interval
excluding one can coexist with a Holm-adjusted p-value above .05.
Ranking many candidates by the same sample is descriptive, not a
held-out guarantee.

Sparse Storage adjusts separately per sparse endpoint or whole forward, across
datasets, native controls, and the planned within-family size
comparisons. A format in its stronger selection table must beat
untouched GATConv **and all three native hybrids** after correction.
The graph reordering study adjusts its primary tests separately by node/edge sweep and endpoint,
across datasets and candidates including random controls; its
extension also has nine secondary forward contrasts to
Destination-grouped. Loop-Nest Optimization adjusts separately by sweep and endpoint
(isolated, in-forward target, whole-forward), across all three
datasets and alternatives in that sweep. Graph Reordering and Loop-Nest Optimization require both
adjusted p < .05 and a confidence lower speedup bound **strictly
above 1.01×** for practical qualification. A nonsignificant finding is
**inconclusive**, not proof of equal performance.

## Datasets and setup

<!-- evidence:datasets:start -->

| Dataset | N | Loader directed E | Timed directed E | Input width | Timed E/N | Max total degree | DGL class |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Cora | 2,708 | 10,556 | 13,264 | 1,433 | 4.898 | 338 | CoraGraphDataset |
| CiteSeer | 3,327 | 9,228 | 12,431 | 3,703 | 3.736 | 200 | CiteseerGraphDataset |
| PubMed | 19,717 | 88,651 | 108,365 | 500 | 5.496 | 344 | PubmedGraphDataset |

<!-- evidence:datasets:end -->

The counts above are **directed DGL records**, not unique undirected
citation pairs. “Loader directed E” is recorded immediately after
loading. “Timed directed E” is the count after
`dgl.add_self_loop(dgl.remove_self_loop(graph))`: exactly one loop per
node. Dataset objects are constructed with `reorder=False`.
Features are the original loader's feature rows made contiguous FP32;
no width changes, training-label input, extra features, or feature
training. The baseline loading code removes unused node and edge
metadata and caches DGL's sparse views. The benchmark intentionally
preserves edge multiplicity and tests parallel edges on fixtures even
though the measured citation sets have no duplicates beyond the first
edge with a given directed endpoint pair in the ordering diagnostics.
The preprocessed edges, features and initialized weight hashes match
across all five acquired result sets above and the older seed-0 baseline
checks. Different timing runs do not mean different graph inputs.

Cora and CiteSeer are small enough that a few hundred microseconds of
sparse time can be masked by dense projection; CiteSeer's input width
is substantially larger than Cora's. PubMed has many more vertices and
edge records, stronger incoming-degree effects, and a narrower input
than either. The larger sparse workload makes changes in softmax and
aggregation visible in the total on that graph. This is a useful
contrast, but three citation graphs do not independently vary size,
degree skew, directionality, and feature width. They cannot establish
which single graph statistic causally predicts the best configuration.

The public DGL data classes retrieve dataset archives from
`data.dgl.ai/dataset` into an ignored cache. Exact loader identities
and workload fingerprints are checked when running the supplied
acquisition scripts. Benchmarks are inference-only and report no
accuracy, training-time, or end-to-end data-ingestion metric.

## Sparse Storage Format Evaluation

### Study design and layouts

The stock row is the unmodified `dgl.nn.pytorch.GATConv`. A native hybrid
row chooses one preferred layout for the sparse operations that support
it, and uses a verified, prebuilt DGL fallback for the unsupported ones.
A custom row uses its selected representation across score,
destination softmax, and aggregation. This is a *whole sparse
representation choice*, not three independently substituted kernels in
the same forward. The native-versus-blocked study contains related one-substitution controls.
Production timing does not include preparing the representation. Every
custom sparse output that has an edge dimension is indexed by original EID,
so no undocumented edge-ID conversion is hidden between kernels.

| Storage | What it stores and how it traverses |
|---|---|
| Native COO | Two endpoint arrays, one relation entry per EID in the COO view. Scores use native generalized SDDMM; aggregation uses native generalized SpMM. CSC fallback for softmax. |
| Native CSR | Source-row pointers, destination indices and entry-to-EID mapping. Native score via GSDDMM on CSR. CSC fallbacks for both incoming softmax and aggregation. |
| Native CSC | Destination-column pointers, source indices and entry-to-EID mapping. Native incoming softmax and aggregation. COO fallback for score. |
| Blocked CSR | A major block-row pointer indexes nonempty source-block/destination-block tiles. Each tile has local source-row pointers, local destination indices and original EIDs. Walk source block, tile, local source row, actual edge. |
| Blocked CSC | Transpose the major/minor roles of the blocked CSR structure. A major destination block indexes source-block tiles with per-tile local destination-row pointers. Each entry retains the local source and its EID. Traversal visits only actual entries. |
| CSB | Nonempty 2D tiles with explicit global tile coordinates, a tile pointer to packed local source and destination coordinate arrays, and original EIDs. No allocated dense grid of unused cells. |
| Block Dense | Every nonempty square tile has a dense row-major int32 EID grid. A uint64 occupancy mask per tile row gates reads. The first EID occupies its coordinate; a sidecar of (src,dst,EID) preserves every parallel edge beyond the first. Padding is never treated as an edge. |
| SELL | Consecutive original destination rows are partitioned into slices. Each slice has width equal to its maximum incoming degree. Positions are stored by within-row edge ordinal with interleaved row lanes. Source/EID arrays use a negative sentinel in unused lanes. No row sorting or silent coalescing. |

The custom families sweep square block sizes 2, 4, 8, 16, 32, 64; SELL
sweeps the same numbers as **row-slice heights**, not square tile sizes.
These are 30 custom choices plus three native hybrids and one stock
reference on each graph. The custom score reads $L_u,R_v$ and writes
LeakyReLU output at the edge's canonical location. Custom softmax scans
valid entries three times: maximum over all destination/head entries,
exponential and denominator accumulation, then normalization. It
maintains global $[N,H]$ max/sum scratch, so an incoming group split across
tiles is still one reduction. Aggregation zero-initializes the destination
output and visits each valid edge weighting the corresponding projected
source row. Block Dense traverses set mask bits and then the sidecar in
all three passes; SELL examines padded lanes and skips invalid EIDs.
The builder independently decodes storage back to canonical endpoints
and verifies exactly one visit per EID.

Only supported **isolated** native cells receive a number. The N/A
cells in [the dispatch notes](../benchmarks/comprehensive_sparse_formats/NA_EXPLANATIONS.md)
do not make hybrid forwards invalid and are not missing samples:
CPU destination softmax asks for CSC; GSDDMM supports COO or CSR; GSpMM
supports COO or CSC. There is no extrapolated time for an unsupported
combination. Design and storage code are
[`benchmark.py`](../benchmarks/comprehensive_sparse_formats/benchmark.py),
[`padded.py`](../benchmarks/comprehensive_sparse_formats/padded.py),
[`padded.cpp`](../benchmarks/comprehensive_sparse_formats/padded.cpp),
[`storage.py`](../benchmarks/sparse_formats/storage.py), and
[`kernels.cpp`](../benchmarks/sparse_formats/kernels.cpp).

### Direct forward observations

<!-- evidence:storage-forward:start -->

| Dataset | Observed winner | Stock ms | Candidate ms | Raw median ratio | Paired speedup | 95% CI | Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Cora | CSB 64×64 | 3.904638 | 3.353436 | 1.16437× | 1.15397× | [1.13584, 1.17239] | 5.8799e-05 |
| CiteSeer | CSB 2×2 | 10.007215 | 9.125802 | 1.09658× | 1.08408× | [1.06245, 1.10614] | 0.0095558 |
| PubMed | SELL 2 | 16.553568 | 14.344164 | 1.15403× | 1.14637× | [1.12324, 1.16997] | 0.00041784 |

<!-- evidence:storage-forward:end -->

The following is the full grid of direct complete-forward medians from
the same acquisition. These values are not a sum of stage cells.

<!-- evidence:storage-grid:start -->

| Implementation | Cora ms | CiteSeer ms | PubMed ms |
| --- | --- | --- | --- |
| Imported DGL GATConv | 3.904638 | 10.007215 | 16.553568 |
| Native COO hybrid | 4.197399 | 9.974854 | 20.055999 |
| Native CSR hybrid | 3.730390 | 9.742386 | 16.278996 |
| Native CSC hybrid | 3.680334 | 9.542912 | 16.100883 |
| Blocked CSR 2×2 | 3.693070 | 9.389166 | 16.465251 |
| Blocked CSR 4×4 | 3.770164 | 9.675932 | 17.253464 |
| Blocked CSR 8×8 | 3.858937 | 9.737176 | 18.275347 |
| Blocked CSR 16×16 | 4.079333 | 9.961793 | 20.741055 |
| Blocked CSR 32×32 | 4.193959 | 10.016387 | 24.587283 |
| Blocked CSR 64×64 | 3.994791 | 10.106813 | 28.704397 |
| Blocked CSC 2×2 | 3.657140 | 9.568428 | 16.481699 |
| Blocked CSC 4×4 | 3.787562 | 9.510600 | 17.380802 |
| Blocked CSC 8×8 | 3.898833 | 9.720939 | 18.363943 |
| Blocked CSC 16×16 | 4.035204 | 9.795138 | 20.825731 |
| Blocked CSC 32×32 | 4.182981 | 10.189232 | 25.181526 |
| Blocked CSC 64×64 | 3.977546 | 9.959783 | 29.264057 |
| CSB 2×2 | 3.401421 | 9.125802 | 14.397993 |
| CSB 4×4 | 3.419682 | 9.169380 | 14.507040 |
| CSB 8×8 | 3.383374 | 9.215504 | 14.447152 |
| CSB 16×16 | 3.408135 | 9.302397 | 14.566944 |
| CSB 32×32 | 3.409411 | 9.186633 | 14.359960 |
| CSB 64×64 | 3.353436 | 9.260628 | 14.692041 |
| Block Dense 2×2 | 3.635789 | 9.446644 | 16.073784 |
| Block Dense 4×4 | 3.738104 | 9.500669 | 17.187417 |
| Block Dense 8×8 | 3.990004 | 9.788507 | 19.855970 |
| Block Dense 16×16 | 4.403125 | 10.363761 | 28.103751 |
| Block Dense 32×32 | 4.758974 | 10.775529 | 37.960748 |
| Block Dense 64×64 | 4.341195 | 10.520167 | 44.460476 |
| SELL 2 | 3.406188 | 9.282819 | 14.344164 |
| SELL 4 | 3.404848 | 9.236375 | 14.657094 |
| SELL 8 | 3.463965 | 9.365690 | 15.260661 |
| SELL 16 | 3.508989 | 9.454563 | 15.830215 |
| SELL 32 | 3.563250 | 9.507414 | 16.160204 |
| SELL 64 | 3.583723 | 9.456531 | 16.542315 |

<!-- evidence:storage-grid:end -->

The highest confidence in these rows is a paired comparison against a
specified control, not an inference that the lowest-looking size beats
all alternatives. The stronger storage evaluation decision is that all four
stock/native comparisons favor a candidate after Holm correction:

<!-- evidence:storage-eligibility:start -->

| Dataset | Number beating all four controls | Best qualifying minimum over controls | Limiting control by speedup | Paired speedup | 95% CI | Holm p |
| --- | --- | --- | --- | --- | --- | --- |
| Cora | 10 | CSB 64×64 | Native CSC hybrid | 1.09689× | [1.07414, 1.12012] | 0.0054961 |
| CiteSeer | 1 | CSB 16×16 | Native CSC hybrid | 1.03260× | [1.02236, 1.04295] | 0.034099 |
| PubMed | 8 | SELL 2 | Native CSC hybrid | 1.11820× | [1.09497, 1.14193] | 0.0017719 |

<!-- evidence:storage-eligibility:end -->

On CiteSeer, **CSB 2×2 is fastest by raw median and beats stock**
but is inconclusive against the strongest native comparisons after
adjustment. The only custom row passing the four-control gate is
**CSB 16×16**. Its minimum paired ratio is against native CSC. The
original size analysis did not establish that CSB 64×64 in Cora or
CSB 2×2 in CiteSeer outruns other CSB sizes: all 45 Sparse Storage whole-forward
within-CSB size comparisons across the three datasets were inconclusive
after correction. On PubMed, SELL 2 beat SELL 8, 16, 32, and 64
after correction. This is better evidence for avoiding large slices
on that graph than simply pointing to the smallest median.

### Where the time changed

<!-- evidence:storage-sparse:start -->

| Dataset | Isolated stage | Fastest observed | ms | Native comparator | ms | Paired speedup | 95% CI | Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cora | score | Native COO hybrid | 0.123544 | Native COO hybrid | 0.123544 | — | — | — |
| Cora | softmax | CSB 64×64 | 0.222386 | Native CSC hybrid | 0.419633 | 1.88369× | [1.86992, 1.89756] | 7.1759e-12 |
| Cora | aggregation | CSB 64×64 | 0.154329 | Native CSC hybrid | 0.424710 | 2.75741× | [2.71098, 2.80463] | 9.5998e-11 |
| CiteSeer | score | Native COO hybrid | 0.124607 | Native COO hybrid | 0.124607 | — | — | — |
| CiteSeer | softmax | CSB 64×64 | 0.214544 | Native CSC hybrid | 0.428719 | 2.00504× | [1.98360, 2.02671] | 5.255e-11 |
| CiteSeer | aggregation | CSB 64×64 | 0.148746 | Native CSC hybrid | 0.423186 | 2.85186× | [2.79554, 2.90931] | 2.2578e-10 |
| PubMed | score | Native COO hybrid | 0.755625 | Native COO hybrid | 0.755625 | — | — | — |
| PubMed | softmax | CSB 8×8 | 2.024847 | Native CSC hybrid | 3.319920 | 1.62964× | [1.60705, 1.65255] | 3.2395e-09 |
| PubMed | aggregation | CSB 4×4 | 1.551239 | Native CSC hybrid | 3.689200 | 2.34281× | [2.28911, 2.39776] | 2.506e-09 |

<!-- evidence:storage-sparse:end -->

These sparse values are **isolated** timings on prepared tensors.
Projection and terms are independently timed common controls in Sparse Storage,
not per-row full-pass regions. Native COO is fastest by observed score
median on all three. Strong isolated gains appear in the custom softmax
and aggregation against native CSC; the best sparse row often differs
from the best whole-forward row. Examples include CiteSeer's 2.852×
paired isolated aggregation gain for CSB 64×64, versus a 1.084× whole
forward speedup for a **different** CSB 2×2 implementation. Do not
combine those two ratios into a single result. Small sparse stages
can be heavily diluted by dense projection. All three sparse stages
also change together in a custom row, so the net forward may reflect
producer and consumer locality, allocator behavior, and the different
cost of scoring, not just savings in aggregate.

### Memory, construction, and important regressions

<!-- evidence:storage-storage:start -->

| Dataset | Format | Index KiB | Build ms | Padded slots per original edge |
| --- | --- | --- | --- | --- |
| Cora | Blocked CSC 2×2 | 238.51 | 0.975341 | — |
| Cora | Blocked CSC 64×64 | 549.41 | 1.122876 | — |
| Cora | CSB 2×2 | 285.03 | 1.179787 | — |
| Cora | CSB 64×64 | 176.01 | 0.890895 | — |
| Cora | Block Dense 2×2 | 431.95 | 1.497074 | 3.33 |
| Cora | Block Dense 64×64 | 28971.21 | 2.415100 | 541.95 |
| Cora | SELL 2 | 140.64 | 0.664818 | 1.31 |
| Cora | SELL 64 | 589.67 | 0.666414 | 5.69 |
| CiteSeer | Blocked CSC 2×2 | 224.73 | 0.892956 | — |
| CiteSeer | Blocked CSC 64×64 | 734.12 | 1.102368 | — |
| CiteSeer | CSB 2×2 | 266.78 | 1.106750 | — |
| CiteSeer | CSB 64×64 | 175.07 | 0.872650 | — |
| CiteSeer | Block Dense 2×2 | 403.67 | 1.392451 | 3.33 |
| CiteSeer | Block Dense 64×64 | 41401.59 | 4.555802 | 826.38 |
| CiteSeer | SELL 2 | 133.75 | 0.627841 | 1.31 |
| CiteSeer | SELL 64 | 485.71 | 0.637728 | 5.00 |
| PubMed | Blocked CSC 2×2 | 2035.12 | 8.502544 | — |
| PubMed | Blocked CSC 64×64 | 14740.30 | 15.356504 | — |
| PubMed | CSB 2×2 | 2419.90 | 11.557964 | — |
| PubMed | CSB 64×64 | 1911.10 | 9.712395 | — |
| PubMed | Block Dense 2×2 | 3833.32 | 14.305479 | 3.62 |
| PubMed | Block Dense 64×64 | 903224.96 | 91.366750 | 2068.13 |
| PubMed | SELL 2 | 1290.14 | 6.156717 | 1.48 |
| PubMed | SELL 64 | 6077.21 | 6.161192 | 7.18 |

<!-- evidence:storage-storage:end -->

“Index KiB” counts exposed representation arrays, masks, sentinel
slots and any sidecars, not transient Python building memory, allocator
metadata, process RSS, or live runtime feature tensors. Native setup
starts from fresh COO and converts a restricted DGL view. Custom setup
starts from canonical arrays and builds a representation. Report these
as separate one-time medians rather than including them in every
forward. Use `ANALYSIS.json.storage` for all sizes. At acquisition there
were 1 GiB per-layout and 2 GiB combined-custom-payload guards; none of
the three datasets hit those guards. More memory is needed during
construction than these static array counts alone imply.

At PubMed 64×64 Block Dense reserves about 882.06 MiB of index
payload (about 2068 allocated tile slots per edge), needs 91.367 ms
to construct, and took 44.460 ms per direct forward. Its paired
speedup relative to stock was about 0.375×, a supported slowdown.
The 2×2 variant uses about 3.74 MiB and its 16.074 ms forward was
inconclusive versus stock after correction. In contrast, SELL2
on PubMed allocates about 1.48 slots per edge and SELL64 about 7.18;
large slices let one high-degree row increase the width of all
rows in that slice. SELL2 versus SELL64 is the stronger within-family
contrast (claims C10). These results quantify repeated latency and
payload tradeoffs. They do not measure cache misses or attribute
every difference to padding. Tile sizes 32 and 64 are not interchangeable:
the number of nonempty tiles also changes, so monotonic latency is
not a rule.

### Matched backend, storage, and timing validation

<!-- evidence:timing-checks:start -->

| Dataset | Candidate | ms | Comparator | ms | Paired speedup | 95% CI | Holm p | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cora | Custom COO matched hybrid | 3.332054 | Native COO hybrid | 4.178306 | 1.25440× | [1.23957, 1.26940] | 6.215e-09 | Supported gain |
| PubMed | SELL 2 | 14.581988 | Custom COO | 13.374672 | 0.90788× | [0.88055, 0.93605] | 0.0079761 | Supported slowdown |

<!-- evidence:timing-checks:end -->

Timing Validation reran selected CSB/SELL configurations and added route-matched custom
and native controls plus custom unblocked COO/CSR/CSC. Its fastest
observed forward was custom COO on all three (3.288, 9.433, and
13.375 ms). All nine route-matched custom-versus-native forward
comparisons favored the custom implementation after correction. No
tested CSB or SELL configuration established a corrected whole-forward
gain over the custom unblocked controls. PubMed SELL2 even lost to
custom COO, while its stock-relative win replicated. This shows why
backend, 32- versus 64-bit indices, activation fusion, dispatch, and
traversal can confound comparisons to native DGL. A good Sparse Storage code path
need not imply a general blocking benefit.

Sparse Storage's isolated CiteSeer projection median was 9.923 ms and the CSB2
complete-forward median was 9.126 ms; the projection median exceeded
the forward median in all eight process rounds. That does **not** mean
projection was timed inside the corresponding 9.126 ms call.
Timing Validation interleaved isolated, in-forward and direct calls and measured
timer calibration. It obtained 8.700 ms isolated projection,
8.574 ms in-forward projection inside the 9.529 ms instrumented
CSB2 execution, and 9.437 ms direct CSB2 forward. Net empty-timer
cost was about 1.71 µs on CiteSeer, much smaller than the phase
discrepancy. Context probes and allocator/cache differences make
execution context a credible explanation; the report does not
identify an exact hardware cause or “correct” the original dataset.
Keep both acquisitions and label the measurement boundary.

## Graph Reordering Experiments

### What is and is not permuted

The two sweeps independently change node or edge order within unchanged
stock DGL GATConv. Graph creation receives canonical-EID endpoint
arrays for the node sweep, with node labels remapped via the inverse
permutation. Every node-associated value is indexed by the new-to-original
node map. Nodes' logical identities are restored for validation. Even
the Original node control allocates feature rows with the **same**
indexing operation; the record order of edges never changes in this
sweep. In the edge sweep, node labels and the exact feature tensor are
unchanged. Endpoint array order and EID-associated values follow the
new-to-original edge permutation. Attention is restored by EID before
comparison. Both sweeps have an Original row and compare only within
the same interleaved sweep. Absolute timings between the two Original
rows need not match because their execution contexts differ.

| Node policy | Exact definition | Present in |
|---|---|---|
| Original | Identity node labels and row-copy control. | Both |
| RCMK | SciPy reverse Cuthill–McKee on symmetric, unweighted structural adjacency; parallel edges do not weight the skeleton. | Both |
| Degree ascending | Stable increasing total in-plus-out degree, original node ID ties. | Both |
| Degree descending | Stable decreasing total in-plus-out degree, original node ID ties. | Both |
| BFS | Use symmetric adjacency, smallest unseen original ID as component root, visit neighbors in ascending original ID; include disconnected components. | Both |
| Random | Seeded NumPy permutations, three per graph, held fixed across process rounds. Diagnostic only. | Both |
| METIS | DGL native k-way 32 partitions, cut objective on a symmetric auxiliary graph; stable assignment-ID order with original node ID ties. Freeze assignments before timing and replay exact map. | Extension |

| Edge policy | Exact definition | Present in |
|---|---|---|
| Original | Identity EID permutation. | Both |
| Source-grouped | Stable single-key sort by source with original EID ties. | Both |
| Destination-grouped | Stable single-key sort by destination with original EID ties. | Both |
| Random | Three seeded EID permutations per graph, fixed across process rounds and reused for shared policies. Diagnostic only. | Both |
| Source then destination | Sort by (source, destination, original EID); changes within-source traversal beyond Source-grouped. | Extension |
| Destination then source | Sort by (destination, source, original EID); changes within-destination traversal beyond Destination-grouped. | Extension |
| Morton | Sort uint64 Z-order keys with source bit $i$ at $2i$, destination bit $i$ at $2i+1$, then original EID. Supports 32 coordinate bits each. | Extension |

The archival first acquisition reused the loader's feature tensor only
for Original in the node sweep, while alternate node orders allocated
new features. It is informative as a sensitivity check, not the main
Graph Reordering experiment. When the entire experiment was repeated using matched
allocation, significant deterministic projection gains decreased
from 11/12 to 3/12. Other time variation between acquisitions is
possible. Use the **matched** original report and the extension; the
extension has a fresh schedule and more comparisons, even though it
preserves permutations of shared policies. The exact saved permutations,
METIS assignments, graph state hashes and raw observations are
preserved in the corresponding directories.

### Requested complete-forward contrasts

<!-- evidence:reordering-forward:start -->

| Dataset | Acquisition | Axis | Candidate | Original ms | Candidate ms | Paired speedup | 95% CI | Holm p | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cora | Matched | node | Degree descending | 4.293674 | 4.228765 | 1.02100× | [1.00259, 1.03975] | 0.27541 | Inconclusive after Holm |
| Cora | Matched | edge | Destination-grouped | 3.816230 | 3.816565 | 1.00516× | [0.99199, 1.01850] | 1 | Inconclusive after Holm |
| Cora | Extension | node | Degree descending | 4.250858 | 4.097715 | 1.04264× | [1.01909, 1.06674] | 0.059023 | Inconclusive after Holm |
| Cora | Extension | edge | Destination-grouped | 3.797769 | 3.755902 | 1.00803× | [0.99927, 1.01687] | 0.87121 | Inconclusive after Holm |
| Cora | Extension | node | METIS (32) | 4.250858 | 4.170966 | 1.02915× | [1.00778, 1.05098] | 0.20021 | Inconclusive after Holm |
| Cora | Extension | edge | Destination then source | 3.797769 | 3.769888 | 1.00528× | [0.99229, 1.01844] | 1 | Inconclusive after Holm |
| CiteSeer | Matched | node | Degree descending | 10.755670 | 10.618062 | 1.01684× | [1.00807, 1.02568] | 0.033869 | Supported gain below practical gate |
| CiteSeer | Matched | edge | Destination-grouped | 9.999194 | 10.066579 | 1.00128× | [0.98520, 1.01762] | 1 | Inconclusive after Holm |
| CiteSeer | Extension | node | Degree ascending | 10.743577 | 10.519621 | 1.01694× | [1.00317, 1.03090] | 0.22502 | Inconclusive after Holm |
| CiteSeer | Extension | edge | Destination-grouped | 9.839732 | 9.792291 | 1.00452× | [0.98055, 1.02907] | 1 | Inconclusive after Holm |
| CiteSeer | Extension | node | METIS (32) | 10.743577 | 10.588574 | 1.01058× | [1.00123, 1.02003] | 0.22626 | Inconclusive after Holm |
| CiteSeer | Extension | edge | Destination then source | 9.839732 | 9.877807 | 0.99773× | [0.98177, 1.01395] | 1 | Inconclusive after Holm |
| PubMed | Matched | node | Degree descending | 17.428137 | 17.100453 | 1.01737× | [1.00558, 1.02930] | 0.11083 | Inconclusive after Holm |
| PubMed | Matched | edge | Destination-grouped | 16.395252 | 15.727640 | 1.05043× | [1.04266, 1.05826] | 9.385e-06 | Meets practical gate |
| PubMed | Extension | node | RCMK | 17.162624 | 16.907439 | 1.01068× | [1.00237, 1.01905] | 0.22502 | Inconclusive after Holm |
| PubMed | Extension | edge | Destination-grouped | 15.864936 | 15.428297 | 1.02800× | [1.01517, 1.04101] | 0.020119 | Meets practical gate |
| PubMed | Extension | node | METIS (32) | 17.162624 | 16.908630 | 1.01112× | [0.99885, 1.02353] | 0.22626 | Inconclusive after Holm |
| PubMed | Extension | edge | Destination then source | 15.864936 | 15.269934 | 1.03648× | [1.02624, 1.04682] | 0.0010826 | Meets practical gate |

<!-- evidence:reordering-forward:end -->

Each “best observed node” row is the lowest forward median among
nonrandom alternate node policies in that acquisition; it is not
proof that the order beats the original. METIS is shown as its own
extension contrast. Destination grouping qualified on **PubMed in both
independent campaigns**, with paired speedup 1.0504× in the matched-allocation run and
1.0280× in the seven-policy extension. These estimates need not be equal;
we did not test a change in effect between the acquisitions. The extension
found Destination-then-source also qualifies versus Original on PubMed.
No Cora or CiteSeer edge policy met the extension's practical threshold.
Keep Original and the two destination-oriented options as candidates.
Source grouping, Source-then-destination, and Morton did not qualify;
random remains a diagnostic. Neither node campaign produced evidence for
retaining node order under the practical gate. The original matched
run had small corrected node gains such as CiteSeer degree descending
and PubMed BFS, with interval lower bounds below 1.01×. METIS can
improve a stage (PubMed aggregation about 1.0767×) without a supported
whole-forward win. Rankings changed among some shared policies between
campaigns; that is not a reason to pool the timings or assume a
universal preferred degree or bandwidth policy.

<!-- evidence:reordering-secondary:start -->

| Dataset | Destination-grouped ms | Destination then source ms | Paired speedup | 95% CI | Holm p | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| Cora | 3.755902 | 3.769888 | 0.99727× | [0.98284, 1.01191] | 1 | Inconclusive after Holm |
| CiteSeer | 9.792291 | 9.877807 | 0.99324× | [0.98268, 1.00391] | 1 | Inconclusive after Holm |
| PubMed | 15.428297 | 15.269934 | 1.00825× | [0.99333, 1.02338] | 1 | Inconclusive after Holm |

<!-- evidence:reordering-secondary:end -->

**Do not say lexicographic destination order is faster than stable
destination grouping.** It is fastest by observed PubMed median, but
the head-to-head interval covers one and Holm p = 1. This separate
secondary family does not establish equivalence either. Both
variants can be considered when tuning a specific graph, with the
original-style stable grouping as the simpler proven option.

### One-time costs and storage visibility

<!-- evidence:reordering-overhead:start -->

| Dataset (extension) | Policy | Plan / replay ms | Rebuild ms | Formats ms |
| --- | --- | --- | --- | --- |
| Cora | METIS (32) | 0.009910 | 1.980347 | 0.062725 |
| Cora | Destination-grouped | 0.414584 | 0.199040 | 0.061909 |
| Cora | Destination then source | 0.482942 | 0.206341 | 0.059941 |
| CiteSeer | METIS (32) | 0.010596 | 21.295174 | 0.097940 |
| CiteSeer | Destination-grouped | 0.378835 | 0.217709 | 0.062877 |
| CiteSeer | Destination then source | 0.461739 | 0.214984 | 0.064469 |
| PubMed | METIS (32) | 0.039929 | 19.580216 | 0.851147 |
| PubMed | Destination-grouped | 4.304770 | 0.904030 | 0.622849 |
| PubMed | Destination then source | 5.077314 | 0.867458 | 0.650545 |

<!-- evidence:reordering-overhead:end -->

Planning and feature/endpoint remapping plus graph reconstruction
and native-format materialization are timed separately. Forward
latencies exclude all three and exclude restoring outputs to
canonical node IDs. METIS replay planning in the table excludes the
one-shot DGL partitioner. The native assignment took 3.545 ms on Cora,
2.673 ms on CiteSeer, and 17.324 ms on PubMed, before a stable sort.
All 32 groups were nonempty; the Python API has no METIS seed parameter.
Saved assignments and source fingerprints address portability of
the plan, not guaranteed identical partition generation across builds.

The audit hashes index/pointer arrays, offsets to current EIDs, and
the traversal expressed in original EIDs separately. Stable
Destination-grouped leaves CSC pointers, neighbor indices and original-EID
traversal unchanged on all three; it changes the map to current EIDs.
Stable Source-grouped does that for CSR. Both lexicographic options and
Morton change the compressed traversal data and mappings in the measured
graphs. Node reorder changes node labels and therefore physical index
arrays while preserving canonical edge sequence; output is in the
current node order until restored. Separately measured bandwidth and
endpoint-step statistics describe adjacency locality, not cache
counters. PubMed Destination-grouped improved both softmax and
aggregation in the corrected **in-forward** stage comparisons. These are
not isolated calls in Graph Reordering. The observed pair of stage gains
supports a sparse-access explanation, but no measurement isolates a
unique hardware mechanism.

Random controls are not tested as recommended configurations. Each
reported random median summarizes three fixed permutations across
processes, and the paired interval conditions on that seed set. The
seed-specific table in each report distinguishes seed-to-seed timing
variation from within-seed process variation.

## Loop-Nest Optimization

### Fixed implementation controls

Public DGL 1.1.3 GATConv has no argument for exchanging these loop
nests. All treatment comparisons use serial C++ kernels in one
implementation family with fixed coo endpoints for scoring and CSC
incoming rows for normalization and aggregation, canonical EIDs,
native graph ordering, and stable tensor layout. The same Python
pipeline performs projection, degree guard and bias. The untouched
imported GATConv is a separate numerical and timing reference.
Comparing custom C3 or D2 directly with stock as though only its loop
order changed would be invalid.

`[N,H,F]` projected features and aggregate are contiguous with
feature stride 1, head stride $F$, node stride $HF$. Terms have
contiguous $[N,H]$ storage and scores/coefficients use contiguous EH
with head stride 1. COO arrays and CSC pointer/source/EID arrays
are int64. The CSC entry $j$ has source `in_src[j]` and original EID
`eid[j]`. The loader's COO/CSC conversion is outside the repeated
forward. No kernel duplicates, coalesces, reshapes or reorders graph
records as a substitute for changing a loop nest.

### Exact implemented variants

| ID | Precise nesting | Inner arithmetic and access consequence |
|---|---|---|
| A1 | node → head → feature | Accumulate L and R together using the same feature order. Natural contiguous scan over the next head and next node. |
| A2 | head → node → feature | Same pair of FP32 reductions. Fixed-head feature runs are separated by 32 floats across node rows. |
| B1 | canonical COO edge → head | Read endpoints once per edge and compute consecutive EH entries. Reuse the endpoints across the four contiguous heads. |
| B2 | head → canonical COO edge | Same lookup and fused LeakyReLU, but walk the edge list once per head with strided EH output. |
| C1 | destination → head → incoming edge | Run the complete three-pass softmax for each destination/head, reading its full CSC row per pass. |
| C2 | head → destination → incoming edge | Same three per-destination/head passes, but visit all destinations for a fixed head first. |
| C3 | destination → incoming edge → head, in each of three passes | Reuse each EID for adjacent heads. Coordinate per-head maxima, exponentials/sums, then normalizations for the entire row. |
| D1 | destination → incoming edge → head → feature | Reuse original EID and source across heads; update all head/feature values for the current edge. |
| D2 | destination → head → incoming edge → feature | Work on a contiguous feature vector for one destination/head across its incoming edges. Revisit the index row for each head. |
| D3 | destination → head → feature → incoming edge | Finish one feature's incoming-edge reduction at a time. Revisit the row for each head/feature. No change to the sequence of floating additions for a given output cell. |
| D4 | head → destination → incoming edge → feature | Similar small feature-vector accumulation to D2 but complete all destinations for one head first. |

For C3, the important pseudocode is:

```text
for v in original destination order:
    maximum[0:H] = -infinity; denominator[0:H] = 0
    for j in CSC row(v): for h: maximum[h] = max(maximum[h], S[eid[j],h])
    for j in CSC row(v): for h:
        alpha[eid[j],h] = exp(S[eid[j],h] - maximum[h])
        denominator[h] += alpha[eid[j],h]
    for j in CSC row(v): for h: alpha[eid[j],h] /= denominator[h]
```

C1 and C2 take the same three steps for one head at a time; C3
reorders arithmetic **between independent heads**. Each
destination/head visits incoming EIDs in exactly the same order in
every pass. All three allocate the same $2H$ FP32 scratch (32 bytes)
and the same EH output. The logical EID visits per pass are $HE$
for C1/C2 and $E$ for C3. C3 does not do chunk-local normalization.
An empty CSC row writes no alpha entries. All aggregation variants
allocate the same NHF output and zero it inside the timed kernel,
then preserve incoming-edge FP32 accumulation order for every
output cell. Their logical source/EID visits are $E$, $HE$, $HFE$,
and $HE$ for D1–D4. These are source-level traversal counts, not
measured cache traffic. Compilers can keep values in registers.

Every case binds a named `extern "C"` specialization to a fixed ctypes
function address during preparation. Forward uses that specialization,
not a runtime switch over a generic wrapper. The acquisition stores
the binary hash, compiled address offset, required libraries and
normalized instruction hash for each named order, and validates
which symbol each measured route uses. No symbol addresses or
normalized instruction bodies matched within a sweep. The profiler
copies the executable forward body by AST instrumentation rather than
writing a subtly different arithmetic path.

One-factor treatment definitions use A1/B1/C1/D1 as the shared
default and replace exactly one of the four. Each sweep measures
its own concurrently interleaved default, so the defaults have the
same operations but separately observed times. The isolated call
receives held default intermediate tensors with fresh output/scratch
allocation; in-forward regions see inputs produced earlier in the
same instrumented pass. A bare factory diagnostic checks allocation
outlays without touching all pages or zero-filling; it is not
subtracted from any headline number. The direct complete pass has
no stage timer.

### Measured results

<!-- evidence:loop-nest-forward:start -->

| Dataset | Sweep | Candidate | Default | Default ms | Candidate ms | Paired speedup | 95% CI | Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cora | aggregation | D2 | D1 | 3.256207 | 3.161590 | 1.03143× | [1.01507, 1.04805] | 0.012754 |
| PubMed | softmax | C3 | C1 | 13.421983 | 12.974664 | 1.02641× | [1.01461, 1.03834] | 0.0065265 |
| PubMed | aggregation | D2 | D1 | 13.294048 | 12.405656 | 1.07613× | [1.06121, 1.09126] | 4.5251e-05 |
| PubMed | aggregation | D4 | D1 | 13.294048 | 12.686727 | 1.05424× | [1.03280, 1.07613] | 0.0035121 |

<!-- evidence:loop-nest-forward:end -->

These are **all** the practical whole-forward gains by the prespecified
rule. No CiteSeer alternative qualifies. An order can have a favorable
median without a supported practical gain. The complete Loop-Nest Optimization report
contains all stage medians and every paired result, including slower
orders and estimates below the practical threshold.

<!-- evidence:loop-nest-ranks:start -->

| Dataset | Sweep | Fastest isolated | Fastest in-forward | Fastest forward |
| --- | --- | --- | --- | --- |
| Cora | terms | A1 | A1 | A1 |
| Cora | score | B1 | B1 | B1 |
| Cora | softmax | C3 | C3 | C1 |
| Cora | aggregation | D2 | D2 | D2 |
| CiteSeer | terms | A1 | A1 | A2 |
| CiteSeer | score | B1 | B1 | B2 |
| CiteSeer | softmax | C3 | C3 | C2 |
| CiteSeer | aggregation | D4 | D2 | D2 |
| PubMed | terms | A1 | A1 | A1 |
| PubMed | score | B1 | B1 | B1 |
| PubMed | softmax | C3 | C3 | C3 |
| PubMed | aggregation | D2 | D2 | D2 |

<!-- evidence:loop-nest-ranks:end -->

The isolated and in-forward rank agrees for most cases. A1 and B1
are fastest for their target stage on all three graphs, so keep them
fixed. C3 improves isolated and in-forward softmax on all three after
correction, but only PubMed has a qualified whole-layer effect. On
Cora C3 gives about 1.091× isolated speedup, with no established
whole-forward improvement. D2 and D4 each improve isolated and
in-forward aggregation on all three after correction, and D3
loses in the target region. D2 has the lowest in-forward aggregation
median on every graph, while D4 narrowly leads the isolated median
on CiteSeer. D2's CiteSeer whole-forward contrast is inconclusive.
Campaign recommendations are C1/C3 and D1/D2/D4, with the
understanding that D4 did not have a tested advantage over D2 and
the combined choices were not run. Retain a default alongside
workload-dependent alternatives rather than promising a single
improving choice for all new graphs.

### PubMed aggregation: the core isolated-versus-forward result

<!-- evidence:loop-nest-pubmed:start -->

| Endpoint | D1 med. ms | D2 med. ms | Paired speedup | 95% CI | Holm p |
| --- | --- | --- | --- | --- | --- |
| isolated | 2.151045 | 1.085058 | 1.97696× | [1.66393, 2.34888] | 9.9819e-05 |
| in_forward | 1.475163 | 0.616297 | 2.34610× | [2.25198, 2.44416] | 2.2358e-09 |
| forward | 13.294048 | 12.405656 | 1.07613× | [1.06121, 1.09126] | 4.5251e-05 |

<!-- evidence:loop-nest-pubmed:end -->

The paired D2 gain is roughly 1.98× in isolation and 2.35× for the
instrumented aggregation stage, but only 1.076× for the uninstrumented
complete layer. These are separate matched measurements. One cannot
subtract two independent medians to isolate an exact stage fraction.
Both direct and instrumented calls include D2's output initialization,
and all other kernels run unchanged. The report checks the rebuilt,
byte-identical ELF. D1's per-edge inner loop emits cross-head
shuffles, stack spills and stores for the output. The D2 loop keeps
two feature vectors in registers for the entire incoming-edge stream
of a head and stores afterward; D4 has a similar feature-vector
inner loop with a different outer scan. D3's scalar add chain
traverses the incoming list for every feature. This explains why
fewer logical index visits alone did not predict the winner. It
does not measure the hardware cost of individual instructions or
prove the whole timing gap was caused by cache behavior.

Projection and the other sparse stages consume most of the complete
layer, so removing a large fraction of the aggregation latency
only removes a fraction of total runtime. The changed isolation
and in-forward magnitudes also show that prepared operands and
fresh intermediates are not interchangeable timing contexts.
Measure both to diagnose; decide from direct forward. On CiteSeer
projection dominates, and even the supported D2/D4 aggregation
improvements do not translate into qualified total gains.

### Stock DGL context (separate acquisition schedule within Loop-Nest Optimization)

<!-- evidence:loop-nest-stock:start -->

| Dataset | projection | terms | score | softmax | aggregation | output | forward |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Cora | 2.496316 | 0.120305 | 0.187456 | 0.433462 | 0.457685 | 0.022508 | 3.852960 |
| CiteSeer | 8.784473 | 0.150778 | 0.210618 | 0.443345 | 0.453988 | 0.027349 | 10.259409 |
| PubMed | 7.516801 | 0.767366 | 0.703230 | 3.359760 | 3.811452 | 0.103889 | 16.668003 |

<!-- evidence:loop-nest-stock:end -->

The stock row is the unchanged import on the exact same graph and
seed-0 parameters, not the loop-family control. DGL and the compiled
pipeline have different orchestration and may accumulate in a
different order, so any ratio of absolute stock and custom totals
would be an implementation comparison. The ratios in the tuning
tables never make that substitution.

## Cross-experiment synthesis

| Choice | What changes | Evidence to retain | What the data cannot establish |
|---|---|---|---|
| Sparse storage | Coordinate encoding, grouping and adjacency traversal; Sparse Storage may also change backend, index width, dispatch and score fusion. | CSB and small SELL rows are viable implementation candidates; Sparse Storage winners beat stock, with size sensitivity for PubMed SELL. Prefer matched custom controls in future tests. | An inherent advantage of blocking alone. Timing Validation unblocked custom COO was fastest by observed forward median on all three. |
| Node order | Labels, feature allocation order, and compressed traversal; original EID sequence stays fixed. | Keep Original as the initial choice. Node alternatives did not meet the practical threshold after matching allocation. | General impossibility of a graph relabeling gain; current CIs and only three graphs do not establish equivalence. |
| Edge order | EID enumeration and index/feature-access relationship. The nodes and feature tensor stay fixed. | Original, Destination-grouped, Destination-then-source. Stable grouping passed independently twice on PubMed. | Lexicographic destination order is better than stable grouping, or that PubMed predicts all datasets. |
| Loop order | Logical traversal and generated register/memory behavior in otherwise matched custom code. | Fix A1/B1; evaluate C1/C3 and D1/D2/D4. Aggregation is the most sensitive isolated region in this family. | A direct comparison with stock isolates loop order, or C3 and D2 combine without interference. |

**Independent effects** answer “what happened when this measured
choice changed while the control conditions stayed fixed?” In Graph Reordering,
this is particularly strong because a single imported kernel path
sees different graph data; in Loop-Nest Optimization it is the same serial family and
ABI with one changed loop order. Sparse Storage spans more simultaneous
implementation differences and needs the Timing Validation matched-backend and
unblocked qualification.

**Cross-stage observations** say which regions covary after one
change. For instance, destination grouping can improve both
softmax and aggregation in the same stock forward, and a shared
format changes multiple sparse stages. Stage observations can
guide explanations and new tests. They do not establish a formal
interaction between two separately controlled choices. A different
graph state entering the next stage and a different allocator/cache
context can both affect timings.

**A cross-axis interaction requires matched combined treatments.**
For two choices A and B running on the same workload and backend,
an additive contrast would be
$T_{11}-T_{10}-T_{01}+T_{00}$ from the four direct forwards, with
block and process pairing retained. A multiplicative contrast
uses the corresponding log ratios. There is no such fixed four
cell experiment spanning a custom storage choice and a loop order or the
winning edge policies in these experiments.
The older exploratory DGL-only joint matrix used its own
compatibility and timing rules and is not evidence for these
combined treatments. **Do not add or multiply speedups from these
campaigns. Do not propose a measured combined winner.**

The strongest comparison is therefore which apparent improvements survive the progressively
stricter evidence: numerical equivalence, matched graph or
implementation control, direct latency, a process-level interval,
and familywise-adjusted qualification. Negative and inconclusive
measurements are essential results. They narrow the set of
plausible future experiments and prevent reporting noisy winners.

## Testing and correctness

For the independently executed verification checks, exact commands,
results, platform and memory-check scope, see
[VERIFICATION.md](VERIFICATION.md). These checks do not replace the
committed acquisition samples. The focused unit tests and the
smoke run use small fixtures or deliberately small process counts.
Do not describe them as a new performance study.

- **Baseline and source.** `benchmarks/gat.py --smoke-test` validates
  CPU stack, shape, numerical properties and stock inference.
  Supporting verification compares the annotated class AST to
  installed DGL 1.1.3. Stock timing uses the actual import. Where a
  separate DGL primitive decomposition supplies intermediate targets,
  its output and attention are independently checked against that
  import. This distinguishes a stock oracle from a reimplementation
  of the same flawed candidate.
- **Storage.** Sparse Storage builders decode physical arrays and recover exact
  node pairs and original EIDs. Tests exercise masks, sentinel slots,
  parallel-edge sidecars, globally normalized softmax, padding
  allocate guards, and the native fallback routes. Every measured
  format checks intermediates and final output, repeatability and
  active destination/head sums. Sparse Storage maximum final absolute error across
  datasets is approximately 1.12e−8; maximum isolated sparse
  discrepancy 4.47e−8; max reported normalization error
  4.72e−7. These are not thresholds.
- **Ordering.** Both Graph Reordering acquisitions verify bijections and restore
  node output and EID attention against stock on the original graph.
  Tests include nontrivial feature and edge attributes, disconnected
  structure, repeat edges, self-loops, METIS replay and Morton high-bit
  boundaries. All three exposed sparse views are decoded with mapping
  checks. The maximum restored output error in the extension is
  approximately 1.86e−8, attention 5.96e−8, normalization 4.72e−7.
  Plain and profiled executions agree bitwise for the same graph.
- **Loop ordering.** Every case checks projected features, L/R
  terms, activated scores, EID attention, aggregate and output
  against the stock-backed reference; additionally compares all
  custom intermediates across variants and bitwise direct, repeated,
  isolated-target and instrumented output where applicable. Tests
  include skewed degree, parallel edges, initial isolation with/without
  self-loops, an empty edge list, and extreme positive/negative logits
  under stable softmax. Under the established full-forward guard,
  a graph with nonempty zero-in-degree nodes is rejected before
  inference; raw CSC kernels still handle empty rows. In the recorded
  dataset measurements, the stage and output value comparisons were
  exact at observed precision, and the largest active attention-sum
  error was 4.72e−7. The claim does not require bitwise equality to
  DGL on arbitrary workloads. The C++ Memcheck fixture is a separate
  correctness test over the compiled sources; see QA for its result
  and limits.

Numerical equivalence is FP32 with **atol 1e−5 and rtol 1e−4**
for mixed tolerance. Normalization uses the recorded tolerance
independent of timing. Reduction order can differ across formats
or across DGL/custom implementations. Near-zero relative errors
require the mixed tolerance; no near-zero division should be
interpreted as an end-to-end failure in isolation. Fingerprints,
raw correctness metadata and deterministic report replays provide
further evidence. N/A native routes and intentionally skipped
controls remain disclosed rather than silently counted as measured.

## Limitations

1. **No completed tensor-based cost model or automatic selector.**
   A future model can use these observations, but there is no
   cross-validation of cost prediction, no learned ranking error,
   and no automatic joint choice presented as an end result.
2. **No tested joint storage, reordering, and loop-nest configuration.** Independent deltas
   are not compositional. Even separately supported orders should
   be measured as combinations with conversions and placement
   costs included before claiming a joint gain.
3. **Only three citation graphs and one untrained layer shape.**
   Four heads and eight output channels, one seed of model weights,
   native feature width per graph, fixed FP32, directed DGL records
   with one self-loop per node. Results do not generalize without
   tests that vary head width, topology, feature dimensions or graph
   size independently. No training, accuracy, backward, batch-size,
   GPU, multi-thread or multi-layer scalability evidence.
4. **One WSL2/Docker CPU and compiler.** Exact code generation
   under GCC 12.2 and pinned DGL/PyTorch is documented. Scheduler
   noise, frequency behavior, and allocator effects can remain on a
   pinned logical CPU. The disassembly explains one implementation's
   structure; other compilers/architectures can make different
   register and vectorization decisions.
5. **Separate acquisition campaigns and distinct controls.** Do
   not pool relative effects or compare absolute medians across
   campaigns as if a controlled factor alone changed. storage evaluation / timing validation
   differences are informative context, not a combined statistical
   population. Fixed random ordering seeds support conditional timing
   inference; three seeds do not describe all random permutations.
6. **Statistics and rank selection.** Eight processes per dataset
   are the replication level. Pointwise intervals and Holm p-values
   answer related but different questions. “Fastest” is an observed
   rank, not a theory of universal superiority. Nonsignificance does
   not establish practical equivalence unless an equivalence test
   was designed and run. The one-factor design tests only the
   declared baselines, not every alternative head-to-head.
7. **Costs have specific boundaries.** Main timing excludes initial
   conversion, graph building, canonical-output restoration, and data
   download. Regression/QA smoke measurements do not replace the raw
   campaign files. Memory diagnostics report allocated representation
   payloads, not full RSS or cache misses. Timer overhead estimates
   and isolated/in-forward differences reflect context; no hardware
   counter study supports a particular miss-rate mechanism.

## Conclusion and future work

The demonstrated contribution is a reproducible, correctness-checked
performance study with stock-backed references, frozen workloads,
raw matched blocks and explicit timing boundaries. Sparse Storage identifies
supported sparse implementation wins over DGL plus severe capacity
and latency regressions for large dense tiles. Timing Validation shows why those
wins are not automatically a blocking or layout win. Graph Reordering establishes
PubMed's destination-oriented edge gains without evidence to retain
node ordering under the same practical threshold. Loop-Nest Optimization identifies
qualifying loop-order gains in softmax and aggregation, and gives a
clear example of a large isolated aggregation gain becoming a
much smaller, but still supported, whole-layer speedup.

Future work should collect compatibility-aware full-plan measurements
before fitting a cost model. Store dataset and architecture features,
kernel/layout/order choices, dtype, and conversion boundaries along
with observed costs and a mask for unsupported or unmeasured cells.
Keep direct full-forward latency as a separate response. Use small
planned factorial experiments and session-level intervals to
validate cross-axis interactions; separate one-time setup and
per-forward conversions so predicted wins can be amortized over
real repeated workloads. Compare held-out ranking and latency
errors against simple baselines prior to claiming automatic
selection, and test transfer to alternative CPUs, feature widths,
thread counts and graph families. Nothing here claims those steps
are finished.

## Evidence index

Key selectors in the JSON below identify records by field value.
The exact form of each quoted result and its inference status are
in [EXPERIMENTAL_FINDINGS.md](EXPERIMENTAL_FINDINGS.md). Direct timing values are
medians of process medians; paired ratios, intervals and p-values
come from the listed contrast arrays, not from rounding report
tables.

| Result or method | Repository-relative file and location | Measurement / interpretation |
|---|---|---|
| Workload dimensions and counts | [Sparse Storage RESULTS.json](../benchmarks/comprehensive_sparse_formats/RESULTS.json) · `datasets.{cora,citeseer,pubmed}[0].graph` | Directed loader versus self-loop-processed graph; raw metadata. |
| Model and platform identity | Sparse Storage, [Graph Reordering RESULTS.json](../benchmarks/ordering_baselines/RESULTS.json), [extension RESULTS.json](../benchmarks/ordering_baselines/extended/RESULTS.json), [Loop-Nest Optimization RESULTS.json](../benchmarks/loop_ordering/RESULTS.json) · `datasets.<dataset>[i].{weights,edge_sha256,environment,graph.feature_sha256}` | Verified feature, edge, model fingerprints; DGL/Torch/CPU/thread provenance. |
| Sparse Storage stock and observed best forwards | [Sparse Storage ANALYSIS.json](../benchmarks/comprehensive_sparse_formats/ANALYSIS.json) · `timings.<dataset>.forward.{stock,csb64,csb2,sell2}`; `comparisons` matching `stage=forward,control=stock` | Direct forward; each dataset's listed winner beat stock after Holm. C01–C03. |
| Stronger Sparse Storage four-control comparison | [Sparse Storage report](../benchmarks/comprehensive_sparse_formats/REPORT.md) · Supported full-forward comparisons; Sparse Storage `comparisons[dataset=citeseer,stage=forward,format=csb2/csb16,control=stock/coo/csr/csc]` | CiteSeer CSB2's native contrasts are inconclusive; CSB16 is the sole four-control qualifier. C04–C05. |
| Purely isolated Sparse Storage sparse results | Sparse Storage ANALYSIS · `timings.<dataset>.stage["softmax/csb64"]`, relevant `aggregation/csb64` or `aggregation/csb4`, comparator CSC; `comparisons` | Prepared operands and separately timed stages, not percentages or timings of a specific forward. C06–C08. |
| Memory and block regressions | Sparse Storage ANALYSIS · `storage.pubmed.{dense2,dense64,sell2,sell64}`, `comparisons[pubmed,forward,dense64,stock]` and `[pubmed,forward,sell64,sell2]` | One-time build / stored bytes or allocated slots; direct regressions supported after correction. C09–C10. |
| Native-format fallback | [Sparse Storage NA_EXPLANATIONS.md](../benchmarks/comprehensive_sparse_formats/NA_EXPLANATIONS.md); Sparse Storage raw `datasets.<dataset>[i].routes` | Supported native calls only, N/A isolation is not a missing forward sample. |
| Backend and storage controls | [Timing Validation ANALYSIS.json](../benchmarks/sparse_format_validation/ANALYSIS.json) · `contrasts` with `family="matched backend",cora,coo→matched_coo` and `family="matched storage",pubmed,custom_coo→sell2` | Independently paired forwards; backend win and storage candidate loss, respectively. C11–C12. |
| Projection timing discrepancy | [Timing Validation report](../benchmarks/sparse_format_validation/REPORT.md) · Projection discrepancy and timing overhead; Timing Validation ANALYSIS `milestone8.citeseer` | Instrumentation and execution context measured separately. Old Sparse Storage isolated medians are not in-forward stages. |
| Graph Reordering allocation sensitivity | [Graph Reordering report](../benchmarks/ordering_baselines/REPORT.md) · Feature allocation sensitivity; [archived RESULTS.json](../benchmarks/ordering_baselines/initial_allocation/RESULTS.json) | Earlier allocation-asymmetric run retained but not used for recommendation. |
| Reproduced stable destination edge effect | [Graph Reordering ANALYSIS.json](../benchmarks/ordering_baselines/ANALYSIS.json) and [extension ANALYSIS.json](../benchmarks/ordering_baselines/extended/ANALYSIS.json) · `datasets.pubmed.edge.table`; `contrasts[pubmed,edge,dst,forward]` | Separate matched stock forwards. Both qualify vs Original. C13–C14. |
| Destination-then-source effect and head-to-head | Extension ANALYSIS · `contrasts[pubmed,edge,dst_src,forward]` and `new_edges_vs_dst[pubmed,dst_src]` | Qualifies against Original; inconclusive against stable destination grouping. C15–C16. |
| METIS plan and node qualification | [METIS.json](../benchmarks/ordering_baselines/extended/METIS.json); extension ANALYSIS `contrasts[cora,node,metis,forward]`; extension report Conclusions | Saved assignments/permutations with measured build times; no node alternative qualifies. C17. |
| Edge traversal hashes | Extension ANALYSIS · `datasets.<dataset>.edge.representations.<policy>.sparse.{coo,csr,csc}`; extension report Edge traversal details | Separates physical index, current EID map, and original-EID visitation; diagnostic, no statistical hypothesis. |
| Loop-Nest Optimization qualifying forwards | [Loop-Nest Optimization ANALYSIS.json](../benchmarks/loop_ordering/ANALYSIS.json) · `contrasts` cases Cora D2, PubMed C3/D2/D4 and `endpoint="forward"` | Direct same-family forward; supported practical changes only. C18–C21 (ordered by dataset and sweep). |
| Isolated versus in-forward evidence | Loop-Nest Optimization ANALYSIS · `datasets.pubmed.aggregation.timing.{D1,D2}`; `contrasts[pubmed,aggregation,D2,isolated/in_forward/forward]` and `[cora,softmax,C3,isolated/forward]` | D2 has strong stage and smaller whole gain. Cora C3 isolated gain, forward inconclusive. C22–C25. |
| Loop-Nest Optimization compiler and native execution | Loop-Nest Optimization RESULTS · `build.{flags,binary_sha256,symbols}` and `datasets.<dataset>[i].{routes,loaded_offsets,checks}`; [Loop-Nest Optimization report](../benchmarks/loop_ordering/REPORT.md) Access order | Distinct symbols and identical arithmetic; no code-identity aliases in the benchmarked build. |
| Correctness and independent QA | [VERIFICATION.md](VERIFICATION.md) plus each report's validation section and raw checks | Commands executed and limitations; never a substitute for raw performance acquisition. |
