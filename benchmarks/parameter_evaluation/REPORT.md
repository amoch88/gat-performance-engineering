# Milestone 4 — Selecting independent tuning parameters for a DGL GAT cost tensor

**Discussion draft for Professor Solomonik.** Reference: DGL 1.1.3, homogeneous CPU inference, float32. This milestone reads existing implementations and recordings; it runs no GAT benchmarks, changes no previous results, implements no kernels and fits no performance model.

## Section 1 — Research objective

The question is **which semantics-preserving implementation choices merit empirical cost-tensor dimensions**, not which GAT architecture is fastest and not whether joint tuning beats independent tuning. For a fixed workload, topology, features, weights and model settings are inputs to preserve. Permutations, sparse execution routes, physical storage, ownership of reductions and resource allocation can change execution while computing the same graph function.

A useful axis has distinct legal implementations, measurable costs and a reason its ranking could depend on graph/feature/hardware characteristics. A demonstrated cross-dataset rank reversal is strong evidence, but is not a prerequisite for investigating a candidate. Conversely, a collection of labels with identical consumers and execution paths is not a useful dimension. A universally dominant value may become a fixed implementation default rather than a permanent search dimension.

**Recommendations in brief:**

- Retain **node ordering and edge ordering** as the current measured implementation axes, with explicit context and interface costs. Node ordering has the clearest existing cross-dataset evidence under a fixed source-grouped edge policy. Their A classifications below mean initial domain eligibility, not universal optimality or confirmatory statistical certification.
- Evaluate **32-/64-bit sparse indices** next: the stock primitives and graph sizes make it a concrete, inexpensive new independent comparison. Measure **1/2/4 threads** separately as a hardware/resource configuration, initially controlling it in every other experiment.
- Before writing new sparse kernels, evaluate **stage-specific graph views using existing DGL primitives**. This can expose real COO-versus-CSR score traversal or CSC-versus-COO aggregation alternatives that a single whole-layer permission mask cannot express.
- Investigate degree-aware softmax/aggregation schedules and a small exact score-addition/LeakyReLU fusion next. Physical layouts, cache tiles and larger softmax/aggregation fusion are promising **conditional implementation-family choices**, not unsupported labels to add to stock GATConv.
- Do not treat the present `all` versus `COO+CSC` masks as established different kernel algorithms. Source-level requests select the same active formats in this forward path. Keep them as memory/materialization controls until tracing demonstrates another real difference.

### Scope and evidence provenance

Repository state reviewed: commit `53a85fe32872c69195d7cb290fb00c72f608dc10`. The baseline measurements predate their first commit; their own source hashes identify that measured code. Milestone 3 records base commit `cd6d04549a2c24537b631e1d4170d9351bec597c`, per-worker source/lock hashes and six fresh processes per measured dataset.

The review read the annotated full class and constructor in [baseline.py](../dgl_baseline/baseline.py); setup, measurement and memory guards in [common.py](../dgl_baseline/common.py); all six parameter scripts; both sets of tests; [joint core](../joint_tuning/core.py), [worker](../joint_tuning/worker.py), [run driver](../joint_tuning/run.py), [analysis](../joint_tuning/analysis.py); both prior reports; and the raw JSON files below. Reports were checked against data rather than accepted as primary numerical evidence.

- [Milestone 2 raw measurements](../dgl_baseline/MEASUREMENTS.json): original one-parameter blocks, raw repetitions, separate controls, correctness and representation hashes.
- [Milestone 3 raw results](../joint_tuning/RESULTS.json), [derived analysis](../joint_tuning/ANALYSIS.json) and [seven-policy pilot](../joint_tuning/PILOT.json): reused for source/compatibility checks and explicitly restricted one-axis slices, **not** joint-winner claims.
- [EVIDENCE.json](EVIDENCE.json) and [EVIDENCE.md](EVIDENCE.md): offline reanalysis, ratios and uncertainty, including source digests. `analyze_existing.py` only reads stored numbers; it does not import benchmark runners or DGL.
- [SOURCES.json](SOURCES.json): version-pinned upstream source URLs/hashes and inspected local files. Source-level dispatch arguments are distinguished from runtime tracing and wheel build-option evidence.

No ModelingWithTensors source tree was located in this checkout or the workspace project directories. Its loop-order collapsing and Composite examples are treated as conceptual references supplied in the request, not as independently inspected implementation findings. Section 9 makes the transfer limits explicit.

## Section 2 — Complete parameter taxonomy and selection rubric

Stage abbreviations: **P** projection; **A** attention contributions/logits; **S** destination softmax; **G** weighted aggregation; **O** output/restoration. “Stock” distinguishes a GATConv constructor/graph API from lower-level primitives that need a separate forward orchestrator.

| Parameter/family | Concrete values or alternatives | Stages | Implementation requirement | Category and tensor decision |
|---|---|---|---|---|
| Whole-layer sparse permissions | COO, CSR, CSC, COO+CSR, COO+CSC, CSR+CSC, all three | A/S/G, degree guard | Official `g.formats`; only COO+CSC/all pass complete forward | **D** memory/materialization policy; do not encode unused CSR as a proven kernel-choice axis; unsupported values excluded for this family |
| Stage-specific sparse routes | score COO or CSR; softmax CSC; aggregation CSC or COO | A/S/G | **Custom forward orchestration** can initially reuse DGL CPU primitives; CSC-free softmax needs a custom kernel | **C** promising implementation family; real route choices, not current mask labels |
| Basic edge order | original EIDs, source grouping, destination grouping | A/S/G | Official `reorder_graph`; preserve original-ID maps/data | **A** retain measured axis; strongest benefit currently PubMed destination order; no proven cross-dataset reversal yet |
| Additional edge order | stable degree-informed neighbor sorting; endpoint-block ordering with one fixed block size initially | A/S/G | Official custom edge permutation, plus representation checks | **B** test only if actual traversal/EID maps change |
| Basic node order | original, RCMK, ascending total degree | P/A/S/G/O | Official `rcmk`/custom permutation, feature mapping and output restoration | **A** retain measured axis under declared edge policy and interface contract |
| Additional node order | descending total degree, deterministic component-wise BFS | P/A/S/G/O | Generate a permutation, apply official API | **B** inexpensive, plausible locality alternatives; no current timing evidence |
| Partition-based node order | 2/4/8 partitions, deterministic intra-partition order | P/A/S/G/O | Official METIS route where available, or external persisted permutation | **D** graph/build/preprocessing dependent; not assumed reproducible or available |
| Projected feature layout | physical NHF, HNF, NFH, justified node-block AoSoA | P/A/G/O | Stride-aware producer/consumers or explicit packing; stock conversion can erase layout | **C** conditional axis only after both layouts execute physically differently |
| Edge score/attention layout | edge-major EH, head-major HE, CSC-position segmented storage | A/S/G | Modified producer/softmax/aggregation interfaces, EID restoration as needed | **C**; segmented EID permutation alone duplicates edge ordering |
| Sparse schedule / loop nesting | outgoing-row owned, incoming-row owned, edge owned; heads/features inside/outside sparse loops | A/S/G | Some route variants via existing primitives; other loops require CPU kernels | **C** family-specific; remove aliases that lower to the same loop nest |
| Softmax scheduling and scratch | row owned; degree-bucketed; split heavy rows; row/head order; per-row vs per-thread scratch | S | Custom CPU softmax path; stable global row normalization | **C** high priority after inexpensive stock axes |
| Aggregation scheduling | destination ownership; source/edge atomics; private partial reductions; degree-aware split rows | G | CSC/COO primitive routes first, custom scheduling thereafter | **C**; synchronization and reduction order are part of the alternative |
| Cache blocks / chunks | node blocks 16/64/256; edge chunks 64/256/1024/4096; feature tiles 2/4/8; head tiles 1/2/4 | P/A/S/G | Kernel- or scheduler-specific controls, correct tails | **C** conditional; cannot be an independent generic tile axis without a consumer |
| Thread count | 1, 2, 4 | P/A/S/G/O | Set PyTorch/OpenMP/MKL coherently before process initialization | **D** outer resource configuration; keep fixed within other comparisons initially |
| CPU affinity | default; fixed allowed mask; close/spread placement where effective | P/A/S/G/O | OS/OpenMP controls, validate effective thread placement | **D** controlled hardware setting before considering a model dimension |
| NUMA placement | first-touch/local/interleaved pages on a real multi-node system | P/A/S/G/O | Multi-node access and verified placement | **D** future hardware axis; no distinct NUMA values in current one-node VM |
| Sparse index dtype | int32/int64 with all index and offset bounds checked | A/S/G, graph prep | Official `g.int()`/`g.long()`; CPU dtype specializations exist | **B** first new stock comparison; legal domain is workload dependent |
| Add + LeakyReLU fusion | materialized add then activation; one fused edge pass | A | Custom CPU pointwise/GSDDMM epilogue | **C** small first fusion experiment; same GAT attention formula |
| Projection + node contributions | separate Z reads/reductions; projection epilogue producing Z and l/r | P/A | Custom epilogue/forward implementation | **C**; share homogeneous projection as stock already does |
| Softmax + aggregation fusion | materialized alpha; stable two-pass fused reduction; stable online state | S/G | Custom CPU reduction and interface restrictions | **C** high potential, higher correctness/scheduling complexity |
| Other actual fusion | score generation + row softmax; aggregation + output bias/activation epilogue | A/S or G/O | Custom family, fixed model options | **C** if intermediates or passes genuinely disappear |
| “Message multiply + sum fusion” label | unfused UDF versus stock builtin `u_mul_e` + `sum` | G | Stock already executes a fused gSpMM path | **E** as a claimed *new* fusion saving over this baseline |
| Logical dimensions, topology, model options | See Section 5.N | All | Change the defined workload or are ineffective labels in eval mode | **E** optimization axes; retain as workload descriptors/fixed inputs |
| Precision / approximation / changed attention algorithm | FP32→BF16, approximate exp/softmax, dot-product edge attention | Multiple | Different numerical/algorithmic contract | **E** for this strictly equivalent study; separate accuracy-tradeoff study only |

### Rubric: ten questions, not an invented scalar score

Use **2 = demonstrated**, **1 = conditional/partial**, **0 = contradicted**, and **? = not measured**. Unknown is not zero. Do not add the scores into a rank that automatically penalizes unimplemented research ideas.

| Criterion | Required assessment / advancement evidence |
|---|---|
| 1. Semantic equivalence | Same logical graph, feature/weight association, stable destination normalization, model settings and output ordering; test float tolerance and failure conditions |
| 2. Actual execution difference | Different sparse arrays/traversals, physical strides honored by consumers, schedule/ownership, eliminated materialization, or verified thread resources; labels alone fail |
| 3. Dataset dependence | Look for stable pairwise rank reversals while other factors stay fixed; otherwise record dominance or insufficient coverage, not “useless” |
| 4. Magnitude | End-to-end improvement relative to noise and intended inference volume, including restoration/conversions; use a predeclared practical threshold (e.g. 3%) as a planning choice, not a discovered fact |
| 5. Reliability | Independent randomized process trials, matching controls, paired intervals and selection bias/multiplicity discussion; repetitions in one block do not supply process-level confidence |
| 6. Applicability | Legal graph sizes, degree distributions, feature shapes, inference/training interface and operator coverage |
| 7. Feasibility | Constructor/API knob vs orchestration vs compiled kernel; required maintenance, tools and scratch memory |
| 8. Reproducibility | Version/commit/source hashes, fixed weights/features, deterministic permutations, raw timings, resource controls and valid execution fingerprints |
| 9. Overheads | One-time graph work and per-forward packing, output restoration, synchronization and extra storage; estimate break-even only from measured savings |
| 10. Cross-kernel relevance | Producer/consumer representation compatibility and changed intermediates must be explicit; prefer meaningful boundary states over a Cartesian product of names |

**A** admits a correct implemented choice with useful measured contrasts in a stated context; it does not require every value to win somewhere or certify a universal speedup. Edge/node ordering pass this admission gate, with reliability still qualified by post-hoc corroboration. **B** has a feasible candidate but insufficient independent measurements. **C** needs a distinct implementation family; this includes a new forward orchestrator using existing compiled operators, and is not synonymous with writing every kernel from scratch. **D** depends on hardware, capacity or a particular deployment policy. **E** is reserved for semantic changes, invalid values within a family, or demonstrated/structurally redundant labels—not for ideas that simply lack benchmarks.

Admission decisions are revisable. If one legal value dominates reliably across a broad workload envelope, use it as a default and remove an unnecessary dimension. If a new workload reverses the ranking, reintroduce the choice with evidence.

## Section 3 — Exact GAT computation and parameter mapping

Let `X` have shape `[N,D]`, with `H` heads and `F` output features per head. The stored PyTorch linear weight has shape `[H*F,D]`. Edges are `e=(u→v)` and `In(v)` includes exactly the preprocessed incoming edges, including one self-loop. For current recordings `H=4`, `F=8`; `D` is dataset native.

| Logical stage | Operations and logical tensors | Reference calls / implementation boundary |
|---|---|---|
| P — Linear projection | `Z = reshape(X W^T, [N,H,F])`; no projection bias in this constructor path | One shared `self.fc` call in homogeneous GAT; `feat_src = feat_dst`. A dense BLAS/backend computation, not a sparse edge dot product |
| A — Attention scores | `l[u,h]=sum_f Z[u,h,f]*a_l[h,f]`, `r[v,h]=sum_f Z[v,h,f]*a_r[h,f]`; both `[N,H,1]`. `q[e,h]=l[u,h]+r[v,h]`, then `s[e,h]=LeakyReLU(q[e,h],0.2)`; edge tensors `[E,H,1]` | PyTorch multiply/reduce for l/r; builtin `u_add_v` via generalized SDDMM **with add**, then `self.leaky_relu` |
| S — Stable edge softmax | `m[v,h]=max_{e∈In(v)} s[e,h]`; `d[v,h]=sum exp(s[e,h]-m[v,h])`; `alpha[e,h]=exp(s[e,h]-m[v,h])/d[v,h]`; alpha `[E,H,1]` | CPU edge-softmax primitive consumes incoming CSC. Current inference has zero attention dropout |
| G — Weighted aggregation | `Y[v,h,f]=sum_{e=(u→v)} alpha[e,h]*Z[u,h,f]`, output `[N,H,F]` | Builtin `u_mul_e` + `sum` lowers to generalized SpMM with fused message formation/reduction; not a required global `[E,H,F]` message tensor |
| O — Output processing | Current workload: `Y+b`, `b` broadcast from `[H*F]`; no residual/activation. General reference can add its configured residual and activation. Restore original logical node order after relabeling | Preserve exact constructor branches: a linear residual can contain the bias instead of a second explicit bias. Return `[N,H,F]`; optionally attention in reference EID order if requested |

If fixed edge weights `w[e]` are supplied, DGL multiplies alpha by w **after** softmax. They scale the aggregation numerator, not the softmax denominator. Current measured workload has no edge weights. Feature and attention dropout are disabled, and `.eval()` plus `inference_mode()` is used.

**Do not substitute `Z[u,h,:]·Z[v,h,:]` for l+r.** The f-reductions are against learned node-attention vectors and are performed at nodes, not once per edge. DGL's generalized “SDDMM” engine is a dispatch abstraction supporting binary operations; its name does not turn `u_add_v` into dot-product attention.

The five rows above are **logical stages, not five compiled CPU kernels**. A stage can contain several PyTorch/DGL calls; gSpMM already combines logical work; CPU softmax has multiple passes within one primitive. BLAS, OpenMP/runtime scheduling and optional compiled library paths can introduce further internal structure. Full-forward timings cannot attribute a gain to an individual kernel.

### What the source exposes

The v1.1.3 source requests COO in `aten::SDDMM`, CSC in `aten::SpMM`, and explicitly passes `GetCSCMatrix(0)` into CPU softmax. `UnitGraph::SelectFormat` first intersects preferred, allowed and created formats. With both COO and CSC materialized, both surviving permission policies select the same named sparse routes. This is source-level reasoning corroborated by representation hashes, **not a recorded dynamic kernel trace**. CPU SpMM has generic and optional library-specific internal paths; wheel build options and executed inner kernels were not established by the existing recordings.

At the Python boundary, `_sparse_ops.to_dgl_nd` calls `zerocopy_to_dgl_ndarray`, whose PyTorch implementation calls `.contiguous()`. Write conversions require contiguous outputs. This can silently repack a proposed noncontiguous logical view before sparse consumers. Stock GATConv exposes no node-feature-layout, attention-layout, row-grain or loop-order argument.

| Parameter group | P | A | S | G | O |
|---|---|---|---|---|---|
| Whole/stage sparse representation | — | traversal | incoming segments | incoming/edge reductions | — |
| Edge ordering | — | endpoint gathers/EID writes | EID gather/scatter and neighbor order | EID and neighbor locality | attention EID restoration if returned |
| Node ordering | row sequence/locality, same GEMM dimensions | node gathers | row identities/traversal | source features and destinations | restore rows |
| Projected layout | producer packing | node f-reductions | indirect effect through produced scalars | main feature consumer | output boundary/next layer |
| Edge layout | — | producer | main consumer | weight consumer | restore attention if observable |
| Sparse schedule / cache blocks | optional projection blocking | edge/head ownership | row/head/chunk ownership | row/edge/head/f ownership | optional epilogue |
| Threads / affinity / NUMA | yes | yes | yes | yes | yes |
| Index dtype | graph setup only | graph addressing | graph addressing | graph addressing | map conversion if needed |
| Fusion family | P+A candidate | score/activation | S+G candidate | S+G/O candidate | epilogue candidate |

## Section 4 — Existing experimental evidence

Only old timing samples are used. No expensive suite or new parameter experiment was run. The full numerical tables below are generated from immutable recorded JSON by the offline analysis script.

| Dataset | N | E after the fixed self-loop policy | Native D | Mean incoming degree, E/N | Four-head ordering/format evidence |
|---|---:|---:|---:|---:|---|
| Cora | 2,708 | 13,264 | 1,433 | 4.90 | Available |
| Citeseer | 3,327 | 12,431 | 3,703 | 3.74 | Available |
| PubMed | 19,717 | 108,365 | 500 | 5.50 | Available |
| Reddit | 232,965 | 114,848,857 | 602 | 492.99 | Memory-skipped |

Means are computed from recorded graph sizes; degree quantiles/skew are not supplied by these recordings. Different native D values can dilute or amplify the relative importance of sparse stages; that is a workload explanation to investigate, not a measured kernel bottleneck attribution.


<!-- BEGIN RECORDED EVIDENCE -->

### 4.1 Original one-parameter-at-a-time measurements

These are the original Milestone 2 forward-only blocks (20 timed repetitions in one process per dataset/sweep). SD describes repetitions within that block, **not** uncertainty across independent trials. Speedup is the separately measured matching control median divided by the candidate median. The original-option repeat block is not forcibly set to one and is not cherry-picked against the separate control.

Edge sweep: original nodes, all formats. Node sweep: source-grouped edges for every value, all formats. Both use native input width, four heads and eight output features/head. Node restoration is checked but **excluded** from these timings.

| Dataset | Axis | Option | Median ms | Mean ms | Within-block SD ms | Control/case |
| --- | --- | --- | --- | --- | --- | --- |
| citeseer | edge_ordering | original | 7.807906 | 7.792543 | 0.248887 | 0.9901× |
| citeseer | edge_ordering | src | 7.784482 | 7.913373 | 0.399787 | 0.9931× |
| citeseer | edge_ordering | dst | 7.879187 | 8.007051 | 0.360588 | 0.9812× |
| citeseer | node_ordering | original | 7.688945 | 7.712537 | 0.129663 | 1.0289× |
| citeseer | node_ordering | rcmk | 7.756571 | 7.804621 | 0.198367 | 1.0199× |
| citeseer | node_ordering | degree | 7.554538 | 7.647828 | 0.229115 | 1.0472× |
| cora | edge_ordering | original | 3.858080 | 3.893681 | 0.117951 | 1.0323× |
| cora | edge_ordering | src | 3.866215 | 3.896296 | 0.180009 | 1.0302× |
| cora | edge_ordering | dst | 3.820015 | 3.845404 | 0.104873 | 1.0426× |
| cora | node_ordering | original | 3.854922 | 3.865095 | 0.069535 | 1.0560× |
| cora | node_ordering | rcmk | 3.955896 | 3.957695 | 0.165879 | 1.0291× |
| cora | node_ordering | degree | 3.766123 | 3.819310 | 0.186516 | 1.0809× |
| pubmed | edge_ordering | original | 19.372352 | 20.683112 | 5.585460 | 0.9768× |
| pubmed | edge_ordering | src | 19.787533 | 20.032991 | 0.781632 | 0.9563× |
| pubmed | edge_ordering | dst | 17.734171 | 17.701262 | 0.619148 | 1.0670× |
| pubmed | node_ordering | original | 19.862437 | 20.774339 | 3.475244 | 0.9793× |
| pubmed | node_ordering | rcmk | 17.348395 | 17.291668 | 0.494477 | 1.1212× |
| pubmed | node_ordering | degree | 18.634942 | 18.708171 | 0.498901 | 1.0438× |

Fastest *candidate blocks* (not a statement of statistical superiority):

| Dataset | Axis | Fastest block | Control/case |
| --- | --- | --- | --- |
| citeseer | edge_ordering | src | 0.9931× |
| citeseer | node_ordering | degree | 1.0472× |
| cora | edge_ordering | dst | 1.0426× |
| cora | node_ordering | degree | 1.0809× |
| pubmed | edge_ordering | dst | 1.0670× |
| pubmed | node_ordering | rcmk | 1.1212× |
| reddit | edge_ordering | not measured | — |
| reddit | node_ordering | not measured | — |

Sparse single-format cases failed on Cora/Citeseer/PubMed; there is no numerical format ranking from Milestone 2. All Reddit node/edge/format sweeps were memory-skipped. Its one-head result is a different workload and cannot fill the four-head evidence gap.

### 4.2 Restricted existing-data slices: other factors fixed

These tables re-read ONLY one-axis rows from the existing six-round Milestone 3 data. They are post-hoc corroboration, not a new benchmark or joint-winner argument. For each round, relative times first use the recorded bracketing controls; the axis-reference normalized time is then divided by the option normalized time. Intervals are unadjusted Student-t 95% intervals of the mean log ratio over **six process rounds**, exponentiated. The six round medians provide the between-process SD. They do not turn 120 within-process repetitions into 120 independent trials. Comparisons are exploratory, selected after inspecting these datasets and not multiple-comparison corrected.

**edge: node=original, format=all**

| Dataset | Option | Forward ms | Combined ms | Between-round SD ms | Combined reference/case [95% CI] |
| --- | --- | --- | --- | --- | --- |
| cora | original | 4.0159 | 4.0169 | 0.0814 | 1.0000 [1.0000, 1.0000] |
| cora | src | 4.0547 | 4.0558 | 0.1561 | 0.9959 [0.9383, 1.0570] |
| cora | dst | 3.9945 | 3.9956 | 0.0952 | 1.0094 [0.9807, 1.0389] |
| citeseer | original | 8.1526 | 8.1538 | 0.1017 | 1.0000 [1.0000, 1.0000] |
| citeseer | src | 8.2367 | 8.2377 | 0.1266 | 0.9936 [0.9800, 1.0074] |
| citeseer | dst | 8.2352 | 8.2362 | 0.1927 | 0.9960 [0.9878, 1.0042] |
| pubmed | original | 19.1330 | 19.1660 | 0.1548 | 1.0000 [1.0000, 1.0000] |
| pubmed | src | 19.6679 | 19.7011 | 0.5082 | 0.9718 [0.9430, 1.0014] |
| pubmed | dst | 17.5223 | 17.5634 | 0.1835 | 1.0884 [1.0643, 1.1130] |

**node: edge=original EIDs, format=all**

| Dataset | Option | Forward ms | Combined ms | Between-round SD ms | Combined reference/case [95% CI] |
| --- | --- | --- | --- | --- | --- |
| cora | original | 4.0159 | 4.0169 | 0.0814 | 1.0000 [1.0000, 1.0000] |
| cora | rcmk | 4.0105 | 4.0602 | 0.0793 | 0.9906 [0.9581, 1.0242] |
| cora | degree | 3.9717 | 4.0218 | 0.0686 | 1.0032 [0.9652, 1.0427] |
| citeseer | original | 8.1526 | 8.1538 | 0.1017 | 1.0000 [1.0000, 1.0000] |
| citeseer | rcmk | 8.1506 | 8.2114 | 0.1702 | 0.9973 [0.9733, 1.0219] |
| citeseer | degree | 7.9117 | 7.9720 | 0.1669 | 1.0198 [1.0038, 1.0361] |
| pubmed | original | 19.1330 | 19.1660 | 0.1548 | 1.0000 [1.0000, 1.0000] |
| pubmed | rcmk | 18.7533 | 19.1550 | 0.1953 | 0.9947 [0.9782, 1.0114] |
| pubmed | degree | 18.7977 | 19.2295 | 0.2810 | 0.9905 [0.9742, 1.0071] |

**node: edge=src grouped, format=all**

| Dataset | Option | Forward ms | Combined ms | Between-round SD ms | Combined reference/case [95% CI] |
| --- | --- | --- | --- | --- | --- |
| cora | original | 4.0547 | 4.0558 | 0.1561 | 1.0000 [1.0000, 1.0000] |
| cora | rcmk | 3.9830 | 4.0330 | 0.0774 | 1.0013 [0.9545, 1.0505] |
| cora | degree | 4.0113 | 4.0637 | 0.0933 | 1.0050 [0.9567, 1.0557] |
| citeseer | original | 8.2367 | 8.2377 | 0.1266 | 1.0000 [1.0000, 1.0000] |
| citeseer | rcmk | 8.1491 | 8.2104 | 0.2696 | 1.0103 [0.9899, 1.0311] |
| citeseer | degree | 7.8066 | 7.8653 | 0.0750 | 1.0388 [1.0222, 1.0557] |
| pubmed | original | 19.6679 | 19.7011 | 0.5082 | 1.0000 [1.0000, 1.0000] |
| pubmed | rcmk | 17.3293 | 17.7443 | 0.2425 | 1.1005 [1.0731, 1.1287] |
| pubmed | degree | 18.9693 | 19.4285 | 0.3481 | 1.0204 [0.9974, 1.0440] |

**format: node=original, edge=original**

| Dataset | Option | Forward ms | Combined ms | Between-round SD ms | Combined reference/case [95% CI] |
| --- | --- | --- | --- | --- | --- |
| cora | all | 4.0159 | 4.0169 | 0.0814 | 1.0000 [1.0000, 1.0000] |
| cora | coo+csc | 4.0176 | 4.0186 | 0.1386 | 1.0020 [0.9597, 1.0462] |
| citeseer | all | 8.1526 | 8.1538 | 0.1017 | 1.0000 [1.0000, 1.0000] |
| citeseer | coo+csc | 8.2729 | 8.2739 | 0.2739 | 0.9893 [0.9723, 1.0066] |
| pubmed | all | 19.1330 | 19.1660 | 0.1548 | 1.0000 [1.0000, 1.0000] |
| pubmed | coo+csc | 19.1507 | 19.1903 | 0.5593 | 0.9891 [0.9568, 1.0225] |

Reference/case=1 for the reference by construction, not an interval claiming zero timing noise. Combined time includes restoring original node order; forward excludes restoration. The node/src slice matches Milestone 2's edge-policy context; the node/original-EID slice does not. Do not pool these two contexts or compare their absolute times as repeated measurements of a single treatment.

<!-- END RECORDED EVIDENCE -->

### 4.3 What these measurements do and do not establish

**Edge ordering:** original OPAT candidate winners are destination order on Cora, source order on Citeseer, and destination order on PubMed. On Citeseer, however, the supposedly winning source block is still slower than its separately measured original control (0.9931× control/case). Original-repeat ratios range from 0.9768× on PubMed to 1.0323× on Cora, exposing drift. Do not present tiny changes in winner labels as established cross-dataset reversals.

In the restricted six-process node-original/all-format slice, destination order versus original is **1.0094× [0.9807,1.0389]** on Cora, **0.9960× [0.9878,1.0042]** on Citeseer, and **1.0884× [1.0643,1.1130]** on PubMed. Only the PubMed contrast clearly excludes one in this exploratory interval analysis. Source-versus-destination rankings on Cora/Citeseer are unresolved. Destination grouping is a credible useful option on PubMed, but neither a universal dominating choice nor a statistically established cross-dataset rank reversal.

**Node ordering:** the original source-grouped-edge OPAT blocks favor ascending degree on Cora/Citeseer and RCMK on PubMed. Repeating the same-context comparison with restricted existing six-round rows gives:

- Citeseer: ascending-degree versus original **1.0388× [1.0222,1.0557]**; degree versus RCMK **1.0283× [1.0121,1.0447]**.
- PubMed: RCMK versus original **1.1005× [1.0731,1.1287]**; RCMK versus degree **1.0785× [1.0568,1.1007]** (inverse of the stored RCMK/degree latency ratio).
- Cora: all source-grouped node-order contrast intervals include one; its small degree winner is unresolved.

Thus there is **exploratory pairwise reversal evidence between degree and RCMK on Citeseer and PubMed with the edge policy fixed**. This is not evidence obtained from the best joint configuration. It is post-hoc, unadjusted and from six rounds on the same CPU cohort; a dedicated predeclared one-factor repetition is the appropriate confirmation.

A fixed **source-grouping policy is not a fixed logical EID sequence**: relabeling nodes changes the source-ID groups and can reorder edges as a consequence. This is a valid one-parameter contrast under that fixed policy, but it does not isolate feature-memory permutation from the induced traversal changes. It cannot be attributed solely to node-feature locality. The original-EID slice is the stricter contrast when preserving logical edge sequence is essential.

With original EIDs fixed instead, node effects mostly disappear: for example PubMed RCMK/original combined speedup is 0.9947× [0.9782,1.0114]. This is a different fixed context, not a contradiction and not permission to pool contexts. Node order is a useful axis whose interpretation must include the fixed edge policy and restoration contract.

**Sparse permission policy:** COO+CSC versus all has combined ratios **1.0020× [0.9597,1.0462]**, **0.9893× [0.9723,1.0066]**, and **0.9891× [0.9568,1.0225]** on Cora/Citeseer/PubMed with both orders fixed original. Every interval includes one. Shared COO/CSC tensors and EID-map hashes match for these original-order policies in each dataset. Together with source dispatch requests, this does not support a distinct runtime-algorithm axis just for retaining unused CSR.

**Reddit:** its recorded 232,965 nodes, 114,615,892 dataset edges, 114,848,857 preprocessed edges and 602 features are real. Four-head ordering/format sweeps were skipped by memory guards; the existing one-head attention experiment cannot be compared as an equivalent four-head implementation. There is no Reddit edge/node/format ranking to invent. Memory estimates are conservative policies, not measured proof that every single four-head configuration would fail.

### 4.4 Overheads and limits of inference

Milestone 2's case preprocessing entries cover generator work and are single observations, not a distribution of deployment preprocessing costs. Examples: PubMed edge destination preprocessing 12.596 ms; node RCMK 64.581 ms. Those are different transformations and must not be reused as matched cost measurements from another suite. Its node restoration is **outside** forward timing. Milestone 3 records inverse-map preparation, restoration and combined time separately, so it is the proper source for the interface cost, always at a fixed axis context.

For a fixed configuration, let P be incremental preparation, R per-forward restoration/packing and T raw forward. Relative to a control T0, break-even inference count is `ceil(P/(T0−T−R))` only if the denominator is positive. If outputs may remain in permuted order through subsequent layers, charge restoration at the actual external boundary instead; do not silently drop it from the current canonical-output contract.

The two suites also differ in graph residency: Milestone 3 holds its prepared configuration matrix within a round, whereas Milestone 2 generates cases sequentially. Do not pool their absolute timings as repeated samples of the same protocol. The node/src slice uses ratios to its matching source-edge reference; each underlying block was bracketed by the original/original/all control, not a newly run dedicated node/src control. This remains retrospective corroboration.

All original OPAT sweeps have one process per workload/sweep; their 20 repetitions are insufficient for process-level confidence. Raw PubMed original-edge and original-node block SDs are 5.585 and 3.475 ms, respectively, versus 0.619 ms for destination edges and 0.494 ms for RCMK nodes. These outliers and repeat-control shifts matter more than a table's last decimal. Six-round slices improve uncertainty assessment but remain retrospective and uncorrected for searching many contrasts. No stage attribution, broad dominance or deployment-optimal policy follows from these full-forward timings alone.

## Section 5 — Additional parameter candidates and technical evaluation

This section specifies candidate values and execution mechanisms, not new measured results. **S** denotes source evidence, **M** recorded measurement and **H** a hypothesis requiring a new experiment. Advantages/disadvantages below are hypotheses unless a recorded comparison is cited.

### A. Sparse representation and stage-specific formats

| Whole-layer allowed/materialized policy | Recorded complete-forward outcome | Why / implication |
|---|---|---|
| COO | Unsupported | CPU softmax requires CSC and cannot create it under this restriction |
| CSR | Unsupported | Stock zero-in-degree check cannot compute incoming degrees with neither CSC nor COO allowed; do not disable the guard |
| CSC | Unsupported | Edge-score generalized SDDMM selects a representation outside its CSR/COO supported paths |
| COO+CSR | Unsupported | No CSC for CPU softmax |
| COO+CSC | Supported, numerically equivalent | COO score route, CSC softmax and aggregation requests |
| CSR+CSC | Unsupported | Preferred COO is absent; format fallback trips SDDMM's supported-format guard despite CSR being permitted |
| COO+CSR+CSC | Supported, numerically equivalent | Same active requested routes as COO+CSC in this fixed homogeneous forward |

These are pilot outcomes for this installed build and reference path, not a statement that CSR is unusable by every DGL primitive. `SDDMMCsr` exists. `SelectFormat` is not a user-controlled per-stage schedule optimizer. Conversely, the C++ function name `Edge_softmax_csr_forward` does **not** mean outgoing CSR: its caller supplies incoming CSC, represented internally as rows of the transposed adjacency.

**A real future alternative:** use a CSR-only graph view for `gsddmm(add,l,r)`, a CSC view for softmax, and a CSC or COO view for `gspmm(mul,sum,Z,alpha)`. This is custom forward orchestration, initially with existing DGL compiled CPU primitives; it need not start by writing a new kernel. Keep the usual degree validity check on the canonical graph. The complete layer must be revalidated against GATConv. A CSC-free softmax implementation would require additional custom code.

CSR score generation uses source-row ownership/neighbor loops; COO has edge-parallel generation. CSC aggregation owns destination reductions; COO sum has edge-parallel atomic accumulation in the generic CPU source. Those are actual route/schedule alternatives. Do not also treat “CSR score” and “source-row score schedule” as freely independent labels if the chosen primitive fixes both.

Build graph views once; charge sparse conversion, index/EID arrays, memory residency and any tensor-order translations. With canonical EID-indexed attention, CSR score output and CSC consumers can communicate through EID maps without permuting EH data each forward. A position-indexed custom layout may instead need an EH gather/scatter. COO-to-compressed conversion and sorting can be substantial; costs depend on implementation and graph order and must be measured, not assumed zero. Extra structures can exceed memory limits even if one forward fits.

**Dataset dependence (H):** high degree or irregular rows may favor edge work sharing over row ownership; compact rows and repeated sources may favor compressed traversal. The current citation graphs have low mean incoming degree, Reddit much higher; degree tails were not characterized by the stored metadata. **Decision:** D for current memory masks, C for stage-specific orchestration/implementations. First test only COO versus CSR **score traversal**, with downstream CSC fixed—not all stages at once.

### B. Edge ordering

Current values are original canonical EIDs, source grouping and destination grouping. Official reordering retains graph topology and copies associated edge data; retain a new-position→original-EID map. If an API returns attention, restore its EID order as well. Existing topology/output tests cover the unweighted benchmark; future weighted-edge/data tests need unique edge payloads, parallel edges and explicit attention-return checks, rather than assuming that coverage already exists.

COO score generation traverses EID-ordered edges, so grouping can reuse l[u] or r[v]. CSC softmax and aggregation traverse destination rows, but gather/scatter EID-indexed scores/weights; changes in EIDs and within-row neighbor order can affect those accesses even when node IDs do not change. Merely relabeling an unchanged CSC traversal is ineffective; inspect both sparse index arrays and EID maps, not just COO metadata. Existing recordings show actual changed arrays/maps for grouped cases.

Two further **B** candidates are (i) destination grouping with a deterministic within-row sort by source degree, source ID and original EID, and (ii) endpoint-block ordering `(destination//64, source//64, destination, source, original_EID)`, with block size fixed for the first comparison. They could improve source-feature reuse or cluster neighboring gathers. Conversely, they can scatter writes, change reduction order and lose locality for the other endpoint. DGL compressed conversion may erase some custom order; discard any candidate that canonicalizes to the same execution representation.

Sorting edges by destination degree alone does not change the numerical order of destination rows in a fixed CSC matrix. It is not automatically a new degree-aware row schedule. That requires node relabeling or a scheduler change. Endpoint-block edge ordering also overlaps “segmented layout”; do not count the same EID permutation twice.

Evidence and uncertainty are in Section 4: A for basic values, B for extras. Dataset-specific degree skew, source reuse, feature footprint and EID-map locality are plausible causes of different preferences, but the current timings do not isolate them. Confirm with an edge-only repeat at original nodes, all formats and fixed threads, retaining conversion cost.

### C. Node ordering

Retain original, official RCMK and stable ascending `(in_degree+out_degree, original_ID)`. Add descending `(-total_degree, original_ID)` and deterministic component-wise BFS (smallest remaining node as component root, neighbors visited by original ID). Generate maps without modifying connectivity; apply official custom permutation. Degree ties must remain deterministic in both directions; reversing an ascending list would also reverse ties and is a different rule.

RCMK targets adjacency bandwidth, not a guaranteed GAT optimum. BFS/partition grouping may put communities' projected features together; ascending/descending degree can change sparse work distribution and cache behavior. All can worsen another stage: dense GEMM keeps the same dimensions and contiguous row shape, but row data order, feature placement and sparse row ownership change. A permutation does not make arbitrary source gathers contiguous or prove dense projection itself becomes faster.

A partition candidate can use official METIS node reordering with 2/4/8 partitions where the build supports it, followed by a fixed intra-partition rule. METIS availability and determinism were not verified. Persist/hash the generated permutation; setting a seed alone is not proof of identical partitions. Its graph traversal, temporary storage and sorting can cost far more than simple degree sorting and need many repeated inferences to amortize. Treat this as D, not an unconditional stock feature.

Features follow their logical nodes, edge weights follow EIDs, and output rows return to original order. Current joint code precomputes an inverse NID map; `index_select` allocates/copies output every call for nonidentity node maps. End-to-end gains can disappear after this cost or at small inference counts. The original-EID policy must explicitly undo node-induced edge reordering using the valid custom edge-permutation API; `edge_permute_algo=None` is not such an API.

Evidence supports A for the three measured values in a stated edge context; B for descending/BFS, D for partitions. Confirm the Citeseer/PubMed degree-versus-RCMK reversal in a **dedicated node-only** test with source-grouped edges fixed. Do not conclude node ordering is universally weak from the original-EID context or universally useful from a joint winner.

### D. Projected node-feature layout

The logical tensor remains `[N,H,F]`; dimensions and values do not change.

| Physical organization | Logical `[n,h,f]` offset / strides in scalar elements | Required execution |
|---|---|---|
| Node-major NHF | `n*H*F+h*F+f`; strides `(H*F,F,1)` | Stock projection output `[N,H*F]` viewed as `[N,H,F]` |
| Head-major HNF | `h*N*F+n*F+f`; logical-view strides `(F,N*F,1)` | Real HNF packing or a producer writing it; consumers honor those strides |
| Node/feature/head NFH | `n*F*H+f*H+h`; logical-view strides `(F*H,1,H)` | Head-contiguous SIMD/consumer path, not ordinary contiguous NHF |
| Node-block AoSoA | `(((n//B)*H+h)*B+n%B)*F+f`, e.g. B=16 or 32 | Block-aware addressing and masked final-node block; not a simple new three-stride label |

A `.transpose()` view alone changes logical indexing/strides without moving data. A valid physical HNF alternative could pack `[H,N,F]` contiguously, then expose the logical view, but stock DGL conversion may immediately pack it back. Inspect actual strides and copies at **every** consumer. A custom layout also cannot be passed to a kernel that assumes flat NHF addresses and still claim equivalence.

Consumers are node-attention f-reductions and aggregation's source-feature gathers; projection must produce the layout, and output/next-layer interfaces may need conversion. Head-major can make head-specific source streams regular while worsening multi-head reuse, packing/GEMM batching or output flattening. NFH can aid head vectorization, but H=4 is small and its feature dimension becomes strided. AoSoA can improve cache reuse but adds address arithmetic, padding and code complexity. These effects plausibly depend on degree/reuse and N versus cache capacity.

A complete repack of float32 Z reads and writes at least `2*4*N*H*F` bytes, before any additional consumer normalization; this is an analytic traffic count, not a measured speedup. Stock PyTorch operations may accept strides, but the sparse boundary's `.contiguous()` is decisive. **C:** introduce only with a stride-aware producer and matching sparse consumer; hold their arithmetic and schedules fixed between the two tested layouts and include conversions. Do not create a tensor axis for a view that is erased.

### E. Edge attention/logit layout

For logical `[E,H]`, edge-major uses strides `(H,1)` and head-major physical `[H,E]` exposed as `[E,H]` uses `(1,E)`. Appending a singleton yields `[E,H,1]`; its last stride is immaterial to values. Edge-major makes each edge's heads adjacent; head-major makes one head's edge stream contiguous. Actual H=4 limits head-vector width, while high-degree rows can make per-head neighbor streams important.

A destination-segmented layout stores entries by **CSC position**, with `EID[position]` mapping back to canonical edges; it is not merely a permission mask. This can avoid repeated random EID gathers in row softmax and aggregation, but requires the score producer, softmax and aggregation to agree on the mapping. Associated fixed edge weights and returned attention must also map correctly. If it is only an EID permutation already expressed by edge ordering, merge the labels instead of adding another axis.

Stock CPU softmax processes destination row → head → incoming edges, reading EID-indexed logits and writing EID-indexed alpha. Head-major or CSC-position streams could help that traversal yet harm edge-parallel score production or multi-head aggregation. Maintaining different layouts for logits and alpha introduces an EH conversion. Each full EH float32 repack reads/writes `2*4*E*H` bytes; a transpose view that DGL normalizes adds this cost without proving a useful consumer change.

Logits and normalized attention are separate logical intermediates; a fusion can eliminate one or both. Layout values for an absent intermediate are invalid. **C:** first compare two real edge layouts within one fixed consumer family, log actual strides/positions and count copies; current repository has no layout timings. Higher degree and E relative to cache/bandwidth motivate the hypothesis, not an invented ranking.

### F. Sparse execution schedules and loop ordering

The meaningful loop axes differ by stage. Score formation after l/r uses edges and heads, **not** a per-edge reduction over F. Aggregation uses destination, incoming edge, head and feature; softmax uses destination/head and reductions over incoming edges.

Candidate execution organizations include:

1. COO edge-owned score tasks: independent edges, inner heads, gather l/r; good load balance, potentially poor endpoint reuse.
2. CSR source-row-owned score tasks: reuse source contribution, inner neighbors/heads; row-length imbalance, potentially scattered EID writes.
3. CSC destination-row-owned reductions: reuse destination state and avoid cross-owner output races; long rows can serialize.
4. COO/source-oriented aggregation: source-feature reuse and edge work sharing, but destination collisions need atomics or private partial outputs.
5. Head-outer versus edge-/row-outer loops, and head-feature blocking within aggregation, with actual compiled nests differing.
6. Static contiguous row ranges versus edge-count-balanced row ownership or split heavy-row tasks.

Generic v1.1.3 CPU source shows CSR score row parallelism, COO score OpenMP edge parallelism, CSC-backed SpMM row ownership and COO sum atomics. Some CPU SpMM paths can use optional library-specific implementations; the recordings do not identify the executed inner variant. High-level GATConv does not expose these loop orders. Per-stage primitive routing can test some alternatives; other schedules require a C family.

Moving a loop outside another only matters if it changes traversal, reuse, task ownership or work. If a wrapper's two schedule names call the same compiled kernel, deduplicate them. GEMM's six loop permutations do not transfer literally: sparse adjacency defines ragged iteration, and a source/destination “swap” without changing graph view can change the mathematical neighborhood. If source-equivalent CSR-transpose and CSC labels use identical arrays/owners, they are aliases, not two axes.

Different degree distributions and graph/feature working sets plausibly change the tradeoff between balancing edges, reusing endpoints and avoiding synchronization. First compare one schedule within fixed layout/format/dtype; do not mix kernel replacement, packing and thread changes into a claimed loop-order effect.

### G. Softmax scheduling

Preserve destination-wise stable softmax: subtract the **global maximum for each destination/head**, sum exponentials across **all** its incoming edges, then divide by that same denominator. Self-loops prevent empty rows in current workloads; a general implementation must preserve the reference's zero-degree handling.

Initial C alternatives: destination-row ownership with static scheduling; degree-bucketed rows with the same arithmetic; heavy-row splitting with deterministic partial merges; row→head→edge versus head→row→edge; and per-thread reusable scratch instead of per-row vectors. The generic CPU source already allocates row-length scratch and processes heads inside each destination. These observations identify real implementation opportunities, not a claim that allocation dominates recorded full time.

Many low-degree rows can favor low task overhead and reuse of small scratch. Skewed/high-degree graphs may benefit from split rows or head parallelism when row parallelism is insufficient, at the cost of extra scratch, barriers and merges. Large per-thread scratch sized to the maximum row can waste substantial memory; bounded chunks need correct global normalization. Independently normalizing each chunk and concatenating is **not equivalent**.

Stable split reductions can compute row maxima and sums in coordinated passes, or merge stable `(maximum, denominator)` states. Head-outer streaming helps a head-major tensor but can walk adjacency H times. Changing both together is a coupled implementation family, not an independent schedule measurement. No new softmax schedule was timed; first test row scheduling or scratch policy alone, retaining layout, exp function and reduction rules.

### H. Aggregation scheduling

Destination ownership is the natural CSC reduction: a worker accumulates all incoming weighted projected features for its output row. The generic source uses flattened head-feature offsets and need not allocate an EHF message tensor. It avoids atomics but suffers long-row imbalance and irregular source gathers.

Source-oriented/edge-parallel alternatives can reuse Z[u] or balance irregular edge counts, but multiple workers update the same destination. Floating atomics incur contention/cache-line bouncing; locks serialize; per-thread partial `[N,H,F]` outputs cost memory plus a reduction and can be impractical for large N or many threads. The existing COO primitive has an atomic sum route, so a first routing experiment can precede a new kernel. Do not assume torch's deterministic flag controls all DGL CPU atomic interleavings.

Degree-aware split heavy rows can accumulate small private partial output vectors and merge them deterministically. This needs extra tasks and a reproducible merge order. Possible nests are row→neighbor→head→feature, row→head/feature-tile→neighbor, or head-tile→rows→neighbors. Rewalking indices can improve vector accumulation locality but increases index/weight loads. Source-rich neighborhoods, high destination degree and F/H relative to vector/cache capacity determine the hypothesis.

All alternatives preserve the weighted sum mathematically, but floating reduction order can change. Validate finite extreme inputs, relative/absolute errors, repeated-run nondeterminism and fixed edge weights; keep the same FP32 contract. **C:** prefer destination ownership versus a degree-aware schedule in one family before a broad atomic/private-buffer sweep. No aggregation schedule timings exist here.

### I. Cache tiling and chunk sizes

Start small and explicit, not with a hardware-independent “optimal” value:

| Candidate control | Initial values | What must actually change / constraint |
|---|---|---|
| Sparse node/row block | 16, 64, 256 rows | Real work ownership or feature-cache block; if it merely aliases scheduling grain, expose one parameter |
| Edge task/chunk | 64, 256, 1024, 4096 edges | Work grouping; split softmax/aggregation rows require partial-state handling, not independent row normalization |
| Feature tile | 2, 4, 8 features | Logical F remains 8; values above F often clip to identical execution and should not inflate the domain |
| Head tile | 1, 2, 4 heads | Logical H remains 4; smaller tiles may repeat graph traversal, larger tiles increase live accumulators |
| Projection N blocking, if exposed by a separate implementation | 32, 64, 128 rows | Changes actual GEMM call/block structure; stock BLAS's internal tile hierarchy is not a GATConv argument |

At H=4,F=8, each projected node is 128 bytes; a 64-node Z block is 8 KiB, before indices, weights, output and raw input X. Large D can make an X block much larger. These are sizing calculations, not performance observations or proof of available private cache. The VM's reported caches are not calibrated dedicated resources.

Tiles can trade cache reuse and vectorization against repeated adjacency walks, scratch size, scheduling and packing. Feature/head tiling, physical blocking and loop order are often one consumer implementation, not three free knobs. Avoid testing a tile size that the selected kernel ignores. Padding a physical tail is permissible only with masks preserving logical N/H/F; changing a logical feature width is not tiling. **C** conditional on a real kernel, with applicability D for hardware/workload. Existing code provides no generic tile sweep or measurements.

### J. CPU thread count

Use **1,2,4** as the initial values on this environment. The VM exposes affinity CPUs 0–4, but `cpu.max=400000 100000` grants four CPU-equivalents; five advertised logical CPUs do not justify an oversubscribed fifth-thread candidate. Additional counts require another allocation or a documented oversubscription experiment, not an assumed benefit.

PyTorch intra-op, DGL OpenMP and MKL/OpenBLAS resources must be configured coherently before worker initialization, with inter-op fixed to one. Record effective runtime parallel settings; a single environment variable does not prove all libraries honor it. Parallel projection/sparse loops can reduce time, while fork/join barriers, small per-row work, load imbalance, bandwidth saturation and competition between libraries can negate it. These are H: every existing sweep used one thread, so no 2/4-thread ranking is available.

Class **D**: treat thread budget as a hardware-specific **outer configuration**, controlled within other parameter experiments. A deployment allowed to select resource usage could later condition a tensor on the thread budget; it should not confuse a four-core solution with a one-core kernel-layout improvement. First run one thread-count-only experiment with default graph/layout/dtype. Use wall time and CPU/resource accounting; do not assume linear speedup or enable nested oversubscription.

### K. CPU affinity and NUMA

Observed current topology: KVM x86-64, five exposed CPUs, one exposed socket, **one NUMA node (node0)**, four-CPU quota. The measurements did not verify host physical-core exclusivity or remote-memory access. These facts cannot support a local-versus-remote NUMA performance claim.

Initially control placement: default OS scheduling versus a documented allowed four-vCPU mask, and verified OpenMP close/spread binding if the runtime honors it. Binding can reduce migration/cache disruption; it can also constrain load balancing or place threads poorly. In a VM, guest pinning does not guarantee host-core pinning. Capture effective thread affinities and library-specific controls rather than assuming OMP settings bind every BLAS worker.

On a genuine permitted multi-NUMA-node machine, compare local first-touch, interleaved pages and deliberate remote placement with the graph/features and thread team fixed. Page placement, memory bandwidth, remote latency, node/edge partitioning and preprocessing ownership can matter greatly. First-touch must be performed by the intended thread team; pinning without page control is not a NUMA experiment.

Class **D** across hardware; keep these as controlled settings now. A “NUMA choice” that always maps to current node0 is an **E alias**, not a second value. No current affinity or NUMA timings were measured, and no additional hardware is assumed available.

### L. Sparse index dtype

DGL offers `g.int()` and `g.long()`; v1.1.3 CPU SDDMM/SpMM include int32/int64 float specializations, and softmax dispatches on graph ID type. This is source/API support, not a newly executed end-to-end dtype test.

All current graphs have fewer than 2^31 nodes and edges, including Reddit after loops. Check **intermediate offsets too**: the generic CPU softmax stores flattened edge/head locations in an IdType vector. For the required Reddit H=4, `E*H=459,395,428`, still below 2^31. Larger H could make int32 unsafe even when E alone fits. Conversely, do not require a hypothetical EHF message offset to fit when the fused baseline never materializes that array; inspect the actual addresses in each implementation family. Validate all row pointers, EID maps, vertex IDs and operator-specific address bounds; user tensors/maps may require explicit integral conversion at framework interfaces.

Int32 halves the storage/traffic of each cast graph index array; it does not halve float features, logits or attention. For illustration only, separate COO plus CSR plus CSC arrays with explicit EID maps contain roughly `6E+2(N+1)` indices; at Reddit's preprocessed E, a 64→32-bit cast saves about **2.57 GiB** in that nonsharing storage model. Sharing/implicit maps change this estimate. Each EH float32 edge tensor at four heads is about **1.71 GiB** and is unaffected. These analytic estimates do not establish measured peak memory or that the full Reddit experiment fits.

A cast copies structures and may trigger materialization; compare materialized steady-state execution and separately charge one-time conversion. Widening arithmetic, cache fit and relative dense-projection work can change the benefit. If int32 always wins where legal, use it as a domain-dependent default rather than retaining a redundant model dimension. **B, highest-priority new stock comparison:** test both complete forwards, topology/EID identity and numerical equivalence before measuring the dtype axis alone.

### M. Kernel fusion candidates

Fusion is an implementation family with explicit surviving interfaces, not a free boolean independent of schedules/layouts.

| Candidate | Unfused reference / actual proposed alternative | Removed intermediates or passes | Risks and dataset dependence | Category |
|---|---|---|---|---|
| Score addition + LeakyReLU | `u_add_v(l,r)` writes q; PyTorch activation reads q and writes s. A fused CPU edge loop writes s directly | Removes q's EH write/read, roughly `2*4*E*H` bytes of traffic, plus one dispatch; s still exists | Preserves additive attention. Endpoint gathers dominate some graphs; larger E can make saved traffic more valuable; head layout/vectorization still matters | C, first small fusion |
| Projection + node contributions | Shared XW writes Z, then two node f-reductions produce l/r. A projection epilogue produces Z plus both scalars while accumulators are resident | Avoids extra Z reads/product temporaries and launches; Z must remain for aggregation | Requires epilogue support or custom producer, handling output/feature tiles and partial dot reductions. GEMM efficiency can be lost; D, N and HF determine balance | C, later producer family |
| Softmax + aggregation | Materialized alpha followed by weighted reduction; stable per-destination fused reducer computes denominator and weighted numerator | Can eliminate alpha write/read and a stage boundary, but not magically every score or source-feature load | Stable global normalization, row imbalance, scratch/parallel merges and optional attention output; most attractive when EH traffic dominates, not guaranteed on small rows | C, higher complexity |
| Score/activation + softmax | Materialize edge scores, then normalize; compute/recompute l+r+LeakyReLU inside incoming-row passes | Can eliminate q/s storage depending on passes | Extra endpoint gathers/arithmetic versus saved EH traffic; compatible CSC-position attention and preserved EIDs required | C, conditional extension |
| Aggregation + output epilogue | Aggregate Y, then bias/activation/residual processing; row owner finishes the fixed epilogue | Avoids an NHF pass/temporary where applicable | Output layout/contiguity, residual source and bias placement must match reference; lower traffic target than EH on high-degree graphs | C |
| Message multiplication + summation | The builtin baseline already uses fused gSpMM | No baseline EHF message tensor to eliminate | Claiming this as a novel saved EHF buffer over stock is incorrect | E as a new axis |

**Stable softmax–aggregation detail.** A correct two-pass alternative first finds each row/head maximum, then accumulates `denom=sum exp(s−max)` and `numerator=sum exp(s−max)*Z`, finally divides. With fixed edge weights, multiply the numerator contributions by w, not the denominator. This preserves the GAT equation with floating-order differences requiring validation.

A stable online alternative maintains `(m,l,o)` and, for a new score s and value z, updates with `m'=max(m,s)`, `a=exp(m−m')`, `b=exp(s−m')`, `l'=a*l+b`, `o'=a*o+b*z`; final output is o/l. Parallel chunks require correspondingly rescaled merges, not averaging independently normalized chunks. Max subtraction protects exponentials, but an unnormalized weighted numerator can still overflow where a normalized-weight reference would remain finite; verify the full FP32 range/accumulation behavior rather than declaring the fusion correct from algebra alone. Rescaling vector accumulators, additional exponentials/branches and deterministic merges can outweigh saved traffic, especially for small F or low-degree rows. Do not claim online one-pass processing is universally faster or bitwise identical.

Dropout/training backward are outside the present inference family. Returning attention can require retaining/recomputing alpha and restoring its EID order, so a no-alpha family cannot silently satisfy that interface. A fused S+G family makes the materialized-alpha layout **absent**, not another selectable EH layout. Kernel ownership, scratch and tiles may also become fixed by the fusion.

**Composite transfer limit:** stock homogeneous GATConv already computes one shared projection (`feat_src=feat_dst`). There are not two independent WX/WY GEMMs to fuse. Projection/contribution epilogues are a different, GAT-specific opportunity. Algebraically precontracting W with attention vectors would be another reassociated implementation requiring its own fixed-weight/update-cost and numerical validation; it is not dot-product edge attention and is not an immediate additional axis here.

### N. Explicit exclusions: descriptors and model settings are not free optimization axes

| Quantity / setting | Why exclude from the equivalent-computation optimizer |
|---|---|
| Input feature dimension D | Changes X, W and available information. Existing prefix/zero-padding experiments also create differently shaped models/weights; their faster times are different workloads |
| Output features per head F | Changes output dimension, projection, scores' reductions and aggregation work |
| Head count H | Changes attention distributions, learned parameterization and total output dimension; the Reddit one-head result is not a four-head optimization |
| Node count N | Changes the graph function/domain and input/output set |
| Edge count E | Changes neighborhoods, normalization and aggregation |
| Connectivity | Even at fixed N/E, changes which features are combined; permutation/isomorphic relabeling with maps is the legitimate exception |
| Dropout probabilities | Training changes stochastic computation; in eval mode different p labels are ineffective no-ops, not a useful execution axis |
| Activation | Changes the graph-layer function; fixed None in current records |
| Residual connection | Changes the output equation and often adds another learned transformation |
| Attention negative slope | Changes edge scores and normalized attention; retain 0.2 |
| Bias setting | Changes the declared model. Initial zero bias does not justify treating bias on/off as a general axis; provable constant-folding of fixed zero values is a separately scoped specialization |
| Edge weights | DGL applies them after normalization; changing them changes weighted aggregation. Reordering associated weights is required, not optimizing their values |
| Feature/weight numerical values | Fixed workload inputs/parameters, not tuner choices; their sparsity/value patterns can be descriptors but may not be altered to manufacture a faster implementation |
| FP32→lower precision, approximate exp/softmax, sampling/pruning, changed attention algorithm | Different accuracy/numerical/algorithmic contracts, not this strict study. Reordered exact reductions may differ within a validated tolerance; arbitrary approximations cannot be smuggled in under that tolerance |

Physical tail padding with masked inactive lanes is not a logical dimension change. Quantization or approximate normalization is. Fixed workload descriptors can condition a future cost tensor across problems, but the optimizer must not choose a cheaper, different problem.

## Section 6 — Cross-dataset tuning usefulness

The strongest available answer is **node-order ranking can vary across workloads under a fixed edge-policy context**, not “joint tuning is better.” Original OPAT degree-versus-RCMK preferences on Citeseer/PubMed are corroborated by the fixed-source-edge six-round slices, with pairwise unadjusted intervals excluding one in opposite directions. The Cora preference remains noise-scale. With original EIDs fixed instead, PubMed node benefits disappear, showing why the context must be recorded rather than generalized away.

Edge ordering has a useful PubMed destination-order effect and weak/inconclusive Cora/Citeseer differences. No pair of edge options is credibly shown to reverse superiority across those smaller graphs. Nor is destination ordering proved to dominate everywhere: Citeseer lacks a positive contrast, and Reddit has no data. Keep the axis, confirm independently and expand workload coverage later instead of discarding it or declaring a universal winner.

The two supported stock masks do not establish a useful active-kernel ranking: all three fixed-order ratio intervals include one and audited source requests select the same COO/CSC routes. They can still affect memory/materialization or an implementation family with different consumers. Distinguish “inactive runtime choice in this family” from “sparse representations are unimportant.”

No performance rankings exist for descending degree, BFS/partition ordering, real alternate tensor layouts, custom schedules/tiling/fusion, 2/4 threads, affinity/NUMA or index dtype. Their taxonomy classifications are feasibility/mechanism judgments, not measured winners. Three citation datasets and one high-degree memory-skipped graph are insufficient to reject promising mechanisms.

Future graph coverage should characterize mean and degree tails, community/locality structure, source reuse, N/E, feature footprint, working-set size and inference count. Across datasets these are **descriptors**. Within each independent experiment, keep the original graph and dimensions fixed. Obtain missing degree quantiles in a future metadata pass, not by asserting that an average describes the tail.

## Section 7 — Recommended shortlist and compatibility dependencies

### Ready to retain now

**A: node ordering and basic edge ordering**, with original as a mandatory control, actual representation fingerprints and original-node/EID output contracts. Retain their correct implementations and measured candidate domains now. Before fitting a general model, independently confirm the node reversal and PubMed edge effect, and include runtime restoration and one-time costs. Do not use current A status as evidence that every value or every graph benefits.

### Implement/measure next

**B: index width**, then the existing **node/edge confirmations**, descending degree and one deterministic locality ordering. **D: 1/2/4-thread response** should establish the resource regime before interpreting new kernel mechanisms. **C: per-stage primitive orchestration** is the best next implementation family: it can reveal real sparse-route choices without immediately replacing all CPU kernels. Next prioritize one degree-aware reduction schedule and the small add/LeakyReLU fusion.

### Defer, rather than reject

Layouts until a matching consumer family exists; softmax+aggregation fusion until stable reduction/interface tests are established; multi-level tiling until there is an actual kernel knob; partition order until build support and amortization justify it; NUMA until a real multi-node resource is available. No timing claims are made for these candidates.

### Exclude / collapse

Workload-changing settings; invalid whole-layer policies in the current family; transpose-only layouts erased by contiguous conversion; kernel/schedule labels that lower to identical execution; unused tiles that clip to the same loop; destination segmentation that is just an already-counted edge permutation; fictitious duplicate homogeneous projections; and claimed message-multiply/reduce fusion already present in stock.

### Legal dependency structure

| Choice | Compatibility condition / required state |
|---|---|
| Stock whole GATConv | Only pilot-valid complete-forward masks; no per-stage schedule/layout control |
| Stage score route | CSR or COO kernel must receive its corresponding view; EID-indexed outputs must match downstream maps |
| Row softmax | Incoming row segmentation and globally stable per-row/head normalization; outgoing CSR is not a substitute unless the graph orientation is explicitly transformed |
| Aggregation ownership | CSC destination owner versus COO edge owner fixes synchronization requirements; some “schedule” names are consequently aliases of the chosen route |
| Z/alpha layout | Every producer and consumer must implement the physical addresses; insert/charge conversions if adjacent states differ |
| S+G fusion | Materialized-alpha layout becomes absent; attention-return interface, reduction schedule and scratch must be compatible |
| Projection epilogue | Must produce the downstream Z layout and both attention contributions; GEMM layout/tiling restrictions constrain its domain |
| Tile size | Active only for the selected kernel; legal head/feature tiles respect fixed H/F and tails; scheduler chunk and cache tile may be the same control |
| Index width | All graph and flattened operator offsets fit; consumer specializations and map casts exist |
| Node/edge permutation | Topology, feature and edge data maps remain consistent; externally visible rows/EIDs restored |
| Threads/placement | Thread budget and placement are compatible with quota/allowed CPUs; NUMA value exists on actual hardware |

Do not form an unconstrained Cartesian product of these names. Use an implementation-family selector with conditional domains, merge aliases and mark unavailable combinations invalid rather than assigning them invented costs. “Independent” in a one-factor experiment means only that the tested parameter changes while all other **legal** choices are fixed; it does not guarantee globally orthogonal implementation knobs.

## Section 8 — Prioritized next experiments (independent only)

These are proposed experiments, **not executed in this milestone**. Each uses Cora/Citeseer/PubMed, native features, H=4,F=8, float32, identical weights, inference/dropout settings, fixed topology/self-loops and original output contracts. Use new result paths. For Reddit, estimate actual working set and retain the guard; use one configuration at a time where safe, never reduce heads/features to claim the same workload.

| Priority | One factor changed; exact initial values | Everything else held fixed | What would confirm usefulness / required correctness |
|---|---|---|---|
| 1 | Sparse indices: int64 vs int32 | Original nodes/EIDs, all formats, 1 thread, stock complete layer | Complete forward for both; exact graph/EID/data mapping, bounds including EH, output tolerance. Record cast/materialization bytes/time, peak memory and independent paired latency; determine whether gains vary or int32 simply dominates |
| 2a | Node order: original/RCMK/ascending degree | Source-grouped edge policy, all formats, 1 thread | Predeclare Citeseer-vs-PubMed RCMK/degree pair. New independent randomized processes; combined-time interval and rank reversal, permutation/preprocessing cost and amortization |
| 2b | Edge order: original/src/dst | Original nodes, all formats, 1 thread | Confirm PubMed destination effect; resolve small Cora/Citeseer differences without borrowing node-tuned results; exact topology and edge payload/attention-EID mapping |
| 3 | CPU threads: 1/2/4 | Original orders, all formats, int64, original layouts | Configure libraries before imports, inter-op=1; effective affinities/resource budget, full timing and stage diagnostics. Keep this hardware response separate from kernel-axis comparisons |
| 4 | Descending degree, then one BFS candidate, each versus original control | Fixed source-grouped edges/all formats, 1 thread | Actual nonidentity maps and traversal changes; output restoration; independent per-dataset benefit large enough to amortize sorting |
| 5 | Score-stage view: COO vs CSR only | Separate audited forward orchestrator; softmax/aggregation CSC; original orders/layouts/indices/threads | Reproduce official additive l+r attention, degree guard and weights; trace actual primitive route. Include one-time views and any EID conversion. Different score loop paths plus reproducible end-to-end difference justify a new format/schedule axis |
| 6 | Aggregation-stage route: CSC ownership vs COO atomic primitive | Same orchestrator, fixed score COO and softmax CSC, original layout/order, one fixed thread regime | Full numerical/repeated-run tests, degree dependence, synchronization and workspace costs. Change only aggregation; do not simultaneously introduce layout or thread tuning |
| 7 | One custom reduction schedule: static row ownership vs degree-balanced ownership | One validated CSC kernel, existing layout, exp/aggregation arithmetic, threads and tile sizes fixed | Exact incoming edge coverage and stable normalization; inspect tail imbalance and per-stage/end-to-end timing. First softmax OR aggregation, not both in the same experiment |
| 8 | Score add/Leaky fusion: unfused vs fused | Same traversal, layout, threads, model and downstream operators | Compare scores as well as final outputs on finite/extreme test inputs; show eliminated q allocation/pass; include compilation/setup outside timing. Graph-dependent benefit determines whether to retain the family |
| 9 | One real physical layout: NHF vs HNF (later EH vs HE in its own experiment) | One stride-aware producer/consumer family, same arithmetic, loop schedule and resources | Log strides, addresses/copies and traffic; full time including all repacks and output contract. A view normalized back to NHF is a failed layout implementation, not a benchmark win |
| 10 | One active chunk/tile variable at a time | One previously validated kernel/schedule/layout family | Use the candidate sets in 5.I, validate tails and unchanged dimensions. Deduplicate clipped/ignored settings; count extra passes and scratch, never assume a larger block is better |

For first confirmations, choose a fixed number of fresh-process paired rounds (e.g. 10) before seeing outcomes, warmups and sufficient repetitions to stabilize each block. Randomize value order, interleave **the same-context** control and report per-round ratios/intervals; do not repeatedly sample until a preferred value becomes significant. Predeclare a practical threshold, treat within-block repetitions separately from independent trials and account for testing multiple candidate pairs. Additional rounds or graph cohorts require a stated follow-up plan, not post-hoc deletion of slow samples.

Correctness tests should cover parallel edges, isolated nodes before the declared self-loop policy, high-degree destinations, distinct edge weights/payloads, nontrivial node permutations, attention-return EID order, nondivisible tile tails and numerical extremes. Keep the complete untouched GATConv oracle, as well as stage outputs for locating failures. Test 1/2/4 threads separately where nondeterministic reductions are possible. Profiling identifies execution/copies but is a diagnostic run separate from wall-time measurement; profiler overhead must not contaminate reported forward time.

No proposed step combines all new parameters or performs another joint/factorial study. No predictive tensor is trained by these experiments.

## Section 9 — Future tensor model and ModelingWithTensors transfer

### What transfers, and what must change for GAT

| Conceptual reference | Useful transfer | GAT-specific adjustment / limit |
|---|---|---|
| GEMM loop ordering | Real reuse, SIMD and ownership patterns matter, not names | Projection is already a BLAS-backed dense GEMM; its internal permutations are not stock GAT knobs. Ragged sparse loops do not inherit six dense permutations |
| GEMM multi-level tiling | Separate register/cache/thread work units and charge packing | Condition tiles on an actual producer/consumer kernel, H/F bounds and graph reuse; sparse neighbor sets are not rectangular dense tiles |
| SDDMM traversal/formats | Distinguish edge ownership from compressed-row ownership; canonicalize equivalent lowered schedules | DGL GAT's edge operation here is **add of node contributions**, not a sparse per-edge dot product. The request notes that some ModelingWithTensors SDDMM labels collapse; without its source the exact collapsing map cannot be verified or copied |
| Composite intermediate layouts | Model producer-layout, conversion and consumer-layout compatibility explicitly | Relevant intermediates are Z, l/r, logits and attention. Stock contiguous normalization can erase proposed layouts; output restoration and EID maps are part of the contract |
| Composite projection fusion | Search for actual redundant passes/intermediates and epilogues | Homogeneous GAT already shares one XW projection. Copying WX/WY fusion would claim savings for work absent from this baseline |
| Min-plus / tensor cost representations | Costs can be composed over legal intermediate states; measured costs can be indexed by implementation choices | Only when stage contracts/dependencies permit composition. Softmax depends on a whole destination neighborhood, fused stages remove boundaries, and caches/thread teams couple costs; summing isolated microbenchmarks need not equal full-forward time |

No ModelingWithTensors repository was present among workspace project directories, and the reviewed DGL repository contains no implementation of those reference experiments. The preceding links are conceptual relationships, not source-verified claims about its exact six loop orders, tile hierarchy, projection code or cost-tensor implementation. Obtaining the specific reference revision is a prerequisite to a later code-level comparison.

### A legal cost-tensor domain, without implementing it

For each workload descriptor `w=(graph statistics/topology, D,H,F, feature/weight contract)` and hardware/resource context `h`, eventual measurements could index cost by **eligible implementation family and independent choices within it**. Begin with measured node/edge domains; add index width after validation. Treat threads/placement as explicit outer context unless resource selection itself is the deployment objective. Keep N/E/D/H/F as descriptors, not minimizable settings.

For separate operators, compatible boundary states can include physical Z/attention layouts, sparse orientation/indices, canonical-EID maps and whether a tensor is materialized. A min-plus composition could conceptually minimize `producer cost + conversion cost + consumer cost` over a legal shared state. Fusion would replace multiple factors by one measured family and remove the intermediate state. This is a planning description only: **no tensor, fitted cost function, predictor or optimization model is built here**.

Missing/unmeasured costs are **unknown**, not zero. Unsupported combinations are masked as invalid; hardware-inapplicable values are absent; aliases share a state. Do not create independent dimensions for format and schedule when a primitive fixes both, or for alpha layout when a fused family never materializes alpha. End-to-end validation is necessary because cache residency, scheduling, dispatch and allocation can prevent additive composition of isolated costs.

### Source references and reproducibility of this review

All existing files remain unchanged. Recreate only this report's numerical evidence with:

```sh
UV_CACHE_DIR=/workspace/.cache/uv uv run --frozen python benchmarks/parameter_evaluation/analyze_existing.py
```

This command reads stored JSON and writes only `EVIDENCE.json`, `EVIDENCE.md` and the marked evidence block in this new report. It executes no benchmark, kernel or model fit.

Version-pinned source inspected:

- [DGL GATConv](https://github.com/dmlc/dgl/blob/v1.1.3/python/dgl/nn/pytorch/conv/gatconv.py), also preserved locally with stage comments.
- [Kernel dispatch](https://github.com/dmlc/dgl/blob/v1.1.3/src/array/kernel.cc): `SDDMM`, `SpMM`, `Edge_softmax_forward`.
- [Format selection](https://github.com/dmlc/dgl/blob/v1.1.3/src/graph/unit_graph.cc): `SelectFormat`, incoming/outgoing compressed storage and restrictions.
- [CPU score traversal](https://github.com/dmlc/dgl/blob/v1.1.3/src/array/cpu/sddmm.h) and [dtype instantiations](https://github.com/dmlc/dgl/blob/v1.1.3/src/array/cpu/sddmm.cc).
- [CPU aggregation and stable softmax](https://github.com/dmlc/dgl/blob/v1.1.3/src/array/cpu/spmm.h) and [SpMM dtype specializations](https://github.com/dmlc/dgl/blob/v1.1.3/src/array/cpu/spmm.cc). Generic source is not proof of which optional inner path a wheel executed.
- [PyTorch sparse wrappers](https://github.com/dmlc/dgl/blob/v1.1.3/python/dgl/backend/pytorch/sparse.py), [conversion helpers](https://github.com/dmlc/dgl/blob/v1.1.3/python/dgl/backend/pytorch/tensor.py), and [DGL sparse-op bridge](https://github.com/dmlc/dgl/blob/v1.1.3/python/dgl/_sparse_ops.py).
- [Graph APIs including index casts](https://github.com/dmlc/dgl/blob/v1.1.3/python/dgl/heterograph.py) and [ID dispatch macros](https://github.com/dmlc/dgl/blob/v1.1.3/include/dgl/aten/macro.h).

Downloaded source excerpts used for inspection remain outside the repository. Failed exploratory source-path lookups are recorded in SOURCES.json as unavailable and are not cited as evidence. No new performance measurements are reported for unimplemented parameters.

### Recommendation to discuss before proceeding

**Retain measured node and edge orderings; evaluate index width next; establish the thread-resource response; then expose per-stage sparse primitives before committing to new kernel families.** These choices have either actual controlled evidence or concrete execution differences supported by source. Degree-aware reductions and small exact fusion are the next custom mechanisms; real layouts and cache tiles follow a matching consumer, not precede it. Confirm node-rank reversal and the PubMed edge effect with dedicated independent tests. Keep stock format masks as memory controls, and leave NUMA, partitions and larger fusion conditional on the appropriate hardware/interface and evidence. This sequence supplies meaningful candidates for a later cost tensor without confusing a changed GAT workload, an ineffective label or a joint winner with an independent tuning dimension.
