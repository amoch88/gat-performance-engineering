# DGL GAT CPU baseline parameter experiments

## Objective and scope

This report contains **82 measured real-dataset case/control records**. Read skipped and unsupported entries separately: they contain no performance measurements.

Measure how public GATConv constructor arguments and public graph representation APIs affect full-forward CPU latency. No prediction, fusion, custom attention algorithm, optimized kernel or training is introduced. Changed dimensions are different workloads; they are not equivalent-computation speedups.

All raw samples and configuration/representation/provenance records are retained in [MEASUREMENTS.json](MEASUREMENTS.json), alongside this report for review.

## Implementation reference and environment

DGL **1.1.3**, [official annotated source reference](https://www.dgl.ai/dgl_docs/en/1.1.x/_modules/dgl/nn/pytorch/conv/gatconv.html). `baseline.py` copies the full installed implementation, cross-checked byte-for-byte against the v1.1.3 upstream source before annotation. The executable class AST is checked against the untouched installed class. Imports/backend/CLI and comments are the only adaptations. The Apache-2.0 license is retained in `DGL-LICENSE`. The five labeled logical stages are not a claim about five compiled CPU kernels.

| Field | Observed value |
| --- | --- |
| cpu_model | INTEL(R) XEON(R) PLATINUM 8573C |
| logical_cpus | 5 |
| cpu_affinity | [0, 1, 2, 3, 4] |
| os | Linux-6.18.57-x86_64-with-glibc2.41 |
| architecture | x86_64 |
| python | 3.11.17 |
| dgl | 1.1.3 |
| torch | 2.2.2+cpu |
| numpy | 1.26.4 |
| torch_threads | 1 |
| torch_interop_threads | 1 |
| thread_environment | {'OMP_NUM_THREADS': '1', 'OMP_DYNAMIC': None, 'OMP_PROC_BIND': None, 'OMP_PLACES': None, 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'} |
| memory_available_bytes | 33850634240 |
| docker_image | None |
| git_commit | None |
| git_dirty | True |

Raw JSON includes per-run environment and source/lockfile SHA-256 hashes. An unset Docker identity means these records came from the native locked environment, not a claimed container measurement. The repository still has no commit when `git_commit` is null; source hashes identify the measured files.

Observed (seed, warmups, timed repetitions): **[(0, 5, 20)]**. Fixed CPU/OpenMP/MKL/OpenBLAS settings are recorded above.

## Datasets

Official DGL dataset classes are used with real topology and features. Citation loader `reorder=False` explicitly disables its default RCM preprocessing. “Original edge order” means the official loader’s resulting edge-ID order, not an asserted order from the original paper. Remove all self-loops, then append exactly one per node; preserve non-loop edges and retain real node features. Loading, feature transforms, permutations and sparse materialization are outside timing.

| Dataset | Nodes | Dataset edges | Edges after loops | Original width | Evidence |
| --- | --- | --- | --- | --- | --- |
| citeseer | 3327 | 9228 | 12431 | 3703 | observed |
| cora | 2708 | 10556 | 13264 | 1433 | observed |
| pubmed | 19717 | 88651 | 108365 | 500 | observed |
| reddit | 232965 | 114615892 | 114848857 | 602 | observed |

| Dataset archive | HTTP status | Redirects | Official loader |
| --- | --- | --- | --- |
| https://data.dgl.ai/dataset/cora_v2.zip | 200 | [] | passed |
| https://data.dgl.ai/dataset/citeseer.zip | 200 | [] | passed |
| https://data.dgl.ai/dataset/pubmed.zip | 200 | [] | passed |
| https://data.dgl.ai/dataset/reddit.zip | 200 | [] | passed |

## Standalone baseline controls

These controls use original dataset features and node/edge order after self-loop preprocessing; four heads, eight output features per head. Each sweep also measures its own in-process control for comparisons.

| Dataset | Median ms | Mean ms | Sample SD ms | Status |
| --- | --- | --- | --- | --- |
| citeseer | 7.654409 | 7.666688 | 0.223821 | measured |
| cora | 4.303292 | 4.508930 | 0.672870 | measured |
| pubmed | 19.422888 | 19.549413 | 0.796431 | measured |
| reddit | — | — | — | skipped_memory |

## Default configuration and measurement protocol

One stock GATConv: dataset-original input width, **4 heads × 8 output features/head**, float32, CPU, bias on, residual off, activation none, zero feature/attention dropout, eval/inference mode. All three formats are allowed/materialized in the ordinary baseline. Each sweep measures its own matching baseline in the same process. Node-ordering uses source-grouped edges for its baseline and every case. Seeded model construction is reset for every case; identical shapes have identical weight hashes. Changed shapes use the same seed but necessarily different weights/shapes.

Each record stores seed, warmups and repetitions. `perf_counter_ns` measures synchronous full forwards. Raw samples, median, mean, sample standard deviation and minimum are saved. Preprocessing/model initialization and setup/check/warmup costs are separate. The representation is checked before validation, before timing and after timing, including compressed index hashes and EID maps. No per-stage timing or compiled-kernel selection claim is made. Numerical comparisons use rtol=1e-4, atol=1e-5 and restore original node IDs.

## Input Features

Input widths 16, 32, 64, 128 and original; prefix-truncate or zero-pad original features before timing.

- citeseer: among measured cases, 16 has the smallest median (2.063128 ms) and 3703 the largest (7.618469 ms). This is descriptive, not proof of monotonic scaling or a causal optimization benefit.
- citeseer: sample coefficient of variation spans 3.5%–5.5%. Ratios are exploratory; this is one process trial per sweep.
- cora: among measured cases, 16 has the smallest median (2.117198 ms) and 1433 the largest (3.943592 ms). This is descriptive, not proof of monotonic scaling or a causal optimization benefit.
- cora: sample coefficient of variation spans 4.3%–10.7%. Ratios are exploratory; this is one process trial per sweep.
- pubmed: among measured cases, 16 has the smallest median (14.608636 ms) and 500 the largest (18.628540 ms). This is descriptive, not proof of monotonic scaling or a causal optimization benefit.
- pubmed: sample coefficient of variation spans 2.7%–3.6%. Ratios are exploratory; this is one process trial per sweep.

| Dataset | Configuration | Median ms | Mean ms | Sample SD ms | Baseline / case | Status |
| --- | --- | --- | --- | --- | --- | --- |
| citeseer | baseline: in=3703, heads=4, out/head=8 | 7.763924 | 7.795644 | 0.218549 | different workloads | passed |
| citeseer | 16: in=16, heads=4, out/head=8 | 2.063128 | 2.055113 | 0.071999 | different workloads | passed |
| citeseer | 32: in=32, heads=4, out/head=8 | 2.106689 | 2.132427 | 0.117707 | different workloads | passed |
| citeseer | 64: in=64, heads=4, out/head=8 | 2.137423 | 2.161598 | 0.080597 | different workloads | passed |
| citeseer | 128: in=128, heads=4, out/head=8 | 2.297160 | 2.316981 | 0.102175 | different workloads | passed |
| citeseer | 3703: in=3703, heads=4, out/head=8 | 7.618469 | 7.750763 | 0.357123 | different workloads | passed |
| cora | baseline: in=1433, heads=4, out/head=8 | 3.984689 | 4.175993 | 0.646153 | different workloads | passed |
| cora | 16: in=16, heads=4, out/head=8 | 2.117198 | 2.205668 | 0.231943 | different workloads | passed |
| cora | 32: in=32, heads=4, out/head=8 | 2.134022 | 2.164935 | 0.111611 | different workloads | passed |
| cora | 64: in=64, heads=4, out/head=8 | 2.167060 | 2.188867 | 0.093346 | different workloads | passed |
| cora | 128: in=128, heads=4, out/head=8 | 2.331255 | 2.450828 | 0.261963 | different workloads | passed |
| cora | 1433: in=1433, heads=4, out/head=8 | 3.943592 | 4.066262 | 0.318483 | different workloads | passed |
| pubmed | baseline: in=500, heads=4, out/head=8 | 19.320988 | 19.461562 | 0.807033 | different workloads | passed |
| pubmed | 16: in=16, heads=4, out/head=8 | 14.608636 | 14.588606 | 0.509854 | different workloads | passed |
| pubmed | 32: in=32, heads=4, out/head=8 | 14.613533 | 14.640313 | 0.519802 | different workloads | passed |
| pubmed | 64: in=64, heads=4, out/head=8 | 15.806769 | 15.770193 | 0.556760 | different workloads | passed |
| pubmed | 128: in=128, heads=4, out/head=8 | 15.449019 | 15.489042 | 0.411269 | different workloads | passed |
| pubmed | 500: in=500, heads=4, out/head=8 | 18.628540 | 18.808835 | 0.652409 | different workloads | passed |
| reddit | 16 | — | — | — | — | skipped_memory |
| reddit | 32 | — | — | — | — | skipped_memory |
| reddit | 64 | — | — | — | — | skipped_memory |
| reddit | 128 | — | — | — | — | skipped_memory |
| reddit | 602 | — | — | — | — | skipped_memory |

## Output Features

Output features per head 8, 16, 32, 64; four heads fixed.

- citeseer: among measured cases, 8 has the smallest median (7.784258 ms) and 64 the largest (50.619870 ms). This is descriptive, not proof of monotonic scaling or a causal optimization benefit.
- citeseer: sample coefficient of variation spans 2.4%–3.4%. Ratios are exploratory; this is one process trial per sweep.
- cora: among measured cases, 8 has the smallest median (4.138171 ms) and 64 the largest (19.633447 ms). This is descriptive, not proof of monotonic scaling or a causal optimization benefit.
- cora: sample coefficient of variation spans 2.9%–4.9%. Ratios are exploratory; this is one process trial per sweep.
- pubmed: among measured cases, 8 has the smallest median (20.022202 ms) and 64 the largest (90.769253 ms). This is descriptive, not proof of monotonic scaling or a causal optimization benefit.
- pubmed: sample coefficient of variation spans 4.5%–5.7%. Ratios are exploratory; this is one process trial per sweep.

| Dataset | Configuration | Median ms | Mean ms | Sample SD ms | Baseline / case | Status |
| --- | --- | --- | --- | --- | --- | --- |
| citeseer | baseline: in=3703, heads=4, out/head=8 | 7.638164 | 7.676719 | 0.184471 | different workloads | passed |
| citeseer | 8: in=3703, heads=4, out/head=8 | 7.784258 | 7.861410 | 0.251970 | different workloads | passed |
| citeseer | 16: in=3703, heads=4, out/head=16 | 14.008686 | 14.154600 | 0.482962 | different workloads | passed |
| citeseer | 32: in=3703, heads=4, out/head=32 | 25.661130 | 25.678598 | 0.612693 | different workloads | passed |
| citeseer | 64: in=3703, heads=4, out/head=64 | 50.619870 | 50.597727 | 1.429271 | different workloads | passed |
| cora | baseline: in=1433, heads=4, out/head=8 | 4.037659 | 4.051112 | 0.185676 | different workloads | passed |
| cora | 8: in=1433, heads=4, out/head=8 | 4.138171 | 4.149146 | 0.200888 | different workloads | passed |
| cora | 16: in=1433, heads=4, out/head=16 | 6.222178 | 6.345282 | 0.262753 | different workloads | passed |
| cora | 32: in=1433, heads=4, out/head=32 | 10.343096 | 10.586623 | 0.520731 | different workloads | passed |
| cora | 64: in=1433, heads=4, out/head=64 | 19.633447 | 19.641329 | 0.574931 | different workloads | passed |
| pubmed | baseline: in=500, heads=4, out/head=8 | 19.451295 | 19.456321 | 0.835106 | different workloads | passed |
| pubmed | 8: in=500, heads=4, out/head=8 | 20.022202 | 20.070505 | 1.056661 | different workloads | passed |
| pubmed | 16: in=500, heads=4, out/head=16 | 29.354941 | 29.805068 | 1.598329 | different workloads | passed |
| pubmed | 32: in=500, heads=4, out/head=32 | 48.018862 | 48.321731 | 2.775502 | different workloads | passed |
| pubmed | 64: in=500, heads=4, out/head=64 | 90.769253 | 91.311124 | 4.115238 | different workloads | passed |
| reddit | 8 | — | — | — | — | skipped_memory |
| reddit | 16 | — | — | — | — | skipped_memory |
| reddit | 32 | — | — | — | — | skipped_memory |
| reddit | 64 | — | — | — | — | skipped_memory |

## Attention Heads

Heads 1, 2, 4, 8; eight output features per head fixed. Total output dimension changes.

- citeseer: among measured cases, 1 has the smallest median (4.991043 ms) and 8 the largest (14.954272 ms). This is descriptive, not proof of monotonic scaling or a causal optimization benefit.
- citeseer: sample coefficient of variation spans 2.3%–11.5%. Ratios are exploratory; this is one process trial per sweep.
- cora: among measured cases, 1 has the smallest median (2.395611 ms) and 8 the largest (7.158593 ms). This is descriptive, not proof of monotonic scaling or a causal optimization benefit.
- cora: sample coefficient of variation spans 4.4%–7.4%. Ratios are exploratory; this is one process trial per sweep.
- pubmed: among measured cases, 1 has the smallest median (8.211536 ms) and 8 the largest (34.663075 ms). This is descriptive, not proof of monotonic scaling or a causal optimization benefit.
- pubmed: sample coefficient of variation spans 3.4%–4.6%. Ratios are exploratory; this is one process trial per sweep.
- reddit/baseline: Conservative configuration working set exceeds 70% of pre-load available memory
- reddit/2: Conservative configuration working set exceeds 70% of pre-load available memory
- reddit/4: Conservative configuration working set exceeds 70% of pre-load available memory
- reddit/8: Conservative configuration working set exceeds 70% of pre-load available memory
- reddit: among measured cases, 1 has the smallest median (3472.990859 ms) and 1 the largest (3472.990859 ms). This is descriptive, not proof of monotonic scaling or a causal optimization benefit.
- reddit: sample coefficient of variation spans 1.4%–1.4%. Ratios are exploratory; this is one process trial per sweep.

| Dataset | Configuration | Median ms | Mean ms | Sample SD ms | Baseline / case | Status |
| --- | --- | --- | --- | --- | --- | --- |
| citeseer | baseline: in=3703, heads=4, out/head=8 | 7.892192 | 7.995734 | 0.358007 | different workloads | passed |
| citeseer | 1: in=3703, heads=1, out/head=8 | 4.991043 | 5.191071 | 0.598755 | different workloads | passed |
| citeseer | 2: in=3703, heads=2, out/head=8 | 5.413042 | 5.455014 | 0.216234 | different workloads | passed |
| citeseer | 4: in=3703, heads=4, out/head=8 | 7.865704 | 7.897383 | 0.338908 | different workloads | passed |
| citeseer | 8: in=3703, heads=8, out/head=8 | 14.954272 | 14.955867 | 0.348492 | different workloads | passed |
| cora | baseline: in=1433, heads=4, out/head=8 | 4.008785 | 4.069239 | 0.281920 | different workloads | passed |
| cora | 1: in=1433, heads=1, out/head=8 | 2.395611 | 2.415678 | 0.106474 | different workloads | passed |
| cora | 2: in=1433, heads=2, out/head=8 | 2.630547 | 2.677133 | 0.169689 | different workloads | passed |
| cora | 4: in=1433, heads=4, out/head=8 | 4.059793 | 4.085884 | 0.301413 | different workloads | passed |
| cora | 8: in=1433, heads=8, out/head=8 | 7.158593 | 7.162608 | 0.333659 | different workloads | passed |
| pubmed | baseline: in=500, heads=4, out/head=8 | 19.714609 | 20.650568 | 3.772913 | different workloads | passed |
| pubmed | 1: in=500, heads=1, out/head=8 | 8.211536 | 8.242519 | 0.320707 | different workloads | passed |
| pubmed | 2: in=500, heads=2, out/head=8 | 11.213377 | 11.302288 | 0.380723 | different workloads | passed |
| pubmed | 4: in=500, heads=4, out/head=8 | 19.015446 | 19.051958 | 0.879555 | different workloads | passed |
| pubmed | 8: in=500, heads=8, out/head=8 | 34.663075 | 34.338083 | 1.217649 | different workloads | passed |
| reddit | baseline | — | — | — | — | skipped_memory |
| reddit | 1: in=602, heads=1, out/head=8 | 3472.990859 | 3483.983896 | 47.450345 | different workloads | passed |
| reddit | 2 | — | — | — | — | skipped_memory |
| reddit | 4 | — | — | — | — | skipped_memory |
| reddit | 8 | — | — | — | — | skipped_memory |

## Sparse Format

Restrict allowed storage to COO, CSR or CSC; materialize before timing and record operator compatibility.

- citeseer/coo: [05:56:22] /opt/dgl/src/graph/unit_graph.cc:1421: The graph have restricted sparse format coo , cannot create CSC matrix.
- citeseer/csr: [05:56:22] /opt/dgl/src/graph/unit_graph.cc:1035: Check failed: fmt == SparseFormat::kCSC || fmt == SparseFormat::kCOO: In degree cannot be computed as neither CSC nor COO format is allowed for this graph. Please enable one of them at least.
- citeseer/csc: [05:56:22] /opt/dgl/src/array/kernel.cc:231: SDDMM only supports CSR and COO formats
- cora/coo: [05:56:04] /opt/dgl/src/graph/unit_graph.cc:1421: The graph have restricted sparse format coo , cannot create CSC matrix.
- cora/csr: [05:56:04] /opt/dgl/src/graph/unit_graph.cc:1035: Check failed: fmt == SparseFormat::kCSC || fmt == SparseFormat::kCOO: In degree cannot be computed as neither CSC nor COO format is allowed for this graph. Please enable one of them at least.
- cora/csc: [05:56:04] /opt/dgl/src/array/kernel.cc:231: SDDMM only supports CSR and COO formats
- pubmed/coo: [05:56:53] /opt/dgl/src/graph/unit_graph.cc:1421: The graph have restricted sparse format coo , cannot create CSC matrix.
- pubmed/csr: [05:56:53] /opt/dgl/src/graph/unit_graph.cc:1035: Check failed: fmt == SparseFormat::kCSC || fmt == SparseFormat::kCOO: In degree cannot be computed as neither CSC nor COO format is allowed for this graph. Please enable one of them at least.
- pubmed/csc: [05:56:53] /opt/dgl/src/array/kernel.cc:231: SDDMM only supports CSR and COO formats

| Dataset | Configuration | Median ms | Mean ms | Sample SD ms | Baseline / case | Status |
| --- | --- | --- | --- | --- | --- | --- |
| citeseer | baseline: in=3703, heads=4, out/head=8 | 7.905181 | 7.902702 | 0.300648 | 1.000× (control) | passed |
| citeseer | coo | — | — | — | — | unsupported_format |
| citeseer | csr | — | — | — | — | unsupported_format |
| citeseer | csc | — | — | — | — | unsupported_format |
| cora | baseline: in=1433, heads=4, out/head=8 | 3.956012 | 3.992225 | 0.155723 | 1.000× (control) | passed |
| cora | coo | — | — | — | — | unsupported_format |
| cora | csr | — | — | — | — | unsupported_format |
| cora | csc | — | — | — | — | unsupported_format |
| pubmed | baseline: in=500, heads=4, out/head=8 | 19.190604 | 19.234812 | 0.670214 | 1.000× (control) | passed |
| pubmed | coo | — | — | — | — | unsupported_format |
| pubmed | csr | — | — | — | — | unsupported_format |
| pubmed | csc | — | — | — | — | unsupported_format |
| reddit | coo | — | — | — | — | skipped_memory |
| reddit | csr | — | — | — | — | skipped_memory |
| reddit | csc | — | — | — | — | skipped_memory |

## Edge Ordering

Original loader edge-ID order, source-grouped and destination-grouped; original node IDs retained.

- citeseer/original/coo: traversal indices distinct from baseline = **False**; EID map distinct = **False**.
- citeseer/original/csc: traversal indices distinct from baseline = **False**; EID map distinct = **False**.
- citeseer/original/csr: traversal indices distinct from baseline = **False**; EID map distinct = **False**.
- citeseer/src/coo: traversal indices distinct from baseline = **True**; EID map distinct = **False**.
- citeseer/src/csc: traversal indices distinct from baseline = **True**; EID map distinct = **True**.
- citeseer/src/csr: traversal indices distinct from baseline = **True**; EID map distinct = **True**.
- citeseer/dst/coo: traversal indices distinct from baseline = **True**; EID map distinct = **False**.
- citeseer/dst/csc: traversal indices distinct from baseline = **True**; EID map distinct = **True**.
- citeseer/dst/csr: traversal indices distinct from baseline = **True**; EID map distinct = **True**.
- citeseer: among measured cases, src has the smallest median (7.784482 ms) and dst the largest (7.879187 ms). This is descriptive, not proof of monotonic scaling or a causal optimization benefit.
- citeseer: sample coefficient of variation spans 3.2%–5.1%. Ratios are exploratory; this is one process trial per sweep.
- cora/original/coo: traversal indices distinct from baseline = **False**; EID map distinct = **False**.
- cora/original/csc: traversal indices distinct from baseline = **False**; EID map distinct = **False**.
- cora/original/csr: traversal indices distinct from baseline = **False**; EID map distinct = **False**.
- cora/src/coo: traversal indices distinct from baseline = **True**; EID map distinct = **False**.
- cora/src/csc: traversal indices distinct from baseline = **True**; EID map distinct = **True**.
- cora/src/csr: traversal indices distinct from baseline = **True**; EID map distinct = **True**.
- cora/dst/coo: traversal indices distinct from baseline = **True**; EID map distinct = **False**.
- cora/dst/csc: traversal indices distinct from baseline = **True**; EID map distinct = **True**.
- cora/dst/csr: traversal indices distinct from baseline = **True**; EID map distinct = **True**.
- cora: among measured cases, dst has the smallest median (3.820015 ms) and src the largest (3.866215 ms). This is descriptive, not proof of monotonic scaling or a causal optimization benefit.
- cora: sample coefficient of variation spans 2.7%–4.6%. Ratios are exploratory; this is one process trial per sweep.
- pubmed/original/coo: traversal indices distinct from baseline = **False**; EID map distinct = **False**.
- pubmed/original/csc: traversal indices distinct from baseline = **False**; EID map distinct = **False**.
- pubmed/original/csr: traversal indices distinct from baseline = **False**; EID map distinct = **False**.
- pubmed/src/coo: traversal indices distinct from baseline = **True**; EID map distinct = **False**.
- pubmed/src/csc: traversal indices distinct from baseline = **True**; EID map distinct = **True**.
- pubmed/src/csr: traversal indices distinct from baseline = **True**; EID map distinct = **True**.
- pubmed/dst/coo: traversal indices distinct from baseline = **True**; EID map distinct = **False**.
- pubmed/dst/csc: traversal indices distinct from baseline = **True**; EID map distinct = **True**.
- pubmed/dst/csr: traversal indices distinct from baseline = **True**; EID map distinct = **True**.
- pubmed: among measured cases, dst has the smallest median (17.734171 ms) and src the largest (19.787533 ms). This is descriptive, not proof of monotonic scaling or a causal optimization benefit.
- pubmed: sample coefficient of variation spans 3.5%–27.0%. Ratios are exploratory; this is one process trial per sweep.

| Dataset | Configuration | Median ms | Mean ms | Sample SD ms | Baseline / case | Status |
| --- | --- | --- | --- | --- | --- | --- |
| citeseer | baseline: in=3703, heads=4, out/head=8 | 7.730745 | 7.804707 | 0.266409 | 1.000× (control) | passed |
| citeseer | original: in=3703, heads=4, out/head=8 | 7.807906 | 7.792543 | 0.248887 | 0.990× (repeat control; same storage) | passed |
| citeseer | src: in=3703, heads=4, out/head=8 | 7.784482 | 7.913373 | 0.399787 | 0.993× | passed |
| citeseer | dst: in=3703, heads=4, out/head=8 | 7.879187 | 8.007051 | 0.360588 | 0.981× | passed |
| cora | baseline: in=1433, heads=4, out/head=8 | 3.982791 | 4.039207 | 0.221263 | 1.000× (control) | passed |
| cora | original: in=1433, heads=4, out/head=8 | 3.858080 | 3.893681 | 0.117951 | 1.032× (repeat control; same storage) | passed |
| cora | src: in=1433, heads=4, out/head=8 | 3.866215 | 3.896296 | 0.180009 | 1.030× | passed |
| cora | dst: in=1433, heads=4, out/head=8 | 3.820015 | 3.845404 | 0.104873 | 1.043× | passed |
| pubmed | baseline: in=500, heads=4, out/head=8 | 18.922911 | 18.998201 | 0.560330 | 1.000× (control) | passed |
| pubmed | original: in=500, heads=4, out/head=8 | 19.372352 | 20.683112 | 5.585460 | 0.977× (repeat control; same storage) | passed |
| pubmed | src: in=500, heads=4, out/head=8 | 19.787533 | 20.032991 | 0.781632 | 0.956× | passed |
| pubmed | dst: in=500, heads=4, out/head=8 | 17.734171 | 17.701262 | 0.619148 | 1.067× | passed |
| reddit | original | — | — | — | — | skipped_memory |
| reddit | src | — | — | — | — | skipped_memory |
| reddit | dst | — | — | — | — | skipped_memory |

## Node Ordering

Original node IDs, official RCM, ascending total degree with node-ID ties; source-grouped edges throughout.

- citeseer: among measured cases, degree has the smallest median (7.554538 ms) and rcmk the largest (7.756571 ms). This is descriptive, not proof of monotonic scaling or a causal optimization benefit.
- citeseer: sample coefficient of variation spans 1.7%–3.0%. Ratios are exploratory; this is one process trial per sweep.
- cora: among measured cases, degree has the smallest median (3.766123 ms) and rcmk the largest (3.955896 ms). This is descriptive, not proof of monotonic scaling or a causal optimization benefit.
- cora: sample coefficient of variation spans 1.8%–4.9%. Ratios are exploratory; this is one process trial per sweep.
- pubmed: among measured cases, rcmk has the smallest median (17.348395 ms) and original the largest (19.862437 ms). This is descriptive, not proof of monotonic scaling or a causal optimization benefit.
- pubmed: sample coefficient of variation spans 2.7%–16.7%. Ratios are exploratory; this is one process trial per sweep.

| Dataset | Configuration | Median ms | Mean ms | Sample SD ms | Baseline / case | Status |
| --- | --- | --- | --- | --- | --- | --- |
| citeseer | baseline: in=3703, heads=4, out/head=8 | 7.910982 | 7.954705 | 0.229860 | 1.000× (control) | passed |
| citeseer | original: in=3703, heads=4, out/head=8 | 7.688945 | 7.712537 | 0.129663 | 1.029× (repeat control; same storage) | passed |
| citeseer | rcmk: in=3703, heads=4, out/head=8 | 7.756571 | 7.804621 | 0.198367 | 1.020× | passed |
| citeseer | degree: in=3703, heads=4, out/head=8 | 7.554538 | 7.647828 | 0.229115 | 1.047× | passed |
| cora | baseline: in=1433, heads=4, out/head=8 | 4.070862 | 4.104983 | 0.182462 | 1.000× (control) | passed |
| cora | original: in=1433, heads=4, out/head=8 | 3.854922 | 3.865095 | 0.069535 | 1.056× (repeat control; same storage) | passed |
| cora | rcmk: in=1433, heads=4, out/head=8 | 3.955896 | 3.957695 | 0.165879 | 1.029× | passed |
| cora | degree: in=1433, heads=4, out/head=8 | 3.766123 | 3.819310 | 0.186516 | 1.081× | passed |
| pubmed | baseline: in=500, heads=4, out/head=8 | 19.451598 | 19.636411 | 0.756256 | 1.000× (control) | passed |
| pubmed | original: in=500, heads=4, out/head=8 | 19.862437 | 20.774339 | 3.475244 | 0.979× (repeat control; same storage) | passed |
| pubmed | rcmk: in=500, heads=4, out/head=8 | 17.348395 | 17.291668 | 0.494477 | 1.121× | passed |
| pubmed | degree: in=500, heads=4, out/head=8 | 18.634942 | 18.708171 | 0.498901 | 1.044× | passed |
| reddit | original | — | — | — | — | skipped_memory |
| reddit | rcmk | — | — | — | — | skipped_memory |
| reddit | degree | — | — | — | — | skipped_memory |

## Correctness, blocked work and limitations

Every measured case must pass annotated-versus-official output checks with identical weights and an official canonical-graph check. Node-order experiments restore output using new-to-old DGL NID mappings and verify feature association. Independent `--verify-only` checks use explicitly synthetic test graphs; their timing samples are discarded and are never reported as dataset performance.

Single-format restrictions can be incompatible with different operators in the complete stock forward. No zero-degree guard or format restriction is disabled to make a case run. A format restriction never establishes that every operator would use that representation.

| Synthetic-only verification | Source AST | Case outcomes |
| --- | --- | --- |
| attention_heads | True | 1:passed; 2:passed; 4:passed; 8:passed |
| baseline | True | original:passed |
| edge_ordering | True | original:passed; src:passed; dst:passed |
| input_features | True | 16:passed; 32:passed; 64:passed; 128:passed; 11:passed |
| node_ordering | True | original:passed; rcmk:passed; degree:passed |
| output_features | True | 8:passed; 16:passed; 32:passed; 64:passed |
| sparse_format | True | coo:unsupported:[06:03:35] /opt/dgl/src/graph/unit_graph.cc:1421: The graph have restricted sparse format coo , cannot create CSC matrix.; csr:unsupported:[06:03:35] /opt/dgl/src/graph/unit_graph.cc:1035: Check failed: fmt == SparseFormat::kCSC // fmt == SparseFormat::kCOO: In degree cannot be computed as neither CSC nor COO format is allowed for this graph. Please enable one of them at least.; csc:unsupported:[06:03:35] /opt/dgl/src/array/kernel.cc:231: SDDMM only supports CSR and COO formats |

The per-script verification also checks residual (identity and linear), bias, activation and returned attention against the untouched official class. Unsupported format cases have no timing or claimed output equivalence. These test-graph outcomes do not replace real-dataset verification.

- **reddit: skipped_memory** — Every requested configuration exceeds the memory safety budget. Affected experiments: baseline, edge_ordering, input_features, node_ordering, output_features, sparse_format. Their requested cases were not measured; other experiments may have successful measurements.

Reddit conservative baseline working-set estimate: **29.53 GiB**, compared against 70% of available host/cgroup memory. The estimate budgets simultaneous graph copies/formats, logits/attention temporaries and input/output buffers. It is a safety policy, not a measured peak. Each existing configuration is checked separately; lower-head-count cases can run while the default baseline is skipped. A missing Reddit baseline prevents a same-work speedup comparison.

Timing results are exploratory within one shared cloud CPU allocation, without repeated independent process trials, pinned CPU frequency or isolated hosts. No training/accuracy conclusions follow from dimension changes. Graph reordering can change reduction order, hence tolerances. Docker does not normalize physical CPU behavior. Tables retain all samples through linked raw JSON; no plots are needed to distinguish missing data.

## Reproduction

From the repository root, using the unchanged uv lockfile and Dockerfile:

```sh
export UV_CACHE_DIR=/workspace/.cache/uv  # writable cache in this cloud; optional elsewhere
export DGLBACKEND=pytorch OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
uv sync --frozen
uv run --frozen python benchmarks/dgl_baseline/run_all.py \
  --datasets cora citeseer pubmed reddit --threads 1 --seed 0 --warmup 5 --repetitions 20
uv run --frozen python benchmarks/dgl_baseline/report.py
```

Each script also supports `--verify-only --warmup 1 --repetitions 2`. Dataset acquisition requires HTTPS access to `data.dgl.ai`; download outcomes and redirect chains are recorded in the dataset-access table when available. JSON is under `results/dgl_baseline/<dataset>/<experiment>.json`; generated data and raw samples remain ignored by Git. The adjacent MEASUREMENTS.json bundles the raw records for GitHub review. Retain the dataset cache for reruns. Docker commands and additional methodology are in the root README.
