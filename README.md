# Graph Attention Network Performance Engineering

A reproducible framework for evaluating sparse storage formats, graph reordering strategies, and loop-nest optimizations in CPU Graph Attention Networks.

## Overview

Sparse access patterns can change the cost of graph attention without
changing its mathematical result. This repository measures those effects
on Cora, CiteSeer, and PubMed. It includes benchmark implementations, raw
timing samples, saved permutations, analysis scripts, and correctness tests.
The [technical analysis](docs/TECHNICAL_ANALYSIS.md) explains the design;
the [experimental findings](docs/EXPERIMENTAL_FINDINGS.md) link quantitative
claims to their measured comparisons.

## Graph attention computation

The layer projects node features, computes separate source and destination
attention terms, adds them at each directed edge and applies LeakyReLU,
normalizes scores across each destination's incoming edges, then aggregates
weighted source features and adds bias. The experiments use FP32, four
attention heads, eight output channels per head, and one pinned CPU thread.
The stock reference is the untouched DGL 1.1.3 `GATConv`.

## Optimization strategies

- **Sparse storage formats.** Compare native DGL COO/CSR/CSC hybrids and
  custom blocked or padded layouts. The [backend and timing controls](benchmarks/sparse_format_validation/README.md)
  separate storage choices from implementation effects more closely.
- **Graph reordering.** Change node order or edge order independently,
  using stock DGL throughout. Preserve the graph and restore the canonical
  node and edge IDs when checking outputs.
- **Loop-nest optimization.** Change one kernel's traversal in a matched
  serial C++ family with fixed storage and FP32 layouts. Compare loop orders
  against the custom default; report stock DGL separately for context.

## Key experimental findings

| Experiment | Measured finding |
|---|---|
| [Sparse Storage Format Evaluation](benchmarks/comprehensive_sparse_formats/REPORT.md) | The fastest observed whole forwards were CSB 64×64 on Cora (3.353 ms), CSB 2×2 on CiteSeer (9.126 ms), and SELL 2 on PubMed (14.344 ms). All beat stock DGL after Holm adjustment. CiteSeer's CSB 16×16, rather than CSB 2×2, cleared the comparison against stock and all three native hybrids. Matched controls do not establish a blocking-only advantage. |
| [Graph Reordering Experiments](benchmarks/ordering_baselines/REPORT.md) and [extended sweeps](benchmarks/ordering_baselines/extended/REPORT.md) | On PubMed, stable destination grouping qualified in both acquisitions. The extension measured 1.0280× for stable destination grouping and 1.0365× for destination-then-source versus the original edge order. The head-to-head comparison was inconclusive, and no node order met the practical criterion. |
| [Loop-Nest Optimization](benchmarks/loop_ordering/REPORT.md) | D2 aggregation qualified on Cora and PubMed. On PubMed it gained 1.9770× in isolation and 1.0761× in direct whole-forward timing versus the same custom family's default. PubMed C3 softmax and D4 aggregation also qualified. |

## Repository structure

| Directory | Purpose |
|---|---|
| [`benchmarks/comprehensive_sparse_formats/`](benchmarks/comprehensive_sparse_formats/) | Sparse format runners, raw data, analysis, native dispatch notes, and tests. |
| [`benchmarks/ordering_baselines/`](benchmarks/ordering_baselines/) | Matched-allocation reordering, an archived allocation-sensitivity run, and extended sweeps with saved METIS plans. |
| [`benchmarks/loop_ordering/`](benchmarks/loop_ordering/) | Compiled loop variants, execution fingerprints, timing samples, and analysis. |
| [`benchmarks/sparse_format_validation/`](benchmarks/sparse_format_validation/) | Matched backend and storage controls and distinct isolated/in-forward measurements. |
| [`benchmarks/dgl_baseline/`](benchmarks/dgl_baseline/), [`benchmarks/tuning_axes/`](benchmarks/tuning_axes/), [`benchmarks/sparse_formats/`](benchmarks/sparse_formats/), [`benchmarks/native_sparse_formats/`](benchmarks/native_sparse_formats/) | Shared loaders, baseline validation, compilation, and sparse kernels. |
| [`docs/`](docs/), [`tools/`](tools/) | Research documentation, experiment verification, and generated evidence tables. |

## Installation and requirements

Use Linux x86-64 (including Docker Desktop on WSL2). The pinned CPU stack
uses Python 3.11, uv 0.13.4, DGL 1.1.3, CPU PyTorch 2.2.2+cpu, NumPy 1.26.4,
and SciPy 1.12.0. Compiled experiments require GNU C++, binutils, and the
OpenMP runtime (`libgomp1` on Debian). Exercise Docker with BuildKit and
a native x86-64 daemon for performance measurement.

```sh
docker build -t gat-performance:cpu .
docker build --build-context dgl-performance-model:cpu=docker-image://gat-performance:cpu \
  -f benchmarks/tuning_axes/Dockerfile -t gat-performance:compiler .
```

On native Linux, install the system tools and
[uv](https://docs.astral.sh/uv/getting-started/installation/), then run
`uv sync --frozen --python 3.11` at the repository root. Keep LF bytes
in the checkout for artifact replay. On a system that translates line
endings, use `git clone --config core.autocrlf=false <repository-url>`.

## Running tests

```sh
docker run --rm --network none gat-performance:cpu --smoke-test
docker run --rm --network none -v "$PWD:/workspace" --entrypoint sh \
  gat-performance:compiler /workspace/tools/validate_experiments.sh
```

On native Linux, run `sh tools/validate_experiments.sh` from the root after
synchronizing dependencies. The suite runs the stock smoke test and the
experiment correctness suites, regenerates five archived reports and
analyses in a temporary directory, compares their bytes, and verifies the
generated evidence. It needs no dataset download. In PowerShell, use
`"${PWD}:/workspace"` for the bind mount.

## Reproducing experiments

DGL downloads its citation-graph datasets from `data.dgl.ai/dataset` on
first use. Put the cache under the ignored `results/datasets/` directory.
For offline Docker, populate the cache before using `--network none`.
For a new online cache, allow the container network on the first
acquisition. No downloaded datasets are in Git. The loaders preserve
native features and graph order, remove pre-existing self-loops, and
append one per node.

```sh
docker run --rm -v "$PWD:/workspace" --entrypoint uv gat-performance:compiler \
  run --frozen --no-sync python /workspace/benchmarks/loop_ordering/acquire_loops.py \
  --datasets cora --rounds 2 --warmup 1 --repetitions 2 \
  --data-dir /workspace/results/datasets --output /workspace/results/cora-smoke.json
```

Choose a fresh output path when repeating the command. With a populated
cache, add `--real-data` to the offline validation command above. The
short run checks correctness and the measurement protocol; use the
experiment READMEs for full acquisition. To regenerate the evidence
tables from committed JSON, run `python tools/build_experiment_evidence.py`
with Python 3.11. Add `--check` to verify without writing.

## Experimental methodology and limitations

The reported acquisitions used eight fresh processes per dataset, ten
warmup blocks, and thirty measured blocks per process. Complete forwards
were timed directly. Isolated kernels use prepared inputs; instrumented
regions run inside a forward with freshly produced intermediates. These
measurements are distinct. Reported latencies are medians of process
medians, with paired speedups, confidence intervals, and Holm-adjusted
tests. A nonsignificant contrast is inconclusive.

The benchmarks cover three citation graphs and one CPU/compiler setup.
The strongest independent choices were not benchmarked together in a
single storage × reordering × loop-nest run. Do not combine their speedups.
A fitted tensor cost model and automatic configuration selector remain
future work.

## References and attribution

See [references in the technical analysis](docs/TECHNICAL_ANALYSIS.md#related-work-and-references),
[source and license attribution](docs/ATTRIBUTION.md), and
[verification](docs/VERIFICATION.md). Archived benchmark reports retain
their original wording so they can still be regenerated byte for byte.
