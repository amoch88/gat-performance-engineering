# Performance Engineering of Graph Attention Networks Through Sparse Storage, Graph Ordering, and Loop-Nest Optimization

**CS 598 APE · Fall 2026 · final project artifact**

This project measures three ways to change how a single CPU GAT layer
executes while preserving its output: sparse storage, node/edge ordering,
and loop order. Experiments use Cora, CiteSeer, and PubMed with one pinned
CPU thread, FP32, four attention heads, and eight channels per head.
The artifact includes the implementations, raw measurements, saved graph
permutations, analyses, and correctness checks. A joint tensor cost model
and automatic configuration selector are future work.

Start with the [technical report handoff](docs/FINAL_REPORT_HANDOFF.md),
[25 source-linked claims](docs/REPORT_CLAIMS.md), and
[submission QA log](docs/SUBMISSION_QA.md).

## Verified findings

| Experiment | Main result |
|---|---|
| [M8 · Sparse storage](benchmarks/comprehensive_sparse_formats/REPORT.md) | Fastest observed forwards were CSB 64×64 on Cora (3.353 ms), CSB 2×2 on CiteSeer (9.126 ms), and SELL 2 on PubMed (14.344 ms). All beat stock DGL after Holm adjustment. CiteSeer's CSB 16×16, not CSB 2×2, cleared the stronger comparison against stock and all native hybrids. [M9 controls](benchmarks/sparse_format_validation/REPORT.md) limit any claim that blocking alone caused those gains. |
| [M10 · Ordering](benchmarks/ordering_baselines/REPORT.md) and [seven-policy extension](benchmarks/ordering_baselines/extended/REPORT.md) | Stable destination grouping qualified on PubMed in both runs. In the extension it gained 1.0280× and destination-then-source gained 1.0365× versus Original. Their head-to-head comparison was inconclusive; no node order qualified. |
| [M11 · Loop ordering](benchmarks/loop_ordering/REPORT.md) | D2 aggregation qualified on Cora and PubMed. On PubMed it gained 1.9770× in isolation and 1.0761× in direct whole-forward timing, versus the same custom family's default. PubMed C3 softmax and D4 aggregation also qualified. |

These are separately acquired experiments. The stage and direct-forward
timings have distinct boundaries, and the independent speedups must not
be combined into a predicted joint speedup. See the handoff for intervals,
controls, and limitations.

## Install and validate

The tested setup is Linux x86-64 (including Docker Desktop on WSL2).
Docker BuildKit uses the preserved pinned CPU base and the preserved GNU
C++ compiler extension. This builds separate tags for the course checkout:

```sh
docker build -t cs598ape-gat:cpu .
docker run --rm --network none cs598ape-gat:cpu --smoke-test
docker build --build-context dgl-performance-model:cpu=docker-image://cs598ape-gat:cpu \
  -f benchmarks/tuning_axes/Dockerfile -t cs598ape-gat:compiler .
docker run --rm --network none -v "$PWD:/submission" --entrypoint sh \
  cs598ape-gat:compiler /submission/tools/validate_submission.sh
```

For native Linux without Docker, install Python 3.11, uv **0.13.4**,
GNU C++ and binutils, then run `uv sync --frozen` and
`sh tools/validate_submission.sh` from the checkout root. The lock
installs DGL **1.1.3**, CPU PyTorch **2.2.2+cpu**, NumPy **1.26.4** and
SciPy **1.12.0**. Use a checkout that preserves LF bytes; when cloning
from GitHub on a host configured to translate line endings, use
`git -c core.autocrlf=false clone <repository-url>`.

The validation script runs the stock smoke test; M8, M9, both M10
suites, and M11 correctness tests; then regenerates all five historical
reports and analyses to a temporary directory and compares their bytes
with the committed versions. It does not replace raw benchmark data.
To regenerate the course claim table and handoff tables, use
`python tools/build_course_evidence.py`; use `--check` to verify them
without modifying anything. The evidence builder uses only the Python
standard library and committed measurements.

### Small real-data smoke

DGL downloads its citation-graph datasets from `data.dgl.ai/dataset` on
first use. Put the cache under the ignored `results/datasets/` directory.
For offline Docker, populate the cache before using `--network none`.
For a new online cache, allow the container network on the first
acquisition. No downloaded datasets are in Git. The loaders preserve
native features and graph order, remove pre-existing self-loops, and
append one per node.

```sh
docker run --rm -v "$PWD:/submission" --entrypoint uv cs598ape-gat:compiler \
  run --frozen --no-sync python /submission/benchmarks/loop_ordering/acquire_loops.py \
  --datasets cora --rounds 2 --warmup 1 --repetitions 2 \
  --data-dir /submission/results/datasets --output /submission/results/cora-smoke.json
```

Choose a fresh output path when repeating the command. With a populated
cache, a fully offline form runs `tools/validate_submission.sh --real-data`
inside the compiler image. This short run is a correctness and protocol
check only. For full acquisition commands and exact schemas, use the
milestone READMEs. The committed runs use eight fresh processes per
dataset, ten warmups and thirty measured blocks.

## Artifact map

| Path | Contents |
|---|---|
| [benchmarks/comprehensive_sparse_formats/](benchmarks/comprehensive_sparse_formats/) | M8 code, raw data, analysis, native-route notes, tests, and reproduction. |
| [benchmarks/sparse_format_validation/](benchmarks/sparse_format_validation/) | M9 matched backend/storage controls and distinct isolated/in-forward measurements. |
| [benchmarks/ordering_baselines/](benchmarks/ordering_baselines/) | Original matched M10, archive for allocation sensitivity, and M10 extension with METIS maps. |
| [benchmarks/loop_ordering/](benchmarks/loop_ordering/) | M11 C++ loop variants, raw blocks, symbol fingerprints, analysis, and tests. |
| [benchmarks/dgl_baseline/](benchmarks/dgl_baseline/), [benchmarks/tuning_axes/](benchmarks/tuning_axes/), [benchmarks/sparse_formats/](benchmarks/sparse_formats/), [benchmarks/native_sparse_formats/](benchmarks/native_sparse_formats/) | Preserved shared loader, model, compiler, measurement, validation, and storage dependencies. |
| [docs/](docs/), [tools/](tools/) | Technical handoff, exact claims, verification log, and standalone QA/evidence helpers. |

The course repository is a tracked snapshot of research commit
`9659684473a0531aa06e40f614e1fbfbbcfc5b76`. Earlier research files
remain present, and the prior top-level README is preserved as
[RESEARCH_README.md](RESEARCH_README.md). See [attribution](docs/ATTRIBUTION.md)
for upstream source and data references. The benchmarks cover three graphs
on a single CPU/compiler environment, not general performance portability.
