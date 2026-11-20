# Exploratory Joint DGL Parameter Study

This module is independent of the preserved `benchmarks/dgl_baseline/` work.
It uses untouched official DGL 1.1.3 GATConv, native dataset features, four heads,
eight output features per head, float32, seed 0 and CPU inference only. No custom
kernels, custom sparse layouts, fusion or predictive model is implemented.

`core.py` contains the pilot, official graph API transformations, exact topology/
EID/feature checks, restoration and timing. `worker.py` is a fresh process per
round; `run.py` executes workers sequentially to avoid benchmark interference.
`analysis.py` computes descriptive factorial contrasts and held-out comparisons;
`report.py` renders results. `plots.py` renders scientific SVG figures separately
using Matplotlib 3.10.8 already available in the host's Python. Plot dependencies
are not imported by or installed into the benchmark environment.

From the repository root, using the existing Docker/uv configuration:

```sh
export DGLBACKEND=pytorch OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export UV_CACHE_DIR=/workspace/.cache/uv
uv sync --frozen
uv run --frozen python benchmarks/joint_tuning/tests.py
uv run --frozen python benchmarks/joint_tuning/run.py \
  --datasets cora citeseer pubmed reddit --rounds 6 --discovery-rounds 3 \
  --threads 1 --seed 0 --order-seed 20261104 --warmup 5 --repetitions 20
uv run --frozen python benchmarks/joint_tuning/report.py
MPLCONFIGDIR=/tmp/dgl-joint-mpl python benchmarks/joint_tuning/plots.py
```

The benchmark runtime needs only existing locked dependencies. Re-render plots
with Python plus Matplotlib 3.10.8; `figures/manifest.json` captures the actual
renderer version and input digest. Main outputs are [REPORT.md](REPORT.md),
[RESULTS.json](RESULTS.json), [ANALYSIS.json](ANALYSIS.json), [PILOT.json](PILOT.json)
and `figures/*.svg`. Raw samples, exact errors and provenance are retained.
Dataset downloads, Python caches and temporary worker files remain under the
existing ignored paths. Use `--output-dir` and `--work-dir` to preserve separate
trials; defaults overwrite only this new module's generated results.

The pilot tests seven nonempty COO/CSR/CSC permission sets. Only successfully
executing, numerically equivalent sets can enter the requested joint matrix.
In the validated build, all and COO+CSC pass; CSR+CSC fails SDDMM despite allowing
CSR. Permissions are not a kernel dispatch contract. No fallback permission is
added and no guard is bypassed. The original-EID edge policy after node relabeling
uses DGL's valid custom edge permutation; topology is verified per original EID,
including parallel edges and self-loops. Maps and features are checked exactly.

Every case is bracketed by warmed controls. Three discovery rounds select joint
and one-factor-axis choices, then three held-out rounds compare them. Both raw
forward and combined original-node-order interface objectives are analyzed. See
the report for the exact normalizations, variability, interaction cross-ratios
and preparation break-even definition. The descriptive all-round rankings must
not be confused with held-out evidence. Full-forward timings do not identify
which compiled DGL kernels caused a difference.

Reddit is not reduced to fewer heads or narrower features. Its fixed-shape
single-case estimate and resident joint-matrix allocation estimate are checked against
70% of available host/cgroup memory, accounting for reclaimable inactive file
cache. Unsafe grids are skipped before loading. Dataset topology/features are
never replaced by synthetic inputs. Tests and pilot use explicitly synthetic
small graphs only for correctness and compatibility.

A small complete end-to-end check without changing the published baseline:

```sh
uv run --frozen python benchmarks/joint_tuning/run.py --datasets cora \
  --rounds 4 --discovery-rounds 2 --warmup 1 --repetitions 2 \
  --output-dir results/joint-smoke --work-dir results/joint-smoke/work
uv run --frozen python benchmarks/joint_tuning/report.py \
  --input results/joint-smoke/RESULTS.json --output results/joint-smoke/REPORT.md
```

Docker uses the unchanged root Dockerfile. Rebuild to copy the new module, then
override its baseline entrypoint. Cached data can be mounted read-only:

```sh
docker build --platform linux/amd64 -t dgl-performance-model:cpu .
docker run --rm --entrypoint uv dgl-performance-model:cpu \
  run --frozen --no-sync python benchmarks/joint_tuning/tests.py
# Use an output volume for measurements you intend to retain.
docker run --rm --entrypoint uv \
  -v "$PWD/results/datasets:/app/cache:ro" \
  -v "$PWD/results/joint-docker:/app/output" dgl-performance-model:cpu \
  run --frozen --no-sync python benchmarks/joint_tuning/run.py \
  --data-dir /app/cache --output-dir /app/output --work-dir /app/output/work \
  --datasets cora citeseer pubmed reddit --rounds 6 --discovery-rounds 3
```

For the managed proxy, use the CA build secret documented in the root README.
Keep TLS/artifact verification enabled. Offline tests need no network/CA mount.
The Docker image pins Python 3.11.11 while the native managed runtime may use a
newer Python 3.11 patch; actual versions are recorded. Docker performance runs
are separate trials, not claimed replicas of the native timing numbers.
