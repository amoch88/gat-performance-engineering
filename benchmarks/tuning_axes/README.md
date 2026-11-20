# Milestone 5: four independent CPU tuning families

This directory preserves all previous modules and measurements. Runtime remains
Python 3.11, DGL 1.1.3, PyTorch 2.2.2+cpu and the existing frozen uv.lock.
No custom attention formula, GPU path, predictive model or factorial study is used.

The seven OPAT contrasts are split among `index_dtype.py`, `traversal.py`,
`layouts.py`, and `scheduling.py`. They share `common.py`, `orchestration.py`, the
compiled `kernels.cpp`/ctypes adapter `native.py`, and independent-round drivers.
Each family file documents the actual EXPERIMENTAL CHANGE.

All original topology/features, self-loop policy, four heads, eight features per
head, float32 parameters, bias, inference behavior and weights are fixed within
each dataset. Canonical EIDs are the preprocessed graph's edge IDs. Test-only
permutations restore logical node/output and attention EID order. Graph views
validate topology, sparse EID maps and all associated graph data.

## Build and test

A C++17 compiler and OpenMP runtime are the only added native requirements.
`g++` with `-O3 -fPIC -shared -fopenmp -ffp-contract=off` builds the small C ABI
library; no Torch extension headers, CUDA or Python build dependencies are needed.
There is no fast-math. Set CXX to a compatible compiler executable if necessary.

```sh
export DGLBACKEND=pytorch OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export UV_CACHE_DIR=/workspace/.cache/uv
uv sync --frozen
uv run --frozen python benchmarks/tuning_axes/native.py
uv run --frozen python benchmarks/tuning_axes/tests.py
```

The binary, compiler logs/first build record and cache live in ignored
`results/tuning_axes_build/`. `TUNING_AXES_BUILD_DIR` can select another writable
build cache. Source/compiler/options determine the cache key. Never commit .so,
.o, dataset archives, virtual environments or worker scratch results.

## Measurements

```sh
uv run --frozen python benchmarks/tuning_axes/run.py \
  --datasets cora citeseer pubmed reddit --rounds 6 --warmup 5 --repetitions 20 \
  --seed 0 --order-seed 20261105
uv run --frozen python benchmarks/tuning_axes/report.py
python benchmarks/tuning_axes/verify_results.py
python benchmarks/tuning_axes/plots.py
```

Plotting uses host Python with Matplotlib 3.10.8 separately; it is not imported by
or installed into the benchmark environment. See `figures/manifest.json`.
Main artifacts are REPORT.md, RESULTS.json (raw data/provenance), ANALYSIS.json,
VALIDATION.json and the SVG figure. `--output-dir` preserves separate trials;
defaults overwrite only this new module's artifacts. Worker files remain ignored.

For a short integration check without contaminating the primary artifacts:

```sh
uv run --frozen python benchmarks/tuning_axes/run.py --datasets cora \
  --rounds 2 --warmup 1 --repetitions 2 \
  --output-dir results/tuning_axes_smoke --work-dir results/tuning_axes_smoke/work
uv run --frozen python benchmarks/tuning_axes/report.py \
  --input results/tuning_axes_smoke/RESULTS.json --output results/tuning_axes_smoke/REPORT.md
```

Each parameter has six independent process rounds in the primary run. To reduce
repeated setup, one fresh process per dataset/round runs the one-thread A/B/C axes;
a separate fresh process runs the four-thread D axes. No timing processes overlap.
Axes/cases are randomized; candidates have before/after within-family controls.
Official complete GATConv is also timed at the same thread budget. Repetitions
inside a process are not independent rounds. Intervals use Student-t log-ratio
uncertainty over rounds and are explicitly unadjusted exploratory comparisons.
A reference normalized to one is a definition, not zero variance in its raw time.

Both scheduling values use four threads. Static vs degree-balanced ownership
changes only the row-to-thread map. Degree balancing greedily assigns intact
rows by degree+1; it does not split high-degree rows or normalize chunks.
Per-thread edge counts and separate service-time diagnostics expose imbalance.
The ordinary full forward never executes profiling clocks or callbacks.
Stage profiles and diagnostic service clocks run separately after ordinary blocks.
The harness fixes OMP_WAIT_POLICY=PASSIVE and KMP_BLOCKTIME=0 before importing
Torch/DGL or loading the native library. This controls idle waiting across Torch,
DGL and native parallel regions. Loaded library paths and these controls are recorded.

The physical layout family is a common compiled implementation. NHF/HNF changes
only projected storage/packing and consumer addresses; EH/HE changes only edge
intermediate addresses. No DGL contiguous conversion erases these layouts.
Projection remains the shared official Torch linear operation; all other
choices stay fixed within each layout comparison. HNF packing is included in
every forward; HE is produced directly. The generic custom family may differ
substantially from stock in overhead; its stock gap is not a layout-only effect.

Stage traversal uses exclusive sole-format graph views: gsddmm(add) COO/CSR,
edge_softmax CSC, gspmm(mul,sum) CSC/COO. Success plus the v1.1.3 source switch and
recorded exact masks establish these routes; there is no claimed dynamic C++
symbol trace. No extra format is silently permitted. Per-stage output stays in
canonical EID order, so views share edge intermediates without per-forward remaps.
Source reference: https://github.com/dmlc/dgl/blob/v1.1.3/src/array/kernel.cc

All live validation tensor payloads are recorded, not claimed as allocator peak.
Sparse graph storage counts deduplicate exposed sparse storages, excluding dense
features and hidden overhead. Process peak RSS spans all setup/profiles/axes in
that process. HNF's end-of-validation payload excludes the transient overlapping
original projection while packing; profile time includes the actual packing.
Reddit retains the conservative four-head full-graph memory preflight. Unsafe
runs are skipped before loading; heads/features/topology are never reduced.

## CPU Docker extension

The root Dockerfile/uv environment remain unchanged. The new extension adds a
compiler to that existing CPU image and copies this module. For the managed
proxy, pass the existing combined CA bundle as a build secret, never disable TLS:

```sh
docker build --secret id=proxy_ca,src=/etc/ssl/certs/ca-certificates.crt \
  -t dgl-performance-model:cpu .
docker build -f benchmarks/tuning_axes/Dockerfile \
  --secret id=proxy_ca,src=/etc/ssl/certs/ca-certificates.crt \
  -t dgl-performance-model:axes .
docker run --rm --entrypoint uv dgl-performance-model:axes \
  run --frozen --no-sync python benchmarks/tuning_axes/tests.py
```

For normal Docker without proxy CA requirements, omit the secret. A full Docker
run is a separate trial: mount cached datasets read-only at /app/cache and an
output directory at /app/output, then pass `--data-dir /app/cache --output-dir
/app/output --work-dir /app/output/work`. Default entrypoint runs run.py. Native
and Docker compiler/Python patch versions can differ; all are recorded.
