# DGL CPU performance modeling

This research project measures the cost of stock DGL graph attention before
exploring performance models, tuning dependent kernels, and comparing fusion
strategies. The first milestone is a single official PyTorch-backend `GATConv`
forward pass. There are no prediction models, tuning transformations or fusion
implementations yet. All tensors, graph storage and computation are on CPU.

## Environment and compatibility

Target: Linux x86-64, Python 3.11, uv **0.13.4**. Direct dependencies:

| Package | Version/source |
| --- | --- |
| DGL | 1.1.3, official CPU wheel on PyPI |
| PyTorch | 2.2.2+cpu, explicit `download.pytorch.org/whl/cpu` index |
| NumPy | 1.26.4 |
| SciPy | 1.12.0 |
| packaging | 24.2 (undeclared DGL import requirement) |

`uv.lock` pins the complete dependency graph and artifact hashes. No accelerator
runtime packages are required. Python is restricted to 3.11 and uv resolution to
Linux x86-64; other platforms are outside this baseline's validated scope.

Before choosing these versions we consulted the official
[DGL installation documentation](https://github.com/dmlc/dgl/blob/master/docs/source/install/index.rst),
[installation selector](https://www.dgl.ai/pages/start.html), and
[release notes](https://github.com/dmlc/dgl/releases).
The installation source lists Python 3.11 and PyTorch >=1.12 support; PyPI
publishes a CPython 3.11 Linux x86-64 CPU wheel for DGL 1.1.3. Newer DGL releases
exist (including 2.4.0), and their wheels can require the separate
`data.dgl.ai` index. That host and the rendered selector were denied by the
onboarding network policy. This deliberately older, official CPU baseline avoids
GraphBolt dependencies and is validated by executing GATConv with the selected
PyTorch wheel. It does **not** represent the latest DGL release. Treat any future
dependency upgrade as a separate baseline and revalidate it. NumPy stays below 2
for compatibility with these binary releases.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) version
0.13.4, then from the repository root:

```sh
uv sync --frozen --python 3.11
uv run --frozen python benchmarks/gat.py --smoke-test
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --frozen python benchmarks/gat.py \
    --nodes 1000 --avg-degree 8 --features 32 --out-features 8 --heads 4 \
    --seed 0 --threads 1 --warmup 5 --repetitions 20 --output results/gat.json
```

The official DGL CPU library needs OpenMP (`libgomp1` on Debian). The Docker image
installs it. Native installations need the equivalent system library.

## Docker

Use Docker with BuildKit and an x86-64 Linux daemon (no emulation for timing):

```sh
docker build --platform linux/amd64 \
  --build-arg GIT_COMMIT="$(git rev-parse HEAD)" -t dgl-performance-model:cpu .
docker run --rm dgl-performance-model:cpu --smoke-test
mkdir -p results
docker run --rm -v "$PWD/results:/app/results" dgl-performance-model:cpu \
  --nodes 1000 --edges 8000 --features 32 --out-features 8 --heads 4 \
  --seed 0 --threads 1 --warmup 5 --repetitions 20 --output results/gat.json
```

For an unborn repository, omit the `GIT_COMMIT` build argument until the first
commit. The Python base tag pins 3.11.11; native `uv` installations may use a newer
3.11 patch release. Results record the actual version. Base and uv images are also pinned by digest. Debian
packages can change: record the built image ID/digest for published experiments,
retain the image, and pass it via `-e BENCHMARK_IMAGE=<image-id-or-digest>`.

In a managed environment requiring proxy CA trust, use the same build command
with `--secret id=proxy_ca,src=/etc/ssl/certs/ca-certificates.crt`. Preserve the
platform's Docker proxy defaults. The bundle is mounted only during installation,
never embedded in the image. TLS verification stays enabled. Runtime benchmarks
need no network access or credentials.

## Methodology and results

`benchmarks/gat.py` exposes a workload dataclass, setup function and measurement
function so later variants can share inputs and reporting. It uses official
`dgl.nn.pytorch.GATConv` unchanged, with float32 inputs, evaluation mode, zero
feature/attention dropout, and `torch.inference_mode()`. Output shape is
`(nodes, heads, out_features)`; output features are **per head**.

Random sources/destinations are sampled uniformly with replacement from a fixed
seed. Parallel edges are retained. Sampled self-loops are removed, then exactly
one self-loop per node is added to avoid zero-in-degree nodes. `--edges` specifies
sampled edges **before** this normalization; `--avg-degree` is an integer sampled
edge count per node, not the final measured degree. JSON records both the
requested and final edge counts. Seeds initialize Python, NumPy, PyTorch and DGL.

Known-answer identity-projection and two-node neighbor-mean tests, plus checks for
shape, finiteness, repeated inference and regenerated-input reproducibility run
before **every** benchmark. `--smoke-test` runs them and exits without timing.
These are correctness smoke checks, not a full numerical oracle for DGL.

Graph generation, self-loops, sparse-format materialization, feature generation
and layer initialization occur before warmup. Graph construction is reported
separately, including sparse-format creation but excluding model/input setup.
Warmup calls are excluded from measurements. Each sample times one synchronous
CPU forward call with `time.perf_counter_ns()`; Python dispatch, output allocation
and ordinary per-call DGL work remain part of stock forward latency. Output
checks, statistics, metadata collection and JSON serialization are outside timing.

JSON contains all raw milliseconds, median, mean, sample standard deviation
(`n-1`), minimum, configuration, seeds and warmup/repetition counts. It also
records CPU model/count/affinity, OS/architecture, Python/DGL/PyTorch/NumPy
versions, PyTorch build information and intra/inter-op thread counts, relevant
threading environment settings, optional image identity, Git commit and dirty
state. In an uncommitted checkout the commit is null; in Docker the build argument
is used, and dirty state is unavailable. Commit the source before reporting
reproducible research results.

Rerun the exact workload using the JSON `rerun_argv` array from the repository
root (pass the array directly to `subprocess.run`, not a shell string). Restore
the recorded threading environment and software/image too. `--threads` controls
PyTorch intra-op threads; inter-op threads are set to one. Set OpenMP/MKL/OpenBLAS
variables **before** launching Python to control the native libraries consistently.
The CLI records rather than silently overwrites these environment choices.

Docker does not guarantee identical performance across physical machines.
Control affinity, competing load, CPU quotas, frequency scaling and thread
settings when comparing results. Small examples validate the harness and are
not performance claims. Keep future experimental implementations separate from
this official baseline and compare identical workloads and timing boundaries.

## Milestone 2: official DGL parameter experiments

The original `benchmarks/gat.py` runner is preserved. The independent scripts in
[`benchmarks/dgl_baseline/`](benchmarks/dgl_baseline/) study input width, output
features per head, head count, sparse formats, edge ordering and node ordering.
[`REPORT.md`](benchmarks/dgl_baseline/REPORT.md) is generated from actual JSON
records, with blocked, skipped, unsupported and measured outcomes kept distinct.
The report now includes real Cora, Citeseer and PubMed measurements and a
memory-admitted one-head Reddit measurement. Strict single-format restrictions
and larger Reddit configurations are explicitly unsupported or skipped. The
official dataset host is accessible, with all four archives downloaded by the
official DGL loaders.

The annotated `baseline.py` contains the full official DGL 1.1.3 GATConv class,
with five logical computation-stage comments, the original behavior, source
attribution, and an adjacent Apache-2.0 license. Its executable class AST must
match the installed class. Do not treat logical stages as individual compiled
kernels. Numerical checks compare identical weights against the untouched
installed class, including residual/bias/activation/attention test cases.

From the repository root, validate all independent scripts:

```sh
export DGLBACKEND=pytorch OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Needed only where the default home cache is read-only:
export UV_CACHE_DIR=/workspace/.cache/uv
for experiment in baseline input_features output_features attention_heads sparse_format edge_ordering node_ordering; do
  uv run --frozen python benchmarks/dgl_baseline/$experiment.py \
    --verify-only --warmup 1 --repetitions 2
done
uv run --frozen python benchmarks/dgl_baseline/tests.py
```

The verification graphs are explicitly synthetic, their times are discarded,
and they are never labeled as dataset measurements. Verification JSON is stored
at `results/dgl_baseline/verification-*.json`.

With `data.dgl.ai` allowed, reproduce the measurements sequentially:

```sh
uv run --frozen python benchmarks/dgl_baseline/run_all.py \
  --datasets cora citeseer pubmed reddit --threads 1 --seed 0 --warmup 5 --repetitions 20
uv run --frozen python benchmarks/dgl_baseline/report.py
```

The driver measures each dataset's baseline first when memory permits. Dataset
access failure is shared across dependent sweeps, avoiding repeated failed
downloads. Reddit memory is checked per existing configuration: safe cases can
run even when the default baseline or other sweeps are skipped. It exits nonzero for blocked datasets or
unexpected experiment failures. Supported measurements and expected single-format
incompatibilities remain separate. For one independent sweep, for example:

```sh
uv run --frozen python benchmarks/dgl_baseline/edge_ordering.py \
  --datasets cora citeseer pubmed --threads 1 --seed 0 --warmup 5 --repetitions 20
```

All six scripts accept that interface. Input-width cases are 16/32/64/128 plus the
original width, deduplicated, using prefix truncation or zero-padding of the real
features outside timing. Output cases are 8/16/32/64 per head with four heads;
head cases are 1/2/4/8 with eight features per head. Baseline inputs use the
original dataset width. Changed dimensions represent changed workloads, not
speedups of equivalent computations.

Citation loaders explicitly use `reorder=False`; "original" refers to the DGL
loader output before experimental permutations. Remove all existing self-loops
and append one per node. Edge ordering retains node IDs; node ordering keeps a
source-grouped edge policy throughout and restores output rows using `dgl.NID`.
Degree ordering is ascending in-degree plus out-degree, with original IDs as ties.
Features remain associated with their logical nodes. Sparse-format conversion and
reordering finish before validation/warmup/timing. Allowed/created formats,
traversal index hashes and EID maps are recorded before and after timing.

Pure COO/CSR/CSC restrictions are tested as requested, without silently allowing
other formats. In this installed stock CPU implementation, all three are
incompatible with at least one full-forward operator. Their error records carry
no timings. The all-format baseline stays intact; no kernel, in-degree check or
assertion is disabled to bypass a restriction.

Reddit's conservative full-graph baseline working-set estimate (graph copies,
formats, attention/logit temporaries and feature/output buffers) is checked
against 70% of pre-load available host/cgroup memory. Inactive file-cache pages
are treated as reclaimable, as in Docker working-set accounting, and the host
available-memory estimate remains an upper bound. Configurations with no safe
cases are skipped before loading the graph for that sweep. The default estimate
is 29.53 GiB; the one-head configuration is 19.20 GiB and fits this approximately
32 GiB allocation. The 2/4/8-head cases and the other Reddit sweeps exceed the
policy budget. These are safety estimates, not measured peak memory or proof
that every skipped configuration would fail. Datasets are never replaced with
synthetic graphs in reported measurements.

Raw results, seeds, configuration, environment, source/lockfile hashes and
separate preprocessing costs are in `results/dgl_baseline/<dataset>/*.json`.
Raw files and dataset caches in results/ are ignored. The generated
`benchmarks/dgl_baseline/MEASUREMENTS.json` retains the raw samples, provenance,
verification outcomes and download-access evidence alongside the report for
GitHub review. Retain the dataset cache for offline reruns. Each invocation overwrites its named records, so use a separate
`--output-dir` for independent trials. Pass `--input-dir` to `report.py` when
rendering a different directory.

The existing Dockerfile and uv dependencies are unchanged. Rebuild to copy in
the new scripts, and override the original benchmark entrypoint:

```sh
docker build --platform linux/amd64 -t dgl-performance-model:cpu .
docker run --rm --entrypoint uv dgl-performance-model:cpu \
  run --frozen --no-sync python benchmarks/dgl_baseline/node_ordering.py \
  --verify-only --warmup 1 --repetitions 2
mkdir -p results
docker run --rm -v "$PWD/results:/app/results" --entrypoint uv dgl-performance-model:cpu \
  run --frozen --no-sync python benchmarks/dgl_baseline/run_all.py \
  --datasets cora citeseer pubmed reddit --threads 1 --seed 0 --warmup 5 --repetitions 20
# Render to a host-visible report; the container's /app/report file is ephemeral.
uv run --frozen python benchmarks/dgl_baseline/report.py
```

Use the CA build secret and runtime read-only CA mount described above when
network access requires the managed proxy trust bundle; keep verification enabled.
For example, add
`--mount type=bind,src=/etc/ssl/certs/ca-certificates.crt,dst=/run/proxy-ca.pem,readonly`
and `-e REQUESTS_CA_BUNDLE=/run/proxy-ca.pem` to the container that downloads data.
No proxy CA mount is needed for offline correctness verification. Docker proxy
settings still need to route to the allowed official dataset host.
