# Sparse Storage Format Evaluation

The [report](REPORT.md) gives the three complete timing tables. The
[N/A notes](NA_EXPLANATIONS.md) identify each native hybrid's fallback. This
evaluation uses each custom format in all three sparse stages of the same
forward. The native-versus-blocked benchmark used one substitution at a time.

## Layouts

- CSB, Blocked CSR and Blocked CSC reuse the existing traversal and encoding.
- Block Dense allocates a row-major int32 EID grid for each nonempty tile.
  One uint64 occupancy word per tile row gates reads from that grid. The
  first edge in a cell occupies the grid slot. Each additional edge retains
  its source, destination and EID in a sidecar consumed by all three kernels.
- SELL slices consecutive original destination rows. Each slice uses the
  maximum incoming degree as its column count and stores columns with
  interleaved lanes. Sentinel EIDs mask padding. Input order within a row
  is canonical EID order.

All formats have size choices 2, 4, 8, 16, 32 and 64 where applicable.
Block sizes count nodes and SELL sizes count rows. Softmax normalizes
across all incoming edges, including edges split across tiles.

## Reproduction

Follow the frozen CPU Docker build and dataset cache instructions in
[the sparse kernel reference](../sparse_formats/README.md). With the same DGL 1.1.3 and Torch
2.2.2+cpu environment, run from the repository root on Linux:

~~~sh
uv run --frozen python benchmarks/gat.py --smoke-test
uv run --frozen python benchmarks/comprehensive_sparse_formats/tests.py
uv run --frozen python benchmarks/comprehensive_sparse_formats/benchmark.py \
  --data-dir results/datasets --output results/storage-formats/RESULTS.json
uv run --frozen python benchmarks/comprehensive_sparse_formats/report.py \
  --input results/storage-formats/RESULTS.json --output results/storage-formats
~~~

With the compiler image, mount the checkout as /checkout, override the
entrypoint to uv, and use absolute paths inside the mount for scripts,
cache and output. Set --cpu to pin to a different allowed logical CPU.
Defaults are eight fresh serial process rounds per dataset, ten warmup
and thirty measured blocks per timed phase, and five randomized
one-time construction observations per layout/round. A smoke run can use
--rounds 2 --warmup 1 --repetitions 2 --build-repetitions 2 and a fresh output.
The runner saves completed rounds as it progresses and rejects an existing
output path. The seed schedule and all raw blocks are stored in RESULTS.json.

For a read-only repeat of the committed report analysis, use the committed
RESULTS.json and specify a fresh --output directory. The analysis checks
source fingerprints, workload identities, execution orders and block
coverage. It writes Markdown and machine-readable paired statistics.
