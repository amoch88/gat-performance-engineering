# Milestone 10 — Independent ordering with imported GATConv

[REPORT.md](REPORT.md) contains the six timing tables, pairing and uncertainty,
ordering diagnostics and interpretation. Each dataset has two independent
one-factor sweeps with identical original/original controls. Each sweep has
its own randomized blocks. Three seeded random permutations per sweep are
crossed with the independent process rounds.

The [seven-by-seven extension report](extended/REPORT.md) uses a separate
acquisition with the same one-factor design. Its raw observations, frozen
METIS plan, permutation maps, and [reproduction commands](extended/README.md)
are kept under `extended/`. The original acquisition below remains replayable.

The endpoint arrays passed to DGL's graph constructor define EID order.
Node relabeling uses the original endpoint sequence and preserves EIDs.
Edge reordering keeps original node IDs and feature rows. Stable source and
destination sorts use original EID ties. Degree sorts use total degree with
original node-ID ties. BFS starts each component at the smallest unseen
original node ID, then visits neighbors in ascending ID over the symmetric
structural adjacency. RCMK uses the SciPy implementation on the same
symmetric adjacency. These auxiliary adjacencies are used to compute
permutations and never replace the measured DGL graph.

Timing uses the unchanged imported GATConv. The profiled copy from
Milestone 9 records six regions from one actual execution. Topology,
aligned data, restored output and attention are checked before timing.
For parity, all node policies allocate their feature tensors using the same
row indexing operation, including the identity control. Edge policies share
the single unchanged feature tensor.
The complete forward has its own direct timer. Permutation planning, graph
rebuild and format construction are timed individually after the forward
sweeps. All created views are audited against the canonical topology.

## Reproduction

Use the pinned CPU and compiler images and cached DGL datasets described
in [Milestone 6](../sparse_formats/README.md). In the frozen Linux environment,
run from repository root:

~~~sh
uv run --frozen python benchmarks/gat.py --smoke-test
uv run --frozen python benchmarks/ordering_baselines/tests.py
uv run --frozen python benchmarks/ordering_baselines/benchmark.py \
  --data-dir results/datasets --output results/m10/RESULTS.json \
  --maps results/m10/PERMUTATIONS.json
uv run --frozen python benchmarks/ordering_baselines/report.py \
  --input results/m10/RESULTS.json --maps results/m10/PERMUTATIONS.json \
  --output results/m10
~~~

For Docker, use the corresponding mounted paths. Defaults are eight fresh
processes per dataset, ten warmups, thirty measured blocks per sweep, and
five separately timed builds per case per process. A smoke trial can use
--rounds 2 --warmup 1 --repetitions 2 --build-repetitions 2. Output paths must
be fresh. Raw timings use nanoseconds, and reports convert them to
milliseconds. Permutation maps use new-to-original orientation, with
identity maps encoded by their length.

To replay the committed report and analysis byte for byte, use
`uv run --frozen python benchmarks/ordering_baselines/report.py --output results/m10-replay`.
The default input includes the archived first acquisition for the allocation
sensitivity check. Its raw timings and exact policy source live in
`initial_allocation/`; its permutation maps match the primary run byte for byte.
Use a separate `--input` to report a new run without comparing it to the archive.

The predeclared practical criterion uses a Holm-adjusted p-value below .05
and a pointwise 95% lower speedup bound above 1.01×. Forward comparisons
form a planned family per sweep across all datasets and policies. Stage
comparisons have separate families per sweep and endpoint. Random controls
are excluded from tuning recommendations. Their intervals quantify process
variability conditional on the three recorded seeds.
