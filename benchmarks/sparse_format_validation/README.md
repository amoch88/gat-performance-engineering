# Milestone 9 — Validate sparse format speedups and kernel timing

[The report](REPORT.md) contains the three timing tables, matched comparisons,
and the fixed Milestone 8 replication. All stage times include output
allocation at their stated boundary. The report records both isolation
and timing from inside a complete execution.

## Reproduce

Use the pinned CPU compiler image and cached datasets from the
[Milestone 6 instructions](../sparse_formats/README.md). This experiment
reuses the Milestone 8 pipeline and padded layout kernels, the Milestone 7
native wrappers and CSB builder, and the Milestone 6 custom traversal kernel.

~~~sh
uv run --frozen python benchmarks/gat.py --smoke-test
uv run --frozen python benchmarks/sparse_format_validation/tests.py
uv run --frozen python benchmarks/sparse_format_validation/runner.py \
  --data-dir results/datasets --output results/m9/RESULTS.json
uv run --frozen python benchmarks/sparse_format_validation/report.py \
  --input results/m9/RESULTS.json --output results/m9
~~~

Use the mounted paths when running in Docker. Defaults are one pinned logical
CPU, eight fresh processes, ten warmup blocks, thirty recorded blocks and five
separate build observations per format per process. For a smoke trial use
--datasets cora --rounds 2 --warmup 1 --repetitions 2 --build-repetitions 2.
Output must be a new path. All timings in the raw blocks are in nanoseconds.
The report converts them to milliseconds.

## Comparisons

The three native hybrid views follow Milestone 8. Three added route-matched
custom hybrids use the identical per-stage COO/CSR/CSC labels with the
Milestone 6 kernels. Custom unblocked single-format rows use one format for
all sparse stages and provide the controls for CSB and SELL.

Each randomized block interleaves plain forwards, instrumented forwards,
isolated calls and projection context probes. The stage profiler builds a
separate executable copy of the forward body, preserving its statements and
inserting two timestamp reads and a record write per region. The stock plain
forward calls the original import. The profiler leaves graph setup, teardown,
and any work outside the named boundaries in an unassigned remainder.
GATConv's graph scope is preserved.

The timer microbenchmark measures an empty call and an empty sequence of six
recorded regions. The projection probes put the existing degree guard, stock
sparse replay or stock forward immediately before the same isolated
projection. Probe preparation is outside the timed region. These are controlled
changes in preceding work with possible cache, memory allocation and frequency
effects. Layouts and weights stay fixed during these probes.

The historical comparison selects Cora CSB64, CiteSeer CSB2 and PubMed SELL2
from the prior report. The fresh rounds are analyzed separately. The report
rebuilds from the committed raw samples with source and workload checks.
