# Working on this repository

This is a CPU-only Linux x86-64 Python 3.11 research baseline. Use the existing
checkout: cloud tasks are isolated; do not create worktrees unless requested.
Use uv 0.13.4 and `uv sync --frozen`. Keep PyTorch bound to its explicit CPU index.
Do not add accelerator dependencies, execution paths, or configurations.

Preserve the stock `dgl.nn.pytorch.GATConv` baseline. Put future experimental
variants behind a separate interface/name; never silently change the baseline.
Do not implement prediction, tuning, or fusion until requested.

Prioritize correctness, reproducibility, minimal dependencies and transparent
methodology. Keep graph/model/input setup, warmups, validation and serialization
outside forward timing. Record raw samples, workload, seeds, versions and threads.
Document edge/self-loop semantics, units, and any methodology changes.

Before measuring, run `uv run --frozen python benchmarks/gat.py --smoke-test`.
After changes run a small benchmark, inspect its JSON, and verify invalid CLI
inputs fail. Build and smoke-test the Docker image when Docker is available.
Report skipped checks honestly. Consult official DGL docs and releases before
changing the DGL/PyTorch/Python combination; regenerate and review uv.lock and
check that it contains no accelerator packages. Do not compare performance
across dependency upgrades as if only the algorithm changed.

The public-API parameter sweeps live under benchmarks/dgl_baseline.
Keep the annotated GATConv class AST identical to installed DGL 1.1.3; retain
DGL-LICENSE and attribution. Graph/feature transforms belong to the separate
experiment files with adjacent EXPERIMENTAL CHANGE comments. Preserve gat.py.
Run all seven --verify-only entrypoints and tests.py; report.py must use measured
JSON and never relabel synthetic verification as real dataset performance.
Do not suppress single-format incompatibilities or correctness failures to
produce timings. Keep raw results, provenance and preprocessing separate from
forward timing. Observe the Reddit memory preflight before full-graph loading.
