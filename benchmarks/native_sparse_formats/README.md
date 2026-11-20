# Native DGL and Blocked Sparse Kernels

See [REPORT.md](REPORT.md) for the measured comparison and the preliminary
annotated/imported equivalence test. This benchmark extends the sparse kernel reference by
reusing its compiled traversal and storage representation, adding 2, 4 and 8
node tiles, and replacing every unblocked comparison with an official DGL op.
The reference kernels, observations, and report are archived unchanged.

## Reproduction

Use the frozen CPU environment and dataset cache instructions in
[the sparse kernel reference](../sparse_formats/README.md). The dataset guard checks the
recorded canonical EIDs, native features and exact seed-0 state_dict hashes.
The measurement host used the existing tuning-axes Docker image with Python
3.11.11, torch 2.2.2+cpu, DGL 1.1.3, and uv 0.13.4. No dependency changes.

Run from the repository root in the pinned Linux environment (or mount the
checkout into the existing compiler image and pass the mounted paths):

~~~sh
uv run --frozen python benchmarks/gat.py --smoke-test
uv run --frozen python benchmarks/native_sparse_formats/tests.py
uv run --frozen python benchmarks/native_sparse_formats/acquire.py preliminary \
  --data-dir results/datasets --output results/native-sparse-trial/PRELIMINARY.json
uv run --frozen python benchmarks/native_sparse_formats/report.py \
  --preliminary results/native-sparse-trial/PRELIMINARY.json
# Review equivalence and investigate meaningful discrepancies before proceeding.
uv run --frozen python benchmarks/native_sparse_formats/acquire.py sparse \
  --data-dir results/datasets --output results/native-sparse-trial/RESULTS.json
uv run --frozen python benchmarks/native_sparse_formats/report.py \
  --preliminary results/native-sparse-trial/PRELIMINARY.json --input results/native-sparse-trial/RESULTS.json \
  --output results/native-sparse-trial
~~~

Defaults: 8 fresh serial processes per dataset in each phase, 10 warmup
blocks, 60 measured preliminary pairs or 30 measured sparse blocks per
endpoint. Dataset order and orders within blocks are randomized. Use --cpu
to select an allowed logical CPU; otherwise the first allowed CPU is pinned.
Raw logs preserve warmups, orders, all measured observations and five build
samples per format/round. Existing output files cannot be overwritten.

For a separate integration smoke use --datasets cora --rounds 2 --warmup 1
--repetitions 2 and a fresh output. Small smoke trials are not the reported
measurements. Figures and report are reproducible from the recorded JSON.

The preliminary analysis uses per-dataset paired TOST (90% Student-t interval)
with a fixed annotated/imported latency band of [0.97, 1.03]. The sparse
analysis uses paired process-median log ratios with pointwise 95% intervals
and separate Holm families for isolated kernel and full-forward endpoints.
The experiment covers all supported native contrasts and all 2/4/8 vs
16/32/64 comparisons within each blocked family, without tuning combinations.

The three native operations are GSDDMM(add) with LeakyReLU, edge_softmax(dst),
and GSpMM(mul,sum). Only the supported sole-format views are dispatched.
Native COO softmax is not inferred from DGL accepting a graph that can
materialize CSC. No unsupported native format receives a substitute timing.
