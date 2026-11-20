# Loop-Nest Optimization

The [report](REPORT.md) contains the twelve custom timing tables, stock DGL
reference, uncertainty, isolated-versus-in-forward comparisons and tuning
recommendations. [Raw samples and binary provenance](RESULTS.json) and
[derived statistics](ANALYSIS.json) are separate from every historical result.

Four one-factor sweeps change only terms (A), score (B), softmax (C), or
aggregation (D). All other compiled kernels retain A1/B1/C1/D1. Every
variant uses the same physical NHF, NH and EH tensors and 64-bit COO/CSC
indices. The only compiler specializations are the distinct loop-order
entry points, at the fixed four heads and eight features. Source and
native symbol fingerprints and actual resolved addresses are audited.

The serial kernels reuse the arithmetic and fixed tensor conventions of
[the independent CPU tuning experiments](../tuning_axes/). The old OpenMP scheduling paths are
inappropriate for an experiment that holds threading fixed. Projection,
degree guard, and bias reuse the existing PyTorch pipeline methods. The
score family fuses add and LeakyReLU identically in both variants.
Report comparisons within the custom family are the loop-order tests.
Stock DGL 1.1.3 is the unchanged correctness and performance reference.

## Reproduction

Build the pinned CPU/compiler images as described in
[the sparse kernel reference](../sparse_formats/README.md). The committed acquisition used
the `dgl-performance-model:axes` image and the unchanged cached citation
datasets. Run from the repository root inside the frozen CPU environment
with GNU C++ and binutils:

~~~sh
p=benchmarks/loop_ordering
uv run --frozen python benchmarks/gat.py --smoke-test
uv run --frozen python "$p/tests.py"
uv run --frozen python "$p/acquire_loops.py" --cpu 0 \
  --data-dir results/datasets --output results/loop-nest/RESULTS.json
uv run --frozen python "$p/report.py" --input results/loop-nest/RESULTS.json \
  --output results/loop-nest
~~~

The output path must be fresh. CPU 0 must be in the process affinity mask;
it is the pinned logical CPU used for all the committed runs. In the
compiler image with a checkout bind mount, keep the image working
directory and use absolute mounted paths for the scripts, data and
outputs, with `uv run --frozen --no-sync`. Temporary .so build products
are under the working directory's ignored `results/loop_order_build/`.
Use `--rounds 2 --warmup 1 --repetitions 2` and a separate output for a
small real-data smoke. Default acquisition uses 8 fresh processes per
dataset, 10 warmup blocks, and 30 measured matched blocks per sweep.

For byte-identical replay of committed artifacts:

~~~sh
uv run --frozen python "$p/report.py" --output results/loop-nest-replay
cmp "$p/REPORT.md" results/loop-nest-replay/REPORT.md
cmp "$p/ANALYSIS.json" results/loop-nest-replay/ANALYSIS.json
~~~

The report checks source fingerprints, stock profile consistency, native
symbol routing, weight/graph provenance, and the randomized acquisition
schedule. Tests cover skewed degree, repeated edges, original IDs,
self-loops, destination-global normalization for extreme logits, empty
rows and edge lists, and stable numerical agreement.
