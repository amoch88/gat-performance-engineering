# Milestone 6: sparse storage formats

See [REPORT.md](REPORT.md) for the measured results. This benchmark substitutes
one sparse kernel at a time into Milestone 5's fixed-stage GAT pipeline. It does
not implement tuning, format selection, a tensor model, or GPU execution.

## Layout definitions

| Layout | Stored coordinates and traversal |
|---|---|
| COO | Source/destination arrays in canonical edge order; EID is implicit. |
| CSR / CSC | Source/destination-major pointer arrays, minor indices, canonical EID map. |
| Blocked CSR / CSC | Compressed block-row/block-column directory of nonempty tiles, opposite block indices, local major pointer per tile, local minor indices, canonical EID map. |
| CSB | Directory with both block coordinates and tile offsets for nonempty tiles; each tile stores two local coordinate arrays and canonical EIDs. |

Blocked CSR and CSC have sparse payloads inside their tiles, not dense BSR
payloads: no padded edges or duplicate coalescing. The local pointer tables
have empty rows/columns and can be larger than the edge arrays. CSB uses sparse
coordinate payloads instead. All blocks are square, at 16, 32 or 64 nodes per
side. No node permutation is applied to increase tile density.

The three operations share the same C++ arithmetic across formats. Stable
softmax uses destination-global max and sum reductions, including for edges
spread across many tiles. Outputs are written directly in canonical EID order;
the index lookup is part of the timed kernel. There is no runtime resort, gather,
or format conversion. Projection, node terms, bias, and untuned sparse stages
are inherited from Milestone 5. The isolated benchmark uses precomputed inputs,
not stages intercepted inside a timed complete forward. Stock GATConv is an
independent oracle and an additional full-forward reference, not a format case.

## Reproduce

Linux x86-64, Python 3.11, uv 0.13.4 and the unchanged repository lockfile.
Build the existing CPU and tuning-axes images:

```sh
docker build -t dgl-performance-model:cpu .
docker build -f benchmarks/tuning_axes/Dockerfile -t dgl-performance-model:axes .
docker run --rm -v "$PWD:/checkout" --entrypoint uv dgl-performance-model:axes \
  run --frozen --no-sync python /checkout/benchmarks/gat.py --smoke-test
docker run --rm -v "$PWD:/checkout" --entrypoint uv dgl-performance-model:axes \
  run --frozen --no-sync python /checkout/benchmarks/sparse_formats/tests.py
docker run --rm -v "$PWD:/checkout" --entrypoint uv dgl-performance-model:axes \
  run --frozen --no-sync python /checkout/benchmarks/sparse_formats/run.py \
  --data-dir /checkout/results/datasets --output /checkout/results/m6-trial/RESULTS.json
docker run --rm -v "$PWD:/checkout" --entrypoint uv dgl-performance-model:axes \
  run --frozen --no-sync python /checkout/benchmarks/sparse_formats/report.py \
  --input /checkout/results/m6-trial/RESULTS.json --output /checkout/results/m6-trial
```

Use the standard DGL cache if already populated. For this acquisition the
Cora archive came from the official DGL S3 bucket; CiteSeer and PubMed raw
files came from [upstream Planetoid](https://github.com/kimiyoung/planetoid/tree/af6abf468a772cc055bec88c52e4d6f51b7c37e3/data)
because their DGL archive endpoints were unavailable. The unchanged DGL parser
and repository loader were used; exact feature, EID and seed-0 weight hashes
are checked against Milestone 5 before timing. The cache, source dataset files,
and compiled binaries are not committed. No trained-weight claim is made.

For offline reproduction, extract the contents of the trusted
`https://dgl-data.s3.amazonaws.com/dataset/cora_v2.zip` into
`results/datasets/cora_v2_d697a464/`. Copy the eight `ind.citeseer.*`
files from the pinned Planetoid source into `results/datasets/citeseer_d6836239/`,
and the eight `ind.pubmed.*` files into `results/datasets/pubmed_35464cad/`.
These are the DGL 1.1.3 raw-path names for the default DGL dataset URL.
DGL processes them normally; the benchmark requires the recorded workload
fingerprints. Do not load untrusted pickle files as dataset inputs.

Defaults are eight rounds, ten warmup blocks, thirty measurement blocks,
model seed 0 and order seed 20261113. Set `--cpu` to pin a specific allowed
logical CPU (default: first allowed). To smoke-test a separate acquisition use
`--datasets cora --rounds 2 --warmup 1 --repetitions 2` and a fresh output
path. Existing raw outputs are never overwritten by the runner.

Raw JSON stores each timing block in execution order. Reports are independently
regenerable from it. CSV summaries are flat, with one format/endpoint or
contrast per row, separate units in column names, and uncertainty alongside
point estimates. SVGs use a zero origin and mark the COO baseline at 1×.
