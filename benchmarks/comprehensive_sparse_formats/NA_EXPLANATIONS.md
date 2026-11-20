# N/A and hybrid pipelines

The native entries use sole-format DGL graph views. The limitations below are from [DGL 1.1.3 CPU dispatch](https://raw.githubusercontent.com/dmlc/dgl/v1.1.3/src/array/kernel.cc). An N/A stage has no timing. Each hybrid forward directly times the exact stages listed below.

| Selected format | Score | Softmax | Aggregation | Reason for N/A |
|---|---|---|---|---|
| COO | COO | CSC fallback | COO | CPU destination edge softmax directly requests a CSC matrix. |
| CSR | CSR | CSC fallback | CSC fallback | CPU destination edge softmax requests CSC. GSpMM supports only COO and CSC. |
| CSC | COO fallback | CSC | CSC | GSDDMM supports only COO and CSR. |

The fallback always uses native DGL on a prebuilt view with the same canonical EIDs. The stock row uses unmodified imported GATConv for its complete forward. Its sparse stage cells replay its public DGL calls in isolation, using the same graph formats and precomputed inputs.

## Memory guards

The recorded per-layout limit is 1,073,741,824 bytes and the combined custom index payload limit is 2,147,483,648 bytes. Construction also needs room for one more layout and temporary arrays. The runner checks the padded layout size before allocation.

- Cora completed all custom formats and sizes within the guard limits.
- CiteSeer completed all custom formats and sizes within the guard limits.
- PubMed completed all custom formats and sizes within the guard limits.
