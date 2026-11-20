# Milestone 11 — Independent CPU GAT loop ordering

Custom orders differ only in the loop nest of the targeted kernel. Within each sweep the other three kernels use the same custom defaults A1/B1/C1/D1. Full forward is separately timed without internal clock reads. Six stage columns come from an execution-equivalent instrumented forward, never from isolated timings. All timing-table values are milliseconds and use the median of process medians.

8 fresh process rounds per dataset, 10 warmup blocks and 30 measured randomized matched blocks per sweep. Each block contains the concurrently measured default and all alternatives. Process-median ratios use geometric means and pointwise 95% Student-t intervals on log ratios. Holm correction is separate by sweep and endpoint (full, isolated, in-forward), across the datasets and alternatives. The forward qualification rule requires Holm p < .05 and a 95% CI lower speedup bound > 1.01×.

## Cora

### Attention terms

| Loop order | Projection | Attention Terms | Scores | Softmax | Aggregation | Output | Total Forward |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1 · N → H → F | 2.459739 | 0.032510 | 0.221035 | 0.229887 | 0.148937 | 0.019524 | 3.233267 |
| A2 · H → N → F | 2.479402 | 0.067091 | 0.222078 | 0.226792 | 0.145120 | 0.018827 | 3.261204 |

### Attention scores

| Loop order | Projection | Attention Terms | Scores | Softmax | Aggregation | Output | Total Forward |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 · E → H | 2.514385 | 0.032643 | 0.237979 | 0.228062 | 0.147788 | 0.019398 | 3.282645 |
| B2 · H → E | 2.454189 | 0.032628 | 0.263726 | 0.232637 | 0.148937 | 0.019444 | 3.308220 |

### Edge softmax

| Loop order | Projection | Attention Terms | Scores | Softmax | Aggregation | Output | Total Forward |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1 · V → H → Eᵢₙ | 2.501040 | 0.032763 | 0.228077 | 0.231688 | 0.149623 | 0.019400 | 3.249559 |
| C2 · H → V → Eᵢₙ | 2.481018 | 0.032730 | 0.225415 | 0.244945 | 0.147938 | 0.019435 | 3.293596 |
| C3 · V → Eᵢₙ → H | 2.505482 | 0.032831 | 0.227172 | 0.215659 | 0.149282 | 0.019380 | 3.267535 |

### Aggregation

| Loop order | Projection | Attention Terms | Scores | Softmax | Aggregation | Output | Total Forward |
|---|---:|---:|---:|---:|---:|---:|---:|
| D1 · V → Eᵢₙ → H → F | 2.508921 | 0.032784 | 0.222287 | 0.227767 | 0.146004 | 0.018811 | 3.256207 |
| D2 · V → H → Eᵢₙ → F | 2.509534 | 0.032862 | 0.224131 | 0.233936 | 0.070423 | 0.019446 | 3.161590 |
| D3 · V → H → F → Eᵢₙ | 2.484184 | 0.032747 | 0.224479 | 0.232629 | 0.244178 | 0.019691 | 3.344735 |
| D4 · H → V → Eᵢₙ → F | 2.490466 | 0.032690 | 0.221506 | 0.234126 | 0.081089 | 0.018844 | 3.192266 |

## CiteSeer

### Attention terms

| Loop order | Projection | Attention Terms | Scores | Softmax | Aggregation | Output | Total Forward |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1 · N → H → F | 8.534513 | 0.040158 | 0.224948 | 0.225558 | 0.151735 | 0.025832 | 9.370821 |
| A2 · H → N → F | 8.634904 | 0.082957 | 0.226497 | 0.223423 | 0.152725 | 0.025452 | 9.370743 |

### Attention scores

| Loop order | Projection | Attention Terms | Scores | Softmax | Aggregation | Output | Total Forward |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 · E → H | 8.655739 | 0.040286 | 0.230870 | 0.223623 | 0.149755 | 0.026160 | 9.520589 |
| B2 · H → E | 8.590045 | 0.040261 | 0.251786 | 0.220239 | 0.148814 | 0.025868 | 9.459848 |

### Edge softmax

| Loop order | Projection | Attention Terms | Scores | Softmax | Aggregation | Output | Total Forward |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1 · V → H → Eᵢₙ | 8.670820 | 0.040121 | 0.228429 | 0.224776 | 0.149128 | 0.026364 | 9.559179 |
| C2 · H → V → Eᵢₙ | 8.670668 | 0.040311 | 0.226138 | 0.238600 | 0.147070 | 0.025584 | 9.417999 |
| C3 · V → Eᵢₙ → H | 8.651499 | 0.040744 | 0.228289 | 0.203930 | 0.151233 | 0.025080 | 9.483758 |

### Aggregation

| Loop order | Projection | Attention Terms | Scores | Softmax | Aggregation | Output | Total Forward |
|---|---:|---:|---:|---:|---:|---:|---:|
| D1 · V → Eᵢₙ → H → F | 8.481896 | 0.040334 | 0.224723 | 0.224488 | 0.149777 | 0.025434 | 9.335412 |
| D2 · V → H → Eᵢₙ → F | 8.626668 | 0.040311 | 0.228307 | 0.225683 | 0.076396 | 0.024781 | 9.201032 |
| D3 · V → H → F → Eᵢₙ | 8.642938 | 0.040477 | 0.226971 | 0.225486 | 0.244016 | 0.025793 | 9.403172 |
| D4 · H → V → Eᵢₙ → F | 8.519816 | 0.040761 | 0.225812 | 0.225694 | 0.088283 | 0.024559 | 9.212002 |

## PubMed

### Attention terms

| Loop order | Projection | Attention Terms | Scores | Softmax | Aggregation | Output | Total Forward |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1 · N → H → F | 7.236045 | 0.122309 | 1.810916 | 2.082334 | 1.478904 | 0.089420 | 13.190990 |
| A2 · H → N → F | 7.283932 | 0.387795 | 1.811389 | 2.094920 | 1.471074 | 0.090025 | 13.369339 |

### Attention scores

| Loop order | Projection | Attention Terms | Scores | Softmax | Aggregation | Output | Total Forward |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 · E → H | 7.383083 | 0.125227 | 1.808597 | 2.121274 | 1.478553 | 0.091401 | 13.287946 |
| B2 · H → E | 7.378045 | 0.123088 | 1.978198 | 2.106435 | 1.512663 | 0.091627 | 13.357464 |

### Edge softmax

| Loop order | Projection | Attention Terms | Scores | Softmax | Aggregation | Output | Total Forward |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1 · V → H → Eᵢₙ | 7.424715 | 0.123467 | 1.805770 | 2.093853 | 1.492411 | 0.098336 | 13.421983 |
| C2 · H → V → Eᵢₙ | 7.337282 | 0.125362 | 1.813861 | 2.512631 | 1.483432 | 0.097862 | 13.775065 |
| C3 · V → Eᵢₙ → H | 7.320369 | 0.122523 | 1.804644 | 1.895439 | 1.503483 | 0.098987 | 12.974664 |

### Aggregation

| Loop order | Projection | Attention Terms | Scores | Softmax | Aggregation | Output | Total Forward |
|---|---:|---:|---:|---:|---:|---:|---:|
| D1 · V → Eᵢₙ → H → F | 7.365378 | 0.124739 | 1.800997 | 2.095481 | 1.475163 | 0.093805 | 13.294048 |
| D2 · V → H → Eᵢₙ → F | 7.352613 | 0.124359 | 1.805337 | 2.133320 | 0.616297 | 0.091597 | 12.405656 |
| D3 · V → H → F → Eᵢₙ | 7.451644 | 0.123442 | 1.805874 | 2.079830 | 2.177422 | 0.095264 | 13.967440 |
| D4 · H → V → Eᵢₙ → F | 7.359249 | 0.122923 | 1.801742 | 2.106773 | 0.847866 | 0.090162 | 12.686727 |

## External reference — untouched imported DGL GATConv

The DGL reference has a separate randomized schedule in each process. Its stage columns use the validated profiled copy. Absolute latencies are contextual: library calls, allocation, and kernel implementation differ, so a difference versus stock is not a loop-order effect.

| Dataset | Projection | Attention Terms | Scores | Softmax | Aggregation | Output | Total Forward |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cora | 2.496316 | 0.120305 | 0.187456 | 0.433462 | 0.457685 | 0.022508 | 3.852960 |
| CiteSeer | 8.784473 | 0.150778 | 0.210618 | 0.443345 | 0.453988 | 0.027349 | 10.259409 |
| PubMed | 7.516801 | 0.767366 | 0.703230 | 3.359760 | 3.811452 | 0.103889 | 16.668003 |

## Paired complete-forward comparisons

### Attention terms

| Dataset | Loop order | Forward ms | Paired speedup vs default | 95% CI | Holm p | Decision |
|---|---|---:|---:|---|---:|---|
| Cora | A1 · N → H → F | 3.233267 | 1.0000× | — | — | Default |
| Cora | A2 · H → N → F | 3.261204 | 0.9927× | [0.9774, 1.0081] | 0.5928 | Inconclusive |
| CiteSeer | A1 · N → H → F | 9.370821 | 1.0000× | — | — | Default |
| CiteSeer | A2 · H → N → F | 9.370743 | 1.0022× | [0.9834, 1.0214] | 0.7917 | Inconclusive |
| PubMed | A1 · N → H → F | 13.190990 | 1.0000× | — | — | Default |
| PubMed | A2 · H → N → F | 13.369339 | 0.9853× | [0.9749, 0.9957] | 0.03819 | Slower |

### Attention scores

| Dataset | Loop order | Forward ms | Paired speedup vs default | 95% CI | Holm p | Decision |
|---|---|---:|---:|---|---:|---|
| Cora | B1 · E → H | 3.282645 | 1.0000× | — | — | Default |
| Cora | B2 · H → E | 3.308220 | 0.9908× | [0.9750, 1.0068] | 0.4866 | Inconclusive |
| CiteSeer | B1 · E → H | 9.520589 | 1.0000× | — | — | Default |
| CiteSeer | B2 · H → E | 9.459848 | 0.9994× | [0.9873, 1.0116] | 0.9058 | Inconclusive |
| PubMed | B1 · E → H | 13.287946 | 1.0000× | — | — | Default |
| PubMed | B2 · H → E | 13.357464 | 0.9881× | [0.9703, 1.0062] | 0.4866 | Inconclusive |

### Edge softmax

| Dataset | Loop order | Forward ms | Paired speedup vs default | 95% CI | Holm p | Decision |
|---|---|---:|---:|---|---:|---|
| Cora | C1 · V → H → Eᵢₙ | 3.249559 | 1.0000× | — | — | Default |
| Cora | C2 · H → V → Eᵢₙ | 3.293596 | 0.9946× | [0.9773, 1.0122] | 0.9559 | Inconclusive |
| Cora | C3 · V → Eᵢₙ → H | 3.267535 | 0.9941× | [0.9785, 1.0099] | 0.9559 | Inconclusive |
| CiteSeer | C1 · V → H → Eᵢₙ | 9.559179 | 1.0000× | — | — | Default |
| CiteSeer | C2 · H → V → Eᵢₙ | 9.417999 | 1.0141× | [1.0063, 1.0219] | 0.01423 | Faster, below practical bound |
| CiteSeer | C3 · V → Eᵢₙ → H | 9.483758 | 1.0105× | [0.9875, 1.0339] | 0.9559 | Inconclusive |
| PubMed | C1 · V → H → Eᵢₙ | 13.421983 | 1.0000× | — | — | Default |
| PubMed | C2 · H → V → Eᵢₙ | 13.775065 | 0.9687× | [0.9533, 0.9843] | 0.01113 | Slower |
| PubMed | C3 · V → Eᵢₙ → H | 12.974664 | 1.0264× | [1.0146, 1.0383] | 0.006527 | Qualifies |

### Aggregation

| Dataset | Loop order | Forward ms | Paired speedup vs default | 95% CI | Holm p | Decision |
|---|---|---:|---:|---|---:|---|
| Cora | D1 · V → Eᵢₙ → H → F | 3.256207 | 1.0000× | — | — | Default |
| Cora | D2 · V → H → Eᵢₙ → F | 3.161590 | 1.0314× | [1.0151, 1.0480] | 0.01275 | Qualifies |
| Cora | D3 · V → H → F → Eᵢₙ | 3.344735 | 0.9775× | [0.9716, 0.9835] | 0.0004006 | Slower |
| Cora | D4 · H → V → Eᵢₙ → F | 3.192266 | 1.0176× | [1.0051, 1.0303] | 0.05075 | Inconclusive |
| CiteSeer | D1 · V → Eᵢₙ → H → F | 9.335412 | 1.0000× | — | — | Default |
| CiteSeer | D2 · V → H → Eᵢₙ → F | 9.201032 | 1.0119× | [0.9885, 1.0359] | 0.2705 | Inconclusive |
| CiteSeer | D3 · V → H → F → Eᵢₙ | 9.403172 | 0.9921× | [0.9831, 1.0012] | 0.2404 | Inconclusive |
| CiteSeer | D4 · H → V → Eᵢₙ → F | 9.212002 | 1.0133× | [0.9954, 1.0316] | 0.2465 | Inconclusive |
| PubMed | D1 · V → Eᵢₙ → H → F | 13.294048 | 1.0000× | — | — | Default |
| PubMed | D2 · V → H → Eᵢₙ → F | 12.405656 | 1.0761× | [1.0612, 1.0913] | 4.525e-05 | Qualifies |
| PubMed | D3 · V → H → F → Eᵢₙ | 13.967440 | 0.9572× | [0.9405, 0.9743] | 0.003808 | Slower |
| PubMed | D4 · H → V → Eᵢₙ → F | 12.686727 | 1.0542× | [1.0328, 1.0761] | 0.003512 | Qualifies |

## Isolated and in-forward target kernels

Both regions include the same tensor allocation and ctypes call. An isolated call receives the same held default intermediates for every order. Inside forward it sees freshly produced buffers, a different cache/allocator context, and profiler bookends. Allocations are separately timed below; their medians are not subtracted. The speedups and Holm p-values are from separate process-paired families for isolated and in-forward timings.

| Dataset | Kernel | Loop order | Isolated ms | In-forward ms | Speedup vs default (isolated / in-forward; 95% CI, Holm p) |
|---|---|---|---:|---:|---|
| Cora | Attention terms | A1 · N → H → F | 0.022161 | 0.032510 | Default |
| Cora | Attention terms | A2 · H → N → F | 0.056589 | 0.067091 | 0.3951× [0.3722, 0.4195], p=2.887e-09 / 0.4849× [0.4835, 0.4863], p=3.284e-17 |
| Cora | Attention scores | B1 · E → H | 0.234897 | 0.237979 | Default |
| Cora | Attention scores | B2 · H → E | 0.261581 | 0.263726 | 0.8935× [0.8831, 0.9040], p=2.362e-07 / 0.9011× [0.8956, 0.9067], p=4.738e-09 |
| Cora | Edge softmax | C1 · V → H → Eᵢₙ | 0.228873 | 0.231688 | Default |
| Cora | Edge softmax | C2 · H → V → Eᵢₙ | 0.238157 | 0.244945 | 0.9592× [0.9387, 0.9801], p=0.00523 / 0.9502× [0.9230, 0.9783], p=0.004314 |
| Cora | Edge softmax | C3 · V → Eᵢₙ → H | 0.208454 | 0.215659 | 1.0911× [1.0623, 1.1206], p=0.0005714 / 1.0796× [1.0531, 1.1067], p=0.0003252 |
| Cora | Aggregation | D1 · V → Eᵢₙ → H → F | 0.149714 | 0.146004 | Default |
| Cora | Aggregation | D2 · V → H → Eᵢₙ → F | 0.071140 | 0.070423 | 2.0928× [2.0274, 2.1603], p=1.218e-09 / 2.0746× [2.0106, 2.1407], p=1.371e-09 |
| Cora | Aggregation | D3 · V → H → F → Eᵢₙ | 0.242358 | 0.244178 | 0.6121× [0.6034, 0.6210], p=1.041e-10 / 0.6085× [0.5874, 0.6303], p=1.127e-08 |
| Cora | Aggregation | D4 · H → V → Eᵢₙ → F | 0.079266 | 0.081089 | 1.8649× [1.8164, 1.9146], p=1.218e-09 / 1.8115× [1.7410, 1.8848], p=1.12e-08 |
| CiteSeer | Attention terms | A1 · N → H → F | 0.029956 | 0.040158 | Default |
| CiteSeer | Attention terms | A2 · H → N → F | 0.073455 | 0.082957 | 0.4009× [0.3872, 0.4150], p=1.447e-10 / 0.4877× [0.4698, 0.5064], p=6.714e-10 |
| CiteSeer | Attention scores | B1 · E → H | 0.227993 | 0.230870 | Default |
| CiteSeer | Attention scores | B2 · H → E | 0.251368 | 0.251786 | 0.9113× [0.8925, 0.9304], p=1.478e-05 / 0.9152× [0.9101, 0.9203], p=4.738e-09 |
| CiteSeer | Edge softmax | C1 · V → H → Eᵢₙ | 0.254404 | 0.224776 | Default |
| CiteSeer | Edge softmax | C2 · H → V → Eᵢₙ | 0.263098 | 0.238600 | 0.9819× [0.9539, 1.0106], p=0.1775 / 0.9374× [0.9238, 0.9513], p=7.873e-05 |
| CiteSeer | Edge softmax | C3 · V → Eᵢₙ → H | 0.236627 | 0.203930 | 1.0912× [1.0463, 1.1382], p=0.00523 / 1.0931× [1.0669, 1.1199], p=0.0001604 |
| CiteSeer | Aggregation | D1 · V → Eᵢₙ → H → F | 0.213489 | 0.149777 | Default |
| CiteSeer | Aggregation | D2 · V → H → Eᵢₙ → F | 0.120327 | 0.076396 | 1.7622× [1.6381, 1.8958], p=1.773e-06 / 1.9395× [1.8804, 2.0004], p=2.144e-09 |
| CiteSeer | Aggregation | D3 · V → H → F → Eᵢₙ | 0.296446 | 0.244016 | 0.7253× [0.6931, 0.7589], p=2.644e-06 / 0.6126× [0.5977, 0.6279], p=2.581e-09 |
| CiteSeer | Aggregation | D4 · H → V → Eᵢₙ → F | 0.120070 | 0.088283 | 1.7681× [1.6574, 1.8861], p=8.813e-07 / 1.6800× [1.6338, 1.7276], p=3.295e-09 |
| PubMed | Attention terms | A1 · N → H → F | 0.139408 | 0.122309 | Default |
| PubMed | Attention terms | A2 · H → N → F | 0.428336 | 0.387795 | 0.3281× [0.3172, 0.3393], p=4.4e-11 / 0.3154× [0.3120, 0.3188], p=7.564e-15 |
| PubMed | Attention scores | B1 · E → H | 1.831709 | 1.808597 | Default |
| PubMed | Attention scores | B2 · H → E | 1.982956 | 1.978198 | 0.9202× [0.9111, 0.9295], p=4.46e-07 / 0.9155× [0.9105, 0.9205], p=4.738e-09 |
| PubMed | Edge softmax | C1 · V → H → Eᵢₙ | 2.470250 | 2.093853 | Default |
| PubMed | Edge softmax | C2 · H → V → Eᵢₙ | 2.794162 | 2.512631 | 0.8764× [0.8449, 0.9090], p=0.000363 / 0.8399× [0.8175, 0.8630], p=7.597e-06 |
| PubMed | Edge softmax | C3 · V → Eᵢₙ → H | 2.172606 | 1.895439 | 1.1353× [1.0746, 1.1995], p=0.003785 / 1.1206× [1.0922, 1.1498], p=7.873e-05 |
| PubMed | Aggregation | D1 · V → Eᵢₙ → H → F | 2.151045 | 1.475163 | Default |
| PubMed | Aggregation | D2 · V → H → Eᵢₙ → F | 1.085058 | 0.616297 | 1.9770× [1.6639, 2.3489], p=9.982e-05 / 2.3461× [2.2520, 2.4442], p=2.236e-09 |
| PubMed | Aggregation | D3 · V → H → F → Eᵢₙ | 2.788241 | 2.177422 | 0.7634× [0.6830, 0.8534], p=0.0007126 / 0.6797× [0.6612, 0.6986], p=1.127e-08 |
| PubMed | Aggregation | D4 · H → V → Eᵢₙ → F | 1.303531 | 0.847866 | 1.6239× [1.4063, 1.8752], p=0.0001869 / 1.7329× [1.7022, 1.7642], p=2.211e-10 |

## Allocation and instrumentation

Allocation is the bare factory call (no filling or page-touch guarantee). Aggregation zero-fills the same Y buffer inside every compiled call; softmax initializes its shared head scratch inside each compiled call. Profile overhead averages the process medians of matched instrumented-minus-plain differences. Timer calibration subtracts the matched empty call from the six clock-and-record operations. The analysis contains every order's overhead interval.

| Dataset | Sweep | Bare allocation µs | Empty six-region net µs | Default instrumented − plain µs (95% CI) |
|---|---|---:|---:|---|
| Cora | Attention terms | 5.172 | 1.421 | -0.719 [-27.835, +26.397] |
| Cora | Attention scores | 4.668 | 1.429 | +10.452 [-10.901, +31.805] |
| Cora | Edge softmax | 6.067 | 1.489 | -11.407 [-55.267, +32.454] |
| Cora | Aggregation | 6.054 | 1.478 | +7.055 [-33.457, +47.567] |
| Cora | Stock DGL | — | 1.401 | +19.347 [-33.318, +72.012] |
| CiteSeer | Attention terms | 6.535 | 1.523 | -20.340 [-137.209, +96.529] |
| CiteSeer | Attention scores | 5.071 | 1.492 | -27.149 [-165.778, +111.480] |
| CiteSeer | Edge softmax | 6.764 | 1.571 | -90.392 [-236.910, +56.126] |
| CiteSeer | Aggregation | 7.299 | 1.550 | -44.552 [-125.185, +36.080] |
| CiteSeer | Stock DGL | — | 1.516 | -83.499 [-215.776, +48.778] |
| PubMed | Attention terms | 16.923 | 1.613 | +1.903 [-119.328, +123.134] |
| PubMed | Attention scores | 25.952 | 1.708 | -60.104 [-271.408, +151.199] |
| PubMed | Edge softmax | 28.614 | 1.782 | +31.132 [-112.626, +174.891] |
| PubMed | Aggregation | 33.081 | 1.859 | -36.593 [-220.386, +147.201] |
| PubMed | Stock DGL | — | 1.595 | -39.725 [-308.876, +229.425] |

## What should be tuned?

Fastest denotes the smallest observed median in that column. Qualification is separate and uses only the matched complete-forward comparisons.

| Dataset | Kernel | Fastest isolated | Fastest in-forward | Fastest complete forward | Practical gain vs default |
|---|---|---|---|---|---|
| Cora | Attention terms | A1 | A1 | A1 | None |
| Cora | Attention scores | B1 | B1 | B1 | None |
| Cora | Edge softmax | C3 | C3 | C1 | None |
| Cora | Aggregation | D2 | D2 | D2 | D2 |
| CiteSeer | Attention terms | A1 | A1 | A2 | None |
| CiteSeer | Attention scores | B1 | B1 | B2 | None |
| CiteSeer | Edge softmax | C3 | C3 | C2 | None |
| CiteSeer | Aggregation | D4 | D2 | D2 | None |
| PubMed | Attention terms | A1 | A1 | A1 | None |
| PubMed | Attention scores | B1 | B1 | B1 | None |
| PubMed | Edge softmax | C3 | C3 | C3 | C3 |
| PubMed | Aggregation | D2 | D2 | D2 | D2, D4 |

- **Attention terms.** Fix default A1 · N → H → F in the initial space. No alternative cleared the forward criterion.
- **Attention scores.** Fix default B1 · E → H in the initial space. No alternative cleared the forward criterion.
- **Edge softmax.** Keep C1 · V → H → Eᵢₙ, C3 · V → Eᵢₙ → H as candidates. Qualifications differ by workload or order.
- **Aggregation.** Keep D1 · V → Eᵢₙ → H → F, D2 · V → H → Eᵢₙ → F, D4 · H → V → Eᵢₙ → F as candidates. Qualifications differ by workload or order.

### Stage sensitivity and survival

The latency spread is the highest-to-lowest isolated median in a dataset's sweep. A large spread can reflect a slow alternative, so it is not evidence of speedup over default.

| Kernel | Largest isolated spread | Dataset | Corrected isolated gains | Corrected in-forward gains | Corrected whole-forward gains |
|---|---:|---|---:|---:|---:|
| Attention terms | 3.073× | PubMed | 0 | 0 | 0 |
| Attention scores | 1.114× | Cora | 0 | 0 | 0 |
| Edge softmax | 1.286× | PubMed | 3 | 3 | 2 |
| Aggregation | 3.407× | Cora | 6 | 6 | 3 |

| Dataset | Isolated gain | Isolated result (95% CI, Holm p) | Forward result (95% CI, Holm p) | Forward decision |
|---|---|---|---|---|
| Cora | C3 · V → Eᵢₙ → H | 1.0911× [1.0623, 1.1206], p=0.0005714 | 0.9941× [0.9785, 1.0099], p=0.9559 | Inconclusive |
| CiteSeer | C3 · V → Eᵢₙ → H | 1.0912× [1.0463, 1.1382], p=0.00523 | 1.0105× [0.9875, 1.0339], p=0.9559 | Inconclusive |
| PubMed | C3 · V → Eᵢₙ → H | 1.1353× [1.0746, 1.1995], p=0.003785 | 1.0264× [1.0146, 1.0383], p=0.006527 | Qualifies |
| Cora | D2 · V → H → Eᵢₙ → F | 2.0928× [2.0274, 2.1603], p=1.218e-09 | 1.0314× [1.0151, 1.0480], p=0.01275 | Qualifies |
| Cora | D4 · H → V → Eᵢₙ → F | 1.8649× [1.8164, 1.9146], p=1.218e-09 | 1.0176× [1.0051, 1.0303], p=0.05075 | Inconclusive |
| CiteSeer | D2 · V → H → Eᵢₙ → F | 1.7622× [1.6381, 1.8958], p=1.773e-06 | 1.0119× [0.9885, 1.0359], p=0.2705 | Inconclusive |
| CiteSeer | D4 · H → V → Eᵢₙ → F | 1.7681× [1.6574, 1.8861], p=8.813e-07 | 1.0133× [0.9954, 1.0316], p=0.2465 | Inconclusive |
| PubMed | D2 · V → H → Eᵢₙ → F | 1.9770× [1.6639, 2.3489], p=9.982e-05 | 1.0761× [1.0612, 1.0913], p=4.525e-05 | Qualifies |
| PubMed | D4 · H → V → Eᵢₙ → F | 1.6239× [1.4063, 1.8752], p=0.0001869 | 1.0542× [1.0328, 1.0761], p=0.003512 | Qualifies |

An isolated gain can disappear inside forward when working-set context changes. Even a gain that survives in-forward need not produce a detectable full-layer win because projection and the other stages still run. Confidence intervals do not establish equal latency.

## Access order, memory and compiled identity

All operands stay in the same 64-bit COO/CSC and FP32 NHF/NH/EH layouts. The table counts logical adjacency visits, not measured cache misses or physical memory traffic. The compiler can retain pointers in registers or unroll/vectorize fixed-size head/feature loops. Loop interchange and unroll-and-jam are disabled for all variants.

| Orders | Index visits | Layout and reuse |
|---|---|---|
| A1 / A2 | No adjacency | A1 traverses Z by consecutive node/head cells. A2 scans one head across nodes with a 32-float stride between feature runs. |
| B1 / B2 | E / H·E COO edge visits | B1 reuses endpoints across contiguous EH heads. B2 follows EIDs at a fixed head with stride H. Same fused add and LeakyReLU. |
| C1 / C2 | 3·H·E CSC EID visits | Per-row/head maximum, exponential/denominator and normalization scans. C2 revisits the destinations for each head. |
| C3 | 3·E CSC EID visits | The same three passes coordinate adjacent heads for each edge; row/head reduction order stays fixed. |
| D1 / D2 / D3 / D4 | E / H·E / H·F·E / H·E CSC entry visits | D1 reuses src/EID across heads; D2 and D4 work over contiguous feature vectors. D3 scans the complete incoming row for each output feature. |

Each row's incoming CSC order is unchanged. Softmax uses the same 32-byte max/denominator scratch for all orders and the same FP32 fmax, exp and division. All aggregation orders do the same FP32 multiply/add per output cell in the same edge order. No packing, initialization shortcut, atomics, or per-forward index construction is used.

| Dataset | Target | Tensor payload KiB incl. output | Index payload KiB | Output KiB | Extra scratch bytes |
|---|---|---:|---:|---:|---:|
| Cora | Attention terms | 423.38 | 0.00 | 84.62 | 0 |
| Cora | Attention scores | 291.88 | 207.25 | 207.25 | 0 |
| Cora | Edge softmax | 414.50 | 124.79 | 207.25 | 32 |
| Cora | Aggregation | 884.25 | 228.41 | 338.50 | 0 |
| CiteSeer | Attention terms | 520.09 | 0.00 | 103.97 | 0 |
| CiteSeer | Attention scores | 298.20 | 194.23 | 194.23 | 0 |
| CiteSeer | Edge softmax | 388.47 | 123.12 | 194.23 | 32 |
| CiteSeer | Aggregation | 1025.98 | 220.23 | 415.88 | 0 |
| PubMed | Attention terms | 3081.03 | 0.00 | 616.16 | 0 |
| PubMed | Attention scores | 2309.36 | 1693.20 | 1693.20 | 0 |
| PubMed | Edge softmax | 3386.41 | 1000.65 | 1693.20 | 32 |
| PubMed | Aggregation | 6622.45 | 1847.25 | 2464.62 | 0 |

Payloads describe fixed live stage operands, not process RSS or actual cache traffic.

Build: `g++ (Debian 12.2.0-14+deb12u1) 12.2.0`; `-std=c++17 -O3 -fPIC -shared -ffp-contract=off -fno-loop-interchange -fno-loop-unroll-and-jam`. Binary SHA-256: `d7d1b5120d0b7767e3f734858e4e415b1866d7694286db95e19d65b4aa905130`. Source SHA-256: `05e661bb53c0b528908fc1dbb6fdea9f0fe584ae098e25f7a87a5fe442e1f1b7`.

Each timed route binds the actual exported address during setup. Every loaded symbol offset matches the ELF, and address and normalized disassembly are checked for aliases.

| Config | C ABI symbol | Body bytes | Instructions | Normalized assembly SHA-256 (prefix) |
|---|---|---:|---:|---|
| A1 | `loop_a1` | 819 | 205 | `7a7da9a47bc25973` |
| A2 | `loop_a2` | 3986 | 731 | `22afcbe2512eaca7` |
| B1 | `loop_b1` | 236 | 57 | `9367372a75b21bd2` |
| B2 | `loop_b2` | 132 | 41 | `a42d8192f00e0d00` |
| C1 | `loop_c1` | 375 | 90 | `d6f23fc9b0e8eaee` |
| C2 | `loop_c2` | 365 | 87 | `93402874471e31a0` |
| C3 | `loop_c3` | 602 | 133 | `dd1f583578deb969` |
| D1 | `loop_d1` | 833 | 211 | `861b4c563766c5ca` |
| D2 | `loop_d2` | 227 | 70 | `343412eb5e2e01d7` |
| D3 | `loop_d3` | 256 | 75 | `71afb8da2428331c` |
| D4 | `loop_d4` | 219 | 69 | `e6d6e03f7fb6245b` |

No addresses or normalized instruction bodies matched within any sweep.

Disassembly of the rebuilt, byte-identical measured ELF helps explain why fewer index visits did not make D1 faster. In D1, the backward branch at 0x2dfb returns to a loop with cross-head shuffles, stack spills and output stores for every incoming edge. D2 holds two feature vectors in registers across its edge loop at 0x2ea0 and stores them after the loop. D4 keeps a similar inner loop but visits all destinations at a fixed head. D3 has a scalar multiply/add chain and advances the feature loop outside each complete edge traversal. This is evidence about this compiler's code generation, not a direct measurement of cache misses or the isolated cost of any individual instruction.

## Numerical verification and limits

AMD Ryzen 7 7800X3D 8-Core Processor, logical CPU 0, one thread, Docker CPU base label dgl-performance-model:cpu, Linux-6.18.58.1-microsoft-standard-WSL2-x86_64-with-glibc2.36, Python 3.11.11, PyTorch 2.2.2+cpu, DGL 1.1.3. The original citation graphs, EID ordering, native features and seed-0 weights were preserved. The existing self-loop rule removes all original loops and appends one per node. Inputs were FP32 with four heads, eight features per head, no dropout, residual, or output activation.

The independent DGL stage reference constructs projection, terms, scores, attention and aggregation with native primitives and checks final output and attention against the untouched imported GATConv. Every custom route verifies these intermediates, the original COO/CSC EID mapping, finite outputs and destination-wise normalization before measurement. All custom intermediates were bitwise equal across orders, and direct, repeated and instrumented outputs matched bitwise within the custom family. The stock comparison uses atol 1e−5 and rtol 1e−4 because its implementation may sum in a different order.

Max final-output absolute error: 0. Max final-output relative error over nonzero reference values: 0. Max error as fraction of tolerance across all stages: 0. Max attention sum error: 4.72e-07. Raw data contain per-stage absolute and relative errors, including errors on zero-reference entries. Ordinary relative error can be large near zero; the mixed tolerance is the correctness test.

The study has one CPU, three workloads, one head and channel configuration, and fixed storage and order. It does not measure cross-axis interactions, cache traffic, or transfer effects. The conclusions concern this compiled family and should be rechecked for a different compiler or CPU.

[Raw blocks and build metadata](RESULTS.json) · [Statistical analysis](ANALYSIS.json) · [Reproduction](README.md)
