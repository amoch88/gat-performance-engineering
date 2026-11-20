# Milestone 8 — Sparse formats for CPU GAT

Each row uses one selected storage layout for every supported sparse stage. The three native rows use fixed native fallbacks where DGL dispatch requires another format. [N/A explanations and exact routes](NA_EXPLANATIONS.md).

All cells are milliseconds and report the median of process medians. Projection and attention terms share one measured value per block across rows because the same operations and inputs are used. Sparse stages are timed separately on reference inputs. The full forward is timed directly. For SELL the size is the slice height.

## Timing tables

### Cora

| Storage Format | Projection | Attention Terms (L/R) | Attention Scores | Edge Softmax | Aggregation | Total Forward |
|---|---:|---:|---:|---:|---:|---:|
| Official imported DGL GATConv | 3.105643 | 0.131799 | 0.181657 | 0.418787 | 0.493401 | 3.904638 |
| Native DGL COO (hybrid) | 3.105643 | 0.131799 | 0.123544 | N/A | 0.872444 | 4.197399 |
| Native DGL CSR (hybrid) | 3.105643 | 0.131799 | 0.147062 | N/A | N/A | 3.730390 |
| Native DGL CSC (hybrid) | 3.105643 | 0.131799 | N/A | 0.419633 | 0.424710 | 3.680334 |
| Blocked CSR 2×2 | 3.105643 | 0.131799 | 0.331557 | 0.390752 | 0.208993 | 3.693070 |
| Blocked CSR 4×4 | 3.105643 | 0.131799 | 0.360952 | 0.442645 | 0.228445 | 3.770164 |
| Blocked CSR 8×8 | 3.105643 | 0.131799 | 0.393427 | 0.483129 | 0.240281 | 3.858937 |
| Blocked CSR 16×16 | 3.105643 | 0.131799 | 0.435097 | 0.575160 | 0.269862 | 4.079333 |
| Blocked CSR 32×32 | 3.105643 | 0.131799 | 0.462790 | 0.651541 | 0.293725 | 4.193959 |
| Blocked CSR 64×64 | 3.105643 | 0.131799 | 0.407533 | 0.545805 | 0.261699 | 3.994791 |
| Blocked CSC 2×2 | 3.105643 | 0.131799 | 0.322981 | 0.381928 | 0.213053 | 3.657140 |
| Blocked CSC 4×4 | 3.105643 | 0.131799 | 0.359389 | 0.429963 | 0.226304 | 3.787562 |
| Blocked CSC 8×8 | 3.105643 | 0.131799 | 0.387687 | 0.473927 | 0.239640 | 3.898833 |
| Blocked CSC 16×16 | 3.105643 | 0.131799 | 0.421850 | 0.557098 | 0.285370 | 4.035204 |
| Blocked CSC 32×32 | 3.105643 | 0.131799 | 0.446944 | 0.613203 | 0.320752 | 4.182981 |
| Blocked CSC 64×64 | 3.105643 | 0.131799 | 0.400225 | 0.518107 | 0.275297 | 3.977546 |
| CSB 2×2 | 3.105643 | 0.131799 | 0.271844 | 0.253221 | 0.171200 | 3.401421 |
| CSB 4×4 | 3.105643 | 0.131799 | 0.275384 | 0.249076 | 0.170003 | 3.419682 |
| CSB 8×8 | 3.105643 | 0.131799 | 0.275542 | 0.246758 | 0.167403 | 3.383374 |
| CSB 16×16 | 3.105643 | 0.131799 | 0.277501 | 0.253056 | 0.166999 | 3.408135 |
| CSB 32×32 | 3.105643 | 0.131799 | 0.275601 | 0.258138 | 0.168093 | 3.409411 |
| CSB 64×64 | 3.105643 | 0.131799 | 0.265863 | 0.222386 | 0.154329 | 3.353436 |
| Block Dense 2×2 | 3.105643 | 0.131799 | 0.326750 | 0.374384 | 0.208494 | 3.635789 |
| Block Dense 4×4 | 3.105643 | 0.131799 | 0.360887 | 0.441438 | 0.231991 | 3.738104 |
| Block Dense 8×8 | 3.105643 | 0.131799 | 0.393055 | 0.524873 | 0.264067 | 3.990004 |
| Block Dense 16×16 | 3.105643 | 0.131799 | 0.422964 | 0.646860 | 0.296021 | 4.403125 |
| Block Dense 32×32 | 3.105643 | 0.131799 | 0.448585 | 0.723689 | 0.326938 | 4.758974 |
| Block Dense 64×64 | 3.105643 | 0.131799 | 0.402104 | 0.623778 | 0.302552 | 4.341195 |
| SELL 2 | 3.105643 | 0.131799 | 0.261083 | 0.251731 | 0.168236 | 3.406188 |
| SELL 4 | 3.105643 | 0.131799 | 0.281251 | 0.256460 | 0.168890 | 3.404848 |
| SELL 8 | 3.105643 | 0.131799 | 0.286436 | 0.287228 | 0.174923 | 3.463965 |
| SELL 16 | 3.105643 | 0.131799 | 0.293133 | 0.312054 | 0.183658 | 3.508989 |
| SELL 32 | 3.105643 | 0.131799 | 0.296209 | 0.326299 | 0.189773 | 3.563250 |
| SELL 64 | 3.105643 | 0.131799 | 0.304752 | 0.348897 | 0.200592 | 3.583723 |

### CiteSeer

| Storage Format | Projection | Attention Terms (L/R) | Attention Scores | Edge Softmax | Aggregation | Total Forward |
|---|---:|---:|---:|---:|---:|---:|
| Official imported DGL GATConv | 9.922955 | 0.155538 | 0.194472 | 0.425087 | 0.503887 | 10.007215 |
| Native DGL COO (hybrid) | 9.922955 | 0.155538 | 0.124607 | N/A | 0.851599 | 9.974854 |
| Native DGL CSR (hybrid) | 9.922955 | 0.155538 | 0.148475 | N/A | N/A | 9.742386 |
| Native DGL CSC (hybrid) | 9.922955 | 0.155538 | N/A | 0.428719 | 0.423186 | 9.542912 |
| Blocked CSR 2×2 | 9.922955 | 0.155538 | 0.317389 | 0.361103 | 0.197595 | 9.389166 |
| Blocked CSR 4×4 | 9.922955 | 0.155538 | 0.349676 | 0.411688 | 0.212328 | 9.675932 |
| Blocked CSR 8×8 | 9.922955 | 0.155538 | 0.384468 | 0.453890 | 0.228571 | 9.737176 |
| Blocked CSR 16×16 | 9.922955 | 0.155538 | 0.430883 | 0.551117 | 0.252178 | 9.961793 |
| Blocked CSR 32×32 | 9.922955 | 0.155538 | 0.482320 | 0.676607 | 0.290570 | 10.016387 |
| Blocked CSR 64×64 | 9.922955 | 0.155538 | 0.451016 | 0.622686 | 0.272834 | 10.106813 |
| Blocked CSC 2×2 | 9.922955 | 0.155538 | 0.323807 | 0.355091 | 0.204013 | 9.568428 |
| Blocked CSC 4×4 | 9.922955 | 0.155538 | 0.354493 | 0.400687 | 0.217162 | 9.510600 |
| Blocked CSC 8×8 | 9.922955 | 0.155538 | 0.383681 | 0.441177 | 0.227905 | 9.720939 |
| Blocked CSC 16×16 | 9.922955 | 0.155538 | 0.423634 | 0.533672 | 0.276888 | 9.795138 |
| Blocked CSC 32×32 | 9.922955 | 0.155538 | 0.468341 | 0.634792 | 0.334073 | 10.189232 |
| Blocked CSC 64×64 | 9.922955 | 0.155538 | 0.441068 | 0.580609 | 0.308301 | 9.959783 |
| CSB 2×2 | 9.922955 | 0.155538 | 0.259530 | 0.233629 | 0.160512 | 9.125802 |
| CSB 4×4 | 9.922955 | 0.155538 | 0.264457 | 0.226542 | 0.156415 | 9.169380 |
| CSB 8×8 | 9.922955 | 0.155538 | 0.269273 | 0.225028 | 0.154223 | 9.215504 |
| CSB 16×16 | 9.922955 | 0.155538 | 0.269530 | 0.226822 | 0.158307 | 9.302397 |
| CSB 32×32 | 9.922955 | 0.155538 | 0.271502 | 0.233729 | 0.159994 | 9.186633 |
| CSB 64×64 | 9.922955 | 0.155538 | 0.264084 | 0.214544 | 0.148746 | 9.260628 |
| Block Dense 2×2 | 9.922955 | 0.155538 | 0.309948 | 0.342763 | 0.193979 | 9.446644 |
| Block Dense 4×4 | 9.922955 | 0.155538 | 0.347541 | 0.405114 | 0.216274 | 9.500669 |
| Block Dense 8×8 | 9.922955 | 0.155538 | 0.387075 | 0.503383 | 0.257777 | 9.788507 |
| Block Dense 16×16 | 9.922955 | 0.155538 | 0.435096 | 0.683846 | 0.321750 | 10.363761 |
| Block Dense 32×32 | 9.922955 | 0.155538 | 0.488791 | 0.819889 | 0.386919 | 10.775529 |
| Block Dense 64×64 | 9.922955 | 0.155538 | 0.468416 | 0.809829 | 0.350066 | 10.520167 |
| SELL 2 | 9.922955 | 0.155538 | 0.265926 | 0.239251 | 0.164806 | 9.282819 |
| SELL 4 | 9.922955 | 0.155538 | 0.283065 | 0.245845 | 0.166168 | 9.236375 |
| SELL 8 | 9.922955 | 0.155538 | 0.283661 | 0.273588 | 0.172694 | 9.365690 |
| SELL 16 | 9.922955 | 0.155538 | 0.288601 | 0.295364 | 0.176645 | 9.454563 |
| SELL 32 | 9.922955 | 0.155538 | 0.291534 | 0.306136 | 0.181993 | 9.507414 |
| SELL 64 | 9.922955 | 0.155538 | 0.298026 | 0.321693 | 0.191188 | 9.456531 |

### PubMed

| Storage Format | Projection | Attention Terms (L/R) | Attention Scores | Edge Softmax | Aggregation | Total Forward |
|---|---:|---:|---:|---:|---:|---:|
| Official imported DGL GATConv | 9.559101 | 0.830955 | 0.857436 | 3.355237 | 3.888108 | 16.553568 |
| Native DGL COO (hybrid) | 9.559101 | 0.830955 | 0.755625 | N/A | 7.246601 | 20.055999 |
| Native DGL CSR (hybrid) | 9.559101 | 0.830955 | 0.825949 | N/A | N/A | 16.278996 |
| Native DGL CSC (hybrid) | 9.559101 | 0.830955 | N/A | 3.319920 | 3.689200 | 16.100883 |
| Blocked CSR 2×2 | 9.559101 | 0.830955 | 2.544639 | 3.102800 | 1.829717 | 16.465251 |
| Blocked CSR 4×4 | 9.559101 | 0.830955 | 2.875884 | 3.617145 | 1.947919 | 17.253464 |
| Blocked CSR 8×8 | 9.559101 | 0.830955 | 3.262367 | 4.137514 | 2.115123 | 18.275347 |
| Blocked CSR 16×16 | 9.559101 | 0.830955 | 3.880691 | 5.251963 | 2.509866 | 20.741055 |
| Blocked CSR 32×32 | 9.559101 | 0.830955 | 4.940339 | 7.557580 | 3.167212 | 24.587283 |
| Blocked CSR 64×64 | 9.559101 | 0.830955 | 5.975640 | 9.766114 | 3.725565 | 28.704397 |
| Blocked CSC 2×2 | 9.559101 | 0.830955 | 2.521132 | 3.186133 | 1.889628 | 16.481699 |
| Blocked CSC 4×4 | 9.559101 | 0.830955 | 2.860542 | 3.624303 | 2.011867 | 17.380802 |
| Blocked CSC 8×8 | 9.559101 | 0.830955 | 3.238995 | 4.229983 | 2.179268 | 18.363943 |
| Blocked CSC 16×16 | 9.559101 | 0.830955 | 3.810213 | 5.240920 | 2.668403 | 20.825731 |
| Blocked CSC 32×32 | 9.559101 | 0.830955 | 4.819345 | 7.320128 | 3.854366 | 25.181526 |
| Blocked CSC 64×64 | 9.559101 | 0.830955 | 5.762080 | 9.200686 | 4.829859 | 29.264057 |
| CSB 2×2 | 9.559101 | 0.830955 | 2.103008 | 2.112664 | 1.582291 | 14.397993 |
| CSB 4×4 | 9.559101 | 0.830955 | 2.123622 | 2.052013 | 1.551239 | 14.507040 |
| CSB 8×8 | 9.559101 | 0.830955 | 2.154336 | 2.024847 | 1.564532 | 14.447152 |
| CSB 16×16 | 9.559101 | 0.830955 | 2.163882 | 2.040361 | 1.572404 | 14.566944 |
| CSB 32×32 | 9.559101 | 0.830955 | 2.178613 | 2.121875 | 1.579645 | 14.359960 |
| CSB 64×64 | 9.559101 | 0.830955 | 2.216277 | 2.298701 | 1.611171 | 14.692041 |
| Block Dense 2×2 | 9.559101 | 0.830955 | 2.478662 | 2.989937 | 1.813719 | 16.073784 |
| Block Dense 4×4 | 9.559101 | 0.830955 | 2.890096 | 3.591840 | 1.945111 | 17.187417 |
| Block Dense 8×8 | 9.559101 | 0.830955 | 3.484653 | 5.039527 | 2.678274 | 19.855970 |
| Block Dense 16×16 | 9.559101 | 0.830955 | 6.141753 | 9.482430 | 7.087721 | 28.103751 |
| Block Dense 32×32 | 9.559101 | 0.830955 | 8.452322 | 14.723026 | 9.371654 | 37.960748 |
| Block Dense 64×64 | 9.559101 | 0.830955 | 9.737715 | 19.319895 | 10.268979 | 44.460476 |
| SELL 2 | 9.559101 | 0.830955 | 1.927421 | 2.263501 | 1.621574 | 14.344164 |
| SELL 4 | 9.559101 | 0.830955 | 2.150025 | 2.281363 | 1.614014 | 14.657094 |
| SELL 8 | 9.559101 | 0.830955 | 2.272416 | 2.524596 | 1.675034 | 15.260661 |
| SELL 16 | 9.559101 | 0.830955 | 2.394865 | 2.861083 | 1.758067 | 15.830215 |
| SELL 32 | 9.559101 | 0.830955 | 2.484373 | 3.135197 | 1.804037 | 16.160204 |
| SELL 64 | 9.559101 | 0.830955 | 2.605761 | 3.293343 | 1.948996 | 16.542315 |

## Fastest observed configurations

The control is the fastest supported native stage format by observed median. Total forward uses untouched imported GATConv as its control. Speedup is the geometric mean of paired process-median ratios, control divided by candidate. Intervals are pointwise 95% Student-t intervals on log ratios. Holm correction is applied separately for each endpoint across datasets, native controls and all within-family size pairs.

| Dataset | Stage | Fastest observed | Milliseconds | Control | Speedup (95% CI) | Holm p |
|---|---|---|---:|---|---|---:|
| Cora | Attention Scores | Native DGL COO (hybrid) | 0.123544 | Native DGL COO (hybrid) | Control | — |
| Cora | Edge Softmax | CSB 64×64 | 0.222386 | Native DGL CSC (hybrid) | 1.884× [1.870, 1.898] | 7.176e-12 |
| Cora | Aggregation | CSB 64×64 | 0.154329 | Native DGL CSC (hybrid) | 2.757× [2.711, 2.805] | 9.6e-11 |
| Cora | Total Forward | CSB 64×64 | 3.353436 | Official imported DGL GATConv | 1.154× [1.136, 1.172] | 5.88e-05 |
| CiteSeer | Attention Scores | Native DGL COO (hybrid) | 0.124607 | Native DGL COO (hybrid) | Control | — |
| CiteSeer | Edge Softmax | CSB 64×64 | 0.214544 | Native DGL CSC (hybrid) | 2.005× [1.984, 2.027] | 5.255e-11 |
| CiteSeer | Aggregation | CSB 64×64 | 0.148746 | Native DGL CSC (hybrid) | 2.852× [2.796, 2.909] | 2.258e-10 |
| CiteSeer | Total Forward | CSB 2×2 | 9.125802 | Official imported DGL GATConv | 1.084× [1.062, 1.106] | 0.009556 |
| PubMed | Attention Scores | Native DGL COO (hybrid) | 0.755625 | Native DGL COO (hybrid) | Control | — |
| PubMed | Edge Softmax | CSB 8×8 | 2.024847 | Native DGL CSC (hybrid) | 1.630× [1.607, 1.653] | 3.24e-09 |
| PubMed | Aggregation | CSB 4×4 | 1.551239 | Native DGL CSC (hybrid) | 2.343× [2.289, 2.398] | 2.506e-09 |
| PubMed | Total Forward | SELL 2 | 14.344164 | Official imported DGL GATConv | 1.146× [1.123, 1.170] | 0.0004178 |

The complete-forward ranks are CSB 64×64 on Cora, CSB 2×2 on CiteSeer, SELL 2 on PubMed. The paired comparisons below test whether those observed differences justify changing formats.

### Evidence for changing size

This table counts all size pairs within each format family across the three datasets. Direction is for the larger size. Individual pairs and their intervals are in ANALYSIS.json.

| Endpoint | Family | Larger faster | Larger slower | Inconclusive |
|---|---|---:|---:|---:|
| Attention Scores | Blocked CSR | 3 | 42 | 0 |
| Attention Scores | Blocked CSC | 3 | 42 | 0 |
| Attention Scores | CSB | 5 | 12 | 28 |
| Attention Scores | Block Dense | 2 | 40 | 3 |
| Attention Scores | SELL | 0 | 39 | 6 |
| Edge Softmax | Blocked CSR | 3 | 42 | 0 |
| Edge Softmax | Blocked CSC | 3 | 42 | 0 |
| Edge Softmax | CSB | 12 | 9 | 24 |
| Edge Softmax | Block Dense | 1 | 42 | 2 |
| Edge Softmax | SELL | 0 | 42 | 3 |
| Aggregation | Blocked CSR | 2 | 42 | 1 |
| Aggregation | Blocked CSC | 3 | 41 | 1 |
| Aggregation | CSB | 12 | 0 | 33 |
| Aggregation | Block Dense | 1 | 40 | 4 |
| Aggregation | SELL | 0 | 32 | 13 |
| Total Forward | Blocked CSR | 1 | 26 | 18 |
| Total Forward | Blocked CSC | 1 | 27 | 17 |
| Total Forward | CSB | 0 | 0 | 45 |
| Total Forward | Block Dense | 0 | 36 | 9 |
| Total Forward | SELL | 0 | 13 | 32 |

For each fastest observed full-forward size, these are the slower sizes within the same family supported by the paired tests after correction.

- On Cora, the size comparisons for CSB 64×64 were inconclusive after correction.
- On CiteSeer, the size comparisons for CSB 2×2 were inconclusive after correction.
- On PubMed, SELL 2 beat SELL 8, SELL 16, SELL 32, SELL 64 after correction.

## Storage and construction

Payload counts the stored index arrays, including masks, padding, EID and parallel-edge sidecar entries. Native indices are 64-bit and custom coordinate/EID indices are 32-bit. Shared DGL graph views, temporary build memory and runtime scratch are outside these payload counts. Native construction starts with a freshly created COO graph. Custom construction starts from canonical endpoint arrays. Forward timing includes intermediate allocation, scratch, and all executed stages. Inputs and outputs stay in canonical EID order throughout.

| Dataset | Format | Index KiB | Construction median ms |
|---|---|---:|---:|
| Cora | Native DGL COO (hybrid) | 207.2 | 0.024892 |
| Cora | Native DGL CSR (hybrid) | 228.4 | 0.056255 |
| Cora | Native DGL CSC (hybrid) | 228.4 | 0.055715 |
| Cora | Blocked CSR 2×2 | 238.5 | 1.301240 |
| Cora | Blocked CSR 4×4 | 297.1 | 1.336218 |
| Cora | Blocked CSR 8×8 | 413.5 | 1.312691 |
| Cora | Blocked CSR 16×16 | 597.8 | 1.241353 |
| Cora | Blocked CSR 32×32 | 728.8 | 1.229579 |
| Cora | Blocked CSR 64×64 | 549.4 | 1.080853 |
| Cora | Blocked CSC 2×2 | 238.5 | 0.975341 |
| Cora | Blocked CSC 4×4 | 297.1 | 1.023621 |
| Cora | Blocked CSC 8×8 | 413.5 | 1.081759 |
| Cora | Blocked CSC 16×16 | 597.8 | 1.103318 |
| Cora | Blocked CSC 32×32 | 728.8 | 1.135635 |
| Cora | Blocked CSC 64×64 | 549.4 | 1.122876 |
| Cora | CSB 2×2 | 285.0 | 1.179787 |
| Cora | CSB 4×4 | 269.9 | 1.185482 |
| Cora | CSB 8×8 | 258.3 | 1.170625 |
| Cora | CSB 16×16 | 242.5 | 1.049203 |
| Cora | CSB 32×32 | 212.2 | 0.983623 |
| Cora | CSB 64×64 | 176.0 | 0.890895 |
| Cora | Block Dense 2×2 | 432.0 | 1.497074 |
| Cora | Block Dense 4×4 | 992.4 | 1.604507 |
| Cora | Block Dense 8×8 | 2811.4 | 1.685805 |
| Cora | Block Dense 16×16 | 8419.1 | 1.825109 |
| Cora | Block Dense 32×32 | 20637.6 | 2.282146 |
| Cora | Block Dense 64×64 | 28971.2 | 2.415100 |
| Cora | SELL 2 | 140.6 | 0.664818 |
| Cora | SELL 4 | 179.6 | 0.672285 |
| Cora | SELL 8 | 235.6 | 0.673836 |
| Cora | SELL 16 | 316.2 | 0.654599 |
| Cora | SELL 32 | 437.8 | 0.671216 |
| Cora | SELL 64 | 589.7 | 0.666414 |
| CiteSeer | Native DGL COO (hybrid) | 194.2 | 0.027036 |
| CiteSeer | Native DGL CSR (hybrid) | 220.2 | 0.054568 |
| CiteSeer | Native DGL CSC (hybrid) | 220.2 | 0.055613 |
| CiteSeer | Blocked CSR 2×2 | 224.7 | 1.205360 |
| CiteSeer | Blocked CSR 4×4 | 280.3 | 1.240131 |
| CiteSeer | Blocked CSR 8×8 | 398.2 | 1.250468 |
| CiteSeer | Blocked CSR 16×16 | 602.9 | 1.253137 |
| CiteSeer | Blocked CSR 32×32 | 824.6 | 1.196559 |
| CiteSeer | Blocked CSR 64×64 | 734.1 | 1.078285 |
| CiteSeer | Blocked CSC 2×2 | 224.7 | 0.892956 |
| CiteSeer | Blocked CSC 4×4 | 280.3 | 1.001805 |
| CiteSeer | Blocked CSC 8×8 | 398.2 | 1.024682 |
| CiteSeer | Blocked CSC 16×16 | 602.9 | 1.053729 |
| CiteSeer | Blocked CSC 32×32 | 824.6 | 1.131158 |
| CiteSeer | Blocked CSC 64×64 | 734.1 | 1.102368 |
| CiteSeer | CSB 2×2 | 266.8 | 1.106750 |
| CiteSeer | CSB 4×4 | 253.6 | 1.127102 |
| CiteSeer | CSB 8×8 | 245.5 | 1.101234 |
| CiteSeer | CSB 16×16 | 234.8 | 1.042948 |
| CiteSeer | CSB 32×32 | 211.8 | 0.958257 |
| CiteSeer | CSB 64×64 | 175.1 | 0.872650 |
| CiteSeer | Block Dense 2×2 | 403.7 | 1.392451 |
| CiteSeer | Block Dense 4×4 | 935.6 | 1.471902 |
| CiteSeer | Block Dense 8×8 | 2728.4 | 1.578043 |
| CiteSeer | Block Dense 16×16 | 8613.9 | 1.770217 |
| CiteSeer | Block Dense 32×32 | 24014.1 | 2.344582 |
| CiteSeer | Block Dense 64×64 | 41401.6 | 4.555802 |
| CiteSeer | SELL 2 | 133.8 | 0.627841 |
| CiteSeer | SELL 4 | 172.7 | 0.602363 |
| CiteSeer | SELL 8 | 224.3 | 0.615130 |
| CiteSeer | SELL 16 | 289.3 | 0.598463 |
| CiteSeer | SELL 32 | 371.2 | 0.622127 |
| CiteSeer | SELL 64 | 485.7 | 0.637728 |
| PubMed | Native DGL COO (hybrid) | 1693.2 | 0.041484 |
| PubMed | Native DGL CSR (hybrid) | 1847.2 | 0.278281 |
| PubMed | Native DGL CSC (hybrid) | 1847.2 | 0.340787 |
| PubMed | Blocked CSR 2×2 | 2035.1 | 12.249312 |
| PubMed | Blocked CSR 4×4 | 2673.8 | 12.833147 |
| PubMed | Blocked CSR 8×8 | 3988.8 | 12.948358 |
| PubMed | Blocked CSR 16×16 | 6507.8 | 13.202535 |
| PubMed | Blocked CSR 32×32 | 10682.4 | 14.474554 |
| PubMed | Blocked CSR 64×64 | 14740.3 | 16.526113 |
| PubMed | Blocked CSC 2×2 | 2035.1 | 8.502544 |
| PubMed | Blocked CSC 4×4 | 2673.8 | 9.339256 |
| PubMed | Blocked CSC 8×8 | 3988.8 | 10.104644 |
| PubMed | Blocked CSC 16×16 | 6507.8 | 11.255419 |
| PubMed | Blocked CSC 32×32 | 10682.4 | 13.686691 |
| PubMed | Blocked CSC 64×64 | 14740.3 | 15.356504 |
| PubMed | CSB 2×2 | 2419.9 | 11.557964 |
| PubMed | CSB 4×4 | 2354.7 | 11.573549 |
| PubMed | CSB 8×8 | 2314.1 | 11.547441 |
| PubMed | CSB 16×16 | 2268.1 | 10.888775 |
| PubMed | CSB 32×32 | 2163.8 | 10.226097 |
| PubMed | CSB 64×64 | 1911.1 | 9.712395 |
| PubMed | Block Dense 2×2 | 3833.3 | 14.305479 |
| PubMed | Block Dense 4×4 | 9401.4 | 15.502712 |
| PubMed | Block Dense 8×8 | 28540.8 | 16.704498 |
| PubMed | Block Dense 16×16 | 96491.8 | 24.636212 |
| PubMed | Block Dense 32×32 | 324798.7 | 37.874142 |
| PubMed | Block Dense 64×64 | 903225.0 | 91.366750 |
| PubMed | SELL 2 | 1290.1 | 6.156717 |
| PubMed | SELL 4 | 1877.1 | 6.080113 |
| PubMed | SELL 8 | 2668.7 | 6.196209 |
| PubMed | SELL 16 | 3658.4 | 6.057828 |
| PubMed | SELL 32 | 4772.9 | 6.175097 |
| PubMed | SELL 64 | 6077.2 | 6.161192 |

### Padding costs

Slots count only allocated dense tile cells or SELL lanes. The slot-to-edge ratio includes padding and uses the canonical edge count as denominator. Dense row masks skip padding during traversal. SELL scans each slice's padded lanes.

| Dataset | Layout | Slots per edge at size 2 | Slots per edge at size 64 | Index MiB at 2 / 64 |
|---|---|---:|---:|---:|
| Cora | Block Dense | 3.33 | 541.95 | 0.422 / 28.292 |
| Cora | SELL | 1.31 | 5.69 | 0.137 / 0.576 |
| CiteSeer | Block Dense | 3.33 | 826.38 | 0.394 / 40.431 |
| CiteSeer | SELL | 1.31 | 5.00 | 0.131 / 0.474 |
| PubMed | Block Dense | 3.62 | 2068.13 | 3.743 / 882.056 |
| PubMed | SELL | 1.48 | 7.18 | 1.260 / 5.935 |

## Supported full-forward comparisons

A custom configuration qualifies below only if it beats imported GATConv and all three native hybrid rows after correction. The speedup is the smallest of those four paired estimates.

| Dataset | Qualifying configurations | Largest minimum speedup |
|---|---:|---|
| Cora | 10 | CSB 64×64 (1.097×) |
| CiteSeer | 1 | CSB 16×16 (1.033×) |
| PubMed | 8 | SELL 2 (1.118×) |

The comparisons estimate the net effect of using a format across its sparse stages. The isolated timings help locate bottlenecks, while the forward comparison also captures allocation and data-access effects across stages. Use the paired size contrasts to evaluate whether a size advantage extends across workloads.

### Interpreting the new padded formats

- Cora. The fastest observed Block Dense configuration was Block Dense 2×2 at 3.636 ms (1.068× vs stock, Holm p=0.000362). At 64×64 it took 4.341 ms (0.888×, Holm p=0.00725).
- CiteSeer. The fastest observed Block Dense configuration was Block Dense 2×2 at 9.447 ms (1.055× vs stock, Holm p=0.029). At 64×64 it took 10.520 ms (0.941×, Holm p=0.00143).
- PubMed. The fastest observed Block Dense configuration was Block Dense 2×2 at 16.074 ms (1.034× vs stock, Holm p=0.941). At 64×64 it took 44.460 ms (0.375×, Holm p=4.97e-07).

Dense tiles trade a large reserved EID grid and a scan of occupancy words for implicit coordinates. Larger tiles can increase both costs even as tile count falls. SELL trades padded lanes for contiguous slice storage. A high-degree destination raises the slice width for all its neighbors in row order. The within-family contrasts quantify the size effect for these specific traversals. The timing and capacity measurements do not isolate cache misses.

## Methods and validation

AMD Ryzen 7 7800X3D 8-Core Processor, logical CPU 0, one thread, DGL 1.1.3 and PyTorch 2.2.2+cpu. 8 fresh serial processes per dataset, 10 warmup and 30 randomized matched blocks per timed phase, plus 5 construction samples per round. All observations were retained.

The raw log contains 24,480 measured forwards, 72,000 measured stage calls and 3,960 construction observations.

FP32 inference with no dropout, 4 heads and 8 channels per head. The loader uses original features and ordering, removes existing self-loops and appends one per node. Canonical endpoint, feature and seed-0 weight hashes match Milestone 5. Every custom layout reconstructs every EID including parallel edges. All intermediates and final output were compared against the stock-backed reference with atol=1e−5 and rtol=1e−4. Every active destination/head sum was checked.

Dense tiles allocate every coordinate in each nonempty square tile. Row masks gate EID reads and a sidecar retains parallel edges. SELL uses lane-interleaved destination rows and scans the padded slots with an invalid-EID guard. The two use stable softmax across every incoming edge of a destination. Blocked sparse traversal reuses Milestones 6 and 7.

Stock sparse stages replay apply_edges, edge_softmax and update_all on a local graph with preassigned tensors. Native-format variants use GSDDMM, edge_softmax and GSpMM on restricted views. The score boundary includes LeakyReLU. Custom score fuses the activation, while DGL calls it separately. Index width, API overhead, and activation fusion vary across families, so comparisons estimate implementation speed rather than a causal effect of layout alone.

- Cora maximum absolute errors. Final output 7.45e-09, isolated sparse stage 2.98e-08, destination normalization 4.72e-07.
- CiteSeer maximum absolute errors. Final output 3.73e-09, isolated sparse stage 2.98e-08, destination normalization 3.09e-07.
- PubMed maximum absolute errors. Final output 1.12e-08, isolated sparse stage 4.47e-08, destination normalization 4.6e-07.

[Raw observations and provenance](RESULTS.json) · [Paired estimates and variability](ANALYSIS.json)
