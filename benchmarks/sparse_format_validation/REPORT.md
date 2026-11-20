# Milestone 9 — Sparse format and timing validation

Each stage cell is **isolated / in-forward**, in milliseconds. Each number is a median of process medians. Total forward uses the plain measured call. The instrumented copy uses the original executable forward statements with six stage boundaries. The stock direct call uses the untouched imported GATConv.

Projection, L/R terms and output bias use common isolated controls with the same mathematics. Stock isolated sparse cells replay its public DGL calls on prepared tensors. In-forward values run on intermediates produced inside that call. N/A marks unsupported native isolated cells. Their in-forward timings belong to the stated native fallback.

## Findings

- **Cora**. The fastest observed forward was Custom COO at 3.288 ms. Custom implementations won 3 of 3 route-matched backend comparisons after correction. All CSB and SELL forward gains over the custom unblocked controls fell short of the corrected threshold.
- **CiteSeer**. The fastest observed forward was Custom COO at 9.433 ms. Custom implementations won 3 of 3 route-matched backend comparisons after correction. All CSB and SELL forward gains over the custom unblocked controls fell short of the corrected threshold.
- **PubMed**. The fastest observed forward was Custom COO at 13.375 ms. Custom implementations won 3 of 3 route-matched backend comparisons after correction. All CSB and SELL forward gains over the custom unblocked controls fell short of the corrected threshold.

## Cora

| Implementation | Projection | Attention Terms (L/R) | Attention Scores | Edge Softmax | Aggregation | Output | Total Forward | Speedup vs stock (95% CI) | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| Imported GATConv | 2.51267 / 2.50964 | 0.11553 / 0.12178 | 0.16605 / 0.19604 | 0.41882 / 0.43460 | 0.50784 / 0.46474 | 0.01969 / 0.02235 | 3.90062 | Control | — |
| Native COO hybrid | 2.51267 / 2.51829 | 0.11553 / 0.12075 | 0.11444 / 0.11768 | N/A / 0.40728 | 0.90624 / 0.88859 | 0.01969 / 0.02263 | 4.17831 | 0.935× [0.923, 0.948] | 0.0001653 |
| Native CSR hybrid | 2.51267 / 2.50128 | 0.11553 / 0.12081 | 0.13331 / 0.13681 | N/A / 0.40174 | N/A / 0.42822 | 0.01969 / 0.02143 | 3.71773 | 1.054× [1.045, 1.063] | 6.906e-05 |
| Native CSC hybrid | 2.51267 / 2.50712 | 0.11553 / 0.12083 | N/A / 0.11762 | 0.42109 / 0.40725 | 0.43609 / 0.42388 | 0.01969 / 0.02021 | 3.70579 | 1.055× [1.045, 1.065] | 0.0001179 |
| Custom COO | 2.51267 / 2.51590 | 0.11553 / 0.12055 | 0.24990 / 0.25522 | 0.17746 / 0.17439 | 0.12831 / 0.12394 | 0.01969 / 0.01752 | 3.28844 | 1.187× [1.180, 1.194] | 3.29e-09 |
| Custom CSR | 2.51267 / 2.52015 | 0.11553 / 0.12111 | 0.24000 / 0.24337 | 0.20632 / 0.20746 | 0.13701 / 0.13449 | 0.01969 / 0.01916 | 3.32839 | 1.174× [1.161, 1.188] | 3.149e-07 |
| Custom CSC | 2.51267 / 2.50979 | 0.11553 / 0.12106 | 0.23131 / 0.23514 | 0.21053 / 0.20991 | 0.16450 / 0.15934 | 0.01969 / 0.01961 | 3.34244 | 1.171× [1.159, 1.183] | 1.306e-07 |
| CSB 2×2 | 2.51267 / 2.51390 | 0.11553 / 0.12081 | 0.27994 / 0.28289 | 0.23586 / 0.23838 | 0.16452 / 0.16351 | 0.01969 / 0.01961 | 3.41817 | 1.141× [1.133, 1.149] | 4.333e-08 |
| CSB 4×4 | 2.51267 / 2.50264 | 0.11553 / 0.12084 | 0.28343 / 0.28740 | 0.23314 / 0.23744 | 0.16137 / 0.15917 | 0.01969 / 0.01973 | 3.42775 | 1.144× [1.137, 1.151] | 1.886e-08 |
| CSB 8×8 | 2.51267 / 2.50780 | 0.11553 / 0.12077 | 0.28493 / 0.28705 | 0.23231 / 0.23084 | 0.16216 / 0.15938 | 0.01969 / 0.01937 | 3.44867 | 1.135× [1.123, 1.148] | 1.104e-06 |
| CSB 16×16 | 2.51267 / 2.53157 | 0.11553 / 0.12092 | 0.28562 / 0.28936 | 0.23498 / 0.24107 | 0.16432 / 0.15953 | 0.01969 / 0.01938 | 3.42183 | 1.141× [1.131, 1.150] | 1.364e-07 |
| CSB 32×32 | 2.51267 / 2.50949 | 0.11553 / 0.12088 | 0.28538 / 0.28808 | 0.24135 / 0.24387 | 0.16477 / 0.16060 | 0.01969 / 0.01965 | 3.44051 | 1.137× [1.128, 1.146] | 1.149e-07 |
| CSB 64×64 | 2.51267 / 2.49776 | 0.11553 / 0.12075 | 0.27373 / 0.27746 | 0.20493 / 0.20136 | 0.15072 / 0.14442 | 0.01969 / 0.01912 | 3.36417 | 1.163× [1.152, 1.174] | 1.125e-07 |
| SELL 2 | 2.51267 / 2.51010 | 0.11553 / 0.12099 | 0.27063 / 0.27403 | 0.23485 / 0.23683 | 0.17098 / 0.16672 | 0.01969 / 0.01964 | 3.42324 | 1.142× [1.133, 1.151] | 8.46e-08 |
| SELL 4 | 2.51267 / 2.49676 | 0.11553 / 0.12084 | 0.28880 / 0.29212 | 0.24074 / 0.24235 | 0.17072 / 0.16532 | 0.01969 / 0.01950 | 3.45777 | 1.135× [1.125, 1.144] | 1.909e-07 |
| SELL 8 | 2.51267 / 2.51469 | 0.11553 / 0.12083 | 0.29682 / 0.29935 | 0.27525 / 0.27657 | 0.17851 / 0.17132 | 0.01969 / 0.01971 | 3.50403 | 1.118× [1.110, 1.127] | 2.633e-07 |
| SELL 16 | 2.51267 / 2.49641 | 0.11553 / 0.12082 | 0.30064 / 0.30329 | 0.30366 / 0.30371 | 0.18587 / 0.18233 | 0.01969 / 0.01978 | 3.56560 | 1.095× [1.082, 1.107] | 1.053e-05 |
| SELL 32 | 2.51267 / 2.49170 | 0.11553 / 0.12065 | 0.30623 / 0.30992 | 0.31908 / 0.32101 | 0.19325 / 0.18782 | 0.01969 / 0.01982 | 3.54580 | 1.098× [1.091, 1.105] | 2.986e-07 |
| SELL 64 | 2.51267 / 2.49434 | 0.11553 / 0.12063 | 0.31510 / 0.32009 | 0.34155 / 0.34423 | 0.20346 / 0.20095 | 0.01969 / 0.01998 | 3.58957 | 1.086× [1.079, 1.092] | 4.232e-07 |
| Custom COO matched hybrid | 2.51267 / 2.49200 | 0.11553 / 0.12098 | 0.24990 / 0.25497 | 0.21053 / 0.20614 | 0.12831 / 0.12361 | 0.01969 / 0.01813 | 3.33205 | 1.173× [1.160, 1.187] | 2.772e-07 |
| Custom CSR matched hybrid | 2.51267 / 2.50074 | 0.11553 / 0.12078 | 0.24000 / 0.24342 | 0.21053 / 0.20520 | 0.16450 / 0.15735 | 0.01969 / 0.01908 | 3.35965 | 1.164× [1.155, 1.172] | 2.25e-08 |
| Custom CSC matched hybrid | 2.51267 / 2.50677 | 0.11553 / 0.12086 | 0.24990 / 0.25606 | 0.21053 / 0.20853 | 0.16450 / 0.15775 | 0.01969 / 0.01875 | 3.36462 | 1.161× [1.153, 1.169] | 1.894e-08 |

## CiteSeer

| Implementation | Projection | Attention Terms (L/R) | Attention Scores | Edge Softmax | Aggregation | Output | Total Forward | Speedup vs stock (95% CI) | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| Imported GATConv | 8.70008 / 8.62073 | 0.14271 / 0.14729 | 0.20837 / 0.23674 | 0.46046 / 0.44668 | 0.58025 / 0.46938 | 0.02171 / 0.02693 | 10.00561 | Control | — |
| Native COO hybrid | 8.70008 / 8.57226 | 0.14271 / 0.14757 | 0.13774 / 0.14105 | N/A / 0.41788 | 0.88565 / 0.85187 | 0.02171 / 0.02624 | 10.25233 | 0.977× [0.972, 0.983] | 0.0005264 |
| Native CSR hybrid | 8.70008 / 8.60120 | 0.14271 / 0.14663 | 0.15490 / 0.16309 | N/A / 0.41926 | N/A / 0.42176 | 0.02171 / 0.02487 | 9.81578 | 1.019× [1.002, 1.037] | 0.1874 |
| Native CSC hybrid | 8.70008 / 8.64200 | 0.14271 / 0.14877 | N/A / 0.14134 | 0.46894 / 0.42172 | 0.49982 / 0.42228 | 0.02171 / 0.02495 | 10.00884 | 1.005× [0.995, 1.015] | 0.5077 |
| Custom COO | 8.70008 / 8.58761 | 0.14271 / 0.14900 | 0.24477 / 0.24689 | 0.16790 / 0.16930 | 0.12714 / 0.12113 | 0.02171 / 0.02353 | 9.43337 | 1.065× [1.048, 1.083] | 0.0008668 |
| Custom CSR | 8.70008 / 8.73903 | 0.14271 / 0.14778 | 0.23270 / 0.23068 | 0.20763 / 0.21055 | 0.13850 / 0.13680 | 0.02171 / 0.02452 | 9.45803 | 1.066× [1.045, 1.089] | 0.002262 |
| Custom CSC | 8.70008 / 8.58949 | 0.14271 / 0.14746 | 0.24119 / 0.24140 | 0.21863 / 0.20104 | 0.21043 / 0.15714 | 0.02171 / 0.02271 | 9.48281 | 1.063× [1.048, 1.078] | 0.0003983 |
| CSB 2×2 | 8.70008 / 8.57392 | 0.14271 / 0.14755 | 0.27461 / 0.27407 | 0.22254 / 0.22494 | 0.16085 / 0.15711 | 0.02171 / 0.02383 | 9.43708 | 1.059× [1.048, 1.071] | 0.000146 |
| CSB 4×4 | 8.70008 / 8.65546 | 0.14271 / 0.14717 | 0.28253 / 0.27763 | 0.21759 / 0.21720 | 0.15881 / 0.15153 | 0.02171 / 0.02370 | 9.52028 | 1.053× [1.042, 1.064] | 0.0001845 |
| CSB 8×8 | 8.70008 / 8.62286 | 0.14271 / 0.14704 | 0.28141 / 0.28248 | 0.21640 / 0.21775 | 0.16280 / 0.15345 | 0.02171 / 0.02406 | 9.62005 | 1.042× [1.026, 1.059] | 0.006424 |
| CSB 16×16 | 8.70008 / 8.51189 | 0.14271 / 0.14783 | 0.28196 / 0.28288 | 0.21778 / 0.22057 | 0.16905 / 0.15395 | 0.02171 / 0.02397 | 9.54421 | 1.059× [1.040, 1.077] | 0.002075 |
| CSB 32×32 | 8.70008 / 8.62977 | 0.14271 / 0.14783 | 0.28121 / 0.28340 | 0.23216 / 0.23202 | 0.18120 / 0.15348 | 0.02171 / 0.02394 | 9.62946 | 1.043× [1.034, 1.052] | 0.0001845 |
| CSB 64×64 | 8.70008 / 8.59725 | 0.14271 / 0.14710 | 0.27300 / 0.27430 | 0.20820 / 0.20161 | 0.18397 / 0.14661 | 0.02171 / 0.02326 | 9.48662 | 1.053× [1.046, 1.060] | 1.758e-05 |
| SELL 2 | 8.70008 / 8.67222 | 0.14271 / 0.14721 | 0.27990 / 0.27959 | 0.24671 / 0.23221 | 0.22845 / 0.16489 | 0.02171 / 0.02320 | 9.56048 | 1.052× [1.035, 1.070] | 0.002544 |
| SELL 4 | 8.70008 / 8.55844 | 0.14271 / 0.14767 | 0.29690 / 0.29296 | 0.25743 / 0.24114 | 0.22717 / 0.16592 | 0.02171 / 0.02348 | 9.66455 | 1.038× [1.019, 1.058] | 0.01989 |
| SELL 8 | 8.70008 / 8.57401 | 0.14271 / 0.14736 | 0.29881 / 0.29881 | 0.28337 / 0.27238 | 0.22440 / 0.17245 | 0.02171 / 0.02384 | 9.53858 | 1.049× [1.034, 1.065] | 0.002165 |
| SELL 16 | 8.70008 / 8.56852 | 0.14271 / 0.14771 | 0.30673 / 0.30229 | 0.30529 / 0.29451 | 0.23754 / 0.17595 | 0.02171 / 0.02362 | 9.77945 | 1.027× [1.016, 1.038] | 0.006424 |
| SELL 32 | 8.70008 / 8.61387 | 0.14271 / 0.14788 | 0.30890 / 0.30691 | 0.32291 / 0.30719 | 0.25435 / 0.18512 | 0.02171 / 0.02382 | 9.76246 | 1.028× [1.012, 1.044] | 0.03241 |
| SELL 64 | 8.70008 / 8.63625 | 0.14271 / 0.14743 | 0.31534 / 0.31367 | 0.33916 / 0.32244 | 0.26418 / 0.19020 | 0.02171 / 0.02446 | 9.75246 | 1.030× [1.014, 1.046] | 0.02831 |
| Custom COO matched hybrid | 8.70008 / 8.58064 | 0.14271 / 0.14845 | 0.24477 / 0.24539 | 0.21863 / 0.20326 | 0.12714 / 0.12269 | 0.02171 / 0.02348 | 9.43811 | 1.060× [1.036, 1.084] | 0.006424 |
| Custom CSR matched hybrid | 8.70008 / 8.59891 | 0.14271 / 0.14751 | 0.23270 / 0.23104 | 0.21863 / 0.20567 | 0.21043 / 0.15847 | 0.02171 / 0.02297 | 9.43339 | 1.057× [1.039, 1.076] | 0.002165 |
| Custom CSC matched hybrid | 8.70008 / 8.61340 | 0.14271 / 0.14745 | 0.24477 / 0.24648 | 0.21863 / 0.20350 | 0.21043 / 0.15846 | 0.02171 / 0.02358 | 9.53539 | 1.052× [1.040, 1.064] | 0.000351 |

## PubMed

| Implementation | Projection | Attention Terms (L/R) | Attention Scores | Edge Softmax | Aggregation | Output | Total Forward | Speedup vs stock (95% CI) | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| Imported GATConv | 7.46076 / 7.49573 | 0.78954 / 0.77199 | 0.79260 / 0.74395 | 3.64588 / 3.28795 | 5.32687 / 3.79465 | 0.11443 / 0.10217 | 16.71540 | Control | — |
| Native COO hybrid | 7.46076 / 7.45319 | 0.78954 / 0.76978 | 0.69912 / 0.64660 | N/A / 3.29063 | 7.46390 / 7.45465 | 0.11443 / 0.10179 | 20.07132 | 0.832× [0.816, 0.848] | 4.003e-06 |
| Native CSR hybrid | 7.46076 / 7.41049 | 0.78954 / 0.77708 | 0.79756 / 0.70466 | N/A / 3.26421 | N/A / 3.76657 | 0.11443 / 0.10016 | 16.28051 | 1.025× [0.997, 1.054] | 0.2394 |
| Native CSC hybrid | 7.46076 / 7.54589 | 0.78954 / 0.77067 | N/A / 0.64412 | 3.63051 / 3.24909 | 5.05322 / 3.72156 | 0.11443 / 0.10099 | 16.40625 | 1.025× [1.000, 1.050] | 0.2394 |
| Custom COO | 7.46076 / 7.47345 | 0.78954 / 0.77517 | 1.93044 / 1.88412 | 1.56081 / 1.53348 | 1.27364 / 1.28109 | 0.11443 / 0.09737 | 13.37467 | 1.256× [1.216, 1.297] | 2.597e-05 |
| Custom CSR | 7.46076 / 7.40680 | 0.78954 / 0.76799 | 1.89040 / 1.83001 | 1.77528 / 1.76392 | 1.37347 / 1.40179 | 0.11443 / 0.09962 | 13.81879 | 1.214× [1.178, 1.251] | 4.661e-05 |
| Custom CSC | 7.46076 / 7.35234 | 0.78954 / 0.77293 | 1.78621 / 1.72276 | 2.31921 / 1.98031 | 2.58431 / 1.55169 | 0.11443 / 0.10172 | 14.08010 | 1.199× [1.175, 1.223] | 4.878e-06 |
| CSB 2×2 | 7.46076 / 7.41429 | 0.78954 / 0.77553 | 2.26651 / 2.18093 | 2.05948 / 2.03392 | 1.61913 / 1.60107 | 0.11443 / 0.10174 | 14.71877 | 1.137× [1.121, 1.153] | 5.254e-06 |
| CSB 4×4 | 7.46076 / 7.30276 | 0.78954 / 0.76677 | 2.30564 / 2.21639 | 2.04598 / 2.00085 | 1.63173 / 1.56927 | 0.11443 / 0.10102 | 14.55507 | 1.149× [1.120, 1.178] | 0.0001268 |
| CSB 8×8 | 7.46076 / 7.39576 | 0.78954 / 0.77449 | 2.32780 / 2.25648 | 2.04770 / 2.00559 | 1.75691 / 1.58656 | 0.11443 / 0.10312 | 14.84273 | 1.140× [1.107, 1.173] | 0.0003364 |
| CSB 16×16 | 7.46076 / 7.37147 | 0.78954 / 0.77423 | 2.34326 / 2.27847 | 2.08456 / 2.03658 | 1.87992 / 1.56835 | 0.11443 / 0.10316 | 14.80183 | 1.138× [1.110, 1.167] | 0.0001653 |
| CSB 32×32 | 7.46076 / 7.40548 | 0.78954 / 0.77036 | 2.36775 / 2.30033 | 2.22173 / 2.07917 | 2.07092 / 1.58479 | 0.11443 / 0.10291 | 14.74243 | 1.135× [1.110, 1.161] | 0.0001114 |
| CSB 64×64 | 7.46076 / 7.33236 | 0.78954 / 0.76647 | 2.38910 / 2.31887 | 2.47508 / 2.18535 | 2.30688 / 1.59612 | 0.11443 / 0.10162 | 14.80196 | 1.119× [1.097, 1.140] | 8.957e-05 |
| SELL 2 | 7.46076 / 7.28767 | 0.78954 / 0.77462 | 2.10281 / 2.03167 | 2.65530 / 2.27379 | 2.81856 / 1.63250 | 0.11443 / 0.10412 | 14.58199 | 1.140× [1.110, 1.170] | 0.0001846 |
| SELL 4 | 7.46076 / 7.43618 | 0.78954 / 0.77356 | 2.32482 / 2.24918 | 2.70708 / 2.32658 | 2.90798 / 1.60891 | 0.11443 / 0.10154 | 15.23636 | 1.102× [1.088, 1.116] | 1.645e-05 |
| SELL 8 | 7.46076 / 7.38057 | 0.78954 / 0.77294 | 2.49282 / 2.38872 | 2.93205 / 2.49790 | 2.88573 / 1.63088 | 0.11443 / 0.10372 | 15.45423 | 1.089× [1.064, 1.114] | 0.0009154 |
| SELL 16 | 7.46076 / 7.35129 | 0.78954 / 0.76804 | 2.61624 / 2.49776 | 3.30030 / 2.88729 | 3.04146 / 1.72137 | 0.11443 / 0.10473 | 16.02380 | 1.052× [1.021, 1.084] | 0.03616 |
| SELL 32 | 7.46076 / 7.37977 | 0.78954 / 0.76422 | 2.68064 / 2.59924 | 3.52444 / 3.12260 | 3.03273 / 1.79772 | 0.11443 / 0.10178 | 16.27350 | 1.025× [1.000, 1.050] | 0.2394 |
| SELL 64 | 7.46076 / 7.44775 | 0.78954 / 0.77836 | 2.83728 / 2.73892 | 3.73756 / 3.33076 | 3.14418 / 1.94869 | 0.11443 / 0.10142 | 16.93715 | 0.990× [0.955, 1.026] | 0.5398 |
| Custom COO matched hybrid | 7.46076 / 7.39945 | 0.78954 / 0.77190 | 1.93044 / 1.88121 | 2.31921 / 1.99383 | 1.27364 / 1.30705 | 0.11443 / 0.10150 | 14.04834 | 1.194× [1.172, 1.217] | 3.719e-06 |
| Custom CSR matched hybrid | 7.46076 / 7.42871 | 0.78954 / 0.77291 | 1.89040 / 1.82638 | 2.31921 / 2.00742 | 2.58431 / 1.55242 | 0.11443 / 0.10237 | 14.31059 | 1.171× [1.137, 1.207] | 0.000145 |
| Custom CSC matched hybrid | 7.46076 / 7.40664 | 0.78954 / 0.76468 | 1.93044 / 1.88439 | 2.31921 / 1.96611 | 2.58431 / 1.54201 | 0.11443 / 0.10010 | 14.32620 | 1.175× [1.157, 1.194] | 2.637e-06 |


## Native DGL versus custom unblocked

Each direct kernel pair uses the same named sparse traversal. The three whole-forward pairs also match the per-stage routes. Across backends the index width, kernel code, library dispatch and score-activation fusion differ. Speedup is native divided by custom.

| Dataset | Stage | Native route | Custom speedup (95% CI) | Holm p |
|---|---|---|---|---:|
| Cora | score | COO | 0.456× [0.448, 0.464] | 1.056e-11 |
| Cora | score | CSR | 0.554× [0.546, 0.562] | 1.169e-11 |
| Cora | softmax | CSC | 2.011× [1.969, 2.053] | 2.789e-11 |
| Cora | aggregation | COO | 7.036× [6.907, 7.167] | 1.779e-14 |
| Cora | aggregation | CSC | 2.662× [2.625, 2.700] | 2.455e-13 |
| Cora | forward | COO | 1.254× [1.240, 1.269] | 6.215e-09 |
| Cora | forward | CSR | 1.104× [1.093, 1.116] | 5.322e-07 |
| Cora | forward | CSC | 1.100× [1.089, 1.112] | 7.609e-07 |
| CiteSeer | score | COO | 0.564× [0.541, 0.588] | 1.305e-08 |
| CiteSeer | score | CSR | 0.666× [0.644, 0.688] | 1.499e-08 |
| CiteSeer | softmax | CSC | 2.127× [2.111, 2.144] | 2.263e-14 |
| CiteSeer | aggregation | COO | 6.959× [6.849, 7.071] | 8.178e-15 |
| CiteSeer | aggregation | CSC | 2.322× [2.169, 2.486] | 2.852e-08 |
| CiteSeer | forward | COO | 1.085× [1.062, 1.108] | 8.827e-05 |
| CiteSeer | forward | CSR | 1.037× [1.019, 1.056] | 0.001873 |
| CiteSeer | forward | CSC | 1.047× [1.040, 1.054] | 3.591e-06 |
| PubMed | score | COO | 0.367× [0.357, 0.377] | 1.795e-11 |
| PubMed | score | CSR | 0.422× [0.415, 0.428] | 3.064e-12 |
| PubMed | softmax | CSC | 1.577× [1.535, 1.621] | 1.806e-09 |
| PubMed | aggregation | COO | 5.875× [5.799, 5.952] | 4.503e-15 |
| PubMed | aggregation | CSC | 1.920× [1.767, 2.086] | 3.251e-07 |
| PubMed | forward | COO | 1.436× [1.402, 1.470] | 2.724e-08 |
| PubMed | forward | CSR | 1.142× [1.124, 1.161] | 1.19e-06 |
| PubMed | forward | CSC | 1.147× [1.127, 1.168] | 1.353e-06 |

## Matched custom storage comparisons

These comparisons share FP32 arithmetic, int32 sparse indices, fused score activation, canonical EID output, destination-global three-pass softmax, and timed output/scratch allocation. SELL uses its own traversal and an equivalent stage interface. All pair estimates are in ANALYSIS.json.

| Dataset | Endpoint | Family | Gains / losses vs custom unblocked after Holm | Fastest candidate | Paired speedup against fastest observed custom unblocked (95% CI) | Holm p |
|---|---|---|---:|---|---|---:|
| Cora | score | CSB | 0 / 18 of 18 | CSB 64×64 | 0.845× [0.835, 0.855] vs Custom CSC | 1.67e-07 |
| Cora | score | SELL | 0 / 18 of 18 | SELL 2 | 0.855× [0.850, 0.861] vs Custom CSC | 1.112e-08 |
| Cora | softmax | CSB | 0 / 16 of 18 | CSB 64×64 | 0.869× [0.856, 0.883] vs Custom COO | 4.289e-06 |
| Cora | softmax | SELL | 0 / 18 of 18 | SELL 2 | 0.755× [0.739, 0.771] vs Custom COO | 4.435e-07 |
| Cora | aggregation | CSB | 1 / 12 of 18 | CSB 64×64 | 0.856× [0.840, 0.872] vs Custom COO | 1.149e-05 |
| Cora | aggregation | SELL | 0 / 18 of 18 | SELL 4 | 0.748× [0.730, 0.767] vs Custom COO | 1.441e-06 |
| Cora | inside/score | CSB | 0 / 18 of 18 | CSB 64×64 | 0.850× [0.840, 0.859] vs Custom CSC | 1.332e-07 |
| Cora | inside/score | SELL | 0 / 18 of 18 | SELL 2 | 0.859× [0.846, 0.872] vs Custom CSC | 6.608e-07 |
| Cora | inside/softmax | CSB | 2 / 16 of 18 | CSB 64×64 | 0.871× [0.863, 0.879] vs Custom COO | 1.954e-07 |
| Cora | inside/softmax | SELL | 0 / 18 of 18 | SELL 2 | 0.742× [0.731, 0.753] vs Custom COO | 3.217e-08 |
| Cora | inside/aggregation | CSB | 1 / 12 of 18 | CSB 64×64 | 0.857× [0.848, 0.867] vs Custom COO | 4.075e-07 |
| Cora | inside/aggregation | SELL | 0 / 16 of 18 | SELL 4 | 0.748× [0.733, 0.763] vs Custom COO | 3.698e-07 |
| Cora | forward | CSB | 0 / 16 of 18 | CSB 64×64 | 0.980× [0.973, 0.986] vs Custom COO | 0.007864 |
| Cora | forward | SELL | 0 / 18 of 18 | SELL 2 | 0.962× [0.953, 0.971] vs Custom COO | 0.001702 |
| CiteSeer | score | CSB | 0 / 18 of 18 | CSB 64×64 | 0.848× [0.839, 0.857] vs Custom CSR | 1.284e-07 |
| CiteSeer | score | SELL | 0 / 18 of 18 | SELL 2 | 0.826× [0.810, 0.843] vs Custom CSR | 7.062e-07 |
| CiteSeer | softmax | CSB | 1 / 12 of 18 | CSB 64×64 | 0.811× [0.795, 0.826] vs Custom COO | 1.355e-06 |
| CiteSeer | softmax | SELL | 0 / 18 of 18 | SELL 2 | 0.681× [0.660, 0.703] vs Custom COO | 8.657e-07 |
| CiteSeer | aggregation | CSB | 6 / 12 of 18 | CSB 4×4 | 0.795× [0.775, 0.816] vs Custom COO | 9.402e-06 |
| CiteSeer | aggregation | SELL | 0 / 14 of 18 | SELL 8 | 0.566× [0.531, 0.603] vs Custom COO | 8.252e-06 |
| CiteSeer | inside/score | CSB | 0 / 18 of 18 | CSB 2×2 | 0.845× [0.836, 0.854] vs Custom CSR | 1.11e-07 |
| CiteSeer | inside/score | SELL | 0 / 18 of 18 | SELL 2 | 0.831× [0.817, 0.845] vs Custom CSR | 4.494e-07 |
| CiteSeer | inside/softmax | CSB | 1 / 16 of 18 | CSB 64×64 | 0.840× [0.831, 0.849] vs Custom COO | 1.589e-07 |
| CiteSeer | inside/softmax | SELL | 0 / 18 of 18 | SELL 2 | 0.729× [0.716, 0.743] vs Custom COO | 1.029e-07 |
| CiteSeer | inside/aggregation | CSB | 1 / 11 of 18 | CSB 64×64 | 0.826× [0.816, 0.835] vs Custom COO | 1.465e-07 |
| CiteSeer | inside/aggregation | SELL | 0 / 18 of 18 | SELL 2 | 0.736× [0.728, 0.744] vs Custom COO | 3.316e-09 |
| CiteSeer | forward | CSB | 0 / 0 of 18 | CSB 2×2 | 0.994× [0.979, 1.011] vs Custom COO | 1 |
| CiteSeer | forward | SELL | 0 / 6 of 18 | SELL 8 | 0.985× [0.969, 1.001] vs Custom COO | 1 |
| PubMed | score | CSB | 0 / 18 of 18 | CSB 2×2 | 0.786× [0.779, 0.794] vs Custom CSC | 9.609e-09 |
| PubMed | score | SELL | 0 / 18 of 18 | SELL 2 | 0.848× [0.842, 0.854] vs Custom CSC | 1.111e-08 |
| PubMed | softmax | CSB | 4 / 13 of 18 | CSB 4×4 | 0.764× [0.750, 0.778] vs Custom COO | 2.769e-07 |
| PubMed | softmax | SELL | 0 / 18 of 18 | SELL 2 | 0.590× [0.578, 0.603] vs Custom COO | 9.852e-09 |
| PubMed | aggregation | CSB | 5 / 12 of 18 | CSB 2×2 | 0.783× [0.773, 0.795] vs Custom COO | 1.248e-07 |
| PubMed | aggregation | SELL | 0 / 14 of 18 | SELL 2 | 0.457× [0.429, 0.486] vs Custom COO | 1.002e-06 |
| PubMed | inside/score | CSB | 0 / 18 of 18 | CSB 2×2 | 0.794× [0.784, 0.803] vs Custom CSC | 3.82e-08 |
| PubMed | inside/score | SELL | 0 / 18 of 18 | SELL 2 | 0.850× [0.836, 0.863] vs Custom CSC | 6.094e-07 |
| PubMed | inside/softmax | CSB | 0 / 16 of 18 | CSB 4×4 | 0.766× [0.759, 0.774] vs Custom COO | 7.599e-09 |
| PubMed | inside/softmax | SELL | 0 / 18 of 18 | SELL 2 | 0.674× [0.667, 0.682] vs Custom COO | 8.462e-10 |
| PubMed | inside/aggregation | CSB | 0 / 12 of 18 | CSB 16×16 | 0.805× [0.782, 0.829] vs Custom COO | 2.273e-05 |
| PubMed | inside/aggregation | SELL | 0 / 18 of 18 | SELL 4 | 0.796× [0.784, 0.809] vs Custom COO | 3.104e-07 |
| PubMed | forward | CSB | 0 / 14 of 18 | CSB 4×4 | 0.915× [0.899, 0.931] vs Custom COO | 0.0005612 |
| PubMed | forward | SELL | 0 / 16 of 18 | SELL 2 | 0.908× [0.881, 0.936] vs Custom COO | 0.007976 |

## Milestone 8 replication

The three formats selected here were the recorded Milestone 8 forward winners. The control in the last column is the fastest native hybrid by observed median in Milestone 9. The adjusted family covers each fixed candidate against stock and all three native hybrids. The old and new process blocks are analyzed separately.

| Dataset | Fixed prior candidate | M8 / M9 forward ms | M8 vs stock (95% CI) | M9 vs stock (95% CI), Holm p | M9 vs fastest native hybrid (95% CI), Holm p |
|---|---|---:|---|---|---|
| Cora | CSB 64×64 | 3.35344 / 3.36417 | 1.154× [1.136, 1.172] | 1.163× [1.152, 1.174], 2.171e-08 | 1.102× [1.087, 1.118], 5.886e-06 |
| CiteSeer | CSB 2×2 | 9.12580 / 9.43708 | 1.084× [1.062, 1.106] | 1.059× [1.048, 1.071], 2.919e-05 | 1.039× [1.022, 1.057], 0.001053 |
| PubMed | SELL 2 | 14.34416 / 14.58199 | 1.146× [1.123, 1.170] | 1.140× [1.110, 1.170], 3.691e-05 | 1.112× [1.081, 1.144], 0.0001243 |

The prior isolated softmax and aggregation winners are compared with the same native CSC kernel.

| Dataset | Stage | Fixed prior candidate | M8 speedup (95% CI) | M9 speedup (95% CI) | M9 Holm p |
|---|---|---|---|---|---:|
| Cora | softmax | CSB 64×64 | 1.884× [1.870, 1.898] | 2.064× [2.035, 2.093] | 1.993e-12 |
| Cora | aggregation | CSB 64×64 | 2.757× [2.711, 2.805] | 2.906× [2.852, 2.961] | 9.932e-13 |
| CiteSeer | softmax | CSB 64×64 | 2.005× [1.984, 2.027] | 2.228× [2.184, 2.273] | 7.627e-12 |
| CiteSeer | aggregation | CSB 64×64 | 2.852× [2.796, 2.909] | 2.744× [2.627, 2.865] | 3.44e-10 |
| PubMed | softmax | CSB 8×8 | 1.630× [1.607, 1.653] | 1.781× [1.729, 1.836] | 6.667e-10 |
| PubMed | aggregation | CSB 4×4 | 2.343× [2.289, 2.398] | 3.059× [2.862, 3.270] | 1.684e-09 |

## Projection discrepancy and timing overhead

Milestone 8 measured complete forwards first and the isolated calls later. Its shared projection statistic came from the latter phase. Each Milestone 9 block interleaves isolated calls, both forward modes, and context probes. The in-forward projection column is a region of the measured call, and all six regions together fit inside its enclosing per-call timer.

| Dataset | M8 isolated projection / selected forward ms | M8 rounds with isolated projection above selected forward | M9 isolated projection ms | M9 selected in-forward projection ms | M9 selected plain / instrumented forward ms |
|---|---:|---:|---:|---:|---:|
| Cora | 3.10564 / 3.35344 | 0/8 | 2.51267 | 2.49776 | 3.36417 / 3.36291 |
| CiteSeer | 9.92296 / 9.12580 | 8/8 | 8.70008 | 8.57392 | 9.43708 / 9.52939 |
| PubMed | 9.55910 / 14.34416 | 0/8 | 7.46076 | 7.28767 | 14.58199 / 14.71934 |

These probes all execute the same projection. The graph guard checks destination degrees. The sparse probe replays the three stock sparse operations on prepared tensors. The forward probe precedes projection with stock GATConv. Preceding work is outside the projection timer.

| Dataset | Isolated projection ms | After guard ms | After sparse replay ms | After stock forward ms | Timer-only six regions net µs |
|---|---:|---:|---:|---:|---:|
| Cora | 2.51267 | 2.49256 | 2.51669 | 2.49308 | 1.613 |
| CiteSeer | 8.70008 | 8.68809 | 8.59705 | 8.69371 | 1.709 |
| PubMed | 7.46076 | 7.46789 | 7.93012 | 7.44253 | 2.051 |

On CiteSeer the old isolated projection exceeded the CSB 2×2 forward by 0.797 ms. Under interleaving the isolated projection was 1.223 ms lower than that old projection, and the in-forward region was 8.574 ms inside a 9.529 ms instrumented call. The net empty timer cost was 1.71 µs. Median observed instrumentation overhead was +18.6 µs for that layout. These measurements quantify the scale difference between clock overhead and the historical phase discrepancy.

| Dataset | Implementation | Instrumented minus plain forward, paired median ms | 95% CI for mean of process median differences, ms |
|---|---|---:|---:|
| Cora | Imported GATConv | -0.01479 | [-0.03137, +0.00337] |
| Cora | CSB 64×64 | +0.00531 | [-0.01681, +0.02847] |
| CiteSeer | Imported GATConv | +0.04901 | [-0.01287, +0.18264] |
| CiteSeer | CSB 2×2 | +0.01859 | [-0.10520, +0.12574] |
| PubMed | Imported GATConv | +0.05297 | [-0.45249, +0.42767] |
| PubMed | SELL 2 | +0.03495 | [-0.27643, +0.30146] |

| Dataset | Probe | Isolated projection / probe projection (95% CI) | Holm p |
|---|---|---|---:|
| Cora | guard | 1.006× [0.997, 1.016] | 1 |
| Cora | sparse | 0.998× [0.988, 1.007] | 1 |
| Cora | forward | 1.008× [1.000, 1.015] | 0.4075 |
| CiteSeer | guard | 1.007× [0.994, 1.021] | 1 |
| CiteSeer | sparse | 1.015× [1.001, 1.029] | 0.3445 |
| CiteSeer | forward | 1.002× [0.986, 1.017] | 1 |
| PubMed | guard | 1.004× [0.990, 1.018] | 1 |
| PubMed | sparse | 0.940× [0.921, 0.960] | 0.001886 |
| PubMed | forward | 1.003× [0.983, 1.024] | 1 |

| Dataset | Fixed prior candidate | Stage | Isolated / in-forward (95% CI) | Holm p |
|---|---|---|---|---:|
| Cora | CSB 64×64 | projection | 1.004× [0.994, 1.015] | 1 |
| Cora | CSB 64×64 | terms | 0.962× [0.947, 0.977] | 0.003987 |
| Cora | CSB 64×64 | score | 0.986× [0.980, 0.992] | 0.005135 |
| Cora | CSB 64×64 | softmax | 1.017× [1.003, 1.031] | 0.102 |
| Cora | CSB 64×64 | aggregation | 1.038× [1.028, 1.049] | 0.0001682 |
| Cora | CSB 64×64 | output | 1.034× [0.960, 1.115] | 0.4673 |
| CiteSeer | CSB 2×2 | projection | 1.019× [1.012, 1.027] | 0.004529 |
| CiteSeer | CSB 2×2 | terms | 0.971× [0.951, 0.992] | 0.05157 |
| CiteSeer | CSB 2×2 | score | 1.004× [0.991, 1.016] | 0.5252 |
| CiteSeer | CSB 2×2 | softmax | 0.994× [0.980, 1.008] | 0.6484 |
| CiteSeer | CSB 2×2 | aggregation | 1.024× [1.011, 1.037] | 0.008277 |
| CiteSeer | CSB 2×2 | output | 0.921× [0.883, 0.961] | 0.009737 |
| PubMed | SELL 2 | projection | 1.019× [0.993, 1.046] | 0.9163 |
| PubMed | SELL 2 | terms | 1.008× [0.972, 1.045] | 0.6194 |
| PubMed | SELL 2 | score | 1.033× [1.023, 1.042] | 0.0004794 |
| PubMed | SELL 2 | softmax | 1.161× [1.132, 1.190] | 1.797e-05 |
| PubMed | SELL 2 | aggregation | 1.702× [1.584, 1.830] | 4.548e-06 |
| PubMed | SELL 2 | output | 1.056× [0.956, 1.167] | 0.4673 |

Empty-call subtraction is used only for the timer microbenchmark. Forward readings are retained as measured. Actual instrumented-minus-plain differences include runtime variability, which can give either sign. Sparse isolation uses already materialized intermediates, while a forward consumes freshly produced tensors and includes graph bookkeeping and allocations. The controlled probes change cache state and allocator history together. They establish context sensitivity without assigning a fraction to either cause. The raw data do not include hardware counters or a reconstruction of the earlier process state.

Per-configuration instrumented totals, unassigned time, paired overhead and round variability are in ANALYSIS.json.

## Unsupported native paths and construction

The [DGL 1.1.3 CPU dispatch](https://raw.githubusercontent.com/dmlc/dgl/v1.1.3/src/array/kernel.cc) supports COO/CSR GSDDMM, CSC destination softmax and COO/CSC GSpMM. COO uses CSC softmax. CSR uses CSC softmax and aggregation. CSC uses COO scoring. The matching custom hybrids follow these same routes using Milestone 6 kernels. Canonical EIDs carry edge tensors directly between stages.

Index payload counts retained sparse arrays only. Construction starts from canonical endpoints for custom layouts and a fresh COO graph for native views. Hybrid rows share views and the table lists the constituent formats separately.

| Dataset | Storage | KiB | Build ms |
|---|---|---:|---:|
| Cora | Native COO hybrid | 207.2 | 0.02473 |
| Cora | Native CSR hybrid | 228.4 | 0.05557 |
| Cora | Native CSC hybrid | 228.4 | 0.05442 |
| Cora | Custom COO | 103.6 | 0.03951 |
| Cora | Custom CSR | 114.2 | 0.11644 |
| Cora | Custom CSC | 114.2 | 0.49357 |
| Cora | CSB 2×2 | 285.0 | 1.21838 |
| Cora | CSB 4×4 | 269.9 | 1.18767 |
| Cora | CSB 8×8 | 258.3 | 1.17463 |
| Cora | CSB 16×16 | 242.5 | 1.07652 |
| Cora | CSB 32×32 | 212.2 | 0.98944 |
| Cora | CSB 64×64 | 176.0 | 0.89238 |
| Cora | SELL 2 | 140.6 | 0.67297 |
| Cora | SELL 4 | 179.6 | 0.66816 |
| Cora | SELL 8 | 235.6 | 0.65951 |
| Cora | SELL 16 | 316.2 | 0.66811 |
| Cora | SELL 32 | 437.8 | 0.68125 |
| Cora | SELL 64 | 589.7 | 0.65804 |
| CiteSeer | Native COO hybrid | 194.2 | 0.02500 |
| CiteSeer | Native CSR hybrid | 220.2 | 0.05250 |
| CiteSeer | Native CSC hybrid | 220.2 | 0.05306 |
| CiteSeer | Custom COO | 97.1 | 0.03859 |
| CiteSeer | Custom CSR | 110.1 | 0.11222 |
| CiteSeer | Custom CSC | 110.1 | 0.45013 |
| CiteSeer | CSB 2×2 | 266.8 | 1.11828 |
| CiteSeer | CSB 4×4 | 253.6 | 1.13734 |
| CiteSeer | CSB 8×8 | 245.5 | 1.12042 |
| CiteSeer | CSB 16×16 | 234.8 | 1.03031 |
| CiteSeer | CSB 32×32 | 211.8 | 0.96771 |
| CiteSeer | CSB 64×64 | 175.1 | 0.88072 |
| CiteSeer | SELL 2 | 133.8 | 0.62204 |
| CiteSeer | SELL 4 | 172.7 | 0.61192 |
| CiteSeer | SELL 8 | 224.3 | 0.62238 |
| CiteSeer | SELL 16 | 289.3 | 0.60985 |
| CiteSeer | SELL 32 | 371.2 | 0.61168 |
| CiteSeer | SELL 64 | 485.7 | 0.62059 |
| PubMed | Native COO hybrid | 1693.2 | 0.02709 |
| PubMed | Native CSR hybrid | 1847.2 | 0.26407 |
| PubMed | Native CSC hybrid | 1847.2 | 0.31260 |
| PubMed | Custom COO | 846.6 | 0.19194 |
| PubMed | Custom CSR | 923.6 | 0.68584 |
| PubMed | Custom CSC | 923.6 | 4.76866 |
| PubMed | CSB 2×2 | 2419.9 | 11.39144 |
| PubMed | CSB 4×4 | 2354.7 | 11.54855 |
| PubMed | CSB 8×8 | 2314.1 | 11.33723 |
| PubMed | CSB 16×16 | 2268.1 | 10.64115 |
| PubMed | CSB 32×32 | 2163.8 | 10.26663 |
| PubMed | CSB 64×64 | 1911.1 | 9.72867 |
| PubMed | SELL 2 | 1290.1 | 6.02979 |
| PubMed | SELL 4 | 1877.1 | 5.97098 |
| PubMed | SELL 8 | 2668.7 | 5.98278 |
| PubMed | SELL 16 | 3658.4 | 5.99808 |
| PubMed | SELL 32 | 4772.9 | 6.07279 |
| PubMed | SELL 64 | 6077.2 | 6.15010 |

## Methods and correctness

AMD Ryzen 7 7800X3D 8-Core Processor, logical CPU 0, one thread. DGL 1.1.3, Torch 2.2.2+cpu, FP32, four heads, eight channels per head, seed-0 weights. Original node and edge ordering, existing self-loop preprocessing, inference mode and dropout disabled.

8 fresh processes per dataset, 10 warmup and 30 measured randomized blocks per process, and 5 separately timed builds per format. Intervals on speedups are pointwise 95% Student-t over paired process log ratios. Holm families are separated by comparison family and endpoint across all datasets. All blocks are retained. The report replays the randomization from the seeds.

The log contains 15,840 measured plain forwards and 15,840 instrumented forwards. Each instrumented call records all six regions as well as its enclosing timer.

The setup verifies canonical endpoints, feature bytes and weight fingerprints against the original workload. All pipeline intermediates, destination softmax sums and final output pass stock-backed checks. The timing copies produce bitwise-identical output to their plain calls at validation. Maximum final-output absolute error is 1.12e-08.

[Raw samples](RESULTS.json) · [Round summaries, paired contrasts and storage](ANALYSIS.json)
