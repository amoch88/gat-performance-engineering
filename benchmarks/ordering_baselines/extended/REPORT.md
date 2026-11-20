# Milestone 10 Extension — Seven node orderings and seven edge orderings

Every forward uses the imported DGL 1.1.3 GATConv. The six stage columns come from a validated execution-equivalent copy of its forward. Total forward is timed directly on the unmodified import. All times are milliseconds and use the median of process medians.

The node sweep relabels nodes and feature rows while preserving every canonical EID. The edge sweep preserves node labels and feature rows. Both re-create the same original topology and order as their within-sweep control. Every node-ordering case allocates its feature rows with the same indexing operation, including the identity control. Every edge-ordering case shares the loader's unchanged feature tensor. Every graph starts from the same DGL constructor and materializes all COO, CSR, and CSC views before timing.

Use the original row within the same sweep as the control. The two sweeps have different case sets and feature working sets, so their absolute medians need not coincide.

## Cora

### Node ordering

| Ordering | Projection | Attention Terms (L/R) | Attention Scores | Edge Softmax | Aggregation | Output | Total Forward |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 2.943264 | 0.120575 | 0.188979 | 0.424287 | 0.454190 | 0.022082 | 4.250858 |
| RCMK | 2.893993 | 0.119635 | 0.187459 | 0.415547 | 0.443953 | 0.021555 | 4.209413 |
| Degree ascending | 2.882726 | 0.120168 | 0.190832 | 0.413168 | 0.453527 | 0.022246 | 4.189194 |
| Degree descending | 2.835369 | 0.119788 | 0.187753 | 0.399216 | 0.447756 | 0.022918 | 4.097715 |
| BFS | 2.839133 | 0.119571 | 0.190364 | 0.430874 | 0.442977 | 0.021926 | 4.138363 |
| Random control | 2.798762 | 0.119775 | 0.188056 | 0.429083 | 0.470944 | 0.022535 | 4.172116 |
| METIS (32 partitions) | 2.822794 | 0.119855 | 0.189211 | 0.421314 | 0.448350 | 0.021750 | 4.170966 |

### Edge ordering

| Ordering | Projection | Attention Terms (L/R) | Attention Scores | Edge Softmax | Aggregation | Output | Total Forward |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 2.500210 | 0.118805 | 0.185168 | 0.426156 | 0.454287 | 0.021074 | 3.797769 |
| Source-grouped | 2.472265 | 0.118927 | 0.185479 | 0.414482 | 0.458309 | 0.022650 | 3.816808 |
| Destination-grouped | 2.493440 | 0.118681 | 0.185321 | 0.408015 | 0.439680 | 0.021838 | 3.755902 |
| Random control | 2.488850 | 0.118972 | 0.186505 | 0.427323 | 0.471983 | 0.022815 | 3.856520 |
| Source then destination | 2.467577 | 0.118695 | 0.184971 | 0.422460 | 0.452201 | 0.021199 | 3.824627 |
| Destination then source | 2.487117 | 0.119325 | 0.185801 | 0.413194 | 0.438904 | 0.021540 | 3.769888 |
| Morton (Z-order) | 2.471306 | 0.118597 | 0.183976 | 0.413790 | 0.450434 | 0.022005 | 3.810904 |

## CiteSeer

### Node ordering

| Ordering | Projection | Attention Terms (L/R) | Attention Scores | Edge Softmax | Aggregation | Output | Total Forward |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 9.198072 | 0.154006 | 0.214871 | 0.439874 | 0.452165 | 0.028468 | 10.743577 |
| RCMK | 9.164506 | 0.154450 | 0.226835 | 0.433439 | 0.439678 | 0.028771 | 10.596366 |
| Degree ascending | 9.092771 | 0.154416 | 0.229658 | 0.415098 | 0.443550 | 0.029300 | 10.519621 |
| Degree descending | 9.046978 | 0.154855 | 0.224977 | 0.416608 | 0.446856 | 0.029234 | 10.533286 |
| BFS | 9.074868 | 0.153591 | 0.225992 | 0.430588 | 0.439020 | 0.028821 | 10.580265 |
| Random control | 9.088390 | 0.155353 | 0.230540 | 0.440333 | 0.467311 | 0.028879 | 10.526382 |
| METIS (32 partitions) | 9.006491 | 0.154119 | 0.230974 | 0.447838 | 0.452187 | 0.028933 | 10.588574 |

### Edge ordering

| Ordering | Projection | Attention Terms (L/R) | Attention Scores | Edge Softmax | Aggregation | Output | Total Forward |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 8.384026 | 0.146568 | 0.198571 | 0.439317 | 0.451289 | 0.026072 | 9.839732 |
| Source-grouped | 8.396131 | 0.147364 | 0.200333 | 0.429538 | 0.450754 | 0.026941 | 9.781487 |
| Destination-grouped | 8.346667 | 0.148294 | 0.201312 | 0.424313 | 0.432956 | 0.026432 | 9.792291 |
| Random control | 8.368075 | 0.147498 | 0.206946 | 0.438335 | 0.462272 | 0.026667 | 9.795352 |
| Source then destination | 8.418578 | 0.150808 | 0.205207 | 0.439638 | 0.453459 | 0.027416 | 9.820382 |
| Destination then source | 8.354883 | 0.146819 | 0.201397 | 0.419487 | 0.435821 | 0.026264 | 9.877807 |
| Morton (Z-order) | 8.433666 | 0.147308 | 0.201540 | 0.431483 | 0.441599 | 0.026722 | 9.789721 |

## PubMed

### Node ordering

| Ordering | Projection | Attention Terms (L/R) | Attention Scores | Edge Softmax | Aggregation | Output | Total Forward |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 8.208197 | 0.753551 | 0.678694 | 3.171391 | 3.660746 | 0.092312 | 17.162624 |
| RCMK | 8.321551 | 0.748976 | 0.695177 | 3.150866 | 3.554290 | 0.094238 | 16.907439 |
| Degree ascending | 8.226351 | 0.749789 | 0.680933 | 3.049700 | 3.632453 | 0.093552 | 16.931210 |
| Degree descending | 8.190065 | 0.744281 | 0.690086 | 3.034573 | 3.626122 | 0.093978 | 16.968679 |
| BFS | 8.215890 | 0.745300 | 0.694330 | 3.219295 | 3.516099 | 0.094965 | 17.043900 |
| Random control | 8.384114 | 0.751190 | 0.712907 | 3.226539 | 3.888941 | 0.092534 | 17.506921 |
| METIS (32 partitions) | 8.368842 | 0.756973 | 0.713303 | 3.189017 | 3.411170 | 0.093529 | 16.908630 |

### Edge ordering

| Ordering | Projection | Attention Terms (L/R) | Attention Scores | Edge Softmax | Aggregation | Output | Total Forward |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 7.217806 | 0.748335 | 0.669305 | 3.228622 | 3.617854 | 0.102714 | 15.864936 |
| Source-grouped | 7.196795 | 0.741977 | 0.673964 | 3.251072 | 3.729709 | 0.101963 | 15.962032 |
| Destination-grouped | 7.381868 | 0.749281 | 0.677699 | 2.917505 | 3.338457 | 0.102652 | 15.428297 |
| Random control | 7.173931 | 0.745781 | 0.717397 | 3.298609 | 3.915346 | 0.102555 | 16.249561 |
| Source then destination | 7.222667 | 0.746856 | 0.684561 | 3.182733 | 3.638092 | 0.094249 | 15.930172 |
| Destination then source | 7.240553 | 0.746463 | 0.675924 | 2.930454 | 3.351301 | 0.099659 | 15.269934 |
| Morton (Z-order) | 7.058097 | 0.745699 | 0.678654 | 3.055937 | 3.428017 | 0.102612 | 15.558688 |

## Forward comparisons against original

Speedup divides the original time by the candidate time. Effects use the geometric mean of 8 paired process-median ratios with pointwise 95% Student-t intervals in log space. Holm correction is separate for the node and edge sweeps across all datasets and planned policies, including random controls. The rules apply separately to each profiled stage. The tuning criterion requires adjusted p < 0.05 and a 95% CI lower speedup bound above 1.01×.

| Dataset | Sweep | Policy | Median forward ms | Paired speedup (95% CI) | Holm p | Supported |
|---|---|---|---:|---|---:|---|
| Cora | node | Original | 4.250858 | 1.0000× (control) | — | — |
| Cora | node | RCMK | 4.209413 | 1.0095× [1.0020, 1.0171] | 0.225 | Inconclusive |
| Cora | node | Degree ascending | 4.189194 | 1.0181× [1.0025, 1.0339] | 0.2263 | Inconclusive |
| Cora | node | Degree descending | 4.097715 | 1.0426× [1.0191, 1.0667] | 0.05902 | Inconclusive |
| Cora | node | BFS | 4.138363 | 1.0251× [1.0006, 1.0501] | 0.2263 | Inconclusive |
| Cora | node | Random control | 4.172116 | 1.0245× [1.0047, 1.0447] | 0.225 | Inconclusive |
| Cora | node | METIS (32 partitions) | 4.170966 | 1.0292× [1.0078, 1.0510] | 0.2002 | Inconclusive |
| Cora | edge | Original | 3.797769 | 1.0000× (control) | — | — |
| Cora | edge | Source-grouped | 3.816808 | 0.9981× [0.9925, 1.0038] | 1 | Inconclusive |
| Cora | edge | Destination-grouped | 3.755902 | 1.0080× [0.9993, 1.0169] | 0.8712 | Inconclusive |
| Cora | edge | Random control | 3.856520 | 0.9876× [0.9793, 0.9959] | 0.1366 | Inconclusive |
| Cora | edge | Source then destination | 3.824627 | 0.9939× [0.9819, 1.0061] | 1 | Inconclusive |
| Cora | edge | Destination then source | 3.769888 | 1.0053× [0.9923, 1.0184] | 1 | Inconclusive |
| Cora | edge | Morton (Z-order) | 3.810904 | 1.0016× [0.9808, 1.0229] | 1 | Inconclusive |
| CiteSeer | node | Original | 10.743577 | 1.0000× (control) | — | — |
| CiteSeer | node | RCMK | 10.596366 | 1.0083× [0.9955, 1.0212] | 0.2263 | Inconclusive |
| CiteSeer | node | Degree ascending | 10.519621 | 1.0169× [1.0032, 1.0309] | 0.225 | Inconclusive |
| CiteSeer | node | Degree descending | 10.533286 | 1.0153× [1.0067, 1.0241] | 0.06421 | Inconclusive |
| CiteSeer | node | BFS | 10.580265 | 1.0134× [1.0019, 1.0251] | 0.2263 | Inconclusive |
| CiteSeer | node | Random control | 10.526382 | 1.0135× [1.0009, 1.0262] | 0.2263 | Inconclusive |
| CiteSeer | node | METIS (32 partitions) | 10.588574 | 1.0106× [1.0012, 1.0200] | 0.2263 | Inconclusive |
| CiteSeer | edge | Original | 9.839732 | 1.0000× (control) | — | — |
| CiteSeer | edge | Source-grouped | 9.781487 | 1.0035× [0.9903, 1.0169] | 1 | Inconclusive |
| CiteSeer | edge | Destination-grouped | 9.792291 | 1.0045× [0.9806, 1.0291] | 1 | Inconclusive |
| CiteSeer | edge | Random control | 9.795352 | 0.9992× [0.9845, 1.0140] | 1 | Inconclusive |
| CiteSeer | edge | Source then destination | 9.820382 | 1.0065× [0.9901, 1.0231] | 1 | Inconclusive |
| CiteSeer | edge | Destination then source | 9.877807 | 0.9977× [0.9818, 1.0140] | 1 | Inconclusive |
| CiteSeer | edge | Morton (Z-order) | 9.789721 | 1.0039× [0.9880, 1.0201] | 1 | Inconclusive |
| PubMed | node | Original | 17.162624 | 1.0000× (control) | — | — |
| PubMed | node | RCMK | 16.907439 | 1.0107× [1.0024, 1.0191] | 0.225 | Inconclusive |
| PubMed | node | Degree ascending | 16.931210 | 1.0142× [1.0051, 1.0235] | 0.1167 | Inconclusive |
| PubMed | node | Degree descending | 16.968679 | 1.0138× [1.0033, 1.0245] | 0.2232 | Inconclusive |
| PubMed | node | BFS | 17.043900 | 1.0086× [0.9991, 1.0182] | 0.2263 | Inconclusive |
| PubMed | node | Random control | 17.506921 | 0.9785× [0.9720, 0.9851] | 0.002216 | Loss |
| PubMed | node | METIS (32 partitions) | 16.908630 | 1.0111× [0.9988, 1.0235] | 0.2263 | Inconclusive |
| PubMed | edge | Original | 15.864936 | 1.0000× (control) | — | — |
| PubMed | edge | Source-grouped | 15.962032 | 0.9981× [0.9862, 1.0101] | 1 | Inconclusive |
| PubMed | edge | Destination-grouped | 15.428297 | 1.0280× [1.0152, 1.0410] | 0.02012 | Gain and meets tuning criterion |
| PubMed | edge | Random control | 16.249561 | 0.9719× [0.9619, 0.9821] | 0.005798 | Loss |
| PubMed | edge | Source then destination | 15.930172 | 0.9994× [0.9936, 1.0053] | 1 | Inconclusive |
| PubMed | edge | Destination then source | 15.269934 | 1.0365× [1.0262, 1.0468] | 0.001083 | Gain and meets tuning criterion |
| PubMed | edge | Morton (Z-order) | 15.558688 | 1.0202× [1.0075, 1.0330] | 0.1048 | Inconclusive |

## Conclusions

### Node ordering

- **Cora**. Degree descending had the lowest observed median (4.0977 ms). The predetermined tuning criterion was unmet.
- **CiteSeer**. Degree ascending had the lowest observed median (10.5196 ms). The predetermined tuning criterion was unmet.
- **PubMed**. RCMK had the lowest observed median (16.9074 ms). The predetermined tuning criterion was unmet.

Keep Original as the default for the node sweep. Omit the tested alternatives from the initial tuning space under this criterion. Keep the measurements and seeded random controls for reference.

### Edge ordering

- **Cora**. Destination-grouped had the lowest observed median (3.7559 ms). The predetermined tuning criterion was unmet.
- **CiteSeer**. Source-grouped had the lowest observed median (9.7815 ms). The predetermined tuning criterion was unmet.
- **PubMed**. Destination then source had the lowest observed median (15.2699 ms). Destination-grouped, Destination then source met the tuning criterion.

Retain edge ordering as an independent candidate axis on the tested workloads. Keep Original and Destination-grouped, Destination then source. Keep the seeded random runs as diagnostic controls.
Omit Source-grouped, Source then destination, Morton (Z-order) from the initial tuning space under this criterion.

## Stage effects

These counts summarize the same paired comparisons for individual profiled stages. Random controls are tabulated separately in the analysis JSON. Stage timings describe the instrumented execution and the full-forward comparison determines the tuning decision.

| Sweep | Stage | Supported policy-dataset gains | Supported losses |
|---|---|---:|---:|
| node | projection | 0/15 | 0/15 |
| node | terms | 0/15 | 0/15 |
| node | score | 0/15 | 1/15 |
| node | softmax | 4/15 | 0/15 |
| node | aggregation | 2/15 | 0/15 |
| node | output | 0/15 | 0/15 |
| edge | projection | 0/15 | 0/15 |
| edge | terms | 0/15 | 0/15 |
| edge | score | 0/15 | 0/15 |
| edge | softmax | 3/15 | 0/15 |
| edge | aggregation | 6/15 | 1/15 |
| edge | output | 0/15 | 0/15 |

The table below lists corrected stage differences for deterministic policies.

| Dataset | Sweep | Policy | Stage | Paired speedup (95% CI) | Holm p |
|---|---|---|---|---|---:|
| Cora | edge | Destination then source | aggregation | 1.0380× [1.0219, 1.0545] | 0.008678 |
| CiteSeer | node | Degree ascending | softmax | 1.0559× [1.0286, 1.0838] | 0.02411 |
| CiteSeer | node | Degree descending | softmax | 1.0560× [1.0383, 1.0740] | 0.002218 |
| CiteSeer | node | METIS (32 partitions) | score | 0.9449× [0.9196, 0.9710] | 0.02727 |
| CiteSeer | edge | Destination-grouped | aggregation | 1.0438× [1.0198, 1.0684] | 0.03341 |
| CiteSeer | edge | Destination then source | aggregation | 1.0355× [1.0236, 1.0474] | 0.002553 |
| PubMed | node | Degree ascending | softmax | 1.0410× [1.0254, 1.0568] | 0.006447 |
| PubMed | node | Degree descending | softmax | 1.0423× [1.0227, 1.0623] | 0.01953 |
| PubMed | node | BFS | aggregation | 1.0464× [1.0319, 1.0612] | 0.00191 |
| PubMed | node | METIS (32 partitions) | aggregation | 1.0767× [1.0632, 1.0903] | 4.152e-05 |
| PubMed | edge | Source-grouped | aggregation | 0.9792× [0.9712, 0.9872] | 0.006109 |
| PubMed | edge | Destination-grouped | softmax | 1.1154× [1.0914, 1.1399] | 0.0001154 |
| PubMed | edge | Destination-grouped | aggregation | 1.0906× [1.0745, 1.1069] | 4.487e-05 |
| PubMed | edge | Destination then source | softmax | 1.1084× [1.0902, 1.1269] | 2.865e-05 |
| PubMed | edge | Destination then source | aggregation | 1.0812× [1.0664, 1.0962] | 4.885e-05 |
| PubMed | edge | Morton (Z-order) | softmax | 1.0622× [1.0401, 1.0847] | 0.004029 |
| PubMed | edge | Morton (Z-order) | aggregation | 1.0536× [1.0326, 1.0750] | 0.006109 |

## New edge policies versus Destination-grouped

This secondary forward-only family uses destination-grouped as the control. Holm correction covers the three new edge policies across all datasets. The same practical bound applies here. The primary retention rule compares with Original. An inconclusive difference does not establish equivalence.

| Dataset | Candidate | Destination-grouped ms | Candidate ms | Paired speedup (95% CI) | Holm p | Supported |
|---|---|---:|---:|---|---:|---|
| Cora | Source then destination | 3.755902 | 3.824627 | 0.9860× [0.9737, 0.9984] | 0.2592 | Inconclusive |
| Cora | Destination then source | 3.755902 | 3.769888 | 0.9973× [0.9828, 1.0119] | 1 | Inconclusive |
| Cora | Morton (Z-order) | 3.755902 | 3.810904 | 0.9936× [0.9734, 1.0143] | 1 | Inconclusive |
| CiteSeer | Source then destination | 9.792291 | 9.820382 | 1.0019× [0.9865, 1.0176] | 1 | Inconclusive |
| CiteSeer | Destination then source | 9.792291 | 9.877807 | 0.9932× [0.9827, 1.0039] | 1 | Inconclusive |
| CiteSeer | Morton (Z-order) | 9.792291 | 9.789721 | 0.9994× [0.9763, 1.0230] | 1 | Inconclusive |
| PubMed | Source then destination | 15.428297 | 15.930172 | 0.9722× [0.9604, 0.9842] | 0.008595 | Loss |
| PubMed | Destination then source | 15.428297 | 15.269934 | 1.0082× [0.9933, 1.0234] | 1 | Inconclusive |
| PubMed | Morton (Z-order) | 15.428297 | 15.558688 | 0.9924× [0.9781, 1.0069] | 1 | Inconclusive |

No new edge policy met the practical gain criterion against Destination-grouped.

## What the added policies contribute

- **METIS (32 partitions)** versus Original. Cora (inconclusive), CiteSeer (inconclusive), PubMed (inconclusive).
- **Source then destination** versus Original. Cora (inconclusive), CiteSeer (inconclusive), PubMed (inconclusive).
- **Destination then source** versus Original. Cora (inconclusive), CiteSeer (inconclusive), PubMed (qualifies).
- **Morton (Z-order)** versus Original. Cora (inconclusive), CiteSeer (inconclusive), PubMed (inconclusive).

The stage comparisons use separate Holm families. A significant projection, attention-score, softmax, or aggregation change alone does not qualify a policy. Rankings refer to these three workloads and one CPU, not to every graph with similar statistics.

## Random permutation variability

The main row uses the median of the three seed-specific medians within each process, followed by the median across processes. Its paired interval instead averages the three seed log ratios within each process. The interval treats the seeds as fixed and describes process variability. The variation across the three seed estimates is descriptive.

| Dataset | Sweep | Seed | Forward ms | SD across process medians, ms | Seed-specific paired speedup |
|---|---|---:|---:|---:|---:|
| Cora | node | 2026122000 | 4.170376 | 0.126580 | 1.0220× |
| Cora | node | 2026122001 | 4.160525 | 0.084046 | 1.0258× |
| Cora | node | 2026122002 | 4.155112 | 0.112120 | 1.0258× |
| Cora | edge | 2026122010 | 3.875075 | 0.037076 | 0.9825× |
| Cora | edge | 2026122011 | 3.813925 | 0.040348 | 0.9939× |
| Cora | edge | 2026122012 | 3.855701 | 0.070561 | 0.9864× |
| CiteSeer | node | 2026122100 | 10.538370 | 0.121117 | 1.0159× |
| CiteSeer | node | 2026122101 | 10.571510 | 0.197004 | 1.0115× |
| CiteSeer | node | 2026122102 | 10.584192 | 0.121714 | 1.0130× |
| CiteSeer | edge | 2026122110 | 9.842080 | 0.134357 | 0.9994× |
| CiteSeer | edge | 2026122111 | 9.868889 | 0.104292 | 0.9977× |
| CiteSeer | edge | 2026122112 | 9.801652 | 0.162208 | 1.0004× |
| PubMed | node | 2026122200 | 17.613497 | 0.221649 | 0.9763× |
| PubMed | node | 2026122201 | 17.439934 | 0.360155 | 0.9830× |
| PubMed | node | 2026122202 | 17.569915 | 0.363895 | 0.9764× |
| PubMed | edge | 2026122210 | 16.285424 | 0.545776 | 0.9748× |
| PubMed | edge | 2026122211 | 16.382903 | 0.584021 | 0.9669× |
| PubMed | edge | 2026122212 | 16.242130 | 0.518505 | 0.9741× |

## Instrumentation overhead

Each block interleaves direct forwards, profiled forwards and timer calibration. Overhead is the paired instrumented-minus-plain interval for the original case. The point estimate averages process medians of the paired differences. The empty six-region calibration measures timestamp reads, loop and record writes. Full per-policy overhead and unassigned graph-scope costs are in ANALYSIS.json.

| Dataset | Sweep | Empty six regions net µs (median) | Original overhead µs (95% CI) |
|---|---|---:|---|
| Cora | node | 1.550 | +68.048 [-7.128, +143.224] |
| Cora | edge | 1.418 | +38.507 [-4.590, +81.604] |
| CiteSeer | node | 1.708 | -9.194 [-145.121, +126.733] |
| CiteSeer | edge | 1.547 | -34.439 [-146.560, +77.681] |
| PubMed | node | 1.749 | -110.993 [-220.308, -1.678] |
| PubMed | edge | 1.631 | +157.660 [-31.312, +346.631] |

## One-time costs and physical representation

Rebuild includes endpoint remapping, DGL graph creation and any data permutation. Materialization times the subsequent create_formats_ call. The original control also goes through this constructor. The node identity control includes a feature copy solely for measurement parity. Deployment can keep the original features without that copy. Random rows show seed medians.

Bandwidth is max |src − dst| in current labels. Endpoint step is the mean absolute ID difference between consecutive COO edges. Compressed-view hashes and locality statistics are recorded in ANALYSIS.json. The change flags compare actual arrays against the original within the sweep. I means stored adjacency indices changed, E means the stored traversal-to-current-EID map changed, and L means traversal in original EIDs changed. Random physical flags cover each of the three seeds in order.

| Dataset | Sweep | Ordering | Bandwidth | COO src / dst step | COO / CSR / CSC changes | Permutation ms | Rebuild ms | Materialization ms |
|---|---|---|---:|---|---|---:|---:|---:|
| Cora | node | Original | 2657 | 0.61 / 551.74 | — / — / — | 0.012205 | 1.919515 | 0.062450 |
| Cora | node | RCMK | 690 | 309.64 / 520.38 | I / IEL / IEL | 0.688209 | 2.088972 | 0.065422 |
| Cora | node | Degree ascending | 2701 | 340.81 / 740.88 | I / IEL / IEL | 0.091123 | 1.992839 | 0.064569 |
| Cora | node | Degree descending | 2638 | 329.76 / 731.17 | I / IEL / IEL | 0.095288 | 1.983990 | 0.063218 |
| Cora | node | BFS | 745 | 299.01 / 534.63 | I / IEL / IEL | 3.295261 | 2.134222 | 0.065828 |
| Cora | node | Random control | 2671 | 373.64 / 902.22 | I,I,I / IEL,IEL,IEL / IEL,IEL,IEL | 0.068551 | 2.066170 | 0.067328 |
| Cora | node | METIS (32 partitions) | 2562 | 306.75 / 461.73 | I / IEL / IEL | 0.009910 | 1.980347 | 0.062725 |
| Cora | edge | Original | 2657 | 0.61 / 551.74 | — / — / — | 0.012149 | 0.189343 | 0.059837 |
| Cora | edge | Source-grouped | 2657 | 0.20 / 681.96 | IL / E / IEL | 0.049586 | 0.184518 | 0.059202 |
| Cora | edge | Destination-grouped | 2657 | 616.20 / 0.20 | IL / IEL / E | 0.414584 | 0.199040 | 0.061909 |
| Cora | edge | Random control | 2657 | 867.57 / 874.75 | IL,IL,IL / IEL,IEL,IEL / IEL,IEL,IEL | 0.142488 | 0.186142 | 0.064462 |
| Cora | edge | Source then destination | 2657 | 0.20 / 616.15 | IL / IEL / IEL | 0.844286 | 0.199311 | 0.059431 |
| Cora | edge | Destination then source | 2657 | 616.15 / 0.20 | IL / IEL / IEL | 0.482942 | 0.206341 | 0.059941 |
| Cora | edge | Morton (Z-order) | 2657 | 19.47 / 14.16 | IL / IEL / IEL | 0.472125 | 0.193709 | 0.061764 |
| CiteSeer | node | Original | 3293 | 0.80 / 669.73 | — / — / — | 0.016314 | 21.225576 | 0.074013 |
| CiteSeer | node | RCMK | 390 | 525.88 / 617.21 | I / IEL / IEL | 0.797837 | 21.579307 | 0.088566 |
| CiteSeer | node | Degree ascending | 3250 | 521.14 / 852.83 | I / IEL / IEL | 0.114115 | 21.400166 | 0.082717 |
| CiteSeer | node | Degree descending | 3249 | 507.27 / 842.29 | I / IEL / IEL | 0.113361 | 21.268043 | 0.083768 |
| CiteSeer | node | BFS | 313 | 526.12 / 613.24 | I / IEL / IEL | 3.629681 | 21.644782 | 0.091756 |
| CiteSeer | node | Random control | 3266 | 591.33 / 1113.94 | I,I,I / IEL,IEL,IEL / IEL,IEL,IEL | 0.101139 | 21.586332 | 0.098673 |
| CiteSeer | node | METIS (32 partitions) | 3081 | 531.97 / 632.67 | I / IEL / IEL | 0.010596 | 21.295174 | 0.097940 |
| CiteSeer | edge | Original | 3293 | 0.80 / 669.73 | — / — / — | 0.019578 | 0.231036 | 0.070246 |
| CiteSeer | edge | Source-grouped | 3293 | 0.27 / 925.43 | IL / E / IEL | 0.061894 | 0.225081 | 0.065052 |
| CiteSeer | edge | Destination-grouped | 3293 | 861.68 / 0.27 | IL / IEL / E | 0.378835 | 0.217709 | 0.062877 |
| CiteSeer | edge | Random control | 3293 | 1095.16 / 1090.23 | IL,IL,IL / IEL,IEL,IEL / IEL,IEL,IEL | 0.171774 | 0.210644 | 0.075344 |
| CiteSeer | edge | Source then destination | 3293 | 0.27 / 861.58 | IL / IEL / IEL | 0.781399 | 0.226403 | 0.066545 |
| CiteSeer | edge | Destination then source | 3293 | 861.58 / 0.27 | IL / IEL / IEL | 0.461739 | 0.214984 | 0.064469 |
| CiteSeer | edge | Morton (Z-order) | 3293 | 24.77 / 17.98 | IL / IEL / IEL | 0.473932 | 0.200314 | 0.061007 |
| PubMed | node | Original | 19482 | 0.55 / 3603.86 | — / — / — | 0.048258 | 17.551161 | 0.690521 |
| PubMed | node | RCMK | 8439 | 2372.61 / 4360.11 | I / IEL / IEL | 5.155799 | 19.585571 | 0.793353 |
| PubMed | node | Degree ascending | 19706 | 2088.58 / 3855.80 | I / IEL / IEL | 0.624102 | 19.116199 | 0.772693 |
| PubMed | node | Degree descending | 19707 | 2045.93 / 3820.37 | I / IEL / IEL | 0.650710 | 18.765973 | 0.843828 |
| PubMed | node | BFS | 7764 | 2359.45 / 4276.65 | I / IEL / IEL | 23.871142 | 20.053124 | 0.864679 |
| PubMed | node | Random control | 19661 | 2392.19 / 6565.76 | I,I,I / IEL,IEL,IEL / IEL,IEL,IEL | 0.262376 | 19.356093 | 0.926783 |
| PubMed | node | METIS (32 partitions) | 19615 | 2361.18 / 3551.49 | I / IEL / IEL | 0.039929 | 19.580216 | 0.851147 |
| PubMed | edge | Original | 19482 | 0.55 / 3603.86 | — / — / — | 0.052976 | 0.745206 | 0.618599 |
| PubMed | edge | Source-grouped | 19482 | 0.18 / 4602.86 | IL / E / IEL | 0.313233 | 0.733553 | 0.639393 |
| PubMed | edge | Destination-grouped | 19482 | 3715.68 / 0.18 | IL / IEL / E | 4.304770 | 0.904030 | 0.622849 |
| PubMed | edge | Random control | 19482 | 6537.82 / 6555.90 | IL,IL,IL / IEL,IEL,IEL / IEL,IEL,IEL | 0.915110 | 0.812894 | 0.828550 |
| PubMed | edge | Source then destination | 19482 | 0.18 / 3715.60 | IL / IEL / IEL | 9.112493 | 0.771297 | 0.655978 |
| PubMed | edge | Destination then source | 19482 | 3715.60 / 0.18 | IL / IEL / IEL | 5.077314 | 0.867458 | 0.650545 |
| PubMed | edge | Morton (Z-order) | 19482 | 55.49 / 39.62 | IL / IEL / IEL | 4.166370 | 0.727117 | 0.684414 |

The node sweep retains canonical EIDs. Its CSR and CSC rows may traverse those EIDs in a different sequence because compressed rows follow renumbered node IDs. The edge sweep changes EIDs directly. A compressed view with unchanged I can still change E because the stored entries refer to new edge labels. The COO EID order is implicit.
Source-grouped preserved the CSR pointers, neighbor indices and traversal in original EIDs on every dataset. Its change within that view was confined to the mapping to current EIDs.
Destination-grouped preserved the CSC pointers, neighbor indices and traversal in original EIDs on every dataset. Its change within that view was confined to the mapping to current EIDs.

The cached loader and common rebuilt baseline are compared below.

| Dataset | COO / CSR / CSC changes relative to loader |
|---|---|
| Cora | — / — / — |
| CiteSeer | — / — / — |
| PubMed | — / — / — |

| Dataset | Nodes | Edges | Self-loops | Parallel edges beyond first | Total degree mean / median / p95 / max |
|---|---:|---:|---:|---:|---|
| Cora | 2,708 | 13,264 | 2,708 | 0 | 9.80 / 8 / 20 / 338 |
| CiteSeer | 3,327 | 12,431 | 3,327 | 0 | 7.47 / 6 / 18 / 200 |
| PubMed | 19,717 | 108,365 | 19,717 | 0 | 10.99 / 6 / 38 / 344 |

## Frozen native METIS plan

DGL 1.1.3's installed `dgl.metis_partition_assignment` makes a symmetric auxiliary graph and partitions it with the native k-way cut objective. We requested exactly 32 partitions with default vertex balancing and no edge or type balancing. Stable ordering by partition ID uses the original node ID to break ties. The auxiliary graph is never timed as a forward graph. Its construction and native assignment are included in the one-shot assignment time. The Python API exposes no METIS seed. Partition labels or assignments can depend on the native implementation and build. All workers used the frozen, verified plan.

Assignment and stable sort below were measured once per dataset before acquisition. The five-repetition construction column for METIS in the main overhead table measures replay planning only, followed by actual node-feature copying, graph rebuild, and format materialization.

| Dataset | Requested parts | Nonempty parts | Min / max nodes per nonempty part | Assignment ms | Stable ordering ms | Replay planning ms |
|---|---:|---:|---|---:|---:|---:|
| Cora | 32 | 32 | 83 / 87 | 3.545121 | 0.187855 | 0.009910 |
| CiteSeer | 32 | 32 | 100 / 107 | 2.672972 | 0.208515 | 0.010596 |
| PubMed | 32 | 32 | 606 / 628 | 17.323822 | 0.737426 | 0.039929 |

## Edge traversal details

Source/Destination-grouped use stable single-key sorts. Source then destination and Destination then source add a second endpoint key before the original EID tie-breaker. Morton uses unsigned 64-bit keys. Source bit i goes to 2i and destination bit i goes to 2i+1. Original EID breaks ties between parallel edges.

The table summarizes the new policies across the three graphs using the same flags as the physical table. I means index arrays changed, E means the map to current EIDs changed, and L means the sequence of original EIDs visited changed.

| Policy | COO flags observed | CSR flags observed | CSC flags observed |
|---|---|---|---|
| Source then destination | IL | IEL | IEL |
| Destination then source | IL | IEL | IEL |
| Morton (Z-order) | IL | IEL | IEL |

## Historical matched-allocation comparison

The [earlier acquisition](../REPORT.md) is preserved at 5b338b132f552697b9569bd2886e686fba9b02a4. Every shared policy used the same maps and physical graph views in both acquisitions, with matching workload fingerprints and pinned environment. Each acquisition has its own blocks, process-level estimates and correction families. No timing samples are pooled. Absolute latencies and ranks can shift across acquisitions or when the interleaved case set changes. Rank changes below are descriptive.

| Dataset | Sweep | Fastest historical | Fastest now among shared policies | Fastest now overall |
|---|---|---|---|---|
| Cora | node | Degree descending | Degree descending | Degree descending |
| Cora | edge | Original | Destination-grouped | Destination-grouped |
| CiteSeer | node | Degree descending | Degree ascending | Degree ascending |
| CiteSeer | edge | Original | Source-grouped | Source-grouped |
| PubMed | node | Degree descending | RCMK | RCMK |
| PubMed | edge | Destination-grouped | Destination-grouped | Destination then source |

The best observed shared policy changed for Cora edge, CiteSeer node, CiteSeer edge, PubMed node.
Added policies changed the overall observed winner: Destination then source on PubMed.
This separates shifts among policies that already existed from shifts caused by having new choices in the table. Inference still comes from each acquisition's paired comparisons.

| Dataset | Sweep | Shared policy | Historical median ms | Extended median ms | Historical speedup (95% CI) | Extended speedup (95% CI) | Decisions (historical → extended) |
|---|---|---|---:|---:|---|---|---|
| Cora | node | RCMK | 4.270547 | 4.209413 | 1.0110× [0.9940, 1.0284] | 1.0095× [1.0020, 1.0171] | Inconclusive → Inconclusive |
| Cora | node | Degree ascending | 4.294763 | 4.189194 | 1.0090× [0.9853, 1.0332] | 1.0181× [1.0025, 1.0339] | Inconclusive → Inconclusive |
| Cora | node | Degree descending | 4.228765 | 4.097715 | 1.0210× [1.0026, 1.0397] | 1.0426× [1.0191, 1.0667] | Inconclusive → Inconclusive |
| Cora | node | BFS | 4.254714 | 4.138363 | 1.0086× [0.9854, 1.0325] | 1.0251× [1.0006, 1.0501] | Inconclusive → Inconclusive |
| Cora | node | Random control | 4.244876 | 4.172116 | 1.0176× [1.0019, 1.0335] | 1.0245× [1.0047, 1.0447] | Inconclusive → Inconclusive |
| Cora | edge | Source-grouped | 3.921687 | 3.816808 | 0.9791× [0.9557, 1.0031] | 0.9981× [0.9925, 1.0038] | Inconclusive → Inconclusive |
| Cora | edge | Destination-grouped | 3.816565 | 3.755902 | 1.0052× [0.9920, 1.0185] | 1.0080× [0.9993, 1.0169] | Inconclusive → Inconclusive |
| Cora | edge | Random control | 3.853843 | 3.856520 | 0.9929× [0.9775, 1.0086] | 0.9876× [0.9793, 0.9959] | Inconclusive → Inconclusive |
| CiteSeer | node | RCMK | 10.822754 | 10.596366 | 0.9947× [0.9854, 1.0041] | 1.0083× [0.9955, 1.0212] | Inconclusive → Inconclusive |
| CiteSeer | node | Degree ascending | 10.648112 | 10.519621 | 1.0135× [1.0039, 1.0232] | 1.0169× [1.0032, 1.0309] | Inconclusive → Inconclusive |
| CiteSeer | node | Degree descending | 10.618062 | 10.533286 | 1.0168× [1.0081, 1.0257] | 1.0153× [1.0067, 1.0241] | Gain below practical criterion → Inconclusive |
| CiteSeer | node | BFS | 10.657926 | 10.580265 | 1.0106× [0.9991, 1.0221] | 1.0134× [1.0019, 1.0251] | Inconclusive → Inconclusive |
| CiteSeer | node | Random control | 10.625410 | 10.526382 | 1.0149× [1.0071, 1.0227] | 1.0135× [1.0009, 1.0262] | Gain (diagnostic) → Inconclusive |
| CiteSeer | edge | Source-grouped | 10.075928 | 9.781487 | 1.0066× [0.9909, 1.0225] | 1.0035× [0.9903, 1.0169] | Inconclusive → Inconclusive |
| CiteSeer | edge | Destination-grouped | 10.066579 | 9.792291 | 1.0013× [0.9852, 1.0176] | 1.0045× [0.9806, 1.0291] | Inconclusive → Inconclusive |
| CiteSeer | edge | Random control | 10.152463 | 9.795352 | 0.9947× [0.9810, 1.0087] | 0.9992× [0.9845, 1.0140] | Inconclusive → Inconclusive |
| PubMed | node | RCMK | 17.277737 | 16.907439 | 1.0105× [0.9991, 1.0221] | 1.0107× [1.0024, 1.0191] | Inconclusive → Inconclusive |
| PubMed | node | Degree ascending | 17.239524 | 16.931210 | 1.0087× [0.9967, 1.0209] | 1.0142× [1.0051, 1.0235] | Inconclusive → Inconclusive |
| PubMed | node | Degree descending | 17.100453 | 16.968679 | 1.0174× [1.0056, 1.0293] | 1.0138× [1.0033, 1.0245] | Inconclusive → Inconclusive |
| PubMed | node | BFS | 17.198402 | 17.043900 | 1.0123× [1.0066, 1.0180] | 1.0086× [0.9991, 1.0182] | Gain below practical criterion → Inconclusive |
| PubMed | node | Random control | 17.638702 | 17.506921 | 0.9859× [0.9797, 0.9920] | 0.9785× [0.9720, 0.9851] | Loss → Loss |
| PubMed | edge | Source-grouped | 16.496920 | 15.962032 | 1.0011× [0.9951, 1.0072] | 0.9981× [0.9862, 1.0101] | Inconclusive → Inconclusive |
| PubMed | edge | Destination-grouped | 15.727640 | 15.428297 | 1.0504× [1.0427, 1.0583] | 1.0280× [1.0152, 1.0410] | Qualifies → Qualifies |
| PubMed | edge | Random control | 16.826203 | 16.249561 | 0.9835× [0.9730, 0.9941] | 0.9719× [0.9619, 0.9821] | Inconclusive → Loss |

## Protocol and correctness

8 fresh processes per dataset, 10 randomized warmup blocks, 30 measured matched blocks per sweep per process, and 5 separately randomized construction observations per case and process. Every block has a plain forward and a separately instrumented forward for every case, including each random seed. Timing schedules used a new recorded seed; random permutations retained their three original seeds per sweep.

AMD Ryzen 7 7800X3D 8-Core Processor, logical CPU 0, one thread, Python 3.11.11, PyTorch 2.2.2+cpu, DGL 1.1.3, Linux-6.18.58.1-microsoft-standard-WSL2-x86_64-with-glibc2.36. FP32, four heads, eight channels per head, identical seed-0 weights, original native features, no dropout. Input self-loops were removed and one per node appended by the unchanged loader.

For every case the restored topology, multiplicities, feature and edge-data alignment, and sparse storage maps were checked. Attention was restored by original EID and checked for destination-wise normalization. Direct and instrumented outputs and attention matched bitwise. Maximum absolute output error 1.86e-08, attention error 5.96e-08, normalization error 4.72e-07. Tolerances were atol 1e−5 and rtol 1e−4.

Complete forward does not include permutation generation, graph construction, or restoring outputs to canonical IDs. The independent sweeps do not measure whether node and edge effects compose. Confidence intervals describe process variation on this system. Random-order intervals condition on three fixed permutations.

[Raw timings](RESULTS.json) · [Permutation maps](PERMUTATIONS.json) · [Frozen METIS metadata](METIS.json) · [Paired statistics](ANALYSIS.json) · [Reproduction](README.md)
