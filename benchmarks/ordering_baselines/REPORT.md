# Milestone 10 — Independent node and edge ordering

Every forward uses the imported DGL 1.1.3 GATConv. The six stage columns come from a validated execution-equivalent copy of its forward. Total forward is timed directly on the unmodified import. All times are milliseconds and use the median of process medians.

The node sweep relabels nodes and feature rows while preserving every canonical EID. The edge sweep preserves node labels and feature rows. Both re-create the same original topology and order as their within-sweep control. Every node-ordering case allocates its feature rows with the same indexing operation, including the identity control. Every edge-ordering case shares the loader's unchanged feature tensor. Every graph starts from the same DGL constructor and materializes all COO, CSR, and CSC views before timing.

Use the original row within the same sweep as the control. The two sweeps have different case sets and feature working sets, so their absolute medians need not coincide.

## Cora

### Node ordering

| Ordering | Projection | Attention Terms (L/R) | Attention Scores | Edge Softmax | Aggregation | Output | Total Forward |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 3.004102 | 0.120842 | 0.189453 | 0.424124 | 0.456960 | 0.022197 | 4.293674 |
| RCMK | 2.942096 | 0.119531 | 0.190006 | 0.423322 | 0.456772 | 0.023004 | 4.270547 |
| Degree ascending | 2.894627 | 0.119160 | 0.189514 | 0.416327 | 0.457555 | 0.022870 | 4.294763 |
| Degree descending | 2.886256 | 0.119795 | 0.191770 | 0.404269 | 0.455437 | 0.022404 | 4.228765 |
| BFS | 2.884709 | 0.119123 | 0.188265 | 0.428926 | 0.451892 | 0.022525 | 4.254714 |
| Random control | 2.852647 | 0.119655 | 0.191139 | 0.436791 | 0.474643 | 0.022908 | 4.244876 |

### Edge ordering

| Ordering | Projection | Attention Terms (L/R) | Attention Scores | Edge Softmax | Aggregation | Output | Total Forward |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 2.482716 | 0.118313 | 0.184681 | 0.422650 | 0.458260 | 0.022603 | 3.816230 |
| Source-grouped | 2.512974 | 0.119306 | 0.184149 | 0.430028 | 0.459734 | 0.022822 | 3.921687 |
| Destination-grouped | 2.503993 | 0.118126 | 0.186062 | 0.407057 | 0.433987 | 0.021474 | 3.816565 |
| Random control | 2.501532 | 0.118332 | 0.186451 | 0.430083 | 0.473419 | 0.022651 | 3.853843 |

## CiteSeer

### Node ordering

| Ordering | Projection | Attention Terms (L/R) | Attention Scores | Edge Softmax | Aggregation | Output | Total Forward |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 9.378066 | 0.156267 | 0.231194 | 0.439010 | 0.452485 | 0.029559 | 10.755670 |
| RCMK | 9.259396 | 0.159522 | 0.232175 | 0.445929 | 0.440161 | 0.029288 | 10.822754 |
| Degree ascending | 9.126389 | 0.155454 | 0.224717 | 0.422455 | 0.442326 | 0.028975 | 10.648112 |
| Degree descending | 9.101258 | 0.157267 | 0.230190 | 0.416900 | 0.448835 | 0.029563 | 10.618062 |
| BFS | 9.104081 | 0.153910 | 0.233376 | 0.437121 | 0.438437 | 0.029391 | 10.657926 |
| Random control | 9.099539 | 0.158686 | 0.230701 | 0.444372 | 0.469174 | 0.029566 | 10.625410 |

### Edge ordering

| Ordering | Projection | Attention Terms (L/R) | Attention Scores | Edge Softmax | Aggregation | Output | Total Forward |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 8.657434 | 0.151874 | 0.213131 | 0.437022 | 0.450358 | 0.027446 | 9.999194 |
| Source-grouped | 8.691131 | 0.153042 | 0.215971 | 0.443456 | 0.462750 | 0.028773 | 10.075928 |
| Destination-grouped | 8.627112 | 0.152972 | 0.216032 | 0.423497 | 0.435519 | 0.027941 | 10.066579 |
| Random control | 8.675320 | 0.151580 | 0.219169 | 0.443959 | 0.465674 | 0.027879 | 10.152463 |

## PubMed

### Node ordering

| Ordering | Projection | Attention Terms (L/R) | Attention Scores | Edge Softmax | Aggregation | Output | Total Forward |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 8.563198 | 0.758234 | 0.701725 | 3.148954 | 3.699364 | 0.095177 | 17.428137 |
| RCMK | 8.329125 | 0.766677 | 0.697039 | 3.154681 | 3.542257 | 0.096504 | 17.277737 |
| Degree ascending | 8.436705 | 0.753170 | 0.690347 | 3.075379 | 3.696967 | 0.094528 | 17.239524 |
| Degree descending | 8.335423 | 0.760667 | 0.702037 | 3.069004 | 3.723873 | 0.096002 | 17.100453 |
| BFS | 8.340849 | 0.761380 | 0.703817 | 3.169537 | 3.508802 | 0.096469 | 17.198402 |
| Random control | 8.317577 | 0.759122 | 0.721264 | 3.292279 | 3.951438 | 0.096165 | 17.638702 |

### Edge ordering

| Ordering | Projection | Attention Terms (L/R) | Attention Scores | Edge Softmax | Aggregation | Output | Total Forward |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 7.627408 | 0.754273 | 0.699168 | 3.250066 | 3.638597 | 0.096284 | 16.395252 |
| Source-grouped | 7.482074 | 0.752842 | 0.694547 | 3.307369 | 3.731764 | 0.095266 | 16.496920 |
| Destination-grouped | 7.594253 | 0.755166 | 0.680744 | 2.926831 | 3.341573 | 0.097801 | 15.727640 |
| Random control | 7.615191 | 0.762790 | 0.733487 | 3.332458 | 3.926025 | 0.096804 | 16.826203 |

## Forward comparisons against original

Speedup divides the original time by the candidate time. Effects use the geometric mean of 8 paired process-median ratios with pointwise 95% Student-t intervals in log space. Holm correction is separate for the node and edge sweeps across all datasets and planned policies, including random controls. The rules apply separately to each profiled stage. The tuning criterion requires adjusted p < 0.05 and a 95% CI lower speedup bound above 1.01×.

| Dataset | Sweep | Policy | Paired speedup (95% CI) | Holm p | Supported |
|---|---|---|---|---:|---|
| Cora | node | RCMK | 1.0110× [0.9940, 1.0284] | 0.6847 | Inconclusive |
| Cora | node | Degree ascending | 1.0090× [0.9853, 1.0332] | 0.8064 | Inconclusive |
| Cora | node | Degree descending | 1.0210× [1.0026, 1.0397] | 0.2754 | Inconclusive |
| Cora | node | BFS | 1.0086× [0.9854, 1.0325] | 0.8064 | Inconclusive |
| Cora | node | Random control | 1.0176× [1.0019, 1.0335] | 0.2754 | Inconclusive |
| Cora | edge | Source-grouped | 0.9791× [0.9557, 1.0031] | 0.5446 | Inconclusive |
| Cora | edge | Destination-grouped | 1.0052× [0.9920, 1.0185] | 1 | Inconclusive |
| Cora | edge | Random control | 0.9929× [0.9775, 1.0086] | 1 | Inconclusive |
| CiteSeer | node | RCMK | 0.9947× [0.9854, 1.0041] | 0.6847 | Inconclusive |
| CiteSeer | node | Degree ascending | 1.0135× [1.0039, 1.0232] | 0.1248 | Inconclusive |
| CiteSeer | node | Degree descending | 1.0168× [1.0081, 1.0257] | 0.03387 | Gain |
| CiteSeer | node | BFS | 1.0106× [0.9991, 1.0221] | 0.4581 | Inconclusive |
| CiteSeer | node | Random control | 1.0149× [1.0071, 1.0227] | 0.03387 | Gain |
| CiteSeer | edge | Source-grouped | 1.0066× [0.9909, 1.0225] | 1 | Inconclusive |
| CiteSeer | edge | Destination-grouped | 1.0013× [0.9852, 1.0176] | 1 | Inconclusive |
| CiteSeer | edge | Random control | 0.9947× [0.9810, 1.0087] | 1 | Inconclusive |
| PubMed | node | RCMK | 1.0105× [0.9991, 1.0221] | 0.4581 | Inconclusive |
| PubMed | node | Degree ascending | 1.0087× [0.9967, 1.0209] | 0.6569 | Inconclusive |
| PubMed | node | Degree descending | 1.0174× [1.0056, 1.0293] | 0.1108 | Inconclusive |
| PubMed | node | BFS | 1.0123× [1.0066, 1.0180] | 0.01972 | Gain |
| PubMed | node | Random control | 0.9859× [0.9797, 0.9920] | 0.01525 | Loss |
| PubMed | edge | Source-grouped | 1.0011× [0.9951, 1.0072] | 1 | Inconclusive |
| PubMed | edge | Destination-grouped | 1.0504× [1.0427, 1.0583] | 9.385e-06 | Gain and meets tuning criterion |
| PubMed | edge | Random control | 0.9835× [0.9730, 0.9941] | 0.06371 | Inconclusive |

## Conclusions

### Node ordering

- **Cora**. Degree descending had the lowest observed median (4.2288 ms). The predetermined tuning criterion was unmet.
- **CiteSeer**. Degree descending had the lowest observed median (10.6181 ms). The predetermined tuning criterion was unmet.
- **PubMed**. Degree descending had the lowest observed median (17.1005 ms). The predetermined tuning criterion was unmet.

Keep Original as the default for the node sweep. Omit the tested alternatives from the initial tuning space under this criterion. Keep the measurements and seeded random controls for reference.

### Edge ordering

- **Cora**. Original had the lowest observed median (3.8162 ms). The predetermined tuning criterion was unmet.
- **CiteSeer**. Original had the lowest observed median (9.9992 ms). The predetermined tuning criterion was unmet.
- **PubMed**. Destination-grouped had the lowest observed median (15.7276 ms). Destination-grouped met the tuning criterion.

Retain edge ordering as an independent candidate axis on the tested workloads. Keep Original and Destination-grouped. Keep the seeded random runs as diagnostic controls.
Omit Source-grouped from the initial tuning space under this criterion.

## Stage effects

These counts summarize the same paired comparisons for individual profiled stages. Random controls are tabulated separately in the analysis JSON. Stage timings describe the instrumented execution and the full-forward comparison determines the tuning decision.

| Sweep | Stage | Supported policy-dataset gains | Supported losses |
|---|---|---:|---:|
| node | projection | 3/12 | 0/12 |
| node | terms | 0/12 | 0/12 |
| node | score | 0/12 | 0/12 |
| node | softmax | 3/12 | 0/12 |
| node | aggregation | 5/12 | 0/12 |
| node | output | 0/12 | 0/12 |
| edge | projection | 0/6 | 0/6 |
| edge | terms | 0/6 | 0/6 |
| edge | score | 0/6 | 0/6 |
| edge | softmax | 3/6 | 0/6 |
| edge | aggregation | 2/6 | 0/6 |
| edge | output | 0/6 | 0/6 |

The table below lists corrected stage differences for deterministic policies.

| Dataset | Sweep | Policy | Stage | Paired speedup (95% CI) | Holm p |
|---|---|---|---|---|---:|
| Cora | node | Degree descending | softmax | 1.0462× [1.0270, 1.0657] | 0.008874 |
| Cora | edge | Destination-grouped | softmax | 1.0386× [1.0190, 1.0587] | 0.01797 |
| Cora | edge | Destination-grouped | aggregation | 1.0498× [1.0362, 1.0635] | 0.0003426 |
| CiteSeer | node | RCMK | aggregation | 1.0272× [1.0112, 1.0434] | 0.03946 |
| CiteSeer | node | Degree ascending | projection | 1.0229× [1.0137, 1.0322] | 0.007822 |
| CiteSeer | node | Degree ascending | aggregation | 1.0252× [1.0142, 1.0363] | 0.0103 |
| CiteSeer | node | Degree descending | projection | 1.0307× [1.0191, 1.0424] | 0.005515 |
| CiteSeer | node | Degree descending | softmax | 1.0590× [1.0389, 1.0795] | 0.00278 |
| CiteSeer | node | BFS | projection | 1.0313× [1.0152, 1.0477] | 0.02602 |
| CiteSeer | node | BFS | aggregation | 1.0335× [1.0175, 1.0498] | 0.01438 |
| CiteSeer | edge | Destination-grouped | softmax | 1.0396× [1.0161, 1.0636] | 0.03576 |
| PubMed | node | RCMK | aggregation | 1.0398× [1.0308, 1.0490] | 0.0002098 |
| PubMed | node | Degree ascending | softmax | 1.0344× [1.0173, 1.0518] | 0.02349 |
| PubMed | node | BFS | aggregation | 1.0564× [1.0424, 1.0706] | 0.0003343 |
| PubMed | edge | Destination-grouped | softmax | 1.1111× [1.0788, 1.1443] | 0.0005772 |
| PubMed | edge | Destination-grouped | aggregation | 1.0999× [1.0751, 1.1252] | 0.0001846 |

## Feature allocation sensitivity

The first full acquisition reused the loader tensor for the original node-order control and allocated a new tensor for every other node policy. That run showed dense projection gains for 11 of 12 deterministic policy/dataset pairs. We then repeated the full experiment with an identical feature-indexing allocation for every node policy, including the identity permutation. The matched-allocation acquisition is primary. The first run is archived and its samples are kept separate. The edge protocol was unchanged. Time variation between acquisitions also prevents attributing the entire difference to the allocator.

| Dataset | Sweep | Policy | First run (95% CI) | Controlled run (95% CI) |
|---|---|---|---|---|
| Cora | node | RCMK | 1.0225× [1.0039, 1.0414] | 1.0110× [0.9940, 1.0284] |
| Cora | node | Degree ascending | 1.0262× [1.0111, 1.0414] | 1.0090× [0.9853, 1.0332] |
| Cora | node | Degree descending | 1.0310× [1.0109, 1.0516] | 1.0210× [1.0026, 1.0397] |
| Cora | node | BFS | 1.0450× [1.0247, 1.0657] | 1.0086× [0.9854, 1.0325] |
| Cora | node | Random control | 1.0441× [1.0250, 1.0636] | 1.0176× [1.0019, 1.0335] |
| Cora | edge | Source-grouped | 1.0024× [0.9827, 1.0224] | 0.9791× [0.9557, 1.0031] |
| Cora | edge | Destination-grouped | 1.0013× [0.9687, 1.0350] | 1.0052× [0.9920, 1.0185] |
| Cora | edge | Random control | 0.9977× [0.9794, 1.0164] | 0.9929× [0.9775, 1.0086] |
| CiteSeer | node | RCMK | 1.0273× [1.0129, 1.0418] | 0.9947× [0.9854, 1.0041] |
| CiteSeer | node | Degree ascending | 1.0380× [1.0190, 1.0574] | 1.0135× [1.0039, 1.0232] |
| CiteSeer | node | Degree descending | 1.0463× [1.0379, 1.0548] | 1.0168× [1.0081, 1.0257] |
| CiteSeer | node | BFS | 1.0413× [1.0272, 1.0556] | 1.0106× [0.9991, 1.0221] |
| CiteSeer | node | Random control | 1.0406× [1.0307, 1.0505] | 1.0149× [1.0071, 1.0227] |
| CiteSeer | edge | Source-grouped | 1.0065× [0.9870, 1.0264] | 1.0066× [0.9909, 1.0225] |
| CiteSeer | edge | Destination-grouped | 1.0029× [0.9871, 1.0189] | 1.0013× [0.9852, 1.0176] |
| CiteSeer | edge | Random control | 0.9943× [0.9816, 1.0073] | 0.9947× [0.9810, 1.0087] |
| PubMed | node | RCMK | 1.0204× [0.9931, 1.0485] | 1.0105× [0.9991, 1.0221] |
| PubMed | node | Degree ascending | 1.0285× [1.0166, 1.0405] | 1.0087× [0.9967, 1.0209] |
| PubMed | node | Degree descending | 1.0287× [1.0067, 1.0513] | 1.0174× [1.0056, 1.0293] |
| PubMed | node | BFS | 1.0343× [1.0168, 1.0521] | 1.0123× [1.0066, 1.0180] |
| PubMed | node | Random control | 1.0057× [0.9900, 1.0217] | 0.9859× [0.9797, 0.9920] |
| PubMed | edge | Source-grouped | 1.0019× [0.9875, 1.0164] | 1.0011× [0.9951, 1.0072] |
| PubMed | edge | Destination-grouped | 1.0439× [1.0290, 1.0590] | 1.0504× [1.0427, 1.0583] |
| PubMed | edge | Random control | 0.9767× [0.9634, 0.9903] | 0.9835× [0.9730, 0.9941] |

Both runs use the same metric, paired process-median speedup. Each interval describes its own acquisition. No hypothesis test compares these intervals. [First raw run](initial_allocation/RESULTS.json) · [First-run policy source](initial_allocation/policies.py).

## Random permutation variability

The main row uses the median of the three seed-specific medians within each process, followed by the median across processes. Its paired interval instead averages the three seed log ratios within each process. The interval treats the seeds as fixed and describes process variability. The variation across the three seed estimates is descriptive.

| Dataset | Sweep | Seed | Forward ms | SD across process medians, ms | Seed-specific paired speedup |
|---|---|---:|---:|---:|---:|
| Cora | node | 2026122000 | 4.263284 | 0.059094 | 1.0139× |
| Cora | node | 2026122001 | 4.225900 | 0.058934 | 1.0205× |
| Cora | node | 2026122002 | 4.246702 | 0.130173 | 1.0183× |
| Cora | edge | 2026122010 | 3.851653 | 0.116144 | 0.9939× |
| Cora | edge | 2026122011 | 3.842028 | 0.104032 | 0.9961× |
| Cora | edge | 2026122012 | 3.868944 | 0.128632 | 0.9888× |
| CiteSeer | node | 2026122100 | 10.725361 | 0.132627 | 1.0086× |
| CiteSeer | node | 2026122101 | 10.526230 | 0.143801 | 1.0196× |
| CiteSeer | node | 2026122102 | 10.613381 | 0.134474 | 1.0165× |
| CiteSeer | edge | 2026122110 | 10.182320 | 0.175683 | 0.9931× |
| CiteSeer | edge | 2026122111 | 10.075802 | 0.219856 | 1.0023× |
| CiteSeer | edge | 2026122112 | 10.180162 | 0.153500 | 0.9888× |
| PubMed | node | 2026122200 | 17.480374 | 0.238756 | 0.9909× |
| PubMed | node | 2026122201 | 17.549351 | 0.212911 | 0.9875× |
| PubMed | node | 2026122202 | 17.695050 | 0.207187 | 0.9792× |
| PubMed | edge | 2026122210 | 16.977094 | 0.712979 | 0.9809× |
| PubMed | edge | 2026122211 | 16.819744 | 0.620902 | 0.9874× |
| PubMed | edge | 2026122212 | 16.893661 | 0.716710 | 0.9822× |

## Instrumentation overhead

Each block interleaves direct forwards, profiled forwards and timer calibration. Overhead is the paired instrumented-minus-plain interval for the original case. The point estimate averages process medians of the paired differences. The empty six-region calibration measures timestamp reads, loop and record writes. Full per-policy overhead and unassigned graph-scope costs are in ANALYSIS.json.

| Dataset | Sweep | Empty six regions net µs (median) | Original overhead µs (95% CI) |
|---|---|---:|---|
| Cora | node | 1.533 | +29.961 [-78.646, +138.568] |
| Cora | edge | 1.478 | -31.316 [-67.547, +4.915] |
| CiteSeer | node | 1.625 | +68.209 [-74.293, +210.711] |
| CiteSeer | edge | 1.563 | +101.793 [-66.689, +270.274] |
| PubMed | node | 1.734 | -90.133 [-288.412, +108.147] |
| PubMed | edge | 1.770 | -38.991 [-172.915, +94.933] |

## One-time costs and physical representation

Rebuild includes endpoint remapping, DGL graph creation and any data permutation. Materialization times the subsequent create_formats_ call. The original control also goes through this constructor. The node identity control includes a feature copy solely for measurement parity. Deployment can keep the original features without that copy. Random rows show seed medians.

Bandwidth is max |src − dst| in current labels. Endpoint step is the mean absolute ID difference between consecutive COO edges. Compressed-view hashes and locality statistics are recorded in ANALYSIS.json. The change flags compare actual arrays against the original within the sweep. I means stored adjacency indices changed, E means the stored traversal-to-current-EID map changed, and L means traversal in original EIDs changed. Random physical flags cover each of the three seeds in order.

| Dataset | Sweep | Ordering | Bandwidth | COO src / dst step | COO / CSR / CSC changes | Permutation ms | Rebuild ms | Materialization ms |
|---|---|---|---:|---|---|---:|---:|---:|
| Cora | node | Original | 2657 | 0.61 / 551.74 | — / — / — | 0.012025 | 1.957492 | 0.064615 |
| Cora | node | RCMK | 690 | 309.64 / 520.38 | I / IEL / IEL | 0.688705 | 2.073968 | 0.068088 |
| Cora | node | Degree ascending | 2701 | 340.81 / 740.88 | I / IEL / IEL | 0.090299 | 1.963971 | 0.062910 |
| Cora | node | Degree descending | 2638 | 329.76 / 731.17 | I / IEL / IEL | 0.092434 | 2.045437 | 0.064621 |
| Cora | node | BFS | 745 | 299.01 / 534.63 | I / IEL / IEL | 3.285325 | 2.020883 | 0.066194 |
| Cora | node | Random control | 2671 | 373.64 / 902.22 | I,I,I / IEL,IEL,IEL / IEL,IEL,IEL | 0.067837 | 2.035115 | 0.073849 |
| Cora | edge | Original | 2657 | 0.61 / 551.74 | — / — / — | 0.011402 | 0.186142 | 0.059892 |
| Cora | edge | Source-grouped | 2657 | 0.20 / 681.96 | IL / E / IEL | 0.049024 | 0.200821 | 0.060292 |
| Cora | edge | Destination-grouped | 2657 | 616.20 / 0.20 | IL / IEL / E | 0.417467 | 0.204867 | 0.060123 |
| Cora | edge | Random control | 2657 | 867.57 / 874.75 | IL,IL,IL / IEL,IEL,IEL / IEL,IEL,IEL | 0.140979 | 0.193555 | 0.065821 |
| CiteSeer | node | Original | 3293 | 0.80 / 669.73 | — / — / — | 0.017175 | 21.409948 | 0.080849 |
| CiteSeer | node | RCMK | 390 | 525.88 / 617.21 | I / IEL / IEL | 0.831237 | 21.749815 | 0.098083 |
| CiteSeer | node | Degree ascending | 3250 | 521.14 / 852.83 | I / IEL / IEL | 0.115052 | 21.539559 | 0.084281 |
| CiteSeer | node | Degree descending | 3249 | 507.27 / 842.29 | I / IEL / IEL | 0.129022 | 21.911689 | 0.083295 |
| CiteSeer | node | BFS | 313 | 526.12 / 613.24 | I / IEL / IEL | 3.643358 | 21.791099 | 0.095585 |
| CiteSeer | node | Random control | 3266 | 591.33 / 1113.94 | I,I,I / IEL,IEL,IEL / IEL,IEL,IEL | 0.105493 | 22.148203 | 0.103067 |
| CiteSeer | edge | Original | 3293 | 0.80 / 669.73 | — / — / — | 0.019841 | 0.247498 | 0.065298 |
| CiteSeer | edge | Source-grouped | 3293 | 0.27 / 925.43 | IL / E / IEL | 0.059757 | 0.209683 | 0.064346 |
| CiteSeer | edge | Destination-grouped | 3293 | 861.68 / 0.27 | IL / IEL / E | 0.383367 | 0.243648 | 0.067347 |
| CiteSeer | edge | Random control | 3293 | 1095.16 / 1090.23 | IL,IL,IL / IEL,IEL,IEL / IEL,IEL,IEL | 0.178175 | 0.232474 | 0.079437 |
| PubMed | node | Original | 19482 | 0.55 / 3603.86 | — / — / — | 0.050833 | 17.593291 | 0.699999 |
| PubMed | node | RCMK | 8439 | 2372.61 / 4360.11 | I / IEL / IEL | 5.364981 | 19.444668 | 0.942203 |
| PubMed | node | Degree ascending | 19706 | 2088.58 / 3855.80 | I / IEL / IEL | 0.656366 | 18.783430 | 0.774470 |
| PubMed | node | Degree descending | 19707 | 2045.93 / 3820.37 | I / IEL / IEL | 0.634938 | 18.811858 | 0.787882 |
| PubMed | node | BFS | 7764 | 2359.45 / 4276.65 | I / IEL / IEL | 23.620531 | 19.569385 | 0.874548 |
| PubMed | node | Random control | 19661 | 2392.19 / 6565.76 | I,I,I / IEL,IEL,IEL / IEL,IEL,IEL | 0.259656 | 19.179803 | 0.926146 |
| PubMed | edge | Original | 19482 | 0.55 / 3603.86 | — / — / — | 0.050330 | 0.738385 | 0.632950 |
| PubMed | edge | Source-grouped | 19482 | 0.18 / 4602.86 | IL / E / IEL | 0.318059 | 0.731595 | 0.642587 |
| PubMed | edge | Destination-grouped | 19482 | 3715.68 / 0.18 | IL / IEL / E | 4.303256 | 0.882192 | 0.597263 |
| PubMed | edge | Random control | 19482 | 6537.82 / 6555.90 | IL,IL,IL / IEL,IEL,IEL / IEL,IEL,IEL | 0.896720 | 0.860774 | 0.798862 |

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

## Protocol and correctness

8 fresh processes per dataset on AMD Ryzen 7 7800X3D 8-Core Processor with logical CPU 0 pinned and one inference thread. 10 warmup and 30 paired randomized blocks per sweep per process. 5 separately randomized construction observations per case and process. The CPU Docker environment reported Python 3.11.11, PyTorch 2.2.2+cpu and Linux-6.18.58.1-microsoft-standard-WSL2-x86_64-with-glibc2.36.

Exact seed-0 weights, FP32, four heads and eight channels per head, native features, inference mode and zero dropout. Existing preprocessing removes input self-loops and appends one per node. Case validation checks exact EID and node permutations, endpoint multiplicity, graph data, restored outputs, restored attention and all per-destination attention sums. Maximum absolute output error 1.86e-08, attention error 5.96e-08, normalization error 4.72e-07. Timing copies matched direct outputs and attention bitwise. Tolerances were atol 1e−5 and rtol 1e−4.

Degree sorts use total incoming plus outgoing degree and original-ID ties. RCMK uses SciPy's reverse Cuthill–McKee on the symmetric structural adjacency. BFS uses the same undirected adjacency with ascending original-ID roots and neighbors, including every disconnected component. Edge grouping uses stable original-EID ties. Both random controls use NumPy default_rng with recorded explicit seeds.

The study covers three graphs on one CPU with two one-factor sweeps. Conclusions apply to the fixed official kernels and graph-format policy used here. Evidence for retaining a choice refers to changing one axis at a time. These measurements cannot establish that node and edge ordering effects compose or are additive. Forward returns the output in the graph's current node order. Restoring canonical IDs for validation is outside forward timing and would cost extra for callers requiring it.

[Raw timings](RESULTS.json) · [Permutation maps](PERMUTATIONS.json) · [Correctness and paired statistics](ANALYSIS.json)
