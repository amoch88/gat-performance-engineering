### 4.1 Original one-parameter-at-a-time measurements

These are the original Milestone 2 forward-only blocks (20 timed repetitions in one process per dataset/sweep). SD describes repetitions within that block, **not** uncertainty across independent trials. Speedup is the separately measured matching control median divided by the candidate median. The original-option repeat block is not forcibly set to one and is not cherry-picked against the separate control.

Edge sweep: original nodes, all formats. Node sweep: source-grouped edges for every value, all formats. Both use native input width, four heads and eight output features/head. Node restoration is checked but **excluded** from these timings.

| Dataset | Axis | Option | Median ms | Mean ms | Within-block SD ms | Control/case |
| --- | --- | --- | --- | --- | --- | --- |
| citeseer | edge_ordering | original | 7.807906 | 7.792543 | 0.248887 | 0.9901× |
| citeseer | edge_ordering | src | 7.784482 | 7.913373 | 0.399787 | 0.9931× |
| citeseer | edge_ordering | dst | 7.879187 | 8.007051 | 0.360588 | 0.9812× |
| citeseer | node_ordering | original | 7.688945 | 7.712537 | 0.129663 | 1.0289× |
| citeseer | node_ordering | rcmk | 7.756571 | 7.804621 | 0.198367 | 1.0199× |
| citeseer | node_ordering | degree | 7.554538 | 7.647828 | 0.229115 | 1.0472× |
| cora | edge_ordering | original | 3.858080 | 3.893681 | 0.117951 | 1.0323× |
| cora | edge_ordering | src | 3.866215 | 3.896296 | 0.180009 | 1.0302× |
| cora | edge_ordering | dst | 3.820015 | 3.845404 | 0.104873 | 1.0426× |
| cora | node_ordering | original | 3.854922 | 3.865095 | 0.069535 | 1.0560× |
| cora | node_ordering | rcmk | 3.955896 | 3.957695 | 0.165879 | 1.0291× |
| cora | node_ordering | degree | 3.766123 | 3.819310 | 0.186516 | 1.0809× |
| pubmed | edge_ordering | original | 19.372352 | 20.683112 | 5.585460 | 0.9768× |
| pubmed | edge_ordering | src | 19.787533 | 20.032991 | 0.781632 | 0.9563× |
| pubmed | edge_ordering | dst | 17.734171 | 17.701262 | 0.619148 | 1.0670× |
| pubmed | node_ordering | original | 19.862437 | 20.774339 | 3.475244 | 0.9793× |
| pubmed | node_ordering | rcmk | 17.348395 | 17.291668 | 0.494477 | 1.1212× |
| pubmed | node_ordering | degree | 18.634942 | 18.708171 | 0.498901 | 1.0438× |

Fastest *candidate blocks* (not a statement of statistical superiority):

| Dataset | Axis | Fastest block | Control/case |
| --- | --- | --- | --- |
| citeseer | edge_ordering | src | 0.9931× |
| citeseer | node_ordering | degree | 1.0472× |
| cora | edge_ordering | dst | 1.0426× |
| cora | node_ordering | degree | 1.0809× |
| pubmed | edge_ordering | dst | 1.0670× |
| pubmed | node_ordering | rcmk | 1.1212× |
| reddit | edge_ordering | not measured | — |
| reddit | node_ordering | not measured | — |

Sparse single-format cases failed on Cora/Citeseer/PubMed; there is no numerical format ranking from Milestone 2. All Reddit node/edge/format sweeps were memory-skipped. Its one-head result is a different workload and cannot fill the four-head evidence gap.

### 4.2 Restricted existing-data slices: other factors fixed

These tables re-read ONLY one-axis rows from the existing six-round Milestone 3 data. They are post-hoc corroboration, not a new benchmark or joint-winner argument. For each round, relative times first use the recorded bracketing controls; the axis-reference normalized time is then divided by the option normalized time. Intervals are unadjusted Student-t 95% intervals of the mean log ratio over **six process rounds**, exponentiated. The six round medians provide the between-process SD. They do not turn 120 within-process repetitions into 120 independent trials. Comparisons are exploratory, selected after inspecting these datasets and not multiple-comparison corrected.

**edge: node=original, format=all**

| Dataset | Option | Forward ms | Combined ms | Between-round SD ms | Combined reference/case [95% CI] |
| --- | --- | --- | --- | --- | --- |
| cora | original | 4.0159 | 4.0169 | 0.0814 | 1.0000 [1.0000, 1.0000] |
| cora | src | 4.0547 | 4.0558 | 0.1561 | 0.9959 [0.9383, 1.0570] |
| cora | dst | 3.9945 | 3.9956 | 0.0952 | 1.0094 [0.9807, 1.0389] |
| citeseer | original | 8.1526 | 8.1538 | 0.1017 | 1.0000 [1.0000, 1.0000] |
| citeseer | src | 8.2367 | 8.2377 | 0.1266 | 0.9936 [0.9800, 1.0074] |
| citeseer | dst | 8.2352 | 8.2362 | 0.1927 | 0.9960 [0.9878, 1.0042] |
| pubmed | original | 19.1330 | 19.1660 | 0.1548 | 1.0000 [1.0000, 1.0000] |
| pubmed | src | 19.6679 | 19.7011 | 0.5082 | 0.9718 [0.9430, 1.0014] |
| pubmed | dst | 17.5223 | 17.5634 | 0.1835 | 1.0884 [1.0643, 1.1130] |

**node: edge=original EIDs, format=all**

| Dataset | Option | Forward ms | Combined ms | Between-round SD ms | Combined reference/case [95% CI] |
| --- | --- | --- | --- | --- | --- |
| cora | original | 4.0159 | 4.0169 | 0.0814 | 1.0000 [1.0000, 1.0000] |
| cora | rcmk | 4.0105 | 4.0602 | 0.0793 | 0.9906 [0.9581, 1.0242] |
| cora | degree | 3.9717 | 4.0218 | 0.0686 | 1.0032 [0.9652, 1.0427] |
| citeseer | original | 8.1526 | 8.1538 | 0.1017 | 1.0000 [1.0000, 1.0000] |
| citeseer | rcmk | 8.1506 | 8.2114 | 0.1702 | 0.9973 [0.9733, 1.0219] |
| citeseer | degree | 7.9117 | 7.9720 | 0.1669 | 1.0198 [1.0038, 1.0361] |
| pubmed | original | 19.1330 | 19.1660 | 0.1548 | 1.0000 [1.0000, 1.0000] |
| pubmed | rcmk | 18.7533 | 19.1550 | 0.1953 | 0.9947 [0.9782, 1.0114] |
| pubmed | degree | 18.7977 | 19.2295 | 0.2810 | 0.9905 [0.9742, 1.0071] |

**node: edge=src grouped, format=all**

| Dataset | Option | Forward ms | Combined ms | Between-round SD ms | Combined reference/case [95% CI] |
| --- | --- | --- | --- | --- | --- |
| cora | original | 4.0547 | 4.0558 | 0.1561 | 1.0000 [1.0000, 1.0000] |
| cora | rcmk | 3.9830 | 4.0330 | 0.0774 | 1.0013 [0.9545, 1.0505] |
| cora | degree | 4.0113 | 4.0637 | 0.0933 | 1.0050 [0.9567, 1.0557] |
| citeseer | original | 8.2367 | 8.2377 | 0.1266 | 1.0000 [1.0000, 1.0000] |
| citeseer | rcmk | 8.1491 | 8.2104 | 0.2696 | 1.0103 [0.9899, 1.0311] |
| citeseer | degree | 7.8066 | 7.8653 | 0.0750 | 1.0388 [1.0222, 1.0557] |
| pubmed | original | 19.6679 | 19.7011 | 0.5082 | 1.0000 [1.0000, 1.0000] |
| pubmed | rcmk | 17.3293 | 17.7443 | 0.2425 | 1.1005 [1.0731, 1.1287] |
| pubmed | degree | 18.9693 | 19.4285 | 0.3481 | 1.0204 [0.9974, 1.0440] |

**format: node=original, edge=original**

| Dataset | Option | Forward ms | Combined ms | Between-round SD ms | Combined reference/case [95% CI] |
| --- | --- | --- | --- | --- | --- |
| cora | all | 4.0159 | 4.0169 | 0.0814 | 1.0000 [1.0000, 1.0000] |
| cora | coo+csc | 4.0176 | 4.0186 | 0.1386 | 1.0020 [0.9597, 1.0462] |
| citeseer | all | 8.1526 | 8.1538 | 0.1017 | 1.0000 [1.0000, 1.0000] |
| citeseer | coo+csc | 8.2729 | 8.2739 | 0.2739 | 0.9893 [0.9723, 1.0066] |
| pubmed | all | 19.1330 | 19.1660 | 0.1548 | 1.0000 [1.0000, 1.0000] |
| pubmed | coo+csc | 19.1507 | 19.1903 | 0.5593 | 0.9891 [0.9568, 1.0225] |

Reference/case=1 for the reference by construction, not an interval claiming zero timing noise. Combined time includes restoring original node order; forward excludes restoration. The node/src slice matches Milestone 2's edge-policy context; the node/original-EID slice does not. Do not pool these two contexts or compare their absolute times as repeated measurements of a single treatment.
