# Graph Reordering Experiments: Extended Sweeps

Read the [extension report](REPORT.md) alongside the [matched-allocation reordering
report](../REPORT.md). The extension does not pool either historical run
with new data. Existing cases use the same maps and seed-0 weights. The
workload, stage profiler, graph formats, feature allocation control, and
single-thread setup are unchanged. The randomized timing schedule uses a
new seed. See [RESULTS.json](RESULTS.json), [PERMUTATIONS.json](PERMUTATIONS.json),
[METIS.json](METIS.json), and [ANALYSIS.json](ANALYSIS.json) for raw blocks,
new-to-original maps, frozen partition data, and the statistical replay.

The fixed native METIS preparation requests 32 k-way cut partitions once per
dataset. It saves partition assignments and the stable partition-then-node-ID
ordering before any timing worker starts. Workers check the saved plan
against dataset fingerprints, replay it, and preserve canonical edge IDs.
Partition generation is one-shot; the five construction observations each
time replay planning, rebuilding, and COO/CSR/CSC materialization separately.

The lexicographic edge policies sort by (primary endpoint, secondary
endpoint, original EID); the original grouped policies only sort on the
primary endpoint, preserving original EID ties. Morton uses a 64-bit
unsigned key with source bits in even positions and destination bits in odd
positions. It preserves parallel edges by breaking key ties with original
EID. All edge cases share the unchanged input feature tensor.

## Reproduction

Use the frozen CPU environment and existing DGL citation datasets, as in
[the sparse kernel reference](../../sparse_formats/README.md). Run from the repository root.
CPU 0 is the logical CPU used for the committed measurements; it must be
available in your affinity mask for the commands below.

~~~sh
ext=benchmarks/ordering_baselines/extended
out=results/graph-reordering-extended
uv run --frozen python benchmarks/gat.py --smoke-test
uv run --frozen python "$ext/tests.py"
uv run --frozen python "$ext/benchmark_extended.py" --cpu 0 +  --data-dir results/datasets --metis "$ext/METIS.json" +  --output "$out/RESULTS.json" --maps "$out/PERMUTATIONS.json"
uv run --frozen python "$ext/report_extended.py" --input "$out/RESULTS.json" +  --maps "$out/PERMUTATIONS.json" --metis "$ext/METIS.json" --output "$out"
~~~

To generate another native METIS plan explicitly, run
`benchmark_extended.py --prepare-metis --cpu 0 --data-dir results/datasets --metis results/graph-reordering-extended/METIS.json`
through the same interpreter before acquiring timings, then pass that path
to the benchmark and report. The output plan path must be fresh. This
regenerates a partition assignment rather than promising an identical
native result across builds. Use the checked-in plan to replay its exact
ordering. The historical report comparison checks matching CPU and runtime;
use `--no-historical` for a fresh acquisition on a different host.

The acquisition defaults are eight processes, ten warmup blocks, thirty
measured blocks, and five separate construction observations. For a quick
smoke use distinct fresh files and `--rounds 2 --warmup 1 --repetitions 2 --build-repetitions 2`.
To check both committed reports from their raw data:

~~~sh
uv run --frozen python "$ext/report_extended.py" --output results/graph-reordering-extended-replay
uv run --frozen python benchmarks/ordering_baselines/report.py --output results/graph-reordering-archived-replay
for f in REPORT.md ANALYSIS.json; do
  cmp "$ext/$f" "results/graph-reordering-extended-replay/$f"
  cmp "benchmarks/ordering_baselines/$f" "results/graph-reordering-archived-replay/$f"
done
~~~

The primary forward test families compare each candidate with Original
within its sweep. We correct separately for node and edge sweeps across all
three datasets and all planned candidates. Stages have separate families,
as in the original report. A separate nine-comparison forward family
compares the new edge policies with Destination-grouped. A qualifying gain
requires Holm p < .05 and a pointwise 95% lower speedup bound above 1.01×.
Random policies are always controls, never tuning choices.
