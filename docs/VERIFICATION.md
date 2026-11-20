# Verification and Artifact Integrity

## Snapshot integrity

Source repository: [amoch88/dgl-performance-model](https://github.com/amoch88/dgl-performance-model)
at `9659684473a0531aa06e40f614e1fbfbbcfc5b76`. An LF-preserving Git
archive seeded this independent repository. The initial import matched
all 161 tracked blobs. Raw observations, saved maps, archived
acquisitions, analyses, acquisition code, and historical report
generators stay unchanged. The earlier synthetic benchmark and
dataset-access instructions remain in the
[archived baseline notes](../RESEARCH_README.md).
The root import diff reports inherited trailing whitespace in
archived SVG plots. It was preserved to keep the research files
byte-identical. Whitespace checks apply to the files changed in this
repository.

Only the tracked snapshot was exported. The original workstation
cache, dependency environments, dataset archives, compiled libraries,
Git credentials, local configuration and temporary results were not
included in the tracked tree. The local **extracted** DGL citation cache
used for smoke validation is separate and ignored. Artifact review
checked file names/extensions, sizes, known credential markers
(filenames-only scan), archived contents and Git state. It is not a
general audit proving absence of every form of secret. The largest
files are committed research data, e.g. the extension's approximately
8.6 MB permutation map, not downloaded datasets.

## Executed validation

The checkout was mounted as `/workspace`, without mounting or invoking
scripts from the research worktree. The image's working directory remained
`/app` for the frozen installed environment. All performance claims
come from archived data. No full campaign was rerun.

```sh
docker build -t gat-performance:cpu .
docker run --rm --network none gat-performance:cpu --smoke-test
docker build --build-context dgl-performance-model:cpu=docker-image://gat-performance:cpu \
  -f benchmarks/tuning_axes/Dockerfile -t gat-performance:compiler .
docker run --rm --network none -v "$PWD:/workspace" --entrypoint sh \
  gat-performance:compiler /workspace/tools/validate_experiments.sh --real-data
```

The CPU and compiler images built successfully and the direct CPU smoke
passed. The CPU build may emit the pinned Dockerfile's fixed-platform
lint warning. The pinned Docker run executes these checks:

| Command inside compiler image | Result |
|---|---|
| `uv run --frozen --no-sync python /workspace/benchmarks/gat.py --smoke-test` | Passed, stock CPU numerical smoke. |
| `uv run --frozen --no-sync python /workspace/benchmarks/comprehensive_sparse_formats/tests.py` | 4 passed, native fallback, sparse/padded execution and correctness. |
| `uv run --frozen --no-sync python /workspace/benchmarks/sparse_format_validation/tests.py` | 1 passed, sparse timing and profiling diagnostics. |
| `uv run --frozen --no-sync python /workspace/benchmarks/ordering_baselines/tests.py` | 2 passed, canonical workload and matched allocation/order controls. |
| `uv run --frozen --no-sync python /workspace/benchmarks/ordering_baselines/extended/tests.py` | 5 passed, including METIS, lexicographic/Morton sorting, parallel edges, and analysis. The installed DGL METIS helper emitted a ResourceWarning about its own /proc/self/status file. |
| `uv run --frozen --no-sync python /workspace/benchmarks/loop_ordering/tests.py` | 4 passed, native specializations, sparse edge cases, and statistics. |

The wrapper regenerated `REPORT.md` and `ANALYSIS.json` for the sparse
storage evaluation, backend and timing validation, matched-allocation
and extended reordering, and loop-nest optimization, using
`uv run --frozen --no-sync python /workspace/benchmarks/<experiment>/report.py --output <temporary-directory>`
(the extension uses `report_extended.py`). A `cmp` against each
archived artifact passed byte for byte, including the storage
evaluation's `NA_EXPLANATIONS.md`. The storage audit replayed 24,480 measured forwards,
72,000 measured isolated-stage calls and 3,960 construction
observations without reacquiring them. These audits report actual
bytes from independently generated temporary outputs, not comparisons of formatting
approximations.

The data smoke ran
`uv run --frozen --no-sync python /workspace/benchmarks/loop_ordering/acquire_loops.py --datasets cora --rounds 2 --warmup 1 --repetitions 2 --data-dir /workspace/results/datasets --output <temporary-directory>/cora-loop-smoke.json`.
Both fresh workers validated and timed every loop variant against stock
DGL, and the same report tool read the small JSON successfully.
The data were loaded from the independent local extracted cache.
This does not verify a fresh download on an arbitrary host.

The shared DGL baseline suite and all seven `--verify-only` entry
points also completed. Run these in the same compiler container:

```sh
b=/workspace/benchmarks/dgl_baseline
uv run --frozen --no-sync python "$b/tests.py"
for experiment in baseline input_features output_features attention_heads sparse_format edge_ordering node_ordering; do
  uv run --frozen --no-sync python "$b/$experiment.py" --verify-only --warmup 1 --repetitions 2
done
```

The deliberately restricted single-format DGL probes reported their
expected unsupported paths. The validation wrapper rejected an unknown
option, and the loop acquisition rejected zero rounds without writing
an output file.

## Native memory check

Valgrind **3.19.0** Memcheck ran the standalone fixture using read-only
source. This was a disposable Debian compiler container with Valgrind
installed temporarily, not a new project dependency. The command:

```sh
docker run --rm -v "$PWD:/workspace:ro" --entrypoint sh gat-performance:compiler -ec '
  apt-get -qq update
  DEBIAN_FRONTEND=noninteractive apt-get -qq install -y --no-install-recommends valgrind
  cd /workspace
  g++ -std=c++17 -O2 -g -ffp-contract=off -fno-loop-interchange -fno-loop-unroll-and-jam \
    tools/native_memory_smoke.cpp benchmarks/sparse_formats/kernels.cpp \
    benchmarks/comprehensive_sparse_formats/padded.cpp benchmarks/loop_ordering/kernels.cpp \
    -o /tmp/native_memory_smoke
  valgrind --version
  valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all \
    --errors-for-leak-kinds=all --error-exitcode=99 /tmp/native_memory_smoke
'
```

Result: fixture passed all **11 loop orders, 6 sparse traversals and
2 padded traversals**. The graph contains two parallel records, self-loops,
an empty incoming row and masked/sentinel lanes. The fixture checks
values against explicit reference reductions and preserves canonical
EIDs. The recorded Memcheck summary:

```text
HEAP SUMMARY:
    in use at exit: 0 bytes in 0 blocks
  total heap usage: 1 allocs, 1 frees, 4,096 bytes allocated
All heap blocks were freed -- no leaks are possible
ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0)
```

This is coverage for a meaningful small case using the same native
sources, **not** a Memcheck run over a full dataset, Python/Torch/DGL
process, every tile/slice size, or the acquired production ELF.
Full reacquisition and leak-checking the dependency runtimes were not
performed.

## Generated evidence

Run `python tools/build_experiment_evidence.py` to regenerate the evidence
tables from archived JSON. Its `--check` mode compares the generated
content byte for byte and resolves local links in the active Markdown
guides. The validation wrapper resolved all 106 local links. A comparison
with the prior evidence documents confirmed that the numbers in all 25
contrasts and 15 generated tables were unchanged. The generator never
rewrites raw measurements or archived reports.

## Retained terminology and provenance

Active guides use descriptive experiment names. The terminology audit
retains older labels in these places:

| Location | Why the wording stays |
|---|---|
| `RESEARCH_README.md` | Byte-identical baseline documentation, including unique historical setup, dataset access, and configuration notes. It is no longer in the main navigation. |
| Archived benchmark reports and parameter evidence under `benchmarks/` | Titles, headings, and experiment labels are historical output, including the exact text checked by report replay. |
| Frozen benchmark, analysis, and replay sources under `benchmarks/` | Legacy comments, CLI descriptions, result labels, and paths are retained with acquisition source fingerprints and historical rendering behavior. |
| Raw results, analyses, and provenance JSON under `benchmarks/` | Recorded keys such as `milestone8.citeseer`, source digests, and measurement labels are part of the archived evidence. |
| `benchmarks/dgl_baseline/DGL-LICENSE` and source attribution | Legal wording and the upstream research identity are preserved. |

The active summaries link to these archives with descriptive labels.
Retaining the archived wording keeps the published measurement evidence
reproducible without presenting it as new experiment output.
