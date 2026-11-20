# Submission validation and handoff

## Snapshot integrity

Source repository: [amoch88/dgl-performance-model](https://github.com/amoch88/dgl-performance-model)
at `9659684473a0531aa06e40f614e1fbfbbcfc5b76`, branch `main`. The
research worktree was clean. An LF-preserving Git archive seeded an
independent Git repository with no inherited remote or private history.
All 161 original tracked blobs matched before course-only additions;
the only replaced original path is the top-level README, which remains
byte-identical as [RESEARCH_README.md](../RESEARCH_README.md). Raw
observations, saved maps, archived acquisition, analyses, benchmarking
source and the legacy report generators are unchanged.
The root import diff reports inherited trailing whitespace in
archived SVG plots. It was preserved to keep the research files
byte-identical. The whitespace check scoped to the new README,
course documents, and QA tools passed.

Only the tracked snapshot was exported. The original workstation
cache, dependency environments, dataset archives, compiled libraries,
Git credentials, local configuration and temporary results were not
copied into Git. The locally staged **extracted** DGL citation cache
used for smoke validation is separate and ignored. Artifact review
checked file names/extensions, sizes, known credential markers
(filenames-only scan), archived contents and Git state. It is not a
general audit proving absence of every form of secret. The largest
files are committed research data, e.g. the extension's approximately
8.6 MB permutation map, not downloaded datasets.

## Executed validation

The checkout was mounted as `/submission`, without mounting or invoking
scripts from the research worktree. The image's working directory remained
`/app` for the frozen installed environment. All performance claims in
the handoff come from historical data. No full campaign was rerun.

```sh
docker build -t cs598ape-gat:cpu .
docker run --rm --network none cs598ape-gat:cpu --smoke-test
docker build --build-context dgl-performance-model:cpu=docker-image://cs598ape-gat:cpu \
  -f benchmarks/tuning_axes/Dockerfile -t cs598ape-gat:compiler .
docker run --rm --network none -v "$PWD:/submission" --entrypoint sh \
  cs598ape-gat:compiler /submission/tools/validate_submission.sh --real-data
```

The CPU and compiler images built successfully and the direct CPU smoke
passed. The CPU build emitted the preserved Dockerfile's fixed-platform
lint warning. An initial draft validation wrapper had a shell line
continuation error after the baseline, M8, M9, and matched M10 tests;
after fixing the wrapper, the entire command above returned 0.
The pinned Docker run executed these tests:

| Command inside compiler image | Result |
|---|---|
| `uv run --frozen --no-sync python /submission/benchmarks/gat.py --smoke-test` | Passed, stock CPU numerical smoke. |
| `uv run --frozen --no-sync python /submission/benchmarks/comprehensive_sparse_formats/tests.py` | 4 passed, native fallback, sparse/padded execution and correctness. |
| `uv run --frozen --no-sync python /submission/benchmarks/sparse_format_validation/tests.py` | 1 passed, historical M9 profiling and diagnostics. |
| `uv run --frozen --no-sync python /submission/benchmarks/ordering_baselines/tests.py` | 2 passed, canonical workload and matched allocation/order controls. |
| `uv run --frozen --no-sync python /submission/benchmarks/ordering_baselines/extended/tests.py` | 5 passed, including METIS, lexicographic/Morton sorting, parallel edges, and analysis. The installed DGL METIS helper emitted a ResourceWarning about its own /proc/self/status file. |
| `uv run --frozen --no-sync python /submission/benchmarks/loop_ordering/tests.py` | 4 passed, native specializations, sparse edge cases, and statistics. |

The wrapper regenerated `REPORT.md` and `ANALYSIS.json` for M8, M9,
matched M10, the M10 extension, and M11, using
`uv run --frozen --no-sync python /submission/benchmarks/<milestone>/report.py --output <temporary-directory>`
(the extension uses `report_extended.py`). A `cmp` against each
original passed byte for byte. M8's `NA_EXPLANATIONS.md` also
matched exactly. The M8 audit replayed 24,480 measured forwards,
72,000 measured isolated-stage calls and 3,960 construction
observations without reacquiring them. These audits report actual
bytes from independently generated temporary outputs, not comparisons of formatting
approximations.

The data smoke ran
`uv run --frozen --no-sync python /submission/benchmarks/loop_ordering/acquire_loops.py --datasets cora --rounds 2 --warmup 1 --repetitions 2 --data-dir /submission/results/datasets --output <temporary-directory>/cora-loop-smoke.json`.
Both fresh workers validated and timed every loop variant against stock
DGL, and the same report tool read the small JSON successfully.
The data were loaded from the independent local extracted cache.
This does not verify a fresh download on an arbitrary host.

## Native memory check

Valgrind **3.19.0** Memcheck ran the standalone fixture using read-only
source. This was a disposable Debian compiler container with Valgrind
installed temporarily, not a new project dependency. The command:

```sh
docker run --rm -v "$PWD:/submission:ro" --entrypoint sh cs598ape-gat:compiler -ec '
  apt-get -qq update
  DEBIAN_FRONTEND=noninteractive apt-get -qq install -y --no-install-recommends valgrind
  cd /submission
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

## Course evidence and publication

The course Markdown tables are generated from the preserved JSON
by `python tools/build_course_evidence.py`. Verification with
`docker run --rm --network none -v "$PWD:/submission:ro" --entrypoint python cs598ape-gat:cpu /submission/tools/build_course_evidence.py --check`
passed byte-for-byte checks and resolved all 77 local links in the
course documents. No original measurement or report is rewritten by this tool.

GitHub CLI's stored login for `amoch88` returned HTTP 401 during
preflight. The local repository is independent and has no research
remote. To publish it privately, run the following **from this course
checkout**, after authenticating as the intended owner:

```sh
gh auth login --hostname github.com --web
gh api user --jq .login
gh repo create amoch88/cs598ape-gat-performance-engineering --private --source . --remote origin --push
gh repo view amoch88/cs598ape-gat-performance-engineering --json nameWithOwner,visibility,url
git ls-remote origin refs/heads/main
```

Verify that the reported visibility is private and the remote SHA
matches `git rev-parse HEAD`. If the private repository was created
already, verify its identity and visibility before adding its URL as
this checkout's remote and pushing. Grant course staff repository
access through GitHub before submitting a private URL.
