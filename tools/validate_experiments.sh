#!/bin/sh
# Run from the repository root on Linux, or keep /app as cwd in the compiler image.
set -eu
src=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
out=$(mktemp -d "${TMPDIR:-/tmp}/gat-experiment-qa.XXXXXX")
b="$src/benchmarks"
experiments="comprehensive_sparse_formats sparse_format_validation ordering_baselines ordering_baselines/extended loop_ordering"

if [ "$#" -gt 1 ] || { [ "$#" -eq 1 ] && [ "$1" != --real-data ]; }; then
    printf 'Usage: sh %s [--real-data]\n' "$0" >&2
    exit 2
fi

run() {
    printf '\n+'
    printf ' %s' "$@"
    printf '\n'
    "$@"
}

run uv run --frozen --no-sync python "$b/gat.py" --smoke-test
for m in $experiments; do
    run uv run --frozen --no-sync python "$b/$m/tests.py"
done

for m in $experiments; do
    runner=report.py
    if [ "$m" = ordering_baselines/extended ]; then runner=report_extended.py; fi
    run uv run --frozen --no-sync python "$b/$m/$runner" --output "$out/$m"
    for f in REPORT.md ANALYSIS.json; do
        run cmp "$b/$m/$f" "$out/$m/$f"
    done
done
run cmp "$b/comprehensive_sparse_formats/NA_EXPLANATIONS.md" "$out/comprehensive_sparse_formats/NA_EXPLANATIONS.md"

if [ "${1:-}" = --real-data ]; then
    # A two-process correctness/protocol check, separate from archived measurements.
    run uv run --frozen --no-sync python "$b/loop_ordering/acquire_loops.py" \
        --datasets cora --rounds 2 --warmup 1 --repetitions 2 \
        --data-dir "$src/results/datasets" --output "$out/cora-loop-smoke.json"
    run uv run --frozen --no-sync python "$b/loop_ordering/report.py" \
        --input "$out/cora-loop-smoke.json" --output "$out/cora-loop-report"
fi
run uv run --frozen --no-sync python "$src/tools/build_experiment_evidence.py" --check
printf '\nAll requested checks passed. Temporary QA reports: %s\n' "$out"
