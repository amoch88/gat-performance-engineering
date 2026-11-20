#!/bin/sh
# Run from the repository root on Linux, or keep /app as cwd in the compiler image.
set -eu
src=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
out=$(mktemp -d "${TMPDIR:-/tmp}/cs598ape-qa.XXXXXX")
b="$src/benchmarks"
milestones="comprehensive_sparse_formats sparse_format_validation ordering_baselines ordering_baselines/extended loop_ordering"

run() {
    printf '\n+'
    printf ' %s' "$@"
    printf '\n'
    "$@"
}

run uv run --frozen --no-sync python "$b/gat.py" --smoke-test
for m in $milestones; do
    run uv run --frozen --no-sync python "$b/$m/tests.py"
done

for m in $milestones; do
    runner=report.py
    if [ "$m" = ordering_baselines/extended ]; then runner=report_extended.py; fi
    run uv run --frozen --no-sync python "$b/$m/$runner" --output "$out/$m"
    for f in REPORT.md ANALYSIS.json; do
        run cmp "$b/$m/$f" "$out/$m/$f"
    done
done
run cmp "$b/comprehensive_sparse_formats/NA_EXPLANATIONS.md" "$out/comprehensive_sparse_formats/NA_EXPLANATIONS.md"

if [ "${1:-}" = --real-data ]; then
    # A two-process correctness/protocol check, not replacement paper measurements.
    run uv run --frozen --no-sync python "$b/loop_ordering/acquire_loops.py" \
        --datasets cora --rounds 2 --warmup 1 --repetitions 2 \
        --data-dir "$src/results/datasets" --output "$out/cora-loop-smoke.json"
    run uv run --frozen --no-sync python "$b/loop_ordering/report.py" \
        --input "$out/cora-loop-smoke.json" --output "$out/cora-loop-report"
fi
printf '\nAll requested checks passed. Temporary QA reports: %s\n' "$out"
