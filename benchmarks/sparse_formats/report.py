"""Verify raw acquisition and generate tables, paired contrasts, and SVG plots."""
import argparse
import csv
import hashlib
import json
from math import ceil, exp, isfinite, log, sqrt
import random
from pathlib import Path
from statistics import median, mean, stdev
from xml.sax.saxutils import escape

from scipy.stats import t

HERE = Path(__file__).resolve().parent
OPS = ("score", "softmax", "aggregation")
FORMATS = ("coo", "csr", "csc", "bcsr16", "bcsr32", "bcsr64",
           "bcsc16", "bcsc32", "bcsc64", "csb16", "csb32", "csb64")


def samples(blocks, expected, count):
    assert len(blocks) == count
    values = {name: [] for name in expected}
    for block in blocks:
        assert len(block) == len(expected) and {k for k, _ in block} == set(expected)
        for key, value in block:
            assert value > 0 and isfinite(value)
            values[key].append(value)
    return values


def paired(control, candidate):
    logs = [log(a/b) for a, b in zip(control, candidate, strict=True)]
    center = mean(logs)
    se = stdev(logs) / sqrt(len(logs))
    radius = float(t.ppf(.975, len(logs)-1)) * se
    p = float(2*t.sf(abs(center/se), len(logs)-1)) if se else float(center == 0)
    return dict(speedup=exp(center), ci_low=exp(center-radius), ci_high=exp(center+radius), p=p)


def holm(rows):
    top = 0
    for i, row in enumerate(sorted(rows, key=lambda r: r["p"])):
        top = max(top, min(1, (len(rows)-i)*row["p"]))
        row["holm_p"] = top


def summarize(raw):
    assert raw["status"] == "complete"
    opts = raw["settings"]
    for name, digest in raw["sources"].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest() == digest
    timing, storage, comparisons, oracle = [], [], [], {}
    observations = 0
    errors = {"output": 0., "normalization": 0., "isolated": 0.}
    keys = [f"{op}/{fmt}" for op in OPS for fmt in FORMATS]
    for dataset, rounds in raw["datasets"].items():
        assert len(rounds) == opts["rounds"]
        assert [r["round"] for r in rounds] == list(range(opts["rounds"]))
        base = rounds[0]
        conversions = []
        data = {"forward": [], "kernel": []}
        for r in rounds:
            assert r["milestone5_identity_verified"] and r["dataset"] == dataset
            for field in ("edge_sha256", "weight_sha256", "raw_dataset_sha256", "storage", "compiler", "cpu"):
                assert r[field] == base[field]
            assert r["graph"]["feature_sha256"] == base["graph"]["feature_sha256"]
            assert set(r["correctness"]) == set(keys)
            for check in r["correctness"].values():
                assert check["bitwise_repeatable"]
                errors["output"] = max(errors["output"], check["stage_max_abs_error"]["output"])
                errors["isolated"] = max(errors["isolated"], check["isolated_max_abs_error"])
                errors["normalization"] = max(errors["normalization"], check["destination_normalization_max_abs_error"])
            conversions.append(samples(r["conversion"], FORMATS, 5))
            for mode in data:
                names = keys + (["stock"] if mode == "forward" else [])
                samples(r[mode+"_warmups"], names, opts["warmup"])
                data[mode].append(samples(r[mode], names, opts["repetitions"]))
                observations += opts["repetitions"]*len(names)
            rng = random.Random(opts["order_seed"] + 1009*r["round"] + ("cora", "citeseer", "pubmed").index(dataset))
            for field in ("forward_warmups", "forward", "kernel_warmups", "kernel", "conversion"):
                names = list(FORMATS) if field == "conversion" else keys + (["stock"] if field.startswith("forward") else [])
                for block in r[field]:
                    order = names.copy()
                    rng.shuffle(order)
                    assert order == [name for name, _ in block]
        oracle[dataset] = median([median(r["stock"]) for r in data["forward"]])
        for fmt in FORMATS:
            storage.append(dict(dataset=dataset, format=fmt, **base["storage"][fmt],
                conversion_ms=median([median(r[fmt]) for r in conversions])))
        for mode, rounds_data in data.items():
            per_round = {name: [median(r[name]) for r in rounds_data] for name in keys}
            for name, times in per_round.items():
                op, fmt = name.split("/")
                ratios = paired(per_round[op+"/coo"], times)
                timing.append(dict(dataset=dataset, mode=mode, kernel=op, format=fmt,
                    ms=median(times), round_min_ms=min(times), round_max_ms=max(times),
                    **{k: v for k, v in ratios.items() if k != "p"}))
                for ctrl in ("coo", "csr", "csc"):
                    if ctrl == fmt or (fmt in ("coo", "csr", "csc") and ("coo", "csr", "csc").index(fmt) < ("coo", "csr", "csc").index(ctrl)):
                        continue
                    comparisons.append(dict(dataset=dataset, mode=mode, kernel=op, format=fmt,
                        control=ctrl, **paired(per_round[op+"/"+ctrl], times)))
    # One family per endpoint; all distinct contrasts against conventional layouts.
    for mode in ("forward", "kernel"):
        holm([r for r in comparisons if r["mode"] == mode])
    return timing, storage, comparisons, oracle, observations, errors


def csv_file(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plots(out, rows):
    for dataset in dict.fromkeys(r["dataset"] for r in rows):
        selected = {(r["mode"], r["kernel"], r["format"]): r for r in rows if r["dataset"] == dataset}
        colors = ("#226184", "#8c4e35", "#36715b")
        s = ['<svg xmlns="http://www.w3.org/2000/svg" width="990" height="690" viewBox="0 0 990 690">',
             '<rect width="990" height="690" fill="white"/>',
             '<g font-family="Arial,sans-serif" fill="#172735" font-size="13">',
             f'<text x="24" y="30" font-size="20" font-weight="bold">{escape(dataset.title())}: sparse storage speedup</text>',
             '<text x="24" y="54">Paired geometric mean speedup vs COO (1×); whiskers show pointwise 95% intervals.</text>']
        for panel, mode in enumerate(("kernel", "forward")):
            top = 102 + panel*286
            height = 210
            maximum = max(1.5, ceil(max(r["ci_high"] for r in rows if r["mode"] == mode)*2.1)/2)
            y = lambda val: top + height - val/maximum*height
            s.append(f'<text x="24" y="{top-13}" font-weight="bold">{"Isolated kernel" if mode=="kernel" else "Complete forward"}</text>')
            for tick in range(int(maximum*2)+1):
                val = tick/2
                yy = y(val)
                color = "#556674" if val == 1 else "#dce3e8"
                s += [f'<path d="M70 {yy:.2f}H965" stroke="{color}"/>',
                      f'<text x="59" y="{yy+4:.2f}" text-anchor="end">{val:g}×</text>']
            for i, fmt in enumerate(FORMATS):
                x = 100 + 73*i
                s.append(f'<text x="{x}" y="{top+height+20}" text-anchor="middle">{fmt.upper()}</text>')
                for j, op in enumerate(OPS):
                    r = selected[mode,op,fmt]
                    px = x + (j-1)*17
                    yy = y(r["speedup"])
                    s += [f'<rect x="{px-6}" y="{yy:.2f}" width="12" height="{y(0)-yy:.2f}" fill="{colors[j]}"/>',
                          f'<path d="M{px} {y(r["ci_low"]):.2f}V{y(r["ci_high"]):.2f}" stroke="#142331"/>']
        for j, op in enumerate(OPS):
            x = 24 + 175*j
            s.append(f'<rect x="{x}" y="663" width="13" height="13" fill="{colors[j]}"/><text x="{x+20}" y="674">{op.title()}</text>')
        (out/f"{dataset}.svg").write_text("\n".join(s+["</g></svg>"])+"\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=HERE/"RESULTS.json")
    p.add_argument("--output", type=Path, default=HERE)
    args = p.parse_args()
    raw = json.loads(args.input.read_text())
    timing, storage, comparisons, oracle, count, errors = summarize(raw)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    csv_file(out/"timings.csv", timing)
    csv_file(out/"storage.csv", [{k:v for k,v in r.items() if k!="arrays"} for r in storage])
    csv_file(out/"contrasts.csv", comparisons)
    plots(out, timing)
    text = ["# Milestone 6: sparse storage and blocking for CPU GAT",
        f"All {len(timing)//2} dataset/kernel/format substitutions passed validation. "
        f"The raw log contains {count:,} measured isolated-kernel and complete-forward observations, plus warmups and conversion samples.",
        "", "## Fastest observed formats", "",
        "Times are medians of process medians (ms). Speedup is the paired geometric mean vs the same-kernel COO case. "
        "A rank is descriptive; it does not establish superiority to every alternative.", "",
        "| Dataset | Substituted kernel | Fastest isolated (ms; speedup) | Fastest full forward (ms; speedup) |",
        "|---|---|---|---|"]
    for dataset in oracle:
        for op in OPS:
            winners = [min((r for r in timing if (r["dataset"],r["kernel"],r["mode"]) == (dataset,op,mode)),key=lambda r:r["ms"])
                       for mode in ("kernel","forward")]
            text.append(f"| {dataset} | {op} | " + " | ".join(f'{w["format"].upper()} ({w["ms"]:.4f}; {w["speedup"]:.3f}×)' for w in winners)+" |")
    text += ["", "## All formats", "", "Cells report isolated / forward latency in ms. See [paired contrasts](contrasts.csv) for CIs and Holm p-values against COO and ordinary CSR/CSC.", "",
             "| Dataset | Format | Score | Softmax | Aggregation | Payload KiB | Build ms |",
             "|---|---|---:|---:|---:|---:|---:|"]
    table = {(r["dataset"],r["format"],r["mode"],r["kernel"]):r["ms"] for r in timing}
    for r in storage:
        d,f = r["dataset"],r["format"]
        text.append(f'| {d} | {f.upper()} | '+" | ".join(f"{table[d,f,'kernel',op]:.4f} / {table[d,f,'forward',op]:.4f}" for op in OPS)
                    + f' | {r["bytes"]/1024:.1f} | {r["conversion_ms"]:.4f} |')
    text += ["", "Stock DGL GATConv reference times (ms): " + ", ".join(f"{k} {v:.4f}" for k,v in oracle.items())+".",
             "The stock reference is a different execution stack, not the COO denominator.",
             "", "## Speedup plots", ""]
    for d in oracle:
        text += [f"![{d} kernel and forward speedups]({d}.svg)"]
    text += ["", "## Blocking and tuning implications", "",
             "Keep the format axis stage-specific. A fast isolated traversal does not by itself justify a full-forward choice. "
             "Use the paired forward contrasts and construction/storage costs before promoting a block size.", "",
             "| Endpoint | Format family | Holm-supported gains vs COO | Holm-supported losses vs COO |",
             "|---|---|---:|---:|"]
    for mode in ("kernel","forward"):
        for family in ("csr","csc","bcsr","bcsc","csb"):
            group = [r for r in comparisons if r["mode"]==mode and r["control"]=="coo" and r["format"].startswith(family)]
            gain = sum(r["holm_p"]<.05 and r["speedup"]>1 for r in group)
            loss = sum(r["holm_p"]<.05 and r["speedup"]<1 for r in group)
            text.append(f"| {mode} | {family.upper()} | {gain}/{len(group)} | {loss}/{len(group)} |")
    text += ["", "### Blocked vs ordinary parent format", "",
             "Counts span the three datasets, three kernels and three block sizes per family.", "",
             "| Endpoint | Comparison | Supported gains | Supported losses |",
             "|---|---|---:|---:|"]
    for mode in ("kernel","forward"):
        for family, parent in (("bcsr","csr"),("bcsc","csc")):
            group = [r for r in comparisons if r["mode"]==mode and r["control"]==parent and r["format"].startswith(family)]
            gain = sum(r["holm_p"]<.05 and r["speedup"]>1 for r in group)
            loss = sum(r["holm_p"]<.05 and r["speedup"]<1 for r in group)
            text.append(f"| {mode} | {family.upper()} vs {parent.upper()} | {gain}/{len(group)} | {loss}/{len(group)} |")
    text += ["", "### Best observed blocked layout and its cost", "",
             "These are descriptive ranks. The paired interval compares each layout with COO, not with other block sizes.", "",
             "| Dataset | Kernel | Best blocked | Kernel ms | Speedup vs COO (95% CI) |",
             "|---|---|---|---:|---|"]
    for dataset in oracle:
        for op in OPS:
            b = min((r for r in timing if (r["dataset"],r["kernel"],r["mode"]) == (dataset,op,"kernel")
                     and r["format"] not in FORMATS[:3]), key=lambda r: r["ms"])
            text.append(f'| {dataset} | {op} | {b["format"].upper()} | {b["ms"]:.4f} | '
                        f'{b["speedup"]:.3f} [{b["ci_low"]:.3f}, {b["ci_high"]:.3f}] |')
    text += ["", "| Dataset | Blocked CSR/CSC payload vs COO | CSB payload vs COO |",
             "|---|---:|---:|"]
    for dataset in oracle:
        sizes = {r["format"]: r["bytes"] for r in storage if r["dataset"] == dataset}
        def span(prefixes):
            v = [b/sizes["coo"] for f,b in sizes.items() if f.startswith(prefixes)]
            return f"{min(v):.2f}–{max(v):.2f}×"
        text.append(f'| {dataset} | {span(("bcsr", "bcsc"))} | {span(("csb",))} |')
    gains = [r for r in comparisons if r["mode"]=="forward" and r["control"]=="coo"
             and r["holm_p"] < .05 and r["speedup"] > 1]
    text += ["", "### Candidates for future tuning", "",
             ("Full-forward gains vs COO supported after correction: "
              + "; ".join(f'{r["dataset"]}/{r["kernel"]}/{r["format"].upper()} ({r["speedup"]:.3f}×)'
                          for r in sorted(gains, key=lambda r: (r["dataset"],r["kernel"],r["format"]))) + "."
              if gains else "No format produced a Holm-supported full-forward gain vs COO in this run. "
              "Retain conventional controls; treat any observed winner or block-size ranking as exploratory."),
             "", "A supported difference from COO does not establish a difference between two block sizes. "
             "The experiment does not time combinations of substitutions."]
    kernel_gains = [r for r in comparisons if r["mode"]=="kernel" and r["control"]=="coo"
                    and r["holm_p"] < .05 and r["speedup"] > 1]
    if kernel_gains:
        text += ["", "Supported isolated gains vs COO: " + "; ".join(
            f'{r["dataset"]}/{r["kernel"]}/{r["format"].upper()} ({r["speedup"]:.3f}×)'
            for r in kernel_gains) + ". These justify retaining conventional formats as kernel-level candidates, "
            "but the forward evidence does not justify promoting a blocked default."]
    env = next(iter(raw["datasets"].values()))[0]
    text += ["", "## Measurement and correctness", "",
             f"CPU: {env['environment']['cpu_model']}, pinned logical CPU {env['cpu']}. "
             "DGL 1.1.3, Torch 2.2.2+cpu, four heads, eight channels/head, FP32, eval/inference mode. "
             "Graph EIDs, feature bytes and seed-0 parameter bytes match Milestone 5. "
             "The prior policy removes all self-loops and appends exactly one per node. No reordering or coalescing.",
             "", f"{raw['settings']['rounds']} fresh serial process rounds per dataset; {raw['settings']['warmup']} warmup and "
             f"{raw['settings']['repetitions']} measured randomized matched blocks per endpoint per process. "
             "Pointwise Student-t intervals use paired log ratios of round medians, not individual repetitions. "
             f"Holm adjustment is separate by endpoint across {len(comparisons)//2} contrasts "
             "against COO, CSR, and CSC. No sample exclusions.",
             "", "Only the selected sparse operation changes. Score timing includes LeakyReLU but not node-term construction. "
             "All formats use canonical EID outputs and the same arithmetic. Softmax uses three full traversals and FP32 "
             "destination max/sum scratch. All scratch/output allocations are timed. Graph construction and conversion are outside forward timing; no runtime format conversion is needed.",
             "", "Build time is the median of round medians for five independently randomized builds from canonical COO arrays. "
             "Payload bytes count sparse index arrays only, not allocator overhead, original graph, dense features or softmax scratch. "
             "Blocked CSR/CSC pay for local pointer rows even when a nonempty tile is very sparse; CSB pays for both local coordinates per edge. "
             "Neither format pays for dense padding or empty tiles.",
             "", f"Maximum absolute error: final output {errors['output']:.3g}; isolated intermediate {errors['isolated']:.3g}; "
             f"destination softmax sum {errors['normalization']:.3g}. Full output and attention reference are checked against untouched stock GATConv.",
             "", "One host and one kernel implementation family: no claim that a storage format is universally faster, "
             "or that these rankings transfer to a destination-specialized softmax, multiple threads, or trained models.",
             "", "Raw samples, environment, hashes and execution order: [RESULTS.json](RESULTS.json). "
             "Machine-readable summaries: [timings.csv](timings.csv), [storage.csv](storage.csv), [contrasts.csv](contrasts.csv)."]
    (out/"REPORT.md").write_text("\n".join(text)+"\n", encoding="utf-8")
    (out/"VALIDATION.json").write_text(json.dumps(dict(measured_samples=count, max_absolute_errors=errors,
        substitutions=len(timing)//2, raw_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        checks="full randomized blocks, data/weight/CPU/storage identity, source hashes, all correctness records"), indent=2)+"\n")
    print("\n".join(text[:20]))


if __name__ == "__main__":
    main()
