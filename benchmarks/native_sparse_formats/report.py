"""Audit matched blocks and generate the Milestone 7 report."""
import argparse
import hashlib
import importlib.util
import json
from math import exp, log, sqrt
from pathlib import Path
import random
from statistics import mean, median, stdev
from scipy.stats import t

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m6_report", HERE.parent/"sparse_formats/report.py")
m6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m6)
NATIVE = {"score": ("coo","csr"), "softmax": ("csc",), "aggregation": ("coo","csc")}
BLOCKS = [f"{f}{b}" for f in ("bcsr","bcsc","csb") for b in (2,4,8,16,32,64)]
FORMATS = ["coo","csr","csc"] + BLOCKS


def audit(raw, phase):
    assert raw["phase"] == phase and raw["status"] == "complete"
    for path,digest in raw["sources"].items():
        assert hashlib.sha256((HERE.parent/path).read_bytes()).hexdigest() == digest
    for dataset,rounds in raw["datasets"].items():
        assert len(rounds) == raw["settings"]["rounds"]
        for i,row in enumerate(rounds):
            assert row["dataset"] == dataset and row["round"] == i
            assert row["environment"]["dgl"] == "1.1.3"
            for field in ("cpu","weights","edge_sha256","graph_formats"):
                assert row[field] == rounds[0][field]
            assert row["graph"]["feature_sha256"] == rounds[0]["graph"]["feature_sha256"]
    assert raw["settings"]["threads"] == 1
    assert raw["settings"]["heads"] == 4 and raw["settings"]["channels"] == 8


def replay(rng, blocks, names, count):
    result = m6.samples(blocks,names,count)
    for block in blocks:
        order = list(names)
        rng.shuffle(order)
        assert order == [k for k,_ in block]
    return {key:median(values) for key,values in result.items()}


def interval(logs, level):
    center = mean(logs)
    half = float(t.ppf((1+level)/2,len(logs)-1))*stdev(logs)/sqrt(len(logs))
    return [exp(center-half),exp(center+half)]


def preliminary(raw):
    audit(raw,"preliminary")
    settings = raw["settings"]
    assert settings["tolerance"] == [.97,1.03]
    result = {}
    for dataset,rounds in raw["datasets"].items():
        pairs, positions = [], [[],[]]
        for r in rounds:
            for key in ("executable_class_ast_equal","same_forward_dependencies","bitwise_output","bitwise_attention"):
                assert r[key]
            rng = random.Random(r["order_seed"])
            replay(rng,r["warmups"],["official","annotated"],settings["warmup"])
            pairs.append(replay(rng,r["forward"],["official","annotated"],settings["repetitions"]))
            for block in r["forward"]:
                v = dict(block)
                positions[int(block[0][0]=="annotated")].append(log(v["annotated"]/v["official"]))
        logs = [log(p["annotated"]/p["official"]) for p in pairs]
        ci = interval(logs,.90)
        status = ("equivalent" if .97 < ci[0] <= ci[1] < 1.03 else
                  "meaningful discrepancy" if ci[1]<.97 or ci[0]>1.03 else "inconclusive")
        result[dataset] = dict(medians={k:median(p[k] for p in pairs) for k in pairs[0]},
            ranges={k:[min(p[k] for p in pairs),max(p[k] for p in pairs)] for k in pairs[0]},
            ratio=exp(mean(logs)), ci90=ci, ci95=interval(logs,.95), status=status,
            round_ratios=[exp(v) for v in logs], order_ratios=[exp(median(v)) if v else None for v in positions])
    return result


def sparse(raw):
    audit(raw,"sparse")
    settings = raw["settings"]
    results, comparisons, memory, stock = [], [], [], {}
    error = 0.
    for dataset, rounds in raw["datasets"].items():
        keys = [f"{op}/{fmt}" for op in NATIVE for fmt in list(NATIVE[op])+BLOCKS]
        unsupported = {f"{op}/{f}" for op in NATIVE for f in FORMATS[:3] if f not in NATIVE[op]}
        data = {k:[] for k in ("forward","kernel","conversion")}
        for r in rounds:
            assert r["storage"] == rounds[0]["storage"]
            assert set(r["unsupported"]) == unsupported and set(r["correctness"]) == set(keys)
            for fmt in FORMATS[:3]:
                assert r["storage"][fmt]["mask"] == {"created":[fmt],"not created":[]}
            for v in r["correctness"].values():
                assert v["bitwise_repeatable"] and v["destination_normalization_max_abs_error"] <= 1e-5
                error = max(error,v["stage_max_abs_error"]["output"])
            rng = random.Random(r["order_seed"])
            for mode in ("forward","kernel"):
                names = keys + (["stock"] if mode=="forward" else [])
                replay(rng,r[mode+"_warmups"],names,settings["warmup"])
                data[mode].append(replay(rng,r[mode],names,settings["repetitions"]))
            data["conversion"].append(replay(rng,r["conversion"],FORMATS,5))
        stock[dataset] = median(r["stock"] for r in data["forward"])
        for f in FORMATS:
            memory.append(dict(dataset=dataset,format=f,**rounds[0]["storage"][f],
                               build_ms=median(r[f] for r in data["conversion"])))
        for mode in ("forward","kernel"):
            times = {key:[r[key] for r in data[mode]] for key in keys}
            for key,values in times.items():
                op,f = key.split("/")
                results.append(dict(dataset=dataset,mode=mode,kernel=op,format=f,
                                    ms=median(values),round_range=[min(values),max(values)]))
                controls = [c for c in NATIVE[op] if c != f and
                            (f not in NATIVE[op] or NATIVE[op].index(c)<NATIVE[op].index(f))]
                family = next((b for b in ("bcsr","bcsc","csb") if f.startswith(b)),None)
                if family and int(f[len(family):]) <= 8:
                    controls += [family+str(b) for b in (16,32,64)]
                for c in controls:
                    comparisons.append(dict(dataset=dataset,mode=mode,kernel=op,format=f,control=c,
                        **m6.paired(times[f"{op}/{c}"],values)))
    for mode in ("forward","kernel"):
        m6.holm([r for r in comparisons if r["mode"]==mode])
    return results,comparisons,memory,stock,error


def plot(path,dataset,rows,mode):
    title = "isolated kernel latency" if mode=="kernel" else "complete forward latency"
    s = ['<svg xmlns="http://www.w3.org/2000/svg" width="1090" height="520" viewBox="0 0 1090 520">',
         '<rect width="1090" height="520" fill="white"/><g font-family="Arial,sans-serif" font-size="12" fill="#20303d">',
         f'<text x="25" y="30" font-size="20" font-weight="bold">{dataset.title()}: {title}</text>',
         '<text x="25" y="52">Median of process medians, ms. Whiskers: process-median range. Blue: native; brown: blocked.</text>']
    for j,op in enumerate(NATIVE):
        selected = {r["format"]:r for r in rows if (r["dataset"],r["kernel"],r["mode"])==(dataset,op,mode)}
        base = 180+j*158
        maximum = max(r["round_range"][1] for r in selected.values())*1.05
        y = lambda v: base-90*v/maximum
        s += [f'<text x="25" y="{base-108}" font-weight="bold">{op.title()}</text>',
              f'<text x="25" y="{y(maximum)+4:.2f}">{maximum:.3f}</text>',
              f'<path d="M65 {base}H1070" stroke="#aab5bd"/><text x="38" y="{base+4}">0</text>']
        for i,f in enumerate(FORMATS):
            x = 86+47*i
            s.append(f'<text x="{x}" y="{base+16}" text-anchor="middle" font-size="10">{f.upper()}</text>')
            if f not in selected:
                s.append(f'<text x="{x}" y="{base-6}" text-anchor="middle">N/A</text>')
                continue
            r = selected[f]
            top = y(r["ms"])
            color = "#225b7b" if f in FORMATS[:3] else "#a6603f"
            s += [f'<rect x="{x-10}" y="{top:.2f}" width="20" height="{base-top:.2f}" fill="{color}"/>',
                  f'<path d="M{x} {y(r["round_range"][0]):.2f}V{y(r["round_range"][1]):.2f}" stroke="#17242c"/>']
    path.write_text("\n".join(s+["</g></svg>"])+"\n",encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preliminary",type=Path,default=HERE/"PRELIMINARY.json")
    p.add_argument("--input",type=Path)
    p.add_argument("--output",type=Path,default=HERE)
    a = p.parse_args()
    pre_raw = json.loads(a.preliminary.read_text())
    pre = preliminary(pre_raw)
    print(json.dumps(pre,indent=2))
    if a.input is None:
        return
    raw = json.loads(a.input.read_text())
    rows,comparisons,memory,stock,error = sparse(raw)
    for d,rounds in raw["datasets"].items():
        for r in rounds:
            for k in ("cpu","weights","edge_sha256","graph_formats"):
                assert r[k] == pre_raw["datasets"][d][0][k]
            for k in ("dgl","torch","python","cpu_model","runtime_thread_controls"):
                assert r["environment"][k] == pre_raw["datasets"][d][0]["environment"][k]
    a.output.mkdir(parents=True,exist_ok=True)
    count = sum(len(b) for rounds in raw["datasets"].values() for r in rounds
                for mode in ("kernel","forward") for b in r[mode])
    pre_count = sum(len(b) for rounds in pre_raw["datasets"].values() for r in rounds for b in r["forward"])
    (a.output/"ANALYSIS.json").write_text(json.dumps(dict(preliminary=pre,timings=rows,contrasts=comparisons,
        storage=memory,stock_ms=stock,max_output_error=error, sparse_timed_samples=count,
        preliminary_timed_samples=pre_count, raw_sha256={
            "preliminary":hashlib.sha256(a.preliminary.read_bytes()).hexdigest(),
            "sparse":hashlib.sha256(a.input.read_bytes()).hexdigest()}),indent=2)+"\n",encoding="utf-8")
    lines = ["# Milestone 7: native DGL versus blocked sparse kernels","",
             f"{len(rows)//2} supported dataset/kernel/format substitutions validated; "
             f"{count:,} sparse timing observations and {pre_count:,} preliminary forward observations. "
             "Warmups and conversion measurements are retained separately.","",
             "## Preliminary: annotated versus imported GATConv","",
             "The full class AST and resolved forward dependencies matched. Output and EID-ordered attention "
             "were bitwise equal on the same pre-materialized graph, features and exact state_dict. "
             "The imported official layer remains the whole-forward reference.","",
             "Latency is the median of process medians, in ms. Ratio is annotated/imported latency. "
             "The ratio is the geometric mean of paired process ratios, not the ratio of the displayed medians. "
             "Equivalence requires the paired 90% t interval on the log ratio to lie wholly within the "
             "predefined [0.97, 1.03] band (TOST at 5% per dataset).","",
             "| Dataset | Imported ms (round range) | Annotated ms (round range) | Ratio | 95% CI | 90% CI | Decision |",
             "|---|---|---|---:|---|---|---|"]
    for d,r in pre.items():
        cells = [f'{r["medians"][k]:.5f} ({r["ranges"][k][0]:.5f}–{r["ranges"][k][1]:.5f})' for k in ("official","annotated")]
        lines.append(f'| {d} | '+" | ".join(cells)+f' | {r["ratio"]:.5f} | '
                     + " | ".join(f'[{r[k][0]:.5f}, {r[k][1]:.5f}]' for k in ("ci95","ci90"))+f' | {r["status"]} |')
    lines += ["", "Order diagnostic: " + "; ".join(
        f'{d} annotated/imported {r["order_ratios"][0]:.4f} when imported ran first, {r["order_ratios"][1]:.4f} when annotated ran first'
        for d,r in pre.items())+". Raw round ratios are in ANALYSIS.json.",
        "", "## Native dispatch and benchmark contract", "",
        "DGL 1.1.3 native GSDDMM supports COO/CSR; destination edge softmax uses CSC; GSpMM supports COO/CSC. "
        "Verified against [pinned DGL source](https://raw.githubusercontent.com/dmlc/dgl/v1.1.3/src/array/kernel.cc) "
        "and successful sole-format views with exact EID checks and unchanged masks. This is not a C++ stack trace.",
        "", "Unsupported: score/CSC, softmax/COO, softmax/CSR, aggregation/CSR. No fallback or timings for these cells.",
        "", "One sparse stage changes at a time; other stages use Milestone 5's fixed DGL pipeline. "
        "Score includes LeakyReLU: native calls DGL then PyTorch; custom fuses activation inside C++. "
        "Native views use 64-bit indices and blocked layouts retain Milestone 6's 32-bit representation. "
        "Contrasts compare implementation stacks, not just blocking. Tile size is unrelated to attention head tiling.",
        "", "All datasets retain exact Milestone 5 features, seed-0 weights and canonical EIDs. "
        "No coalescing or node reorder. Self-loop policy: remove existing loops, append one per node. "
        "Four heads, eight channels/head, FP32, no dropout, inference mode, one pinned CPU.",
        "", "## Observed winners", "",
        "| Dataset | Stage | Fastest isolated (ms) | Fastest substituted forward (ms) |",
        "|---|---|---|---|"]
    for d in stock:
        for op in NATIVE:
            best = [min((r for r in rows if (r["dataset"],r["kernel"],r["mode"])==(d,op,m)),key=lambda r:r["ms"])
                    for m in ("kernel","forward")]
            lines.append(f'| {d} | {op} | '+" | ".join(f'{r["format"].upper()} ({r["ms"]:.5f})' for r in best)+" |")
    lines += ["","Untouched imported GATConv in the sparse sessions (ms): "+", ".join(f"{d} {v:.5f}" for d,v in stock.items())+".",
              "", "## Every format and tile size", "",
              "Cells give isolated / substituted forward ms. ANALYSIS.json includes variability and all paired speedups and CIs.",
              "", "| Dataset | Format | Score | Softmax | Aggregation | Index KiB | Build ms |",
              "|---|---|---:|---:|---:|---:|---:|"]
    idx = {(r["dataset"],r["format"],r["kernel"],r["mode"]):r["ms"] for r in rows}
    for r in memory:
        d,f = r["dataset"],r["format"]
        cells = [" / ".join(f'{idx[d,f,op,m]:.5f}' for m in ("kernel","forward"))
                 if (d,f,op,"kernel") in idx else "Unsupported" for op in NATIVE]
        lines.append(f'| {d} | {f.upper()} | '+" | ".join(cells)+f' | {r["bytes"]/1024:.1f} | {r["build_ms"]:.5f} |')
    lines += ["","## Paired comparisons","",
              "Speedup = control/candidate. Pointwise 95% t intervals use paired log ratios of process medians. "
              "Holm correction is separate per endpoint across native and small-vs-large within-family contrasts.","",
              "| Endpoint | Contrast family | Supported gains | Supported losses | Total |",
              "|---|---|---:|---:|---:|"]
    for mode in ("kernel","forward"):
        for label,ctrl in (("blocked vs native",FORMATS[:3]),("small vs large",BLOCKS)):
            items = [r for r in comparisons if r["mode"]==mode and r["format"] in BLOCKS and r["control"] in ctrl]
            gain = sum(r["holm_p"]<.05 and r["speedup"]>1 for r in items)
            loss = sum(r["holm_p"]<.05 and r["speedup"]<1 for r in items)
            lines.append(f"| {mode} | {label} | {gain} | {loss} | {len(items)} |")
    lines += ["", f"Each endpoint's Holm family contains {len(comparisons)//2} tests, including native-vs-native controls."]
    lines += ["", "### Small versus large tiles within the same family", "",
              "| Endpoint | Small tile | Comparisons vs 16/32/64 | Supported gains | Supported losses |",
              "|---|---:|---:|---:|---:|"]
    for mode in ("kernel","forward"):
        for b in (2,4,8):
            vals = [r for r in comparisons if r["mode"]==mode and r["control"] in BLOCKS
                    and r["format"] in {f+str(b) for f in ("bcsr","bcsc","csb")}]
            gain = sum(r["holm_p"]<.05 and r["speedup"]>1 for r in vals)
            loss = sum(r["holm_p"]<.05 and r["speedup"]<1 for r in vals)
            lines.append(f"| {mode} | {b} | {len(vals)} | {gain} | {loss} |")
    lines += ["","### Best blocked versus best native by observed median","",
              "Descriptive winner selection; p-values come from the full corrected family.","",
              "| Dataset | Stage | Endpoint | Best blocked | Best native | Speedup (95% CI) | Holm p |",
              "|---|---|---|---|---|---|---:|"]
    blocked_winners = {}
    for d in stock:
        for op in NATIVE:
            for mode in ("kernel","forward"):
                opts = [r for r in rows if (r["dataset"],r["kernel"],r["mode"])==(d,op,mode)]
                b = min((r for r in opts if r["format"] in BLOCKS),key=lambda r:r["ms"])
                n = min((r for r in opts if r["format"] in NATIVE[op]),key=lambda r:r["ms"])
                blocked_winners[d,op,mode] = b["format"].upper()
                c = next(r for r in comparisons if (r["dataset"],r["kernel"],r["mode"],r["format"],r["control"])==
                         (d,op,mode,b["format"],n["format"]))
                lines.append(f'| {d} | {op} | {mode} | {b["format"].upper()} | {n["format"].upper()} | '
                             f'{c["speedup"]:.3f} [{c["ci_low"]:.3f}, {c["ci_high"]:.3f}] | {c["holm_p"]:.4g} |')
    lines += ["", "Observed blocked winners vary by endpoint and workload: " + "; ".join(
        f"{d} ({', '.join(blocked_winners[d,op,'kernel'] for op in NATIVE)} isolated; "
        f"{', '.join(blocked_winners[d,op,'forward'] for op in NATIVE)} forward)"
        for d in stock)+". Within each list the order is score, softmax, aggregation. "
        "These ranks do not test pairwise differences between all sizes."]
    lines += ["","## Storage and measurement notes","",
              "Small tiles pay more directory entries; large compressed tiles may pay for empty local rows. "
              "CSB uses per-edge coordinates instead of local row pointers. Construction is one-time from canonical COO "
              "arrays; native restriction/materialization starts with a fresh COO graph already created.",
              "The 2/4/8 builder uses the existing COO input validator, including temporary int32 endpoint arrays. "
              "Its measured construction cost includes that implementation overhead; those temporary arrays are not retained payload.",
              "", "All runtime allocations and scratch are timed. Canonical EID output avoids separate runtime reordering. "
              "Isolated inputs are prepared before timing; complete forwards are measured independently.",
              "", f"Maximum validated final-output absolute error: {error:.3g}. All variants were checked against "
              "the stock-backed intermediate reference, destination sums and exact topology/EID reconstruction.",
              "", f"Preliminary: {pre_raw['settings']['rounds']} processes per dataset, "
              f"{pre_raw['settings']['warmup']} warmup and {pre_raw['settings']['repetitions']} measured paired blocks. "
              f"Sparse: {raw['settings']['rounds']} separate processes per dataset, {raw['settings']['warmup']} warmup "
              f"and {raw['settings']['repetitions']} measured matched blocks per endpoint, five builds per format per process. "
              "All raw observations are retained, and block orders are replayed during report validation.",
              "", "One CPU and one model size; no extrapolation to multithreading, graph reordering, or joint substitutions.",
              "", "Raw: [preliminary](PRELIMINARY.json), [sparse](RESULTS.json). "
              "All timings, uncertainty and storage: [ANALYSIS.json](ANALYSIS.json).",""]
    env = next(iter(raw["datasets"].values()))[0]
    lines += [f"Host: {env['environment']['cpu_model']}, logical CPU {env['cpu']}; "
              f"Python {env['environment']['python']}, Torch {env['environment']['torch']}, DGL 1.1.3.","",
              "## Measured blocking overhead","",
              "| Dataset | Family | Tiles at 2 / 64 | KiB at 2 / 64 | Build ms at 2 / 64 |",
              "|---|---|---:|---:|---:|"]
    for d in stock:
        for family in ("bcsr","bcsc","csb"):
            small,large = [next(r for r in memory if r["dataset"]==d and r["format"]==family+str(size))
                   for size in (2,64)]
            lines.append(f'| {d} | {family.upper()} | {small["nonempty_tiles"]:,} / {large["nonempty_tiles"]:,} | '
                         f'{small["bytes"]/1024:.1f} / {large["bytes"]/1024:.1f} | {small["build_ms"]:.4f} / {large["build_ms"]:.4f} |')
    lines += ["", "Tile counts expose the directory/traversal cost of small blocks. For compressed tiles, "
              "the local pointer budget grows with tile size times nonempty tile count, not just tile count. "
              "For CSB it is primarily the number of tiles and stored edges. Those measured costs explain "
              "why reducing tile count does not necessarily reduce compressed payload. Cache effects are "
              "plausible but not measured with hardware counters."]
    lines += ["","## Tuning implications",""]
    for op in NATIVE:
        candidates = [r for r in comparisons if r["kernel"]==op and r["mode"]=="forward"
                      and r["format"] in BLOCKS and r["control"] in NATIVE[op]]
        eligible = []
        for d in stock:
            for f in BLOCKS:
                vals = [r for r in candidates if (r["dataset"],r["format"])==(d,f)]
                if len(vals)==len(NATIVE[op]) and all(r["holm_p"]<.05 and r["speedup"]>1 for r in vals):
                    eligible.append((d,f,min(r["speedup"] for r in vals)))
        if eligible:
            winners = [max((r for r in eligible if r[0]==d),key=lambda r:r[2])
                       for d in stock if any(r[0]==d for r in eligible)]
            lines.append(f"- **{op.title()}:** {len(eligible)} dataset/format choices beat every supported native control "
                         "after correction. Strongest conservative speedup by dataset: "
                         + "; ".join(f"{d} {f.upper()} ({v:.3f}×)" for d,f,v in winners)+".")
        else:
            lines.append(f"- **{op.title()}:** no blocked variant showed a corrected forward gain against every "
                         "supported native control. Keep native defaults for this workload; do not infer equality from a null result.")
    lines += ["","Tile sizes are candidates only for the measured kernel and workload. The blocked score comparison "
              "also changes activation fusion and index width; it cannot attribute a gain to tiling alone.",""]
    lines += ["## Plots","",
              "Whiskers show process-median ranges, not confidence intervals. Paired intervals are in the tables and analysis JSON.",""]
    for d in stock:
        for mode in ("kernel","forward"):
            name = f"{d}-{mode}.svg"
            plot(a.output/name,d,rows,mode)
            lines += [f"![{d} {mode} latencies]({name})",""]
    (a.output/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")


if __name__=="__main__":
    main()
