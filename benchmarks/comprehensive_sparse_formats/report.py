"""Audit the raw blocks and generate the Milestone 8 tables."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
from statistics import median
from itertools import combinations

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m7_report",HERE.parent/"native_sparse_formats/report.py")
m7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m7)
SIZES = (2,4,8,16,32,64)
FAMILIES = dict(bcsr="Blocked CSR",bcsc="Blocked CSC",csb="CSB",
                dense="Block Dense",sell="SELL")
NATIVE = m7.NATIVE
FORMATS = ["coo","csr","csc"] + [f"{kind}{size}" for kind in FAMILIES for size in SIZES]
NAMES = {"cora":"Cora","citeseer":"CiteSeer","pubmed":"PubMed"}
STAGES = dict(projection="Projection",terms="Attention Terms (L/R)",
              score="Attention Scores",softmax="Edge Softmax",aggregation="Aggregation",forward="Total Forward")
FALLBACK = dict(score="coo",softmax="csc",aggregation="csc")


def label(name):
    if name == "stock":
        return "Official imported DGL GATConv"
    if name in FORMATS[:3]:
        return f"Native DGL {name.upper()} (hybrid)"
    for prefix,public in FAMILIES.items():
        if name.startswith(prefix):
            size = name[len(prefix):]
            return f"{public} {size}" if prefix=="sell" else f"{public} {size}×{size}"
    raise ValueError(name)


def audit(raw):
    assert raw["status"]=="complete"
    for name,digest in raw["sources"].items():
        content = (HERE.parent/name).read_bytes()
        lf = content.replace(b"\r\n",b"\n")
        # Historical files can be checked out with autocrlf on Windows.
        assert digest in {hashlib.sha256(x).hexdigest()
                          for x in (content,lf,lf.replace(b"\n",b"\r\n"))}, name
    settings = raw["settings"]
    assert (settings["heads"],settings["channels"],settings["threads"])==(4,8,1)
    environments = {(row["cpu"],row["environment"]["cpu_model"],
                     row["environment"]["python"],row["environment"]["torch"])
                    for rounds in raw["datasets"].values() for row in rounds}
    assert len(environments)==1
    counts = dict(forward=0,stage=0,construction=0)
    data,storage,errors,comparisons = {},{},{},[]
    for dataset,rounds in raw["datasets"].items():
        first = rounds[0]
        active = [f for f in FORMATS if f not in first["guards"]]
        assert len(rounds)==settings["rounds"]
        assert len({r["pid"] for r in rounds})==len(rounds)
        assert set(first["storage"])==set(active)
        forward_columns = ["stock",*active]
        stage_columns = ["projection","terms",*(f"{operation}/stock" for operation in NATIVE)]
        stage_columns += [f"{op}/{f}" for f in active for op in NATIVE
                          if f not in FORMATS[:3] or f in NATIVE[op]]
        acc = {mode:{k:[] for k in cols} for mode,cols in
               (("forward",forward_columns),("stage",stage_columns),("construction",active))}
        errors[dataset] = dict(output=0.,normalized=0.,isolated=0.)
        for i,row in enumerate(rounds):
            assert row["round"]==i and row["dataset"]==dataset
            assert row["environment"]["dgl"]=="1.1.3"
            assert row["environment"]["torch"]=="2.2.2+cpu"
            for key in ("storage","compiler","padded_compiler","guards","routes","cpu","weights","edge_sha256"):
                assert row[key]==first[key], (dataset,i,key)
            assert row["graph"]["feature_sha256"]==first["graph"]["feature_sha256"]
            assert row["environment"]["cpu_model"]==first["environment"]["cpu_model"]
            assert set(row["checks"])==set(active)
            assert set(row["isolated_errors"])==set(stage_columns)-{"projection","terms"}
            for f in active:
                expected = {op:(FALLBACK[op] if f in FORMATS[:3] and f not in NATIVE[op] else f)
                            for op in NATIVE}
                assert row["routes"][f]==expected
                if f in FORMATS[:3]:
                    assert row["storage"][f]["mask"]=={"created":[f],"not created":[]}
                check = row["checks"][f]
                assert check["bitwise_repeatable"] and check["normalization_error"]<=1e-5
                errors[dataset]["output"] = max(errors[dataset]["output"],check["errors"]["output"])
                errors[dataset]["normalized"] = max(errors[dataset]["normalized"],check["normalization_error"])
            errors[dataset]["isolated"] = max(errors[dataset]["isolated"],*row["isolated_errors"].values())
            seed = settings["order_seed"]+1009*i+list(NAMES).index(dataset)
            assert row["order_seed"]==seed
            rng = random.Random(seed)
            for mode,cols in (("forward",forward_columns),("stage",stage_columns),("construction",active)):
                if mode!="construction":
                    m7.replay(rng,row[mode+"_warmups"],cols,settings["warmup"])
                repeat = settings["build_repetitions"] if mode=="construction" else settings["repetitions"]
                values = m7.replay(rng,row[mode],cols,repeat)
                for key,value in values.items():
                    acc[mode][key].append(value)
                counts[mode] += len(cols)*repeat
        data[dataset] = {mode:{key:dict(ms=median(values),round_medians_ms=values,
                                            round_range_ms=[min(values),max(values)])
                               for key,values in values_by_key.items()}
                         for mode,values_by_key in acc.items() if mode!="construction"}
        storage[dataset] = {f:dict(**first["storage"][f],build_ms=median(acc["construction"][f]),
                                   build_round_medians_ms=acc["construction"][f]) for f in active}
        for op in (*NATIVE,"forward"):
            def timings(fmt):
                return acc["forward"][fmt] if op=="forward" else acc["stage"][f"{op}/{fmt}"]

            native = ["stock",*FORMATS[:3]] if op=="forward" else ["stock",*NATIVE[op]]
            custom = [f for f in active if f not in FORMATS[:3]]
            pairs = [(f,c) for f in custom for c in native]
            pairs += [(f,c) for c,f in combinations(native,2)]
            for family in FAMILIES:
                pairs += [(f,c) for c,f in combinations([f for f in custom if f.startswith(family)],2)]
            for f,c in pairs:
                comparisons.append(dict(dataset=dataset,stage=op,format=f,control=c,
                                         **m7.m6.paired(timings(c),timings(f))))
    for op in (*NATIVE,"forward"):
        m7.m6.holm([c for c in comparisons if c["stage"]==op])
    return dict(timings=data,storage=storage,comparisons=comparisons,errors=errors,counts=counts)


def na_text(raw):
    lines = ["# N/A and hybrid pipelines","",
        "The native entries use sole-format DGL graph views. The limitations below are from "
        "[DGL 1.1.3 CPU dispatch](https://raw.githubusercontent.com/dmlc/dgl/v1.1.3/src/array/kernel.cc). "
        "An N/A stage has no timing. Each hybrid forward directly times the exact stages listed below.","",
        "| Selected format | Score | Softmax | Aggregation | Reason for N/A |",
        "|---|---|---|---|---|",
        "| COO | COO | CSC fallback | COO | CPU destination edge softmax directly requests a CSC matrix. |",
        "| CSR | CSR | CSC fallback | CSC fallback | CPU destination edge softmax requests CSC. GSpMM supports only COO and CSC. |",
        "| CSC | COO fallback | CSC | CSC | GSDDMM supports only COO and CSR. |","",
        "The fallback always uses native DGL on a prebuilt view with the same canonical EIDs. "
        "The stock row uses unmodified imported GATConv for its complete forward. Its sparse stage cells "
        "replay its public DGL calls in isolation, using the same graph formats and precomputed inputs.","",
        "## Memory guards","",
        f"The recorded per-layout limit is {raw['settings']['layout_budget']:,} bytes and the combined custom "
        f"index payload limit is {raw['settings']['resident_budget']:,} bytes. Construction also needs room "
        "for one more layout and temporary arrays. The runner checks the padded layout size before allocation.",""]
    for d,rounds in raw["datasets"].items():
        guards = rounds[0]["guards"]
        if guards:
            for f,info in guards.items():
                lines.append(f"- {NAMES[d]}, {label(f)}. {info['reason']}. All three sparse cells and total are N/A.")
        else:
            lines.append(f"- {NAMES[d]} completed all custom formats and sizes within the guard limits.")
    return "\n".join(lines)+"\n"


def render(raw,result):
    st = raw["settings"]
    lines = ["# Milestone 8 — Sparse formats for CPU GAT","",
        "Each row uses one selected storage layout for every supported sparse stage. The three native "
        "rows use fixed native fallbacks where DGL dispatch requires another format. "
        "[N/A explanations and exact routes](NA_EXPLANATIONS.md).","",
        "All cells are milliseconds and report the median of process medians. Projection and attention "
        "terms share one measured value per block across rows because the same operations and inputs "
        "are used. Sparse stages are timed separately on reference inputs. The full forward is timed directly. "
        "For SELL the size is the slice height.",
        "", "## Timing tables",""]
    all_best = []
    for d,table in result["timings"].items():
        lines += [f"### {NAMES[d]}","",
                  "| Storage Format | Projection | Attention Terms (L/R) | Attention Scores | Edge Softmax | Aggregation | Total Forward |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        stage = table["stage"]
        for f in ["stock",*FORMATS]:
            cells = [stage["projection"],stage["terms"],
                     *(stage.get(f"{op}/{f}") for op in NATIVE),table["forward"].get(f)]
            lines.append("| "+label(f)+" | "+" | ".join(f'{v["ms"]:.6f}' if v else "N/A" for v in cells)+" |")
        lines.append("")
        for op in (*NATIVE,"forward"):
            choices = (table["forward"] if op=="forward" else
                       {key.split("/")[1]:v for key,v in stage.items() if key.startswith(op+"/")})
            winner = min(choices,key=lambda f: choices[f]["ms"])
            control = "stock" if op=="forward" else min(NATIVE[op],key=lambda f:choices[f]["ms"])
            comparison = find(result["comparisons"],d,op,winner,control)
            all_best.append((d,op,winner,choices[winner]["ms"],control,comparison))
    lines += ["## Fastest observed configurations","",
        "The control is the fastest supported native stage format by observed median. Total forward "
        "uses untouched imported GATConv as its control. Speedup is the geometric mean of paired "
        "process-median ratios, control divided by candidate. Intervals are pointwise 95% Student-t "
        "intervals on log ratios. Holm correction is applied separately for each endpoint across datasets, "
        "native controls and all within-family size pairs.","",
        "| Dataset | Stage | Fastest observed | Milliseconds | Control | Speedup (95% CI) | Holm p |",
        "|---|---|---|---:|---|---|---:|"]
    for d,op,f,ms,control,comparison in all_best:
        effect = "Control"
        p = "—"
        if comparison:
            effect = f'{comparison["speedup"]:.3f}× [{comparison["ci_low"]:.3f}, {comparison["ci_high"]:.3f}]'
            p = f'{comparison["holm_p"]:.4g}'
        lines.append(f'| {NAMES[d]} | {STAGES[op]} | {label(f)} | {ms:.6f} | {label(control)} | {effect} | {p} |')
    forward_winners = [(NAMES[d],label(f)) for d,op,f,_,_,_ in all_best if op=="forward"]
    lines += ["", "The complete-forward ranks are " + ", ".join(f"{f} on {d}" for d,f in forward_winners)
              + ". The paired comparisons below test whether those observed differences justify changing formats."]
    lines += ["", "### Evidence for changing size","",
        "This table counts all size pairs within each format family across the three datasets. "
        "Direction is for the larger size. Individual pairs and their intervals are in ANALYSIS.json.","",
        "| Endpoint | Family | Larger faster | Larger slower | Inconclusive |",
        "|---|---|---:|---:|---:|"]
    for op in (*NATIVE,"forward"):
        for family,name in FAMILIES.items():
            rows = [r for r in result["comparisons"] if r["stage"]==op
                    and r["format"].startswith(family) and r["control"].startswith(family)]
            faster = sum(r["holm_p"]<.05 and r["speedup"]>1 for r in rows)
            slower = sum(r["holm_p"]<.05 and r["speedup"]<1 for r in rows)
            lines.append(f"| {STAGES[op]} | {name} | {faster} | {slower} | {len(rows)-faster-slower} |")
    lines += ["","For each fastest observed full-forward size, these are the slower sizes within "
              "the same family supported by the paired tests after correction.",""]
    for d,op,winner,_,_,_ in all_best:
        family = next((name for name in FAMILIES if winner.startswith(name)),None)
        if op!="forward" or family is None:
            continue
        others = [f for f in result["timings"][d]["forward"] if f.startswith(family) and f!=winner]
        supported = []
        for f in others:
            c = find(result["comparisons"],d,op,winner,f)
            if c["holm_p"]<.05 and c["speedup"]>1:
                supported.append(label(f))
        if supported:
            lines.append(f"- On {NAMES[d]}, {label(winner)} beat {', '.join(supported)} after correction.")
        else:
            lines.append(f"- On {NAMES[d]}, the size comparisons for {label(winner)} were inconclusive after correction.")
    lines += ["","## Storage and construction","",
        "Payload counts the stored index arrays, including masks, padding, EID and parallel-edge "
        "sidecar entries. Native indices are 64-bit and custom coordinate/EID indices are 32-bit. "
        "Shared DGL graph views, temporary build memory and runtime scratch are outside these payload "
        "counts. Native construction starts with a freshly created COO graph. Custom construction starts "
        "from canonical endpoint arrays. Forward timing includes intermediate allocation, scratch, and "
        "all executed stages. Inputs and outputs stay in canonical EID order throughout.","",
        "| Dataset | Format | Index KiB | Construction median ms |",
        "|---|---|---:|---:|"]
    for d,formats in result["storage"].items():
        for f,entry in formats.items():
            lines.append(f'| {NAMES[d]} | {label(f)} | {entry["bytes"]/1024:.1f} | {entry["build_ms"]:.6f} |')
    lines += ["", "### Padding costs", "",
              "Slots count only allocated dense tile cells or SELL lanes. The slot-to-edge ratio "
              "includes padding and uses the canonical edge count as denominator. Dense row masks skip "
              "padding during traversal. SELL scans each slice's padded lanes.",
              "", "| Dataset | Layout | Slots per edge at size 2 | Slots per edge at size 64 | Index MiB at 2 / 64 |",
              "|---|---|---:|---:|---:|"]
    for d,formats in result["storage"].items():
        edges = raw["datasets"][d][0]["graph"]["edges_with_self_loops"]
        for family in ("dense","sell"):
            small = formats.get(family+"2")
            large = formats.get(family+"64")
            slots = [f'{f["slots"]/edges:.2f}' if f else "N/A" for f in (small,large)]
            sizes = [f'{f["bytes"]/2**20:.3f}' if f else "N/A" for f in (small,large)]
            lines.append(f'| {NAMES[d]} | {FAMILIES[family]} | '+" | ".join(slots)+" | "+" / ".join(sizes)+" |")
    lines += ["","## Supported full-forward comparisons","",
              "A custom configuration qualifies below only if it beats imported GATConv and all three "
              "native hybrid rows after correction. The speedup is the smallest of those four paired estimates.","",
              "| Dataset | Qualifying configurations | Largest minimum speedup |",
              "|---|---:|---|"]
    for d in result["timings"]:
        qualified = {}
        for f in FORMATS[3:]:
            pairs = [r for r in result["comparisons"]
                     if (r["dataset"],r["stage"],r["format"])==(d,"forward",f)
                     and r["control"] in ("stock",*FORMATS[:3])]
            if len(pairs)==4 and all(r["holm_p"]<.05 and r["speedup"]>1 for r in pairs):
                qualified[f] = min(r["speedup"] for r in pairs)
        strongest = max(qualified,key=qualified.get) if qualified else None
        best = f"{label(strongest)} ({qualified[strongest]:.3f}×)" if strongest else "No supported gain"
        lines.append(f"| {NAMES[d]} | {len(qualified)} | {best} |")
    lines += ["","The comparisons estimate the net effect of using a format across its sparse stages. "
              "The isolated timings help locate bottlenecks, while the forward comparison also captures "
              "allocation and data-access effects across stages. Use the paired size contrasts to evaluate "
              "whether a size advantage extends across workloads."]
    lines += ["","### Interpreting the new padded formats",""]
    for d,table in result["timings"].items():
        legal = [f for f in FORMATS if f.startswith("dense") and f in table["forward"]]
        if not legal:
            continue
        best = min(legal,key=lambda f:table["forward"][f]["ms"])
        best_result = find(result["comparisons"],d,"forward",best,"stock")
        if "dense64" in legal:
            largest = find(result["comparisons"],d,"forward","dense64","stock")
            lines.append(f'- {NAMES[d]}. The fastest observed Block Dense configuration was '
                         f'{label(best)} at {table["forward"][best]["ms"]:.3f} ms '
                         f'({best_result["speedup"]:.3f}× vs stock, Holm p={best_result["holm_p"]:.3g}). '
                         f'At 64×64 it took {table["forward"]["dense64"]["ms"]:.3f} ms '
                         f'({largest["speedup"]:.3f}×, Holm p={largest["holm_p"]:.3g}).')
    lines += ["","Dense tiles trade a large reserved EID grid and a scan of occupancy words for implicit "
              "coordinates. Larger tiles can increase both costs even as tile count falls. SELL trades "
              "padded lanes for contiguous slice storage. A high-degree destination raises the slice width "
              "for all its neighbors in row order. The within-family contrasts quantify the size effect "
              "for these specific traversals. The timing and capacity measurements do not isolate cache misses."]
    env = next(iter(raw["datasets"].values()))[0]
    lines += ["","## Methods and validation","",
        f"{env['environment']['cpu_model']}, logical CPU {env['cpu']}, one thread, DGL 1.1.3 and "
        f"PyTorch {env['environment']['torch']}. {st['rounds']} fresh serial processes per dataset, "
        f"{st['warmup']} warmup and {st['repetitions']} randomized matched blocks per timed phase, "
        f"plus {st['build_repetitions']} construction samples per round. All observations were retained.","",
        f"The raw log contains {result['counts']['forward']:,} measured forwards, "
        f"{result['counts']['stage']:,} measured stage calls and "
        f"{result['counts']['construction']:,} construction observations.","",
        "FP32 inference with no dropout, 4 heads and 8 channels per head. The loader uses original "
        "features and ordering, removes existing self-loops and appends one per node. Canonical endpoint, "
        "feature and seed-0 weight hashes match Milestone 5. Every custom layout reconstructs every "
        "EID including parallel edges. All intermediates and final output were compared against the "
        "stock-backed reference with atol=1e−5 and rtol=1e−4. Every active destination/head sum was checked.","",
        "Dense tiles allocate every coordinate in each nonempty square tile. Row masks gate EID reads "
        "and a sidecar retains parallel edges. SELL uses lane-interleaved destination rows and scans "
        "the padded slots with an invalid-EID guard. The two use stable softmax across every incoming "
        "edge of a destination. Blocked sparse traversal reuses Milestones 6 and 7.","",
        "Stock sparse stages replay apply_edges, edge_softmax and update_all on a local graph with "
        "preassigned tensors. Native-format variants use GSDDMM, edge_softmax and GSpMM on restricted "
        "views. The score boundary includes LeakyReLU. Custom score fuses the activation, while DGL "
        "calls it separately. Index width, API overhead, and activation fusion vary across families, "
        "so comparisons estimate implementation speed rather than a causal effect of layout alone.",""]
    for d,error in result["errors"].items():
        lines.append(f'- {NAMES[d]} maximum absolute errors. Final output {error["output"]:.3g}, '
                     f'isolated sparse stage {error["isolated"]:.3g}, destination normalization {error["normalized"]:.3g}.')
    lines += ["","[Raw observations and provenance](RESULTS.json) · [Paired estimates and variability](ANALYSIS.json)",""]
    return "\n".join(lines)


def find(rows,d,op,candidate,control):
    if candidate==control:
        return None
    for row in rows:
        if (row["dataset"],row["stage"])!=(d,op):
            continue
        if (row["format"],row["control"])==(candidate,control):
            return row
        if (row["format"],row["control"])==(control,candidate):
            return dict(**{k:v for k,v in row.items() if k not in ("speedup","ci_low","ci_high")},
                        speedup=1/row["speedup"],ci_low=1/row["ci_high"],ci_high=1/row["ci_low"])
    raise ValueError((d,op,candidate,control))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=HERE/"RESULTS.json")
    parser.add_argument("--output",type=Path,default=HERE)
    args = parser.parse_args()
    raw = json.loads(args.input.read_text())
    result = audit(raw)
    result["raw_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    args.output.mkdir(parents=True,exist_ok=True)
    files = {"REPORT.md":render(raw,result),"NA_EXPLANATIONS.md":na_text(raw),
             "ANALYSIS.json":json.dumps(result,allow_nan=False,separators=(",",":"))+"\n"}
    for name,content in files.items():
        (args.output/name).write_text(content,encoding="utf-8")
    print(json.dumps(result["counts"]),flush=True)


if __name__ == "__main__":
    main()
