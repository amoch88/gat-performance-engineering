"""Generate audited tables and paired estimates from Milestone 9 raw samples."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
from statistics import mean, median, stdev
from math import isfinite, sqrt

from scipy.stats import t

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m6_statistics",HERE.parent/"sparse_formats/report.py")
stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)
NATIVE = ("coo","csr","csc")
SUPPORT = dict(score=("coo","csr"),softmax=("csc",),aggregation=("coo","csc"))
FALLBACK = dict(score="coo",softmax="csc",aggregation="csc")
STAGES = ("projection","terms","score","softmax","aggregation","output")
SIZES = (2,4,8,16,32,64)
UNBLOCKED = [f"custom_{f}" for f in NATIVE]
BLOCKED = [f"{f}{s}" for f in ("csb","sell") for s in SIZES]
MATCHED = [f"matched_{f}" for f in NATIVE]
CONFIGS = ["stock",*NATIVE,*UNBLOCKED,*BLOCKED,*MATCHED]
NAMES = {"cora":"Cora","citeseer":"CiteSeer","pubmed":"PubMed"}
PREVIOUS = dict(cora="csb64",citeseer="csb2",pubmed="sell2")
PREVIOUS_STAGES = {
    "cora": dict(softmax="csb64",aggregation="csb64"),
    "citeseer": dict(softmax="csb64",aggregation="csb64"),
    "pubmed": dict(softmax="csb8",aggregation="csb4"),
}


def label(key):
    if key=="stock":
        return "Imported GATConv"
    if key in NATIVE:
        return f"Native {key.upper()} hybrid"
    if key in UNBLOCKED:
        return f"Custom {key[-3:].upper()}"
    if key in MATCHED:
        return f"Custom {key[-3:].upper()} matched hybrid"
    if key.startswith("sell"):
        return f"SELL {key[4:]}"
    size = key[3:]
    return f"CSB {size}×{size}"


def sha(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n",b"\n")).hexdigest()


def audit(raw):
    assert raw["status"]=="complete"
    settings = raw["settings"]
    assert (settings["threads"],settings["heads"],settings["channels"],settings["dtype"])==(1,4,8,"float32")
    for path,digest in raw["sources"].items():
        assert sha(HERE.parent/path)==digest, path
    results = {}
    for dataset,rounds in raw["datasets"].items():
        first = rounds[0]
        assert len(rounds)==settings["rounds"] and len({r["pid"] for r in rounds})==len(rounds)
        collected, build_ms = {},{}
        errors = 0.
        for i,row in enumerate(rounds):
            assert (row["dataset"],row["round"])==(dataset,i)
            assert row["environment"]["dgl"]=="1.1.3"
            for field in ("cpu","weights","edge_sha256","graph_formats","storage","routes","compilers"):
                assert row[field]==first[field], field
            assert row["graph"]["feature_sha256"]==first["graph"]["feature_sha256"]
            assert row["environment"]["cpu_model"]==first["environment"]["cpu_model"]
            assert set(row["checks"])==set(CONFIGS)
            errors = max(errors,max(r["output"] for r in row["checks"].values()))
            seed = settings["seed"]+1009*i+list(NAMES).index(dataset)
            assert row["order_seed"]==seed
            rng = random.Random(seed)
            values = {}
            keys = row["schedule_columns"]
            isolated = {f"isolated/{op}/{f}" for op in SUPPORT for f in ("stock",*SUPPORT[op],*UNBLOCKED,*BLOCKED)}
            assert set(keys)=={*(f"forward/{f}" for f in CONFIGS),*(f"profile/{f}" for f in CONFIGS),*isolated,
                               *(f"isolated/{s}/shared" for s in ("projection","terms","output")),
                               "timer/empty","timer/six_regions","context/guard","context/sparse","context/forward"}
            assert len(row["blocks"])==settings["warmup"]+settings["repetitions"]
            for b,block in enumerate(row["blocks"]):
                order = list(keys)
                rng.shuffle(order)
                assert [r[0] for r in block]==order
                record = {r[0]:r[1] for r in block}
                for entry in block:
                    key,elapsed,*stages = entry
                    assert isfinite(elapsed) and elapsed>0
                    if key.startswith("profile/"):
                        regions, = stages
                        assert len(regions)==6 and sum(regions)<=elapsed and min(regions)>0
                    else:
                        assert not stages
                if b<settings["warmup"]:
                    continue
                for entry in block:
                    key,elapsed,*stages = entry
                    values.setdefault(key,[]).append(elapsed/1e6)
                    if stages:
                        f = key.split("/")[1]
                        for stage,value in zip(STAGES,stages[0],strict=True):
                            values.setdefault(f"inside/{stage}/{f}",[]).append(value/1e6)
                        values.setdefault(f"overhead/{f}",[]).append((elapsed-record[f"forward/{f}"])/1e6)
                        values.setdefault(f"unassigned/{f}",[]).append((elapsed-sum(stages[0]))/1e6)
                values.setdefault("timer/net",[]).append((record["timer/six_regions"]-record["timer/empty"])/1e6)
            for k,v in values.items():
                collected.setdefault(k,[]).append(median(v))
            for key,medians in construction(rng,row,settings).items():
                build_ms.setdefault(key,[]).append(medians)
        timing = {key:dict(median_ms=median(v),round_ms=v,min_ms=min(v),max_ms=max(v))
                  for key,v in collected.items()}
        storage = {key:dict(**v,build_median_ms=median(build_ms[key]),build_round_ms=build_ms[key])
                   for key,v in first["storage"].items()}
        results[dataset] = dict(timing=timing,storage=storage,routes=first["routes"],max_output_error=errors)
    return results


def construction(rng,row,settings):
    values = {}
    assert len(row["construction"])==settings["build_repetitions"]
    for block in row["construction"]:
        order = list(row["storage"])
        rng.shuffle(order)
        assert order==[f for f,_ in block]
        for f,value in block:
            assert isfinite(value) and value>0
            values.setdefault(f,[]).append(value/1e6)
    return {f:median(v) for f,v in values.items()}


def contrasts(results):
    result = []
    for d,row in results.items():
        times = row["timing"]
        def compare(family,endpoint,control,candidate,control_key,candidate_key):
            paired = stats.paired(times[control_key]["round_ms"],times[candidate_key]["round_ms"])
            result.append(dict(dataset=d,family=family,endpoint=endpoint,control=control,
                               candidate=candidate,**paired))

        for op,natives in SUPPORT.items():
            for f in natives:
                compare("matched backend",op,f,f"custom_{f}",f"isolated/{op}/{f}",f"isolated/{op}/custom_{f}")
        for f in NATIVE:
            compare("matched backend","forward",f,f"matched_{f}",f"forward/{f}",f"forward/matched_{f}")
        for f in BLOCKED:
            for c in UNBLOCKED:
                for op in (*SUPPORT,"forward"):
                    prefix = f"isolated/{op}" if op in SUPPORT else "forward"
                    compare("matched storage",op,c,f,f"{prefix}/{c}",f"{prefix}/{f}")
                for op in SUPPORT:
                    compare("matched storage",f"inside/{op}",c,f,f"inside/{op}/{c}",f"inside/{op}/{f}")
        prior = PREVIOUS[d]
        for c in ("stock",*NATIVE):
            compare("Milestone 8 replication","forward",c,prior,f"forward/{c}",f"forward/{prior}")
        for op,f in PREVIOUS_STAGES[d].items():
            compare("Milestone 8 replication",op,"csc",f,f"isolated/{op}/csc",f"isolated/{op}/{f}")
        for key in CONFIGS:
            compare("instrumentation","forward",key,key,f"forward/{key}",f"profile/{key}")
            if key!="stock":
                compare("whole-forward reference","forward","stock",key,"forward/stock",f"forward/{key}")
        for key in ("stock",prior,"custom_coo"):
            for op in STAGES:
                isolated_key = key if op in SUPPORT else "shared"
                compare("timing context",op,key,key,f"isolated/{op}/{isolated_key}",f"inside/{op}/{key}")
        for context in ("guard","sparse","forward"):
            compare("projection context","projection","isolated",context,
                    "isolated/projection/shared",f"context/{context}")
    for family,endpoint in {(v["family"],v["endpoint"]) for v in result}:
        stats.holm([v for v in result if (v["family"],v["endpoint"])==(family,endpoint)])
    return result


def ms(times,key):
    return times[key]["median_ms"]


def effect(row):
    return f'{row["speedup"]:.3f}× [{row["ci_low"]:.3f}, {row["ci_high"]:.3f}]'


def ci_difference(values):
    half = float(t.ppf(.975,len(values)-1))*stdev(values)/sqrt(len(values))
    center = mean(values)
    return center-half,center+half


def historical(raw,results,previous):
    assert sha(HERE.parent/"comprehensive_sparse_formats/RESULTS.json")==raw["previous_raw_sha256"]
    diagnostic = {}
    for d,rounds in previous["datasets"].items():
        if d not in results:
            continue
        projection = []
        total = {f:[] for f in ("stock",*NATIVE,PREVIOUS[d])}
        stages = {f"{op}/{f}":[] for op,prior in PREVIOUS_STAGES[d].items() for f in ("csc",prior)}
        inverted = 0
        for i,row in enumerate(rounds):
            current = raw["datasets"][d][i%len(raw["datasets"][d])]
            for key in ("weights","edge_sha256","cpu"):
                assert current[key]==row[key]
            assert current["graph"]["feature_sha256"]==row["graph"]["feature_sha256"]
            assert current["environment"]["cpu_model"]==row["environment"]["cpu_model"]
            projection.append(median(dict(block)["projection"] for block in row["stage"]))
            for f in total:
                total[f].append(median(dict(block)[f] for block in row["forward"]))
            for key in stages:
                stages[key].append(median(dict(block)[key] for block in row["stage"]))
            inverted += projection[-1]>total[PREVIOUS[d]][-1]
        diagnostic[d] = dict(projection_ms=median(projection),winner_forward_ms=median(total[PREVIOUS[d]]),
                             inverted=inverted,rounds=len(rounds),old_total=total,old_stages=stages)
    return diagnostic


def comparison_tables(lines,results,comparisons):
    lines += ["","## Native DGL versus custom unblocked","",
              "Each direct kernel pair uses the same named sparse traversal. The three whole-forward "
              "pairs also match the per-stage routes. Across backends the index width, kernel code, "
              "library dispatch and score-activation fusion differ. Speedup is native divided by custom.","",
              "| Dataset | Stage | Native route | Custom speedup (95% CI) | Holm p |",
              "|---|---|---|---|---:|"]
    for r in comparisons:
        if r["family"]=="matched backend":
            lines.append(f'| {NAMES[r["dataset"]]} | {r["endpoint"]} | {r["control"].upper()} | {effect(r)} | {r["holm_p"]:.4g} |')
    lines += ["","## Matched custom storage comparisons","",
              "These comparisons share FP32 arithmetic, int32 sparse indices, fused score activation, "
              "canonical EID output, destination-global three-pass softmax, and timed output/scratch allocation. "
              "SELL uses its own traversal and an equivalent stage interface. All pair estimates are in ANALYSIS.json.","",
              "| Dataset | Endpoint | Family | Gains / losses vs custom unblocked after Holm | Fastest candidate | Paired speedup against fastest observed custom unblocked (95% CI) | Holm p |",
              "|---|---|---|---:|---|---|---:|"]
    for d,row in results.items():
        times = row["timing"]
        for op in (*SUPPORT,*(f"inside/{op}" for op in SUPPORT),"forward"):
            prefix = f"isolated/{op}" if op in SUPPORT else op
            control = min(UNBLOCKED,key=lambda f:ms(times,f"{prefix}/{f}"))
            for family in ("csb","sell"):
                group = [r for r in comparisons if (r["dataset"],r["family"],r["endpoint"])==(d,"matched storage",op)
                         and r["candidate"].startswith(family)]
                gains = sum(r["holm_p"]<.05 and r["speedup"]>1 for r in group)
                losses = sum(r["holm_p"]<.05 and r["speedup"]<1 for r in group)
                winner = min((f for f in BLOCKED if f.startswith(family)),key=lambda f:ms(times,f"{prefix}/{f}"))
                contrast = next(r for r in group if r["candidate"]==winner and r["control"]==control)
                lines.append(f'| {NAMES[d]} | {op} | {family.upper()} | {gains} / {losses} of {len(group)} | '
                             f'{label(winner)} | {effect(contrast)} vs {label(control)} | {contrast["holm_p"]:.4g} |')


def diagnostics(lines,results,comparisons,history):
    lines += ["","## Milestone 8 replication","",
              "The three formats selected here were the recorded Milestone 8 forward winners. "
              "The control in the last column is the fastest native hybrid by observed median in Milestone 9. "
              "The adjusted family covers each fixed candidate against stock and all three native hybrids. "
              "The old and new process blocks are analyzed separately.","",
              "| Dataset | Fixed prior candidate | M8 / M9 forward ms | M8 vs stock (95% CI) | M9 vs stock (95% CI), Holm p | M9 vs fastest native hybrid (95% CI), Holm p |",
              "|---|---|---:|---|---|---|"]
    for d,row in results.items():
        times = row["timing"]
        f = PREVIOUS[d]
        items = {r["control"]:r for r in comparisons
                 if (r["dataset"],r["family"],r["endpoint"])==(d,"Milestone 8 replication","forward")}
        native = min(NATIVE,key=lambda key:ms(times,f"forward/{key}"))
        hist = history[d]
        old = stats.paired(hist["old_total"]["stock"],hist["old_total"][f])
        lines.append(f'| {NAMES[d]} | {label(f)} | {hist["winner_forward_ms"]:.5f} / {ms(times,f"forward/{f}"):.5f} | '
                     f'{effect(old)} | {effect(items["stock"])}, {items["stock"]["holm_p"]:.4g} | '
                     f'{effect(items[native])}, {items[native]["holm_p"]:.4g} |')
    lines += ["", "The prior isolated softmax and aggregation winners are compared with the same native CSC kernel.",
              "", "| Dataset | Stage | Fixed prior candidate | M8 speedup (95% CI) | M9 speedup (95% CI) | M9 Holm p |",
              "|---|---|---|---|---|---:|"]
    for r in comparisons:
        if r["family"]=="Milestone 8 replication" and r["endpoint"]!="forward":
            d,op,f = r["dataset"],r["endpoint"],r["candidate"]
            old = history[d]["old_stages"]
            old_effect = stats.paired(old[f"{op}/csc"],old[f"{op}/{f}"])
            lines.append(f'| {NAMES[d]} | {op} | {label(f)} | {effect(old_effect)} | {effect(r)} | {r["holm_p"]:.4g} |')
    lines += ["","## Projection discrepancy and timing overhead","",
              "Milestone 8 measured complete forwards first and the isolated calls later. Its shared projection "
              "statistic came from the latter phase. Each Milestone 9 block interleaves isolated calls, both "
              "forward modes, and context probes. The in-forward projection column is a region of the "
              "measured call, and all six regions together fit inside its enclosing per-call timer.","",
              "| Dataset | M8 isolated projection / selected forward ms | M8 rounds with isolated projection above selected forward | M9 isolated projection ms | M9 selected in-forward projection ms | M9 selected plain / instrumented forward ms |",
              "|---|---:|---:|---:|---:|---:|"]
    for d,row in results.items():
        x = row["timing"]
        h = history[d]
        f = PREVIOUS[d]
        lines.append(f'| {NAMES[d]} | {h["projection_ms"]:.5f} / {h["winner_forward_ms"]:.5f} | {h["inverted"]}/{h["rounds"]} | '
                     f'{ms(x,"isolated/projection/shared"):.5f} | {ms(x,f"inside/projection/{f}"):.5f} | '
                     f'{ms(x,f"forward/{f}"):.5f} / {ms(x,f"profile/{f}"):.5f} |')
    lines += ["","These probes all execute the same projection. The graph guard checks destination degrees. "
              "The sparse probe replays the three stock sparse operations on prepared tensors. The forward "
              "probe precedes projection with stock GATConv. Preceding work is outside the projection timer.","",
              "| Dataset | Isolated projection ms | After guard ms | After sparse replay ms | After stock forward ms | Timer-only six regions net µs |",
              "|---|---:|---:|---:|---:|---:|"]
    for d,row in results.items():
        x = row["timing"]
        lines.append(f'| {NAMES[d]} | {ms(x,"isolated/projection/shared"):.5f} | {ms(x,"context/guard"):.5f} | '
                     f'{ms(x,"context/sparse"):.5f} | {ms(x,"context/forward"):.5f} | {1000*ms(x,"timer/net"):.3f} |')
    if "citeseer" in results:
        x = results["citeseer"]["timing"]
        old = history["citeseer"]
        gap = old["projection_ms"]-old["winner_forward_ms"]
        change = old["projection_ms"]-ms(x,"isolated/projection/shared")
        overhead = ms(x,"overhead/csb2")
        lines += ["", f"On CiteSeer the old isolated projection exceeded the CSB 2×2 forward by {gap:.3f} ms. "
                  f"Under interleaving the isolated projection was {change:.3f} ms lower than that old projection, "
                  f"and the in-forward region was {ms(x,'inside/projection/csb2'):.3f} ms inside "
                  f"a {ms(x,'profile/csb2'):.3f} ms instrumented call. "
                  f"The net empty timer cost was {1000*ms(x,'timer/net'):.2f} µs. Median observed instrumentation "
                  f"overhead was {overhead*1000:+.1f} µs for that layout. These measurements quantify the "
                  "scale difference between clock overhead and the historical phase discrepancy."]
    lines += ["","| Dataset | Implementation | Instrumented minus plain forward, paired median ms | 95% CI for mean of process median differences, ms |",
              "|---|---|---:|---:|"]
    for d,row in results.items():
        for f in ("stock",PREVIOUS[d]):
            timing = row["timing"][f"overhead/{f}"]
            low,high = ci_difference(timing["round_ms"])
            lines.append(f'| {NAMES[d]} | {label(f)} | {timing["median_ms"]:+.5f} | [{low:+.5f}, {high:+.5f}] |')
    lines += ["","| Dataset | Probe | Isolated projection / probe projection (95% CI) | Holm p |",
              "|---|---|---|---:|"]
    for row in comparisons:
        if row["family"]=="projection context":
            lines.append(f'| {NAMES[row["dataset"]]} | {row["candidate"]} | {effect(row)} | {row["holm_p"]:.4g} |')
    lines += ["","| Dataset | Fixed prior candidate | Stage | Isolated / in-forward (95% CI) | Holm p |",
              "|---|---|---|---|---:|"]
    for row in comparisons:
        d = row["dataset"]
        if row["family"]=="timing context" and row["candidate"]==PREVIOUS[d]:
            lines.append(f'| {NAMES[d]} | {label(PREVIOUS[d])} | {row["endpoint"]} | {effect(row)} | {row["holm_p"]:.4g} |')
    lines += ["","Empty-call subtraction is used only for the timer microbenchmark. Forward readings are retained as measured. "
              "Actual instrumented-minus-plain differences include runtime variability, which can give either sign. "
              "Sparse isolation uses already materialized intermediates, while a forward consumes freshly produced "
              "tensors and includes graph bookkeeping and allocations. The controlled probes change cache state and "
              "allocator history together. They establish context sensitivity without assigning a fraction to either cause. "
              "The raw data do not include hardware counters or a reconstruction of the earlier process state.",
              "", "Per-configuration instrumented totals, unassigned time, paired overhead and round variability are in ANALYSIS.json."]


def render(raw,results,comparisons,history):
    s = raw["settings"]
    lines = ["# Milestone 9 — Sparse format and timing validation","",
        "Each stage cell is **isolated / in-forward**, in milliseconds. Each number is a median of "
        "process medians. Total forward uses the plain measured call. The instrumented copy uses the "
        "original executable forward statements with six stage boundaries. The stock direct call uses "
        "the untouched imported GATConv.","",
        "Projection, L/R terms and output bias use common isolated controls with the same mathematics. "
        "Stock isolated sparse cells replay its public DGL calls on prepared tensors. In-forward values "
        "run on intermediates produced inside that call. N/A marks unsupported native isolated cells. "
        "Their in-forward timings belong to the stated native fallback.","",
        "## Findings",""]
    for d,row in results.items():
        times = row["timing"]
        backend = [r for r in comparisons if (r["dataset"],r["family"],r["endpoint"])==(d,"matched backend","forward")]
        faster_backends = [label(r["candidate"]) for r in backend if r["holm_p"]<.05 and r["speedup"]>1]
        selected = min(CONFIGS,key=lambda f:ms(times,f"forward/{f}"))
        fastest_plain = min(UNBLOCKED,key=lambda f:ms(times,f"forward/{f}"))
        proven = []
        for f in BLOCKED:
            subset = [r for r in comparisons if (r["dataset"],r["family"],r["endpoint"],r["candidate"])==
                      (d,"matched storage","forward",f)]
            if len(subset)==len(UNBLOCKED) and all(r["holm_p"]<.05 and r["speedup"]>1 for r in subset):
                proven.append(label(f))
        sentence = f"- **{NAMES[d]}**. The fastest observed forward was {label(selected)} "
        sentence += f"at {ms(times,f'forward/{selected}'):.3f} ms. "
        if selected!=fastest_plain:
            sentence += f"The fastest custom unblocked row was {label(fastest_plain)} at {ms(times,f'forward/{fastest_plain}'):.3f} ms. "
        sentence += f"Custom implementations won {len(faster_backends)} of 3 route-matched backend comparisons after correction. "
        if proven:
            sentence += f"{', '.join(proven)} beat all three custom unblocked forwards after correction."
        else:
            sentence += "All CSB and SELL forward gains over the custom unblocked controls fell short of the corrected threshold."
        lines.append(sentence)
    lines.append("")
    for d,row in results.items():
        lines += [f"## {NAMES[d]}","",
            "| Implementation | Projection | Attention Terms (L/R) | Attention Scores | Edge Softmax | Aggregation | Output | Total Forward | Speedup vs stock (95% CI) | Holm p |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
        times = row["timing"]
        stock_effects = {r["candidate"]:r for r in comparisons
                         if r["dataset"]==d and r["family"]=="whole-forward reference"}
        for f in CONFIGS:
            cells = []
            for op in STAGES:
                key = "shared" if op not in SUPPORT else (row["routes"][f][op] if f in MATCHED else f)
                entry = times.get(f"isolated/{op}/{key}")
                value = f'{entry["median_ms"]:.5f}' if entry else "N/A"
                cells.append(f'{value} / {ms(times,f"inside/{op}/{f}"):.5f}')
            stock = stock_effects.get(f)
            improvement,p = (effect(stock),f'{stock["holm_p"]:.4g}') if stock else ("Control","—")
            lines.append("| "+label(f)+" | "+" | ".join(cells)+
                         f' | {ms(times,f"forward/{f}"):.5f} | {improvement} | {p} |')
        lines.append("")
    comparison_tables(lines,results,comparisons)
    diagnostics(lines,results,comparisons,history)
    lines += ["","## Unsupported native paths and construction","",
              "The [DGL 1.1.3 CPU dispatch](https://raw.githubusercontent.com/dmlc/dgl/v1.1.3/src/array/kernel.cc) "
              "supports COO/CSR GSDDMM, CSC destination softmax and COO/CSC GSpMM. "
              "COO uses CSC softmax. CSR uses CSC softmax and aggregation. CSC uses COO scoring. "
              "The matching custom hybrids follow these same routes using Milestone 6 kernels. "
              "Canonical EIDs carry edge tensors directly between stages.","",
              "Index payload counts retained sparse arrays only. Construction starts from canonical "
              "endpoints for custom layouts and a fresh COO graph for native views. Hybrid rows share views "
              "and the table lists the constituent formats separately.","",
              "| Dataset | Storage | KiB | Build ms |","|---|---|---:|---:|"]
    for d,row in results.items():
        for f,v in row["storage"].items():
            lines.append(f'| {NAMES[d]} | {label(f)} | {v["bytes"]/1024:.1f} | {v["build_median_ms"]:.5f} |')
    first = next(iter(raw["datasets"].values()))[0]
    processes = sum(len(rounds) for rounds in raw["datasets"].values())
    forwards = processes*s["repetitions"]*len(CONFIGS)
    lines += ["","## Methods and correctness","",
              f"{first['environment']['cpu_model']}, logical CPU {first['cpu']}, one thread. DGL 1.1.3, "
              f"Torch {first['environment']['torch']}, FP32, four heads, eight channels per head, seed-0 weights. "
              "Original node and edge ordering, existing self-loop preprocessing, inference mode and dropout disabled.","",
              f"{s['rounds']} fresh processes per dataset, {s['warmup']} warmup and {s['repetitions']} measured "
              f"randomized blocks per process, and {s['build_repetitions']} separately timed builds per format. "
              "Intervals on speedups are pointwise 95% Student-t over paired process log ratios. "
              "Holm families are separated by comparison family and endpoint across all datasets. "
              "All blocks are retained. The report replays the randomization from the seeds.","",
              f"The log contains {forwards:,} measured plain forwards and {forwards:,} instrumented forwards. "
              "Each instrumented call records all six regions as well as its enclosing timer.","",
              "The setup verifies canonical endpoints, feature bytes and weight fingerprints against the original workload. "
              "All pipeline intermediates, destination softmax sums and final output pass stock-backed checks. "
              "The timing copies produce bitwise-identical output to their plain calls at validation. "
              f"Maximum final-output absolute error is {max(v['max_output_error'] for v in results.values()):.3g}.",
              "", "[Raw samples](RESULTS.json) · [Round summaries, paired contrasts and storage](ANALYSIS.json)",""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=HERE/"RESULTS.json")
    parser.add_argument("--output",type=Path,default=HERE)
    args = parser.parse_args()
    raw = json.loads(args.input.read_text())
    previous = json.loads((HERE.parent/"comprehensive_sparse_formats/RESULTS.json").read_text())
    results = audit(raw)
    pairs = contrasts(results)
    history = historical(raw,results,previous)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"REPORT.md").write_text(render(raw,results,pairs,history),encoding="utf-8")
    (args.output/"ANALYSIS.json").write_text(json.dumps(dict(datasets=results,contrasts=pairs,
        milestone8=history,raw_sha256=sha(args.input)),separators=(",",":"),allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
