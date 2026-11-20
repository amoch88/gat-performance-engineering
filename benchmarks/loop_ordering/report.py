"""Replay timing schedules, audit correctness and report per-sweep loop effects."""
import argparse
import importlib.util
import json
from math import isfinite, sqrt
from pathlib import Path
import random
from statistics import mean, median, stdev

from scipy.stats import t

from compiled import FLAGS, HERE, sha
from design import DATASETS, FLOOR, ORDERS, SEED, STAGES, SWEEPS, SYMBOLS, columns, design, memory

spec = importlib.util.spec_from_file_location("m11_paired",HERE.parent/"sparse_formats/report.py")
statistics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(statistics)
NAMES = dict(cora="Cora",citeseer="CiteSeer",pubmed="PubMed")
LABELS = dict(terms="Attention terms",score="Attention scores",softmax="Edge softmax",aggregation="Aggregation")
ENDPOINTS = ("forward","isolated","in_forward")


def read_round(row,settings,rng):
    sweeps = [*SWEEPS,"stock"]
    rng.shuffle(sweeps)
    assert row["sweep_order"]==sweeps and list(row["blocks"])==sweeps
    result = {}
    for stage in sweeps:
        keys = columns(stage)
        codes = ("DGL",) if stage=="stock" else SWEEPS[stage]
        blocks = row["blocks"][stage]
        assert len(blocks)==settings["warmup"]+settings["repetitions"]
        metrics = {code:{k:[] for k in (*STAGES,"forward","profile","overhead","unassigned",
                       *(("isolated","in_forward") if stage!="stock" else ()))} for code in codes}
        timers = []
        allocations = []
        for index,block in enumerate(blocks):
            order = keys.copy()
            rng.shuffle(order)
            assert [event[0] for event in block]==order
            observations = {}
            for key,ns,*rest in block:
                assert isinstance(ns,int) and ns>0
                if key.startswith("profile/"):
                    stages, = rest
                    assert len(stages)==len(STAGES) and min(stages)>0 and sum(stages)<=ns
                    observations[key] = ns,stages
                else:
                    assert not rest
                    observations[key] = ns
            if index<settings["warmup"]:
                continue
            timers.append((observations["timer/six"]-observations["timer/empty"])/1e6)
            if stage!="stock":
                allocations.append(observations["allocation"]/1e6)
            for code in codes:
                plain = observations[f"plain/{code}"]
                prof,stages = observations[f"profile/{code}"]
                values = dict(zip(STAGES,stages,strict=True),forward=plain,profile=prof,
                              overhead=prof-plain,unassigned=prof-sum(stages))
                if stage!="stock":
                    values.update(isolated=observations[f"isolated/{code}"],in_forward=values[stage])
                for key,value in values.items():
                    metrics[code][key].append(value/1e6)
        result[stage] = dict(medians={c:{k:median(v) for k,v in m.items()} for c,m in metrics.items()},
                             timer=median(timers),allocation=median(allocations) if allocations else None)
    return result


def mean_ci(values):
    mid = mean(values)
    margin = float(t.ppf(.975,len(values)-1))*stdev(values)/sqrt(len(values))
    return [mid,mid-margin,mid+margin]


def audit(raw):
    assert raw["status"]=="complete"
    settings = raw["settings"]
    assert tuple(settings[k] for k in ("threads","model_seed","heads","channels","dtype","order_seed","practical_speedup_floor"))==(
        1,0,4,8,"float32",SEED,FLOOR)
    for file,digest in raw["sources"].items():
        assert sha(HERE.parent/file,text=True)==digest,file
    assert raw["lock_sha256"]==sha(HERE.parents[1]/"uv.lock",text=True)
    build = raw["build"]
    assert build["source_sha256"]==raw["sources"]["loop_ordering/kernels.cpp"]
    assert build["flags"]==list(FLAGS)
    assert set(build["symbols"])==set(SYMBOLS)
    assert all(rec["symbol"]==SYMBOLS[c] and rec["instructions"]>0 and rec["size_bytes"]>0
               for c,rec in build["symbols"].items())
    offset = {code:v["address"]-build["symbols"]["A1"]["address"] for code,v in build["symbols"].items()}
    aliases = {}
    for stage,codes in SWEEPS.items():
        aliases[stage] = [(a,b) for i,a in enumerate(codes) for b in codes[i+1:]
            if offset[a]==offset[b] or build["symbols"][a]["code_sha256"]==build["symbols"][b]["code_sha256"]]
    parent = random.Random(SEED)
    assert len(raw["dataset_orders"])==settings["rounds"]
    for seen in raw["dataset_orders"]:
        ordering = list(raw["datasets"])
        parent.shuffle(ordering)
        assert seen==ordering
    canonical = next(iter(raw["datasets"].values()))[0]
    env_keys = ("cpu_model","cpu_affinity","os","python","dgl","torch","numpy","torch_threads",
                "torch_interop_threads","thread_environment","runtime_thread_controls")
    assert canonical["environment"]["dgl"]=="1.1.3" and canonical["environment"]["torch"]=="2.2.2+cpu"
    trials = {}
    for dataset,rounds in raw["datasets"].items():
        assert len(rounds)==len({r["pid"] for r in rounds})==settings["rounds"]
        first = rounds[0]
        per_round = []
        for i,row in enumerate(rounds):
            assert row["dataset"]==dataset and row["round"]==i and row["cpu"]==canonical["cpu"]
            assert all(row["environment"][k]==canonical["environment"][k] for k in env_keys)
            assert row["environment"]["cpu_affinity"]==[row["cpu"]]
            assert row["environment"]["torch_threads"]==row["environment"]["torch_interop_threads"]==1
            assert row["loaded_offsets"]==offset and row["binary_sha256"]==build["binary_sha256"]
            assert row["stock_forward_sha256"]==canonical["stock_forward_sha256"]
            assert row["stock_profile_bitwise"]
            assert set(row["graph_formats"]["created"])=={"coo","csr","csc"}
            assert not row["graph_formats"]["not created"]
            for k in ("index_sha256","edge_sha256","weights","stage_payload","checks","routes","parallel_edges"):
                assert row[k]==first[k],k
            assert row["graph"]["feature_sha256"]==first["graph"]["feature_sha256"]
            n,e = (row["graph"][k] for k in ("nodes","edges_with_self_loops"))
            assert row["self_loops"]==n and row["stage_payload"]==memory(n,e)
            assert [row["index_sha256"][k] for k in ("src","dst")]==row["edge_sha256"]
            for stage,codes in SWEEPS.items():
                assert list(row["checks"][stage])==list(codes)
                for code in codes:
                    check = row["checks"][stage][code]
                    assert row["routes"][stage][code]=={s:SYMBOLS[c] for s,c in design(stage,code).items()}
                    assert all(check[k] for k in ("custom_bitwise","repeat_bitwise","profile_bitwise","isolated_bitwise"))
                    assert check["max_normalization_error"]<=1e-5
                    assert set(check["errors"])=={"projected","source_contribution","destination_contribution",
                                                  "logits","attention","aggregated","output"}
                    for errors in check["errors"].values():
                        assert all(isfinite(v) and v>=0 for v in errors.values())
                        assert errors["max_tolerance_fraction"]<=1
            seed = SEED+1009*i+DATASETS.index(dataset)
            assert row["order_seed"]==seed
            per_round.append(read_round(row,settings,random.Random(seed)))
        trials[dataset] = {}
        for stage in (*SWEEPS,"stock"):
            records = [r[stage] for r in per_round]
            values = {code:{k:[r["medians"][code][k] for r in records] for k in metrics}
                      for code,metrics in records[0]["medians"].items()}
            trials[dataset][stage] = dict(process_medians=values,
                timing={c:{k:median(v) for k,v in m.items()} for c,m in values.items()},
                allocation_ms=[r["allocation"] for r in records] if stage!="stock" else [],
                timer_ms=[r["timer"] for r in records],
                overhead_mean_ci_ms={c:mean_ci(m["overhead"]) for c,m in values.items()})
    return trials,aliases


def comparisons(trials,aliases):
    results = []
    for stage,orders in SWEEPS.items():
        for endpoint in ENDPOINTS:
            family = []
            for dataset,sweeps in trials.items():
                values = sweeps[stage]["process_medians"]
                for code in orders[1:]:
                    effect = statistics.paired(values[orders[0]][endpoint],values[code][endpoint])
                    alias = any(code in pair for pair in aliases[stage])
                    family.append(dict(dataset=dataset,sweep=stage,code=code,endpoint=endpoint,alias=alias,**effect))
            statistics.holm(family)
            results.extend(family)
    return results


def decision(row):
    if row["alias"]:
        return "Machine-code alias"
    if row["holm_p"]>=.05:
        return "Inconclusive"
    if row["speedup"]<1:
        return "Slower"
    if row["ci_low"]>FLOOR:
        return "Qualifies"
    return "Faster, below practical bound"


def effect(row):
    return f'{row["speedup"]:.4f}× [{row["ci_low"]:.4f}, {row["ci_high"]:.4f}], p={row["holm_p"]:.4g}'


def name(code):
    return f"{code} · {ORDERS[code]}"


def primary(lines,trials,rows):
    for d,sweeps in trials.items():
        lines += [f"## {NAMES[d]}",""]
        for stage,orders in SWEEPS.items():
            lines += [f"### {LABELS[stage]}","",
                      "| Loop order | Projection | Attention Terms | Scores | Softmax | Aggregation | Output | Total Forward |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for code in orders:
                stats = sweeps[stage]["timing"][code]
                lines.append(f"| {name(code)} | "+" | ".join(f"{stats[k]:.6f}" for k in (*STAGES,"forward"))+" |")
            lines.append("")
    lines += ["## External reference — untouched imported DGL GATConv","",
              "The DGL reference has a separate randomized schedule in each process. Its stage columns "
              "use the validated profiled copy. Absolute latencies are contextual: library calls, allocation, "
              "and kernel implementation differ, so a difference versus stock is not a loop-order effect.","",
              "| Dataset | Projection | Attention Terms | Scores | Softmax | Aggregation | Output | Total Forward |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for d,sweeps in trials.items():
        stats = sweeps["stock"]["timing"]["DGL"]
        lines.append(f"| {NAMES[d]} | "+" | ".join(f"{stats[k]:.6f}" for k in (*STAGES,"forward"))+" |")
    lines += ["","## Paired complete-forward comparisons",""]
    for stage,orders in SWEEPS.items():
        lines += [f"### {LABELS[stage]}","",
                  "| Dataset | Loop order | Forward ms | Paired speedup vs default | 95% CI | Holm p | Decision |",
                  "|---|---|---:|---:|---|---:|---|"]
        by_case = {(r["dataset"],r["code"]):r for r in rows if r["sweep"]==stage and r["endpoint"]=="forward"}
        for d,sweeps in trials.items():
            for code in orders:
                ms = sweeps[stage]["timing"][code]["forward"]
                row = by_case.get((d,code))
                stats = (f'{row["speedup"]:.4f}× | [{row["ci_low"]:.4f}, {row["ci_high"]:.4f}] | '
                         f'{row["holm_p"]:.4g} | {decision(row)}' if row else "1.0000× | — | — | Default")
                lines.append(f"| {NAMES[d]} | {name(code)} | {ms:.6f} | {stats} |")
        lines.append("")


def diagnostic_tables(lines,trials,rows):
    lines += ["## Isolated and in-forward target kernels","",
              "Both regions include the same tensor allocation and ctypes call. An isolated call receives the "
              "same held default intermediates for every order. Inside forward it sees freshly produced "
              "buffers, a different cache/allocator context, and profiler bookends. Allocations are separately "
              "timed below; their medians are not subtracted. The speedups and Holm p-values are from "
              "separate process-paired families for isolated and in-forward timings.","",
              "| Dataset | Kernel | Loop order | Isolated ms | In-forward ms | Speedup vs default (isolated / in-forward; 95% CI, Holm p) |",
              "|---|---|---|---:|---:|---|"]
    lookup = {(r["dataset"],r["sweep"],r["code"],r["endpoint"]):r for r in rows}
    for d,sweeps in trials.items():
        for stage,orders in SWEEPS.items():
            for code in orders:
                stats = sweeps[stage]["timing"][code]
                desc = "Default" if code==orders[0] else " / ".join(
                    effect(lookup[d,stage,code,k]) for k in ("isolated","in_forward"))
                lines.append(f'| {NAMES[d]} | {LABELS[stage]} | {name(code)} | {stats["isolated"]:.6f} | '
                             f'{stats["in_forward"]:.6f} | {desc} |')
    lines += ["","## Allocation and instrumentation","",
              "Allocation is the bare factory call (no filling or page-touch guarantee). Aggregation zero-fills "
              "the same Y buffer inside every compiled call; softmax initializes its shared head scratch "
              "inside each compiled call. Profile overhead averages the process medians of matched "
              "instrumented-minus-plain differences. Timer calibration subtracts the matched empty call from "
              "the six clock-and-record operations. The analysis contains every order's overhead interval.",
              "",
              "| Dataset | Sweep | Bare allocation µs | Empty six-region net µs | Default instrumented − plain µs (95% CI) |",
              "|---|---|---:|---:|---|"]
    for d,sweeps in trials.items():
        for stage,entry in sweeps.items():
            base = SWEEPS[stage][0] if stage!="stock" else "DGL"
            mid,lo,hi = (1000*v for v in entry["overhead_mean_ci_ms"][base])
            alloc = f'{median(entry["allocation_ms"])*1000:.3f}' if stage!="stock" else "—"
            lines.append(f'| {NAMES[d]} | {LABELS.get(stage,"Stock DGL")} | {alloc} | '
                         f'{median(entry["timer_ms"])*1000:.3f} | {mid:+.3f} [{lo:+.3f}, {hi:+.3f}] |')


def scientific_conclusions(lines,trials,rows):
    lines += ["","## What should be tuned?","",
              "Fastest denotes the smallest observed median in that column. Qualification is separate "
              "and uses only the matched complete-forward comparisons.","",
              "| Dataset | Kernel | Fastest isolated | Fastest in-forward | Fastest complete forward | Practical gain vs default |",
              "|---|---|---|---|---|---|"]
    qualifiers = {stage:{} for stage in SWEEPS}
    for d,sweeps in trials.items():
        for stage,codes in SWEEPS.items():
            times = sweeps[stage]["timing"]
            best = [min(codes,key=lambda c:times[c][metric]) for metric in ("isolated","in_forward","forward")]
            keep = [r["code"] for r in rows if r["dataset"]==d and r["sweep"]==stage and r["endpoint"]=="forward"
                    and decision(r)=="Qualifies"]
            qualifiers[stage][d] = keep
            lines.append(f"| {NAMES[d]} | {LABELS[stage]} | "+" | ".join(best)+" | "+(", ".join(keep) or "None")+" |")
    lines.append("")
    for stage,codes in SWEEPS.items():
        hits = qualifiers[stage]
        union = [c for c in codes if any(c in choices for choices in hits.values())]
        if not union:
            text = f"Fix default {name(codes[0])} in the initial space. No alternative cleared the forward criterion."
        elif len(union)==1 and all(union[0] in choices for choices in hits.values()) and all(
                min(codes,key=lambda c:sweeps[stage]["timing"][c]["forward"])==union[0]
                for sweeps in trials.values()):
            text = f"Fix {name(union[0])} on the tested workloads. It qualified everywhere and had the lowest forward median on each."
        else:
            text = "Keep "+", ".join(name(c) for c in (codes[0],*union))+" as candidates. Qualifications differ by workload or order."
        lines.append(f"- **{LABELS[stage]}.** {text}")
    lines += ["","### Stage sensitivity and survival","",
              "The latency spread is the highest-to-lowest isolated median in a dataset's sweep. "
              "A large spread can reflect a slow alternative, so it is not evidence of speedup over default.","",
              "| Kernel | Largest isolated spread | Dataset | Corrected isolated gains | Corrected in-forward gains | Corrected whole-forward gains |",
              "|---|---:|---|---:|---:|---:|"]
    for stage,codes in SWEEPS.items():
        spreads = {}
        for d,sweeps in trials.items():
            samples = [sweeps[stage]["timing"][c]["isolated"] for c in codes]
            spreads[d] = max(samples)/min(samples)
        d = max(spreads,key=spreads.get)
        gains = [sum(r["sweep"]==stage and r["endpoint"]==metric and not r["alias"]
                     and r["holm_p"]<.05 and r["speedup"]>1 for r in rows)
                 for metric in ("isolated","in_forward","forward")]
        lines.append(f"| {LABELS[stage]} | {spreads[d]:.3f}× | {NAMES[d]} | "+" | ".join(map(str,gains))+" |")
    forward = {(r["dataset"],r["sweep"],r["code"]):r for r in rows if r["endpoint"]=="forward"}
    lines += ["","| Dataset | Isolated gain | Isolated result (95% CI, Holm p) | Forward result (95% CI, Holm p) | Forward decision |",
              "|---|---|---|---|---|"]
    gains = [r for r in rows if r["endpoint"]=="isolated" and not r["alias"] and r["holm_p"]<.05 and r["speedup"]>1]
    for row in gains:
        target = forward[row["dataset"],row["sweep"],row["code"]]
        lines.append(f'| {NAMES[row["dataset"]]} | {name(row["code"])} | {effect(row)} | {effect(target)} | {decision(target)} |')
    if not gains:
        lines.append("| — | None after correction | — | — | — |")
    lines += ["","An isolated gain can disappear inside forward when working-set context changes. Even a gain "
              "that survives in-forward need not produce a detectable full-layer win because projection and "
              "the other stages still run. Confidence intervals do not establish equal latency."]


def mechanism(lines,raw,aliases):
    lines += ["","## Access order, memory and compiled identity","",
              "All operands stay in the same 64-bit COO/CSC and FP32 NHF/NH/EH layouts. The table counts "
              "logical adjacency visits, not measured cache misses or physical memory traffic. The "
              "compiler can retain pointers in registers or unroll/vectorize fixed-size head/feature loops. "
              "Loop interchange and unroll-and-jam are disabled for all variants.","",
              "| Orders | Index visits | Layout and reuse |",
              "|---|---|---|",
              "| A1 / A2 | No adjacency | A1 traverses Z by consecutive node/head cells. A2 scans one head across nodes with a 32-float stride between feature runs. |",
              "| B1 / B2 | E / H·E COO edge visits | B1 reuses endpoints across contiguous EH heads. B2 follows EIDs at a fixed head with stride H. Same fused add and LeakyReLU. |",
              "| C1 / C2 | 3·H·E CSC EID visits | Per-row/head maximum, exponential/denominator and normalization scans. C2 revisits the destinations for each head. |",
              "| C3 | 3·E CSC EID visits | The same three passes coordinate adjacent heads for each edge; row/head reduction order stays fixed. |",
              "| D1 / D2 / D3 / D4 | E / H·E / H·F·E / H·E CSC entry visits | D1 reuses src/EID across heads; D2 and D4 work over contiguous feature vectors. D3 scans the complete incoming row for each output feature. |",
              "",
              "Each row's incoming CSC order is unchanged. Softmax uses the same 32-byte max/denominator "
              "scratch for all orders and the same FP32 fmax, exp and division. All aggregation orders "
              "do the same FP32 multiply/add per output cell in the same edge order. No packing, "
              "initialization shortcut, atomics, or per-forward index construction is used.","",
              "| Dataset | Target | Tensor payload KiB incl. output | Index payload KiB | Output KiB | Extra scratch bytes |",
              "|---|---|---:|---:|---:|---:|"]
    for dataset,rounds in raw["datasets"].items():
        for stage,v in rounds[0]["stage_payload"].items():
            lines.append(f'| {NAMES[dataset]} | {LABELS[stage]} | {v["tensor_bytes"]/1024:.2f} | '
                         f'{v["index_bytes"]/1024:.2f} | {v["output_bytes"]/1024:.2f} | {v["scratch_bytes"]} |')
    build = raw["build"]
    lines += ["","Payloads describe fixed live stage operands, not process RSS or actual cache traffic.","",
              f"Build: `{build['compiler']}`; `{' '.join(build['flags'])}`. "
              f"Binary SHA-256: `{build['binary_sha256']}`. "
              f"Source SHA-256: `{build['source_sha256']}`.","",
              "Each timed route binds the actual exported address during setup. Every loaded symbol offset "
              "matches the ELF, and address and normalized disassembly are checked for aliases.","",
              "| Config | C ABI symbol | Body bytes | Instructions | Normalized assembly SHA-256 (prefix) |",
              "|---|---|---:|---:|---|"]
    for code,v in build["symbols"].items():
        lines.append(f'| {code} | `{v["symbol"]}` | {v["size_bytes"]} | {v["instructions"]} | `{v["code_sha256"][:16]}` |')
    found = [f"{a}/{b}" for pairs in aliases.values() for a,b in pairs]
    lines += ["", ("Aliases found: "+", ".join(found)+". Aliases do not qualify as separate tuning choices."
                   if found else "No addresses or normalized instruction bodies matched within any sweep.")]
    if build["binary_sha256"]=="d7d1b5120d0b7767e3f734858e4e415b1866d7694286db95e19d65b4aa905130":
        lines += ["", "Disassembly of the rebuilt, byte-identical measured ELF helps explain why fewer index "
                  "visits did not make D1 faster. In D1, the backward branch at 0x2dfb returns to a loop with "
                  "cross-head shuffles, stack spills and output stores for every incoming edge. D2 holds two "
                  "feature vectors in registers across its edge loop at 0x2ea0 and stores them after the loop. "
                  "D4 keeps a similar inner loop but visits all destinations at a fixed head. D3 has a scalar "
                  "multiply/add chain and advances the feature loop outside each complete edge traversal. "
                  "This is evidence about this compiler's code generation, not a direct measurement of cache "
                  "misses or the isolated cost of any individual instruction."]


def render(raw,trials,rows,aliases):
    s = raw["settings"]
    lines = ["# Milestone 11 — Independent CPU GAT loop ordering","",
             "Custom orders differ only in the loop nest of the targeted kernel. Within each sweep the other "
             "three kernels use the same custom defaults A1/B1/C1/D1. Full forward is separately timed "
             "without internal clock reads. Six stage columns come from an execution-equivalent instrumented "
             "forward, never from isolated timings. All timing-table values are milliseconds and use the "
             "median of process medians.","",
             f"{s['rounds']} fresh process rounds per dataset, {s['warmup']} warmup blocks and "
             f"{s['repetitions']} measured randomized matched blocks per sweep. Each block contains "
             "the concurrently measured default and all alternatives. Process-median ratios use "
             "geometric means and pointwise 95% Student-t intervals on log ratios. Holm correction is "
             "separate by sweep and endpoint (full, isolated, in-forward), across the datasets and alternatives. "
             "The forward qualification rule requires Holm p < .05 and a 95% CI lower speedup bound > 1.01×.",""]
    primary(lines,trials,rows)
    diagnostic_tables(lines,trials,rows)
    scientific_conclusions(lines,trials,rows)
    mechanism(lines,raw,aliases)
    row = next(iter(raw["datasets"].values()))[0]
    env = row["environment"]
    checks = [v for rr in raw["datasets"].values() for stage in rr[0]["checks"].values() for v in stage.values()]
    outputs = [c["errors"]["output"] for c in checks]
    all_errors = [m for c in checks for m in c["errors"].values()]
    lines += ["","## Numerical verification and limits","",
              f"{env['cpu_model']}, logical CPU {row['cpu']}, one thread, Docker CPU base label {env['docker_image']}, "
              f"{env['os']}, Python {env['python']}, PyTorch {env['torch']}, DGL {env['dgl']}. "
              "The original citation graphs, EID ordering, native features and seed-0 weights were preserved. "
              "The existing self-loop rule removes all original loops and appends one per node. "
              "Inputs were FP32 with four heads, eight features per head, no dropout, residual, or output activation.","",
              "The independent DGL stage reference constructs projection, terms, scores, attention and "
              "aggregation with native primitives and checks final output and attention against the "
              "untouched imported GATConv. Every custom route verifies these intermediates, the original "
              "COO/CSC EID mapping, finite outputs and destination-wise normalization before measurement. "
              "All custom intermediates were bitwise equal across orders, and direct, repeated and "
              "instrumented outputs matched bitwise within the custom family. The stock comparison uses "
              "atol 1e−5 and rtol 1e−4 because its implementation may sum in a different order.","",
              f"Max final-output absolute error: {max(v['max_absolute'] for v in outputs):.3g}. "
              f"Max final-output relative error over nonzero reference values: "
              f"{max(v['max_relative_nonzero'] for v in outputs):.3g}. "
              f"Max error as fraction of tolerance across all stages: "
              f"{max(v['max_tolerance_fraction'] for v in all_errors):.3g}. "
              f"Max attention sum error: {max(c['max_normalization_error'] for c in checks):.3g}. "
              "Raw data contain per-stage absolute and relative errors, including errors on zero-reference entries. "
              "Ordinary relative error can be large near zero; the mixed tolerance is the correctness test.","",
              "The study has one CPU, three workloads, one head and channel configuration, and fixed "
              "storage and order. It does not measure cross-axis interactions, cache traffic, or transfer "
              "effects. The conclusions concern this compiled family and should be rechecked for a different compiler or CPU.","",
              "[Raw blocks and build metadata](RESULTS.json) · [Statistical analysis](ANALYSIS.json) · "
              "[Reproduction](README.md)",""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,default=HERE/"RESULTS.json")
    p.add_argument("--output",type=Path,default=HERE)
    a = p.parse_args()
    raw = json.loads(a.input.read_text())
    trials,aliases = audit(raw)
    rows = comparisons(trials,aliases)
    a.output.mkdir(parents=True,exist_ok=True)
    a.output.joinpath("REPORT.md").write_text(render(raw,trials,rows,aliases),encoding="utf-8",newline="\n")
    data = dict(raw_sha256=sha(a.input,text=True),build=raw["build"],datasets=trials,
                contrasts=rows,aliases=aliases,
                analysis_source_sha256={p.name:sha(p,text=True) for p in (Path(__file__),Path(spec.origin))})
    a.output.joinpath("ANALYSIS.json").write_text(
        json.dumps(data,separators=(",",":"),allow_nan=False)+"\n",encoding="utf-8",newline="\n")


if __name__=="__main__":
    main()
