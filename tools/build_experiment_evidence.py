"""Regenerate and verify source-linked experiment evidence (stdlib only)."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
STORAGE = "comprehensive_sparse_formats"
TIMING = "sparse_format_validation"
REORDERING = "ordering_baselines"
REORDERING_EXT = "ordering_baselines/extended"
LOOP_NEST = "loop_ordering"
NAMES = dict(cora="Cora", citeseer="CiteSeer", pubmed="PubMed")
SIZES = (2, 4, 8, 16, 32, 64)
FAMILIES = dict(bcsr="Blocked CSR", bcsc="Blocked CSC", csb="CSB", dense="Block Dense", sell="SELL")
ORDERS = dict(original="Original", rcmk="RCMK", degree_asc="Degree ascending", degree_desc="Degree descending",
              bfs="BFS", random="Random control", metis="METIS (32)", src="Source-grouped", dst="Destination-grouped",
              src_dst="Source then destination", dst_src="Destination then source", morton="Morton")
LOOPS = dict(terms=("A1","A2"), score=("B1","B2"), softmax=("C1","C2","C3"), aggregation=("D1","D2","D3","D4"))
STAGES = ("projection","terms","score","softmax","aggregation","output","forward")


def from_bench(experiment, file):
    return json.loads((ROOT / "benchmarks" / experiment / file).read_text(encoding="utf-8"))


def path(m, file="ANALYSIS.json"):
    return f"benchmarks/{m}/{file}"


def source(m, selector, file="ANALYSIS.json"):
    return f"[{m}]({f'../{path(m,file)}'}) · `{selector}`"


def table(headers, rows):
    def line(cells):
        return "| " + " | ".join(str(c).replace("|", r"\|") for c in cells) + " |"
    return "\n".join((line(headers), line(["---"]*len(headers)), *(line(r) for r in rows)))


def choose(rows, **key):
    result, = [r for r in rows if all(r[k] == v for k,v in key.items())]
    return result


def ms(t):
    return f"{t:.6f}"


def stat(row):
    if row is None:
        return ("—", "—", "—")
    return (f"{row['speedup']:.5f}×", f"[{row['ci_low']:.5f}, {row['ci_high']:.5f}]",
            f"{row['holm_p']:.5g}")


def label(case):
    if case=="stock":
        return "Imported DGL GATConv"
    if case in ("coo","csr","csc"):
        return "Native "+case.upper()+" hybrid"
    if case=="custom_coo":
        return "Custom COO"
    if case=="matched_coo":
        return "Custom COO matched hybrid"
    for prefix,title in FAMILIES.items():
        if case.startswith(prefix):
            k = case[len(prefix):]
            return f"{title} {k}" if prefix=="sell" else f"{title} {k}×{k}"
    raise ValueError(case)


def result(row, floor=None):
    if row["holm_p"]>=.05:
        return "Inconclusive after Holm"
    if row["speedup"]<1:
        return "Supported slowdown"
    if floor is None:
        return "Supported gain"
    return "Meets practical gate" if row["ci_low"]>floor else "Supported gain below practical gate"


def evidence():
    campaigns = (STORAGE,TIMING,REORDERING,REORDERING_EXT,LOOP_NEST)
    raw = {m:from_bench(m,"RESULTS.json") for m in campaigns}
    a = {m:from_bench(m,"ANALYSIS.json") for m in campaigns}
    claims, blocks = [], {}
    for m in campaigns:
        assert raw[m]["status"]=="complete"
        assert all(len(rows)==8 for rows in raw[m]["datasets"].values())
        assert (raw[m]["settings"]["rounds"],raw[m]["settings"]["warmup"],raw[m]["settings"]["repetitions"])==(8,10,30)
    for d in NAMES:
        base = raw[STORAGE]["datasets"][d][0]
        for m in campaigns:
            run = raw[m]["datasets"][d][0]
            assert run["weights"]==base["weights"] and run["edge_sha256"]==base["edge_sha256"]
            assert run["graph"]["feature_sha256"]==base["graph"]["feature_sha256"]

    def record(statement,m,selector,candidate,control,comparison,note,floor=None):
        claims.append([f"C{len(claims)+1:02}",statement,source(m,selector),ms(candidate),ms(control),
                       f"{control/candidate:.5f}×",*stat(comparison),result(comparison,floor),note])

    dt = []
    for d,name in NAMES.items():
        g = raw[STORAGE]["datasets"][d][0]["graph"]
        n,e = g["nodes"],g["edges_with_self_loops"]
        degree = a[REORDERING_EXT]["datasets"][d]["node"]["representations"]["original"]["graph"]["total_degree"]
        dt.append([name,f"{n:,}",f"{g['dataset_edges']:,}",f"{e:,}",f"{g['original_features']:,}",
                  f"{e/n:.3f}",f"{degree['max']:,}",g["source"].removeprefix("DGL 1.1.3 ")])
    blocks["datasets"] = table(("Dataset","N","Loader directed E","Timed directed E","Input width","Timed E/N",
                                "Max total degree","DGL class"),dt)
    blocks["protocol"] = table(("Acquisition","Fresh processes per graph","Warmup blocks","Measured blocks",
                                "Build observations","Order seed"),[
        (tag,s["rounds"],s["warmup"],s["repetitions"],s.get("build_repetitions","not swept"),
         s.get("order_seed",s.get("seed"))) for tag,m in zip(("Sparse Storage Format Evaluation","Sparse Backend and Timing Validation",
             "Graph Reordering (matched allocation)","Graph Reordering (extended sweeps)","Loop-Nest Optimization"),campaigns)
        for s in (raw[m]["settings"],)
    ])

    def storage_row(d,stage,candidate,control):
        ep = a[STORAGE]["timings"][d]
        values = ep["forward"] if stage=="forward" else {
            k.split("/")[1]:v for k,v in ep["stage"].items() if k.startswith(stage+"/")}
        cmp = None if candidate==control else choose(a[STORAGE]["comparisons"],dataset=d,stage=stage,format=candidate,control=control)
        return values[candidate]["ms"],values[control]["ms"],cmp

    def storage_claim(d,stage,candidate,control,statement,note):
        record(statement,STORAGE,f"comparisons[{d},{stage},{candidate},{control}]",*storage_row(d,stage,candidate,control),note)

    results, stage_rows, capacity, eligibility = [], [], [], []
    full_grid = ["stock","coo","csr","csc",*(f"{kind}{size}" for kind in FAMILIES for size in SIZES)]
    for d,name in NAMES.items():
        forward = a[STORAGE]["timings"][d]["forward"]
        winner = min(forward,key=lambda f:forward[f]["ms"])
        v,b,c = storage_row(d,"forward",winner,"stock")
        results.append([name,label(winner),ms(b),ms(v),f"{b/v:.5f}×",*stat(c)])
        storage_claim(d,"forward",winner,"stock",f"{label(winner)}, the fastest observed storage configuration on {name}, beat stock DGL in direct forward.",
                "An implementation contrast; not a layout-only claim.")
        controls = ("stock","coo","csr","csc")
        min_speedups = {}
        for f in full_grid[4:]:
            cs = [storage_row(d,"forward",f,control)[2] for control in controls]
            if all(c["holm_p"]<.05 and c["speedup"]>1 for c in cs):
                min_speedups[f] = min(cs,key=lambda c:c["speedup"])
        best = max(min_speedups,key=lambda f:min_speedups[f]["speedup"])
        r = min_speedups[best]
        eligibility.append([name,len(min_speedups),label(best),label(r["control"]),*stat(r)])
        for stage,controls in (("score",("coo","csr")),("softmax",("csc",)),("aggregation",("coo","csc"))):
            times = {k.split("/")[1]:x["ms"] for k,x in a[STORAGE]["timings"][d]["stage"].items()
                     if k.startswith(stage+"/")}
            f,ref = min(times,key=times.get),min(controls,key=times.get)
            v,b,c = storage_row(d,stage,f,ref)
            stage_rows.append([name,stage,label(f),ms(v),label(ref),ms(b),*stat(c)])
        e = raw[STORAGE]["datasets"][d][0]["graph"]["edges_with_self_loops"]
        for f in ("bcsc2","bcsc64","csb2","csb64","dense2","dense64","sell2","sell64"):
            m = a[STORAGE]["storage"][d][f]
            capacity.append([name,label(f),f"{m['bytes']/1024:.2f}",ms(m["build_ms"]),
                             f"{m['slots']/e:.2f}" if m.get("slots") is not None and f.startswith(("sell","dense")) else "—"])
    blocks["storage-forward"] = table(("Dataset","Observed winner","Stock ms","Candidate ms","Raw median ratio",
                                  "Paired speedup","95% CI","Holm p"),results)
    blocks["storage-eligibility"] = table(("Dataset","Number beating all four controls","Best qualifying minimum over controls",
                                      "Limiting control by speedup","Paired speedup","95% CI","Holm p"),eligibility)
    blocks["storage-grid"] = table(("Implementation","Cora ms","CiteSeer ms","PubMed ms"),
        [(label(f),*(ms(a[STORAGE]["timings"][d]["forward"][f]["ms"]) for d in NAMES)) for f in full_grid])
    blocks["storage-sparse"] = table(("Dataset","Isolated stage","Fastest observed","ms","Native comparator","ms",
                                 "Paired speedup","95% CI","Holm p"),stage_rows)
    blocks["storage-storage"] = table(("Dataset","Format","Index KiB","Build ms","Padded slots per original edge"),capacity)
    storage_claim("citeseer","forward","csb2","csc",
            "CiteSeer CSB 2×2 versus native CSC was inconclusive after adjustment.",
            "Fastest raw forward median and a stock win did not clear the four-control gate.")
    storage_claim("citeseer","forward","csb16","csc","CiteSeer CSB 16×16 beat native CSC.",
            "This was the weakest paired speedup of its four qualifying control comparisons.")
    storage_claim("cora","softmax","csb64","csc","Cora CSB 64×64 improved isolated softmax versus native CSC.",
            "Prepared inputs; neither whole-forward nor layout-only.")
    storage_claim("citeseer","aggregation","csb64","csc","CiteSeer CSB 64×64 improved isolated aggregation versus native CSC.",
            "This was not CiteSeer's fastest measured whole-forward format.")
    storage_claim("pubmed","aggregation","csb4","csc","PubMed CSB 4×4 improved isolated aggregation versus native CSC.",
            "This was not PubMed's fastest measured whole-forward format.")
    storage_claim("pubmed","forward","dense64","stock","PubMed Block Dense 64×64 slowed whole forward relative to stock.",
            "Large masked-tile payload is measured; per-instruction and cache costs are not.")
    storage_claim("pubmed","forward","sell64","sell2","PubMed SELL 64 slowed whole forward relative to SELL 2.",
            "Same implementation family. The source contrast uses the larger size as candidate.")

    def timing_row(d,category,candidate,control):
        row = choose(a[TIMING]["contrasts"],dataset=d,family=category,endpoint="forward",candidate=candidate,control=control)
        times = a[TIMING]["datasets"][d]["timing"]
        return times["forward/"+candidate]["median_ms"],times["forward/"+control]["median_ms"],row

    timing_comparisons = []
    for d,cat,f,c,wording,note in (
        ("cora","matched backend","matched_coo","coo","On Cora, the custom COO-matched hybrid beat the native COO hybrid.",
         "The route pattern is matched; the backend implementation changes."),
        ("pubmed","matched storage","sell2","custom_coo","On PubMed, SELL 2 was slower than custom COO in the matched-storage study.",
         "Independent controlled rerun. A SELL-versus-stock speedup does not establish a layout-only gain."),
    ):
        v,b,row = timing_row(d,cat,f,c)
        record(wording,TIMING,f"contrasts[{d},{cat},forward,{f},{c}]",v,b,row,note)
        timing_comparisons.append([NAMES[d],label(f),ms(v),label(c),ms(b),*stat(row),result(row)])
    blocks["timing-checks"] = table(("Dataset","Candidate","ms","Comparator","ms","Paired speedup","95% CI","Holm p","Decision"),
                                timing_comparisons)

    def ordering_row(m,d,axis,case,control="original"):
        rows = a[m]["datasets"][d][axis]["table"]
        if control=="original":
            row = choose(a[m]["contrasts"],dataset=d,axis=axis,policy=case,endpoint="forward")
        else:
            row = choose(a[m]["new_edges_vs_dst"],dataset=d,policy=case,control=control)
        return rows[case]["forward"],rows[control]["forward"],row

    ordering_rows, setup = [], []
    for d,name in NAMES.items():
        for m,desc in ((REORDERING,"Matched"),(REORDERING_EXT,"Extension")):
            ns = a[m]["datasets"][d]["node"]["table"]
            node = min((p for p in ns if p in ORDERS and p not in ("original","random")),
                       key=lambda p:ns[p]["forward"])
            for axis,case in (("node",node),("edge","dst"),*(
                    (("node","metis"),("edge","dst_src")) if m==REORDERING_EXT and node!="metis" else
                    (("edge","dst_src"),) if m==REORDERING_EXT else ())):
                v,b,row = ordering_row(m,d,axis,case)
                ordering_rows.append([name,desc,axis,ORDERS[case],ms(b),ms(v),*stat(row),result(row,1.01)])
        for axis,case in (("node","metis"),("edge","dst"),("edge","dst_src")):
            r = a[REORDERING_EXT]["datasets"][d][axis]["table"][case]
            setup.append([name,ORDERS[case],*(ms(r[k]) for k in ("permutation","rebuild","formats"))])
    blocks["reordering-forward"] = table(("Dataset","Acquisition","Axis","Candidate","Original ms","Candidate ms",
                                   "Paired speedup","95% CI","Holm p","Decision"),ordering_rows)
    blocks["reordering-overhead"] = table(("Dataset (extension)","Policy","Plan / replay ms","Rebuild ms","Formats ms"),setup)
    secondaries = []
    for d,name in NAMES.items():
        v,b,row = ordering_row(REORDERING_EXT,d,"edge","dst_src","dst")
        secondaries.append([name,ms(b),ms(v),*stat(row),result(row,1.01)])
    blocks["reordering-secondary"] = table(("Dataset","Destination-grouped ms","Destination then source ms",
                     "Paired speedup","95% CI","Holm p","Decision"),secondaries)
    for m,d,axis,case,control,words,note in (
        (REORDERING,"pubmed","edge","dst","original","PubMed Destination-grouped qualified in the matched-allocation reordering experiment.",
         "Stock GATConv within the edge sweep; historic magnitude is not pooled."),
        (REORDERING_EXT,"pubmed","edge","dst","original","PubMed Destination-grouped qualified again in the extension.",
         "The extension measured a smaller relative gain; both acquisitions passed independently."),
        (REORDERING_EXT,"pubmed","edge","dst_src","original","PubMed Destination then source qualified against Original.",
         "It is the observed extension winner, not an established winner against Destination-grouped."),
        (REORDERING_EXT,"pubmed","edge","dst_src","dst","PubMed Destination then source versus Destination-grouped is inconclusive.",
         "The nine-test secondary Holm family is separate. Do not claim equivalence."),
        (REORDERING_EXT,"cora","node","metis","original","Cora METIS versus Original was inconclusive after Holm.",
         "Fixed 32-partition plan; point estimate or stage wins do not qualify node ordering."),
    ):
        sel = f"contrasts[{d},{axis},{case},forward]" if control=="original" else f"new_edges_vs_dst[{d},{case}]"
        record(words,m,sel,*ordering_row(m,d,axis,case,control),note,1.01)

    def loop_row(d,sweep,code,endpoint):
        base = LOOPS[sweep][0]
        values = a[LOOP_NEST]["datasets"][d][sweep]["timing"]
        row = choose(a[LOOP_NEST]["contrasts"],dataset=d,sweep=sweep,code=code,endpoint=endpoint)
        return values[code][endpoint],values[base][endpoint],row

    loop_qualified, winners, stocks, pub = [], [], [], []
    for d,name in NAMES.items():
        stocks.append([name,*(ms(a[LOOP_NEST]["datasets"][d]["stock"]["timing"]["DGL"][s]) for s in STAGES)])
        for sweep,codes in LOOPS.items():
            times = a[LOOP_NEST]["datasets"][d][sweep]["timing"]
            winners.append([name,sweep,*(min(codes,key=lambda c:times[c][k]) for k in ("isolated","in_forward","forward"))])
            for code in codes[1:]:
                v,b,r = loop_row(d,sweep,code,"forward")
                if r["holm_p"]<.05 and r["ci_low"]>1.01 and not r["alias"]:
                    loop_qualified.append([name,sweep,code,codes[0],ms(b),ms(v),*stat(r)])
                    record(f"{name} {code} met the practical whole-forward criterion.",LOOP_NEST,
                           f"contrasts[{d},{sweep},{code},forward]",v,b,r,
                           "Only the target loop order changed; control is custom, not stock DGL.",1.01)
    for endpoint in ("isolated","in_forward","forward"):
        v,b,r = loop_row("pubmed","aggregation","D2",endpoint)
        pub.append([endpoint,ms(b),ms(v),*stat(r)])
    for d,sweep,code,endpoint, words,note in (
        ("pubmed","aggregation","D2","isolated","PubMed D2 improved isolated aggregation versus D1.",
         "Identical prepared operands and allocations; this is not whole-layer speedup."),
        ("pubmed","aggregation","D2","in_forward","PubMed D2 improved aggregation inside forward versus D1.",
         "Instrumented region has freshly produced operands. Complete-forward speedup is a separate contrast."),
        ("cora","softmax","C3","isolated","Cora C3 improved isolated softmax versus C1.",
         "The corrected whole-forward comparison on Cora was inconclusive."),
        ("cora","softmax","C3","forward","Cora C3 versus C1 whole forward was inconclusive.",
         "Do not read this as equivalence or negate the supported isolated gain."),
    ):
        record(words,LOOP_NEST,f"contrasts[{d},{sweep},{code},{endpoint}]",*loop_row(d,sweep,code,endpoint),note,
               1.01 if endpoint=="forward" else None)
    assert len(claims)==25,len(claims)
    blocks["loop-nest-forward"] = table(("Dataset","Sweep","Candidate","Default","Default ms","Candidate ms",
                                    "Paired speedup","95% CI","Holm p"),loop_qualified)
    blocks["loop-nest-ranks"] = table(("Dataset","Sweep","Fastest isolated","Fastest in-forward","Fastest forward"),winners)
    blocks["loop-nest-stock"] = table(("Dataset",*STAGES),stocks)
    blocks["loop-nest-pubmed"] = table(("Endpoint","D1 med. ms","D2 med. ms","Paired speedup","95% CI","Holm p"),pub)

    md = "# Experimental Findings\n\n"
    md += ("Latency is the median of the within-process medians, in milliseconds. **Raw ratio** divides "
           "the two published medians; **paired speedup** is the geometric mean of within-process "
           "control/candidate ratios. Above 1 favors the candidate. Intervals are pointwise 95% "
           "Student-t intervals on log ratios; p-values are Holm-adjusted within the source's own family. "
           "The reordering and loop-nest studies require adjusted p < .05 and a lower confidence bound > 1.01×. "
           "The storage evaluation uses a distinct four-reference rule; backend and timing validation is a separate follow-up. Inconclusive "
           "does not mean equivalent. Source selectors identify records in JSON lists by their fields rather than by row number.\n\n")
    md += table(("ID","Exact claim","Source","Candidate ms","Baseline ms","Raw ratio","Paired speedup",
                 "95% CI","Holm p","Result","Caveat"),claims)+"\n\n"
    md += "Source SHA-256 digests below pin the analysis artifacts used for these values.\n\n"
    md += table(("Repository-relative path","SHA-256"),[
        (f"`{path(m)}`","`"+hashlib.sha256((ROOT/path(m)).read_bytes()).hexdigest()+"`") for m in campaigns])+"\n"
    return blocks,md


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true",help="Verify regenerated evidence and local documentation links")
    args = parser.parse_args()
    blocks,claims = evidence()
    analysis = DOCS/"TECHNICAL_ANALYSIS.md"
    original = analysis.read_text(encoding="utf-8")
    target = original
    for key,body in blocks.items():
        start,end = f"<!-- evidence:{key}:start -->",f"<!-- evidence:{key}:end -->"
        pattern = re.escape(start)+r".*?"+re.escape(end)
        target,count = re.subn(pattern,lambda m:f"{start}\n\n{body}\n\n{end}",target,flags=re.S)
        assert count==1,key
    findings = DOCS/"EXPERIMENTAL_FINDINGS.md"
    if args.check:
        assert analysis.read_bytes()==target.encode("utf-8"),"Technical analysis tables differ from their evidence"
        assert findings.read_bytes()==claims.encode("utf-8"),"Experimental findings differ from their evidence"
        links = 0
        active_docs = [ROOT/"README.md", *sorted(DOCS.glob("*.md")), *sorted((ROOT/"benchmarks").rglob("README.md"))]
        for md in active_docs:
            for uri in re.findall(r"\]\(([^\s)]+)\)",md.read_text(encoding="utf-8")):
                if "://" in uri or uri.startswith("#"):
                    continue
                relative = uri.split("#",1)[0]
                assert (md.parent/relative).exists(),f"Missing link target: {md.name}: {relative}"
                links += 1
        print(f"Experiment evidence is reproducible; {links} local links resolve")
    else:
        analysis.write_text(target,encoding="utf-8",newline="\n")
        findings.write_text(claims,encoding="utf-8",newline="\n")


if __name__=="__main__":
    main()
