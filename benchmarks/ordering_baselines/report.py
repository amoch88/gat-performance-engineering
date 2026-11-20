"""Replay the raw blocks and report separate node and edge contrasts."""
import argparse
import hashlib
import importlib.util
import json
from math import exp, isfinite, log, sqrt
from pathlib import Path
import random
from statistics import mean, median, stdev

import numpy as np
from scipy.stats import t

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("prior_statistics",HERE.parent/"sparse_formats/report.py")
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
NAMES = {"cora":"Cora","citeseer":"CiteSeer","pubmed":"PubMed"}
AXES = dict(node=("original","rcmk","degree_asc","degree_desc","bfs","random"),
            edge=("original","src","dst","random"))
POLICIES = dict(original="Original",rcmk="RCMK",degree_asc="Degree ascending",
                degree_desc="Degree descending",bfs="BFS",src="Source-grouped",
                dst="Destination-grouped",random="Random control")
STAGES = ("projection","terms","score","softmax","aggregation","output")
FORMATS = ("coo","csr","csc")
RANDOM = ("random_0","random_1","random_2")


def sha(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n",b"\n")).hexdigest()


def flags(actual, baseline):
    changed = [label for key,label in (("index_sha256","I"),("stored_eid_sha256","E"),
                                      ("original_eid_traversal_sha256","L"))
               if actual[key]!=baseline[key]]
    return "".join(changed) or "—"


def mean_interval(values):
    center = mean(values)
    radius = float(t.ppf(.975,len(values)-1))*stdev(values)/sqrt(len(values))
    return [center,center-radius,center+radius]


def decode(mapping,n):
    if set(mapping)=={"identity"}:
        assert mapping["identity"]==n
        return np.arange(n,dtype=np.int64)
    assert set(mapping)=={"new_to_original"}
    values = np.asarray(mapping["new_to_original"],dtype=np.int64)
    np.testing.assert_array_equal(np.sort(values),np.arange(n))
    return values


def check_maps(saved,row,dataset,axis,settings):
    cases = row["cases"][axis]
    n = row["graph"]["nodes"]
    e = row["graph"]["edges_with_self_loops"]
    assert list(saved)==list(cases)
    for f,case in cases.items():
        record = saved[f]
        nodes,edges = decode(record["nodes"],n),decode(record["edges"],e)
        ids = [hashlib.sha256(a.tobytes()).hexdigest() for a in (nodes,edges)]
        assert ids==case["maps_sha256"]
        assert "identity" in record["edges" if axis=="node" else "nodes"]
        assert record["seed"]==case["permutation_seed"]
        if f in RANDOM:
            seed = settings.get("permutation_seed_base",settings["order_seed"])+10000+100*list(NAMES).index(dataset)+10*list(AXES).index(axis)+RANDOM.index(f)
            assert seed==record["seed"]
            values = nodes if axis=="node" else edges
            np.testing.assert_array_equal(values,np.random.default_rng(seed).permutation(len(values)))
        else:
            assert record["seed"] is None
        assert all(case["after_materialization"]==v["after_materialization"] for v in cases.values())
        assert all(case["before_materialization"]==v["before_materialization"] for v in cases.values())
        check = case["correctness"]
        assert check["bitwise_profile_match"] and check["finite"]
        assert check["normalization_abs"]<=check["atol"]
        assert set(case["after_materialization"]["created"])==set(FORMATS)
        assert not case["after_materialization"]["not created"]


def round_data(row,settings,rng,catalog=AXES):
    results = {}
    for axis in row["axis_order"]:
        keys = [f for f in catalog[axis] if f!="random"]+list(RANDOM)
        assert list(row["cases"][axis])==keys
        schedule = [(mode,f) for mode in ("plain","profile") for f in keys]
        schedule += [("timer","empty"),("timer","six")]
        blocks = row["blocks"][axis]
        assert len(blocks)==settings["warmup"]+settings["repetitions"]
        samples = {f:{metric:[] for metric in (*STAGES,"forward","profile","overhead","unassigned")} for f in keys}
        calibration = []
        for i,block in enumerate(blocks):
            order = schedule.copy()
            rng.shuffle(order)
            assert [tuple(r[:2]) for r in block]==order
            plain,profile,timer = {},{},{}
            for mode,f,elapsed,*extra in block:
                assert elapsed>0 and isfinite(elapsed)
                if mode=="profile":
                    stages, = extra
                    assert len(stages)==len(STAGES) and min(stages)>0 and sum(stages)<=elapsed
                    profile[f] = elapsed,stages
                else:
                    assert not extra
                    (plain if mode=="plain" else timer)[f] = elapsed
            if i<settings["warmup"]:
                continue
            calibration.append((timer["six"]-timer["empty"])/1e6)
            for f,elapsed in plain.items():
                full,stages = profile[f]
                metrics = dict(zip(STAGES,stages,strict=True))
                metrics.update(forward=elapsed,profile=full,overhead=full-elapsed,unassigned=full-sum(stages))
                for metric,value in metrics.items():
                    samples[f][metric].append(value/1e6)
        results[axis] = dict(times={f:{key:median(values) for key,values in v.items()} for f,v in samples.items()},
                             timer_net_ms=median(calibration))
    build_schedule = [(axis,f) for axis in row["axis_order"] for f in row["cases"][axis]]
    assert len(row["construction"])==settings["build_repetitions"]
    costs = {key:[[],[],[],[]] for key in build_schedule}
    for block in row["construction"]:
        order = build_schedule.copy()
        rng.shuffle(order)
        assert [tuple(r[:2]) for r in block]==order
        for axis,f,plan,graph,formats in block:
            assert min(plan,graph,formats)>0
            for dest,value in zip(costs[axis,f],(plan,graph,formats,plan+graph+formats),strict=True):
                dest.append(value/1e6)
    for (axis,f),vals in costs.items():
        results[axis]["times"][f].update({metric:median(v) for metric,v in
            zip(("permutation","rebuild","formats","setup"),vals,strict=True)})
    return results


def summary(data,perm_path,initial_policy=None,*,catalog=AXES):
    assert data["status"]=="complete"
    assert sha(perm_path)==data["maps_sha256"]
    assert sha(HERE.parents[1]/"uv.lock")==data["lock_sha256"]
    maps = json.loads(perm_path.read_text())
    for f,digest in data["sources"].items():
        source = initial_policy if f=="ordering_baselines/policies.py" and initial_policy else HERE.parent/f
        assert sha(source)==digest, f
    settings = data["settings"]
    assert (settings["heads"],settings["channels"],settings["threads"],settings["model_seed"])==(4,8,1,0)
    exemplar = next(iter(data["datasets"].values()))[0]
    trials,contrasts = {},[]
    for d,rounds in data["datasets"].items():
        assert len(rounds)==settings["rounds"] and len({r["pid"] for r in rounds})==len(rounds)
        assert all(r["environment"]["dgl"]=="1.1.3" and r["environment"]["torch"]=="2.2.2+cpu" for r in rounds)
        first = rounds[0]
        collected = {axis:[] for axis in catalog}
        for i,row in enumerate(rounds):
            assert (row["dataset"],row["round"])==(d,i)
            env = row["environment"]
            assert row["cpu"]==exemplar["cpu"] and env["cpu_affinity"]==[row["cpu"]]
            assert env["torch_threads"]==env["torch_interop_threads"]==1
            for key in ("cpu_model","os","python","thread_environment","runtime_thread_controls"):
                assert env[key]==exemplar["environment"][key],key
            for key in ("cpu","weights","edge_sha256","graph_formats","loader_sparse","cases"):
                assert row[key]==first[key],key
            assert row["graph"]["feature_sha256"]==first["graph"]["feature_sha256"]
            assert row["cases"]["node"]["original"]==row["cases"]["edge"]["original"]
            for axis in catalog:
                check_maps(maps[d][axis],row,d,axis,settings)
            seed = settings["order_seed"]+1009*i+list(NAMES).index(d)
            assert row["order_seed"]==seed
            axes = list(catalog)
            rng = random.Random(seed)
            rng.shuffle(axes)
            assert row["axis_order"]==axes
            by_axis = round_data(row,settings,rng,catalog)
            for axis,entry in by_axis.items():
                collected[axis].append(entry)
        trials[d] = {}
        for axis,entries in collected.items():
            metrics = list(entries[0]["times"]["original"])
            values = {f:{metric:[r["times"][f][metric] for r in entries] for metric in metrics}
                      for f in first["cases"][axis]}
            values["random"] = {metric:[median(values[f][metric][i] for f in RANDOM) for i in range(len(entries))]
                                for metric in metrics}
            table = {f:{metric:median(v) for metric,v in m.items()} for f,m in values.items()}
            for policy in catalog[axis][1:]:
                for endpoint in (*STAGES,"forward"):
                    control = values["original"][endpoint]
                    candidate = values[policy][endpoint]
                    if policy=="random":
                        # Equal-weight log contrasts over the three seeds inside each process.
                        candidate = [exp(mean(log(values[f][endpoint][i]) for f in RANDOM)) for i in range(len(entries))]
                    item = prior.paired(control,candidate)
                    contrasts.append(dict(dataset=d,axis=axis,policy=policy,endpoint=endpoint,**item))
            trials[d][axis] = dict(table=table,round_medians_ms=values,representations=first["cases"][axis],
                timers_ms=[r["timer_net_ms"] for r in entries],loader_sparse=first["loader_sparse"],
                overhead_mean_and_ci_ms={f:mean_interval(v["overhead"]) for f,v in values.items()})
    for axis in catalog:
        for endpoint in (*STAGES,"forward"):
            prior.holm([r for r in contrasts if (r["axis"],r["endpoint"])==(axis,endpoint)])
    return trials,contrasts


def display_effect(row):
    return f'{row["speedup"]:.4f}× [{row["ci_low"]:.4f}, {row["ci_high"]:.4f}]'


def render(data,trials,contrasts,*,catalog=AXES,labels=POLICIES,include_forward_median=False):
    settings = data["settings"]
    floor = settings["practical_speedup_floor"]
    lines = ["# Milestone 10 — Independent node and edge ordering","",
             "Every forward uses the imported DGL 1.1.3 GATConv. The six stage columns come from a "
             "validated execution-equivalent copy of its forward. Total forward is timed directly on "
             "the unmodified import. All times are milliseconds and use the median of process medians.","",
             "The node sweep relabels nodes and feature rows while preserving every canonical EID. "
             "The edge sweep preserves node labels and feature rows. Both re-create the same original "
             "topology and order as their within-sweep control. Every node-ordering case allocates "
             "its feature rows with the same indexing operation, including the identity control. "
             "Every edge-ordering case shares the loader's unchanged feature tensor. Every graph starts "
             "from the same DGL constructor and materializes all COO, CSR, and CSC views before timing.","",
             "Use the original row within the same sweep as the control. The two sweeps have different "
             "case sets and feature working sets, so their absolute medians need not coincide.",""]
    for d,axes in trials.items():
        lines += [f"## {NAMES[d]}",""]
        for axis,result in axes.items():
            lines += [f"### {axis.title()} ordering","",
                "| Ordering | Projection | Attention Terms (L/R) | Attention Scores | Edge Softmax | Aggregation | Output | Total Forward |",
                "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for f in catalog[axis]:
                values = result["table"][f]
                lines.append(f"| {labels[f]} | "+" | ".join(f"{values[m]:.6f}" for m in (*STAGES,"forward"))+" |")
            lines.append("")
    lines += ["## Forward comparisons against original","",
             "Speedup divides the original time by the candidate time. Effects use the geometric mean "
             f"of {settings['rounds']} paired process-median ratios with pointwise 95% Student-t intervals in log space. "
             "Holm correction is separate for the node and edge sweeps across all datasets and planned "
             "policies, including random controls. The rules apply separately to each profiled stage. "
             f"The tuning criterion requires adjusted p < 0.05 and a 95% CI lower speedup bound above {floor:.2f}×.","",
             ("| Dataset | Sweep | Policy | Median forward ms | Paired speedup (95% CI) | Holm p | Supported |"
              if include_forward_median else "| Dataset | Sweep | Policy | Paired speedup (95% CI) | Holm p | Supported |"),
             "|---|---|---|"+("---:|" if include_forward_median else "")+"---|---:|---|"]
    seen = set()
    for r in contrasts:
        if r["endpoint"]!="forward":
            continue
        decision = "Gain" if r["holm_p"]<.05 and r["speedup"]>1 else "Loss" if r["holm_p"]<.05 else "Inconclusive"
        if decision=="Gain" and r["ci_low"]>floor and r["policy"]!="random":
            decision += " and meets tuning criterion"
        start = f'| {NAMES[r["dataset"]]} | {r["axis"]} | '
        timing = ""
        if include_forward_median:
            key = (r["dataset"],r["axis"])
            table = trials[key[0]][key[1]]["table"]
            if key not in seen:
                lines.append(start+f'Original | {table["original"]["forward"]:.6f} | 1.0000× (control) | — | — |')
                seen.add(key)
            timing = f'{table[r["policy"]]["forward"]:.6f} | '
        lines.append(start+f'{labels[r["policy"]]} | {timing}{display_effect(r)} | {r["holm_p"]:.4g} | {decision} |')
    lines += ["","## Conclusions",""]
    for axis in catalog:
        lines += [f"### {axis.title()} ordering",""]
        hits = set()
        for d,axes in trials.items():
            table = axes[axis]["table"]
            best = min(catalog[axis],key=lambda f:table[f]["forward"])
            recommended = [r["policy"] for r in contrasts if (r["dataset"],r["axis"],r["endpoint"])==(d,axis,"forward")
                           and r["policy"]!="random" and r["holm_p"]<.05 and r["ci_low"]>floor]
            hits.update(recommended)
            gain_text = (", ".join(labels[f] for f in recommended)+" met the tuning criterion."
                         if recommended else "The predetermined tuning criterion was unmet.")
            lines.append(f'- **{NAMES[d]}**. {labels[best]} had the lowest observed median '
                         f'({table[best]["forward"]:.4f} ms). {gain_text}')
        if hits:
            retained = [labels[f] for f in catalog[axis] if f in hits]
            lines += ["",f"Retain {axis} ordering as an independent candidate axis on the tested workloads. "
                      f"Keep Original and {', '.join(retained)}. Keep the seeded random runs as diagnostic controls."]
        else:
            lines += ["",f"Keep Original as the default for the {axis} sweep. "
                      "Omit the tested alternatives from the initial tuning space under this criterion. "
                      "Keep the measurements and seeded random controls for reference."]
        excluded = [labels[f] for f in catalog[axis][1:] if f!="random" and f not in hits]
        if excluded and hits:
            lines.append(f"Omit {', '.join(excluded)} from the initial tuning space under this criterion.")
        lines.append("")
    lines += ["## Stage effects","",
              "These counts summarize the same paired comparisons for individual profiled stages. "
              "Random controls are tabulated separately in the analysis JSON. Stage timings describe "
              "the instrumented execution and the full-forward comparison determines the tuning decision.","",
              "| Sweep | Stage | Supported policy-dataset gains | Supported losses |",
              "|---|---|---:|---:|"]
    for axis in catalog:
        for endpoint in STAGES:
            subset = [r for r in contrasts if (r["axis"],r["endpoint"])==(axis,endpoint) and r["policy"]!="random"]
            gains = sum(r["holm_p"]<.05 and r["speedup"]>1 for r in subset)
            losses = sum(r["holm_p"]<.05 and r["speedup"]<1 for r in subset)
            lines.append(f"| {axis} | {endpoint} | {gains}/{len(subset)} | {losses}/{len(subset)} |")
    lines += ["","The table below lists corrected stage differences for deterministic policies.","",
              "| Dataset | Sweep | Policy | Stage | Paired speedup (95% CI) | Holm p |",
              "|---|---|---|---|---|---:|"]
    supported = [r for r in contrasts if r["endpoint"]!="forward" and r["policy"]!="random" and r["holm_p"]<.05]
    for r in supported:
        lines.append(f'| {NAMES[r["dataset"]]} | {r["axis"]} | {labels[r["policy"]]} | {r["endpoint"]} | '
                     f'{display_effect(r)} | {r["holm_p"]:.4g} |')
    if not supported:
        lines.append("| — | — | — | — | Inconclusive after correction | — |")
    return lines


def describe_allocation(lines,initial,controlled):
    """Compare protocol variants without pooling them or testing between runs."""
    keys = ("dataset","axis","policy","endpoint")
    before = {tuple(row[k] for k in keys):row for row in initial}
    projection = [r for r in initial if r["axis"]=="node" and r["endpoint"]=="projection" and r["policy"]!="random"]
    gains = sum(r["holm_p"]<.05 and r["speedup"]>1 for r in projection)
    lines += ["","## Feature allocation sensitivity","",
              "The first full acquisition reused the loader tensor for the original node-order "
              "control and allocated a new tensor for every other node policy. That run showed "
              f"dense projection gains for {gains} of {len(projection)} deterministic policy/dataset pairs. "
              "We then repeated the full experiment with an identical feature-indexing allocation "
              "for every node policy, including the identity permutation. The matched-allocation "
              "acquisition is primary. The first run is archived and its samples are kept separate. "
              "The edge protocol was unchanged. Time variation between acquisitions also prevents "
              "attributing the entire difference to the allocator.","",
              "| Dataset | Sweep | Policy | First run (95% CI) | Controlled run (95% CI) |",
              "|---|---|---|---|---|"]
    for row in controlled:
        if row["endpoint"]!="forward":
            continue
        earlier = before[tuple(row[k] for k in keys)]
        lines.append(f'| {NAMES[row["dataset"]]} | {row["axis"]} | {POLICIES[row["policy"]]} | '
                     f'{display_effect(earlier)} | {display_effect(row)} |')
    lines += ["","Both runs use the same metric, paired process-median speedup. "
              "Each interval describes its own acquisition. No hypothesis test compares these intervals. "
              "[First raw run](initial_allocation/RESULTS.json) · "
              "[First-run policy source](initial_allocation/policies.py)."]


def describe_details(lines,trials,*,catalog=AXES,labels=POLICIES):
    lines += ["","## Random permutation variability","",
              "The main row uses the median of the three seed-specific medians within each process, followed "
              "by the median across processes. Its paired interval instead averages the three seed log "
              "ratios within each process. The interval treats the seeds as fixed and describes process variability. "
              "The variation across the three seed estimates is descriptive.","",
              "| Dataset | Sweep | Seed | Forward ms | SD across process medians, ms | Seed-specific paired speedup |",
              "|---|---|---:|---:|---:|---:|"]
    for d,axes in trials.items():
        for axis,entry in axes.items():
            for f in RANDOM:
                seed = entry["representations"][f]["permutation_seed"]
                values = entry["round_medians_ms"][f]["forward"]
                effect = prior.paired(entry["round_medians_ms"]["original"]["forward"],values)
                lines.append(f'| {NAMES[d]} | {axis} | {seed} | {median(values):.6f} | '
                             f'{stdev(values):.6f} | {effect["speedup"]:.4f}× |')
    lines += ["","## Instrumentation overhead","",
              "Each block interleaves direct forwards, profiled forwards and timer calibration. "
              "Overhead is the paired instrumented-minus-plain interval for the original case. "
              "The point estimate averages process medians of the paired differences. "
              "The empty six-region calibration measures timestamp reads, loop and record writes. "
              "Full per-policy overhead and unassigned graph-scope costs are in ANALYSIS.json.","",
              "| Dataset | Sweep | Empty six regions net µs (median) | Original overhead µs (95% CI) |",
              "|---|---|---:|---|"]
    for d,axes in trials.items():
        for axis,entry in axes.items():
            center,low,high = (1000*v for v in entry["overhead_mean_and_ci_ms"]["original"])
            lines.append(f'| {NAMES[d]} | {axis} | {median(entry["timers_ms"])*1000:.3f} | '
                         f'{center:+.3f} [{low:+.3f}, {high:+.3f}] |')
    lines += ["","## One-time costs and physical representation","",
              "Rebuild includes endpoint remapping, DGL graph creation and any data permutation. "
              "Materialization times the subsequent create_formats_ call. The original control also "
              "goes through this constructor. The node identity control includes a feature copy solely "
              "for measurement parity. Deployment can keep the original features without that copy. "
              "Random rows show seed medians.","",
              "Bandwidth is max |src − dst| in current labels. Endpoint step is the mean absolute "
              "ID difference between consecutive COO edges. Compressed-view hashes and locality "
              "statistics are recorded in ANALYSIS.json. The change flags compare actual arrays "
              "against the original within the sweep. I means stored adjacency indices changed, "
              "E means the stored traversal-to-current-EID map changed, and L means traversal in "
              "original EIDs changed. Random physical flags cover each of the three seeds in order.","",
              "| Dataset | Sweep | Ordering | Bandwidth | COO src / dst step | COO / CSR / CSC changes | Permutation ms | Rebuild ms | Materialization ms |",
              "|---|---|---|---:|---|---|---:|---:|---:|"]
    for d,axes in trials.items():
        for axis,entry in axes.items():
            base = entry["representations"]["original"]["sparse"]
            for policy in catalog[axis]:
                keys = RANDOM if policy=="random" else (policy,)
                reps = [entry["representations"][f] for f in keys]
                bandwidth = median(v["graph"]["bandwidth"] for v in reps)
                steps = [median(v["sparse"]["coo"][direction]["mean_step"] for v in reps) for direction in ("src","dst")]
                change = " / ".join(",".join(flags(v["sparse"][fmt],base[fmt]) for v in reps) for fmt in FORMATS)
                v = entry["table"][policy]
                lines.append(f'| {NAMES[d]} | {axis} | {labels[policy]} | {bandwidth:g} | '
                             f'{steps[0]:.2f} / {steps[1]:.2f} | {change} | '
                             f'{v["permutation"]:.6f} | {v["rebuild"]:.6f} | {v["formats"]:.6f} |')
    lines += ["","The node sweep retains canonical EIDs. Its CSR and CSC rows may traverse those EIDs "
              "in a different sequence because compressed rows follow renumbered node IDs. The edge "
              "sweep changes EIDs directly. A compressed view with unchanged I can still change E because "
              "the stored entries refer to new edge labels. The COO EID order is implicit."]
    for policy,fmt in (("src","csr"),("dst","csc")):
        if all(flags(axes["edge"]["representations"][policy]["sparse"][fmt],
                     axes["edge"]["representations"]["original"]["sparse"][fmt])=="E"
               for axes in trials.values()):
            lines.append(f"{labels[policy]} preserved the {fmt.upper()} pointers, neighbor indices and "
                         "traversal in original EIDs on every dataset. Its change within that view "
                         "was confined to the mapping to current EIDs.")
    lines += ["","The cached loader and common rebuilt baseline are compared below.","",
              "| Dataset | COO / CSR / CSC changes relative to loader |",
              "|---|---|"]
    for d,axes in trials.items():
        e = axes["node"]
        comparison = " / ".join(flags(e["representations"]["original"]["sparse"][fmt],e["loader_sparse"][fmt])
                                for fmt in FORMATS)
        lines.append(f"| {NAMES[d]} | {comparison} |")
    lines += ["","| Dataset | Nodes | Edges | Self-loops | Parallel edges beyond first | Total degree mean / median / p95 / max |",
              "|---|---:|---:|---:|---:|---|"]
    for d,axes in trials.items():
        rep = axes["node"]["representations"]["original"]
        g = rep["graph"]
        degree = g["total_degree"]
        n = rep["sparse"]["csr"]["exposed_array_lengths"][0]-1
        e = rep["sparse"]["coo"]["exposed_array_lengths"][0]
        lines.append(f'| {NAMES[d]} | {n:,} | {e:,} | {g["self_loops"]:,} | {g["parallel_edges"]:,} | '
                     f'{degree["mean"]:.2f} / {degree["median"]:.0f} / {degree["p95"]:.0f} / {degree["max"]} |')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,default=HERE/"RESULTS.json")
    p.add_argument("--maps",type=Path,default=HERE/"PERMUTATIONS.json")
    p.add_argument("--output",type=Path,default=HERE)
    p.add_argument("--initial-input",type=Path,
                   help="Separate first acquisition for the feature-allocation sensitivity table")
    a = p.parse_args()
    data = json.loads(a.input.read_text())
    trials,contrasts = summary(data,a.maps)
    lines = render(data,trials,contrasts)
    archive = HERE/"initial_allocation"
    initial_input = a.initial_input or (archive/"RESULTS.json" if a.input.resolve()==HERE/"RESULTS.json" else None)
    initial_details = {}
    if initial_input:
        initial_data = json.loads(initial_input.read_text())
        assert initial_data["settings"]==data["settings"] and initial_data["datasets"].keys()==data["datasets"].keys()
        for d,rounds in data["datasets"].items():
            first = initial_data["datasets"][d][0]
            for field in ("cpu","weights","edge_sha256"):
                assert first[field]==rounds[0][field],field
            assert first["graph"]["feature_sha256"]==rounds[0]["graph"]["feature_sha256"]
            assert first["environment"]["cpu_model"]==rounds[0]["environment"]["cpu_model"]
            for axis in AXES:
                for key,case in rounds[0]["cases"][axis].items():
                    for field in ("policy","maps_sha256","sparse"):
                        assert first["cases"][axis][key][field]==case[field],field
        _,initial = summary(initial_data,a.maps,initial_policy=archive/"policies.py")
        describe_allocation(lines,initial,contrasts)
        initial_details.update(initial_raw_sha256=sha(initial_input),initial_contrasts=initial)
    describe_details(lines,trials)
    cfg = data["settings"]
    env = next(iter(data["datasets"].values()))[0]
    finishes = [r["correctness"] for axes in trials.values() for e in axes.values() for r in e["representations"].values()]
    lines += ["","## Protocol and correctness","",
              f"{cfg['rounds']} fresh processes per dataset on {env['environment']['cpu_model']} with logical CPU "
              f"{env['cpu']} pinned and one inference thread. {cfg['warmup']} warmup and "
              f"{cfg['repetitions']} paired randomized blocks per sweep per process. "
              f"{cfg['build_repetitions']} separately randomized construction observations per case and process. "
              f"The CPU Docker environment reported Python {env['environment']['python']}, "
              f"PyTorch {env['environment']['torch']} and {env['environment']['os']}.","",
              "Exact seed-0 weights, FP32, four heads and eight channels per head, native features, inference mode "
              "and zero dropout. Existing preprocessing removes input self-loops and appends one per node. "
              "Case validation checks exact EID and node permutations, endpoint multiplicity, graph data, "
              "restored outputs, restored attention and all per-destination attention sums. "
              f"Maximum absolute output error {max(x['output_abs'] for x in finishes):.3g}, attention error "
              f"{max(x['attention_abs'] for x in finishes):.3g}, normalization error "
              f"{max(x['normalization_abs'] for x in finishes):.3g}. Timing copies matched direct outputs "
              "and attention bitwise. Tolerances were atol 1e−5 and rtol 1e−4.","",
              "Degree sorts use total incoming plus outgoing degree and original-ID ties. RCMK uses "
              "SciPy's reverse Cuthill–McKee on the symmetric structural adjacency. BFS uses the same "
              "undirected adjacency with ascending original-ID roots and neighbors, including every "
              "disconnected component. Edge grouping uses stable original-EID ties. "
              "Both random controls use NumPy default_rng with recorded explicit seeds.","",
              "The study covers three graphs on one CPU with two one-factor sweeps. "
              "Conclusions apply to the fixed official kernels and graph-format policy used here. "
              "Evidence for retaining a choice refers to changing one axis at a time. These measurements "
              "cannot establish that node and edge ordering effects compose or are additive. "
              "Forward returns the output in the graph's current node order. Restoring canonical IDs "
              "for validation is outside forward timing and would cost extra for callers requiring it.","",
              "[Raw timings](RESULTS.json) · [Permutation maps](PERMUTATIONS.json) · "
              "[Correctness and paired statistics](ANALYSIS.json)",""]
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    (a.output/"ANALYSIS.json").write_text(json.dumps(dict(datasets=trials,contrasts=contrasts,
                raw_sha256=sha(a.input),maps_sha256=sha(a.maps),**initial_details),
                allow_nan=False,separators=(",",":"))+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
