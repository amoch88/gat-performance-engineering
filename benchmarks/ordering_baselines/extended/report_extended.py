"""Validate, replay and compare the expanded ordering experiment."""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import random

from benchmark_extended import HERE, dgl, saved_metis_nodes
from policies_ext import ADDED_EDGE, AXES, LABELS, MORTON_CONVENTION, PARTITIONS, np

HISTORY = HERE.parent
HISTORICAL_COMMIT = "5b338b132f552697b9569bd2886e686fba9b02a4"
spec = importlib.util.spec_from_file_location("ordering_baseline_report", HISTORY / "report.py")
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
NAMES = common.NAMES
LABELS = {**common.POLICIES, **LABELS}


def verify_protocol(raw, maps_path, metis_path):
    assert common.sha(metis_path) == raw["metis_sha256"]
    settings = raw["settings"]
    assert settings["metis_partitions"] == PARTITIONS and settings["morton_key"] == MORTON_CONVENTION
    rng = random.Random(settings["order_seed"])
    assert len(raw["dataset_orders"]) == settings["rounds"]
    for measured in raw["dataset_orders"]:
        order = list(raw["datasets"])
        rng.shuffle(order)
        assert measured == order
    metis = json.loads(metis_path.read_text())
    maps = json.loads(maps_path.read_text())
    for dataset, rounds in raw["datasets"].items():
        row, plan = rounds[0], metis[dataset]
        assert plan["cpu"] == row["cpu"]
        assert plan["input_edge_sha256"] == row["edge_sha256"]
        assert plan["feature_sha256"] == row["graph"]["feature_sha256"]
        for key in ("dgl", "torch", "cpu_model", "cpu_affinity"):
            assert plan["environment"][key] == row["environment"][key]
        assert plan["api_source_sha256"] == hashlib.sha256(
            inspect.getsource(dgl.metis_partition_assignment).encode()).hexdigest()
        assert (plan["api"], plan["mode"], plan["objective"], plan["balance_edges"]) == (
            "dgl.metis_partition_assignment", "k-way", "cut", False)
        n = row["graph"]["nodes"]
        planned = saved_metis_nodes(plan, n)
        np.testing.assert_array_equal(common.decode(maps[dataset]["node"]["metis"]["nodes"], n), planned)
        for axis in AXES:
            for case in row["cases"][axis].values():
                assert case["graph"]["self_loops"] == n
    return {d: metis[d] for d in raw["datasets"]}


def secondary_edge_contrasts(trials):
    rows = []
    for dataset, sweeps in trials.items():
        values = sweeps["edge"]["round_medians_ms"]
        for policy in ADDED_EDGE:
            effect = common.prior.paired(values["dst"]["forward"], values[policy]["forward"])
            rows.append(dict(dataset=dataset, policy=policy, control="dst", **effect))
    common.prior.holm(rows)
    return rows


def decision(row, floor, diagnostic=False):
    if row["holm_p"] >= .05:
        return "Inconclusive"
    if row["speedup"] < 1:
        return "Loss"
    if row["ci_low"] > floor and not diagnostic:
        return "Qualifies"
    return "Gain (diagnostic)" if diagnostic else "Gain below practical criterion"


def describe_additions(lines, contrasts, secondary, trials, floor):
    lines += ["", "## New edge policies versus Destination-grouped", "",
              "This secondary forward-only family uses destination-grouped as the control. "
              "Holm correction covers the three new edge policies across all datasets. "
              "The same practical bound applies here. The primary retention rule compares with Original. "
              "An inconclusive difference does not establish equivalence.", "",
              "| Dataset | Candidate | Destination-grouped ms | Candidate ms | Paired speedup (95% CI) | Holm p | Supported |",
              "|---|---|---:|---:|---|---:|---|"]
    for row in secondary:
        d, policy = row["dataset"], row["policy"]
        table = trials[d]["edge"]["table"]
        lines.append(f'| {NAMES[d]} | {LABELS[policy]} | {table["dst"]["forward"]:.6f} | '
                     f'{table[policy]["forward"]:.6f} | {common.display_effect(row)} | '
                     f'{row["holm_p"]:.4g} | {decision(row,floor)} |')
    qualified = [f"{LABELS[r['policy']]} on {NAMES[r['dataset']]}" for r in secondary
                 if decision(r,floor)=="Qualifies"]
    lines += ["", ("Practical improvements over Destination-grouped: "+", ".join(qualified)+"."
                   if qualified else "No new edge policy met the practical gain criterion against Destination-grouped.")]
    lines += ["", "## What the added policies contribute", ""]
    for policy in ("metis", *ADDED_EDGE):
        relevant = [r for r in contrasts if r["policy"] == policy and r["endpoint"] == "forward"]
        findings = ", ".join(f"{NAMES[r['dataset']]} ({decision(r,floor).lower()})" for r in relevant)
        lines.append(f"- **{LABELS[policy]}** versus Original. {findings}.")
    lines += ["", "The stage comparisons use separate Holm families. A significant projection, "
              "attention-score, softmax, or aggregation change alone does not qualify a policy. "
              "Rankings refer to these three workloads and one CPU, not to every graph with similar statistics."]


def describe_metis(lines, plans, trials):
    lines += ["", "## Frozen native METIS plan", "",
              "DGL 1.1.3's installed `dgl.metis_partition_assignment` makes a symmetric auxiliary "
              "graph and partitions it with the native k-way cut objective. We requested exactly 32 "
              "partitions with default vertex balancing and no edge or type balancing. Stable ordering "
              "by partition ID uses the original node ID to break ties. The auxiliary graph is never timed "
              "as a forward graph. Its construction and native assignment are included in the one-shot "
              "assignment time. The Python API exposes no METIS seed. Partition labels or assignments "
              "can depend on the native implementation and build. All workers used the frozen, verified plan.", "",
              "Assignment and stable sort below were measured once per dataset before acquisition. The "
              "five-repetition construction column for METIS in the main overhead table measures replay "
              "planning only, followed by actual node-feature copying, graph rebuild, and format materialization.", "",
              "| Dataset | Requested parts | Nonempty parts | Min / max nodes per nonempty part | Assignment ms | Stable ordering ms | Replay planning ms |",
              "|---|---:|---:|---|---:|---:|---:|"]
    for d, plan in plans.items():
        sizes = [s for s in plan["partition_sizes"] if s]
        replay = trials[d]["node"]["table"]["metis"]["permutation"]
        lines.append(f'| {NAMES[d]} | {plan["partitions"]} | {len(sizes)} | {min(sizes)} / {max(sizes)} | '
                     f'{plan["assignment_ns"]/1e6:.6f} | {plan["ordering_ns"]/1e6:.6f} | {replay:.6f} |')
    lines += ["", "## Edge traversal details", "",
              "Source/Destination-grouped use stable single-key sorts. Source then destination and "
              "Destination then source add a second endpoint key before the original EID tie-breaker. "
              "Morton uses unsigned 64-bit keys. Source bit i goes to 2i and destination bit i goes "
              "to 2i+1. Original EID breaks ties between parallel edges.", "",
              "The table summarizes the new policies across the three graphs using the same flags as "
              "the physical table. I means index arrays changed, E means the map to current EIDs changed, "
              "and L means the sequence of original EIDs visited changed.", "",
              "| Policy | COO flags observed | CSR flags observed | CSC flags observed |",
              "|---|---|---|---|"]
    for policy in ADDED_EDGE:
        views = [
            sorted({common.flags(sweeps["edge"]["representations"][policy]["sparse"][fmt],
                                 sweeps["edge"]["representations"]["original"]["sparse"][fmt])
                    for sweeps in trials.values()})
            for fmt in common.FORMATS
        ]
        lines.append(f"| {LABELS[policy]} | "+" | ".join(", ".join(flags) for flags in views)+" |")


def history_comparison(lines, raw, maps_path, trials, contrasts, floor):
    input_path, old_maps_path = HISTORY / "RESULTS.json", HISTORY / "PERMUTATIONS.json"
    previous = json.loads(input_path.read_text())
    prior_trials, prior_contrasts = common.summary(previous, old_maps_path)
    old_maps = json.loads(old_maps_path.read_text())
    new_maps = json.loads(maps_path.read_text())
    for dataset, rounds in raw["datasets"].items():
        old, new = previous["datasets"][dataset][0], rounds[0]
        for key in ("cpu", "weights", "edge_sha256", "graph_formats"):
            assert old[key] == new[key], key
        assert old["graph"]["feature_sha256"] == new["graph"]["feature_sha256"]
        for key in ("cpu_model", "cpu_affinity", "os", "dgl", "torch", "thread_environment", "runtime_thread_controls"):
            assert old["environment"][key] == new["environment"][key], key
        for axis in AXES:
            for policy, mapping in old_maps[dataset][axis].items():
                assert mapping == new_maps[dataset][axis][policy]
                assert old["cases"][axis][policy]["sparse"] == new["cases"][axis][policy]["sparse"]
    old_effects = {(r["dataset"], r["axis"], r["policy"]): r for r in prior_contrasts if r["endpoint"] == "forward"}
    lines += ["", "## Historical matched-allocation comparison", "",
              f"The [earlier acquisition](../REPORT.md) is preserved at {HISTORICAL_COMMIT}. "
              "Every shared policy used the same maps and physical graph views in both acquisitions, "
              "with matching workload fingerprints and pinned environment. Each acquisition has its own "
              "blocks, process-level estimates and correction families. No timing samples are pooled. "
              "Absolute latencies and ranks can shift across acquisitions or when the interleaved case "
              "set changes. Rank changes below are descriptive.", "",
              "| Dataset | Sweep | Fastest historical | Fastest now among shared policies | Fastest now overall |",
              "|---|---|---|---|---|"]
    shared_rank_changes = []
    new_winners = []
    for d, sweeps in trials.items():
        for axis in AXES:
            shared = common.AXES[axis]
            old, new = prior_trials[d][axis]["table"], sweeps[axis]["table"]
            previous_winner = min(shared, key=lambda p: old[p]["forward"])
            shared_winner = min(shared, key=lambda p: new[p]["forward"])
            extended_winner = min(AXES[axis], key=lambda p: new[p]["forward"])
            if shared_winner != previous_winner:
                shared_rank_changes.append(f"{NAMES[d]} {axis}")
            if extended_winner not in shared:
                new_winners.append(f"{LABELS[extended_winner]} on {NAMES[d]}")
            lines.append(f"| {NAMES[d]} | {axis} | {LABELS[previous_winner]} | "
                         f"{LABELS[shared_winner]} | {LABELS[extended_winner]} |")
    lines += ["", ("The best observed shared policy changed for "+", ".join(shared_rank_changes)+"."
                   if shared_rank_changes else "The observed winners among shared policies were unchanged."),
              ("Added policies changed the overall observed winner: "+", ".join(new_winners)+"."
               if new_winners else "The added policies did not change the best observed policy in any sweep."),
              "This separates shifts among policies that already existed from shifts caused by having new "
              "choices in the table. Inference still comes from each acquisition's paired comparisons."]
    lines += ["", "| Dataset | Sweep | Shared policy | Historical median ms | Extended median ms | Historical speedup (95% CI) | Extended speedup (95% CI) | Decisions (historical → extended) |",
              "|---|---|---|---:|---:|---|---|---|"]
    for r in contrasts:
        key = (r["dataset"], r["axis"], r["policy"])
        if r["endpoint"] != "forward" or key not in old_effects:
            continue
        d, axis, policy = key
        old = old_effects[key]
        earlier = prior_trials[d][axis]["table"][policy]["forward"]
        current = trials[d][axis]["table"][policy]["forward"]
        lines.append(f"| {NAMES[d]} | {axis} | {LABELS[policy]} | {earlier:.6f} | {current:.6f} | "
                     f"{common.display_effect(old)} | {common.display_effect(r)} | "
                     f"{decision(old,floor,policy=='random')} → {decision(r,floor,policy=='random')} |")
    return dict(commit=HISTORICAL_COMMIT, historical_raw_sha256=common.sha(input_path),
                historical_maps_sha256=common.sha(old_maps_path), historical_contrasts=prior_contrasts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE/"RESULTS.json")
    parser.add_argument("--maps", type=Path, default=HERE/"PERMUTATIONS.json")
    parser.add_argument("--metis", type=Path, default=HERE/"METIS.json")
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--no-historical", action="store_true")
    args = parser.parse_args()
    if args.output.resolve() == HISTORY:
        parser.error("The extension cannot write into the historical output directory")
    raw = json.loads(args.input.read_text())
    plans = verify_protocol(raw, args.maps, args.metis)
    trials, contrasts = common.summary(raw, args.maps, catalog=AXES)
    secondary = secondary_edge_contrasts(trials)
    lines = common.render(raw, trials, contrasts, catalog=AXES, labels=LABELS, include_forward_median=True)
    lines[0] = "# Milestone 10 Extension — Seven node orderings and seven edge orderings"
    floor = raw["settings"]["practical_speedup_floor"]
    describe_additions(lines, contrasts, secondary, trials, floor)
    common.describe_details(lines, trials, catalog=AXES, labels=LABELS)
    describe_metis(lines, plans, trials)
    archive = {} if args.no_historical else history_comparison(lines, raw, args.maps, trials, contrasts, floor)
    cfg = raw["settings"]
    record = next(iter(raw["datasets"].values()))[0]
    env = record["environment"]
    errors = [case["correctness"] for data in trials.values() for sweep in data.values()
              for case in sweep["representations"].values()]
    lines += ["", "## Protocol and correctness", "",
              f"{cfg['rounds']} fresh processes per dataset, {cfg['warmup']} randomized warmup blocks, "
              f"{cfg['repetitions']} measured matched blocks per sweep per process, and "
              f"{cfg['build_repetitions']} separately randomized construction observations per case and process. "
              "Every block has a plain forward and a separately instrumented forward for every case, "
              "including each random seed. Timing schedules used a new recorded seed; random permutations "
              "retained their three original seeds per sweep.", "",
              f"{env['cpu_model']}, logical CPU {record['cpu']}, one thread, Python {env['python']}, "
              f"PyTorch {env['torch']}, DGL {env['dgl']}, {env['os']}. FP32, four heads, eight channels "
              "per head, identical seed-0 weights, original native features, no dropout. Input self-loops "
              "were removed and one per node appended by the unchanged loader.", "",
              "For every case the restored topology, multiplicities, feature and edge-data alignment, "
              "and sparse storage maps were checked. Attention was restored by original EID and checked "
              "for destination-wise normalization. Direct and instrumented outputs and attention matched "
              f"bitwise. Maximum absolute output error {max(x['output_abs'] for x in errors):.3g}, "
              f"attention error {max(x['attention_abs'] for x in errors):.3g}, normalization error "
              f"{max(x['normalization_abs'] for x in errors):.3g}. Tolerances were atol 1e−5 and rtol 1e−4.", "",
              "Complete forward does not include permutation generation, graph construction, or restoring "
              "outputs to canonical IDs. The independent sweeps do not measure whether node and edge "
              "effects compose. Confidence intervals describe process variation on this system. "
              "Random-order intervals condition on three fixed permutations.", "",
              "[Raw timings](RESULTS.json) · [Permutation maps](PERMUTATIONS.json) · "
              "[Frozen METIS metadata](METIS.json) · [Paired statistics](ANALYSIS.json) · "
              "[Reproduction](README.md)", ""]
    args.output.mkdir(parents=True, exist_ok=True)
    args.output.joinpath("REPORT.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    analysis = dict(datasets=trials, contrasts=contrasts, new_edges_vs_dst=secondary,
                    raw_sha256=common.sha(args.input), maps_sha256=common.sha(args.maps),
                    metis_sha256=common.sha(args.metis), analysis_sources={
                        "report_extended.py": common.sha(Path(__file__)),
                        "../report.py": common.sha(HISTORY/"report.py")}, **archive)
    args.output.joinpath("ANALYSIS.json").write_text(
        json.dumps(analysis, allow_nan=False, separators=(",", ":"))+"\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
