"""Acquire two independent 7-policy sweeps, using frozen native METIS assignments."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from uuid import uuid4

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
from benchmark import (encode, run_axis, setup, save, sha, torch, dgl, fingerprint,
                       stock_profile, validate_output)
from policies import DATASETS, SEED as PERMUTATION_SEED, digest, graph_diagnostics, sparse_diagnostics, validate_graph
from policies_ext import AXES, MORTON_CONVENTION, PARTITIONS, case_specs, nodes_from_assignment, np, permutation

ORDER_SEED = 2026112100


def metis_plan(base):
    """Run native DGL once per dataset; retain only a graph-independent map."""
    if base.num_nodes() < PARTITIONS:
        raise ValueError("The fixed METIS partition count exceeds the graph size")
    api = dgl.metis_partition_assignment
    start = perf_counter_ns()
    # The native API requires int64 and symmetrizes its auxiliary graph itself.
    assignment = api(base.long(), PARTITIONS, balance_edges=False, mode="k-way", objtype="cut")
    assigned = perf_counter_ns()
    if assignment is None:
        raise RuntimeError("Native DGL METIS returned no assignment")
    assignments = assignment.numpy()
    nodes = nodes_from_assignment(assignments, base.num_nodes(), PARTITIONS)
    planned = perf_counter_ns()
    np.testing.assert_array_equal(np.sort(nodes), np.arange(base.num_nodes()))
    return dict(partitions=PARTITIONS, mode="k-way", objective="cut", balance_edges=False,
                assignment=assignments.tolist(), new_to_original=nodes.tolist(),
                partition_sizes=np.bincount(assignments, minlength=PARTITIONS).tolist(),
                assignment_ns=assigned-start, ordering_ns=planned-assigned,
                assignment_sha256=digest(assignments)[0], node_permutation_sha256=digest(nodes)[0],
                api="dgl.metis_partition_assignment",
                api_source_sha256=hashlib.sha256(inspect.getsource(api).encode()).hexdigest(),
                nondeterminism="The Python API exposes no METIS seed. Native implementation and build "
                               "may affect partition assignments and numbering. Workers replay the saved map.")


def saved_metis_nodes(entry, num_nodes):
    assert entry["partitions"] == PARTITIONS
    assignment = np.asarray(entry["assignment"])
    nodes = nodes_from_assignment(assignment, num_nodes, PARTITIONS)
    np.testing.assert_array_equal(nodes, entry["new_to_original"])
    assert np.bincount(assignment, minlength=PARTITIONS).tolist() == entry["partition_sizes"]
    assert digest(assignment, nodes) == [entry["assignment_sha256"], entry["node_permutation_sha256"]]
    assert min(entry["assignment_ns"], entry["ordering_ns"]) > 0
    return nodes


def prepare(args):
    if len(args.datasets) > 1:
        records = {}
        # The shared setup sets PyTorch's interop threads once per process.
        with TemporaryDirectory() as directory:
            for dataset in args.datasets:
                path = Path(directory) / f"{dataset}.json"
                cmd = [sys.executable, __file__, "--prepare-metis", "--datasets", dataset,
                       "--metis", str(path), "--data-dir", str(args.data_dir)]
                if args.cpu is not None:
                    cmd += ["--cpu", str(args.cpu)]
                subprocess.run(cmd, check=True)
                records.update(json.loads(path.read_text()))
        save(args.metis, records)
        return
    dataset, = args.datasets
    base, _, _, _, _, meta = setup(dataset, args.data_dir, args.cpu)
    record = dict(metis_plan(base), input_edge_sha256=meta["edge_sha256"],
                  feature_sha256=meta["graph"]["feature_sha256"], cpu=meta["cpu"], environment=meta["environment"])
    save(args.metis, {dataset: record})
    print(f"{dataset}: froze the native METIS partition assignment", flush=True)


def build(base, src, dst, axis, policy, seed, metis):
    """Same materialization, data indexing and timer boundaries as Milestone 10."""
    start = perf_counter_ns()
    nodes, edges = permutation(base.num_nodes(), src, dst, axis, policy, seed, metis)
    planned = perf_counter_ns()
    inverse = np.empty_like(nodes)
    inverse[nodes] = np.arange(len(nodes))
    graph = dgl.graph((inverse[src[edges]], inverse[dst[edges]]),
                      num_nodes=len(nodes), idtype=base.idtype, device="cpu")
    node_ids, edge_ids = torch.from_numpy(nodes), torch.from_numpy(edges)
    for key, value in base.ndata.items():
        graph.ndata[key] = value[node_ids] if axis == "node" else value
    for key, value in base.edata.items():
        graph.edata[key] = value[edge_ids] if axis == "edge" and policy != "original" else value
    before = graph.formats()
    rebuilt = perf_counter_ns()
    graph.create_formats_()
    complete = perf_counter_ns()
    return graph, nodes, edges, [planned-start, rebuilt-planned, complete-rebuilt], before


def worker(args):
    base, x, layer, src, dst, meta = setup(args.dataset, args.data_dir, args.cpu)
    stock_type = dgl.nn.pytorch.GATConv
    assert type(layer) is stock_type and layer.forward.__func__ is stock_type.forward
    assert os.sched_getaffinity(0) == {meta["cpu"]}
    plan = json.loads(args.metis.read_text())[args.dataset]
    assert (plan["input_edge_sha256"] == meta["edge_sha256"]
            and plan["feature_sha256"] == meta["graph"]["feature_sha256"]
            and plan["cpu"] == meta["cpu"])
    metis = saved_metis_nodes(plan, base.num_nodes())
    profiled, regions = stock_profile(layer)
    seed = ORDER_SEED + 1009 * args.round + DATASETS.index(args.dataset)
    rng = random.Random(seed)
    axes = list(AXES)
    rng.shuffle(axes)
    cases, timings, maps = {}, {}, {}
    loader_sparse = sparse_diagnostics(base, np.arange(len(x)), np.arange(len(src)), src, dst)
    with torch.inference_mode():
        expected, alpha = layer(base, x, get_attention=True)
        for axis in axes:
            cases[axis], maps[axis], candidates = {}, {}, {}
            for label, policy, perm_seed in case_specs(args.dataset, axis):
                g, nodes, edges, _, before = build(base, src, dst, axis, policy, perm_seed, metis)
                sparse = validate_graph(base, g, nodes, edges, src, dst, axis)
                diag = graph_diagnostics(g)
                assert diag["self_loops"] == base.num_nodes()
                errors = validate_output(g, g.ndata["feat"], layer, profiled, nodes, edges, expected, alpha)
                cases[axis][label] = dict(policy=policy, permutation_seed=perm_seed, maps_sha256=digest(nodes, edges),
                    graph=diag, sparse=sparse, before_materialization=before,
                    after_materialization=g.formats(), correctness=errors)
                maps[axis][label] = dict(nodes=encode(nodes), edges=encode(edges), seed=perm_seed)
                candidates[label] = g, g.ndata["feat"]
            timings[axis] = run_axis(candidates, layer, profiled, regions, rng, args.warmup + args.repetitions)
            for label, (g, _) in candidates.items():
                assert g.formats() == cases[axis][label]["after_materialization"]
            del candidates
        for model in (layer, profiled):
            assert {k: fingerprint(t) for k, t in model.state_dict().items()} == meta["weights"]
        assert cases["node"]["original"] == cases["edge"]["original"]
        construction = []
        builds = [(axis, f, policy, seed) for axis in axes for f, policy, seed in case_specs(args.dataset, axis)]
        for _ in range(args.build_repetitions):
            order = builds.copy()
            rng.shuffle(order)
            block = []
            for axis, label, policy, perm_seed in order:
                g, nodes, edges, elapsed, before = build(base, src, dst, axis, policy, perm_seed, metis)
                assert before == cases[axis][label]["before_materialization"]
                assert g.formats() == cases[axis][label]["after_materialization"]
                block.append([axis, label, *elapsed])
                del g
            construction.append(block)
    save(args.output, dict(**meta, round=args.round, order_seed=seed, axis_order=axes,
                          cases=cases, maps=maps, loader_sparse=loader_sparse, blocks=timings,
                          construction=construction))
    print(f"{args.dataset} process {args.round+1}: both extended sweeps validated and timed", flush=True)


def acquire(args):
    historical = json.loads((HERE.parent / "RESULTS.json").read_text())
    sources = list(historical["sources"]) + ["ordering_baselines/extended/benchmark_extended.py",
                                             "ordering_baselines/extended/policies_ext.py"]
    order = random.Random(ORDER_SEED)
    plan_hash = sha(args.metis)
    saves = Path("results/m10_extended_workers") / uuid4().hex
    saves.mkdir(parents=True)
    records = dict(status="running", settings=dict(rounds=args.rounds, warmup=args.warmup,
        repetitions=args.repetitions, build_repetitions=args.build_repetitions, order_seed=ORDER_SEED,
        permutation_seed_base=PERMUTATION_SEED, model_seed=0, threads=1, heads=4, channels=8,
        practical_speedup_floor=1.01, metis_partitions=PARTITIONS, morton_key=MORTON_CONVENTION),
        sources={f: sha(ROOT / "benchmarks" / f) for f in sources}, lock_sha256=sha(ROOT / "uv.lock"),
        metis_sha256=plan_hash, datasets={d: [] for d in args.datasets}, dataset_orders=[])
    maps = {}
    for i in range(args.rounds):
        names = list(records["datasets"])
        order.shuffle(names)
        records["dataset_orders"].append(names)
        for dataset in names:
            output = saves / f"{dataset}-{i}.json"
            command = [sys.executable, __file__, "--dataset", dataset, "--round", str(i),
                       "--output", str(output), "--metis", str(args.metis), "--data-dir", str(args.data_dir),
                       "--warmup", str(args.warmup), "--repetitions", str(args.repetitions),
                       "--build-repetitions", str(args.build_repetitions)]
            if args.cpu is not None:
                command += ["--cpu", str(args.cpu)]
            subprocess.run(command, check=True)
            row = json.loads(output.read_text())
            this_map = row.pop("maps")
            if dataset in maps:
                assert maps[dataset] == this_map
            else:
                maps[dataset] = this_map
                save(args.maps, maps)
            records["datasets"][dataset].append(row)
            save(args.output, records)
    assert sha(args.metis) == plan_hash
    records.update(status="complete", maps_sha256=sha(args.maps))
    save(args.output, records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--metis", type=Path, required=True)
    parser.add_argument("--prepare-metis", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--maps", type=Path)
    parser.add_argument("--datasets", choices=DATASETS, nargs="+", default=DATASETS)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--build-repetitions", type=int, default=5)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--round", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--dataset", choices=DATASETS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("Dataset names must be unique")
    if args.prepare_metis:
        if args.metis.exists() or any((args.output, args.maps, args.dataset)) or args.round is not None:
            parser.error("METIS preparation requires a fresh --metis path and no timing outputs")
        return prepare(args)
    if (not args.metis.is_file() or args.output is None or args.output.exists()
            or min(args.rounds, args.repetitions, args.build_repetitions) < 2 or args.warmup < 1):
        parser.error("Timing requires a saved METIS plan, fresh output, rounds/repetitions/builds >= 2 and warmup >= 1")
    if args.round is not None:
        return worker(args)
    if args.maps is None or args.maps.exists() or args.maps.resolve() == args.output.resolve():
        parser.error("Timing requires a distinct fresh --maps path")
    return acquire(args)


if __name__ == "__main__":
    main()
