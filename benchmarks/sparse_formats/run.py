"""Serial fresh processes; randomized matched blocks; raw observations retained."""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from time import perf_counter_ns
from uuid import uuid4

os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                  OMP_DYNAMIC="FALSE")

from experiment import HERE, Kernel, OPERATIONS, Variant, build_library
from common import configure, fingerprint, model_for, old, provenance, reference_stages, save, torch
from orchestration import Pipeline
from storage import FORMATS, build

DATASETS = ("cora", "citeseer", "pubmed")


def measure(function):
    start = perf_counter_ns()
    value = function()
    return (perf_counter_ns() - start) / 1e6, value


def blocks(functions, count, rng):
    result = []
    for _ in range(count):
        order = list(functions)
        rng.shuffle(order)
        result.append([[name, measure(functions[name])[0]] for name in order])
    return result


def worker(args):
    allowed = os.sched_getaffinity(0)
    cpu = min(allowed) if args.cpu is None else args.cpu
    if cpu not in allowed:
        raise ValueError("selected CPU is outside the available affinity mask")
    os.sched_setaffinity(0, {cpu})
    configure(1)
    g, x, graph = old.load_dataset(args.dataset, args.data_dir)
    src, dst = (t.long().numpy() for t in g.edges(order="eid"))
    n = g.num_nodes()
    layer = model_for(x.shape[1], args.seed)
    library, compile_info = build_library()
    lib = ctypes.CDLL(str(library.resolve()))
    rng = random.Random(args.order_seed + 1009 * args.worker + DATASETS.index(args.dataset))
    builders = {label: (lambda f=f, b=b: build(n, src, dst, f, b))
                for f, b in FORMATS for label in [f"{f}{b or ''}"]}
    stores = {label: create() for label, create in builders.items()}
    for store in stores.values():
        store.validate(src, dst)
    fixed = Pipeline(g)
    result = dict(dataset=args.dataset, round=args.worker, cpu=cpu,
                  graph=graph, environment=provenance(), compiler=compile_info,
                  edge_sha256=[hashlib.sha256(t.tobytes()).hexdigest() for t in (src, dst)],
                  weight_sha256={k: fingerprint(v) for k, v in layer.state_dict().items()},
                  raw_dataset_sha256={str(p.relative_to(args.data_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for directory in args.data_dir.glob(old.DATASETS[args.dataset][1]+"_*")
                      for p in sorted(directory.glob("ind.*"))},
                  storage={key: dict(bytes=s.bytes, nonempty_tiles=s.tiles, arrays=[a.size for a in s.arrays])
                           for key, s in stores.items()}, correctness={})
    prior = json.loads((HERE.parent / "tuning_axes/RESULTS.json").read_text())["datasets"][args.dataset]["rounds"][0]
    expected_edges = prior["axes"]["score_traversal"]["cases"]["coo"]["implementation"]["views"]["score"]["sparse_hashes"]["coo"]["indices"]
    if (result["edge_sha256"] != expected_edges
            or graph["feature_sha256"] != prior["dataset_info"]["feature_sha256"]
            or (args.seed == prior["model"]["seed"] and result["weight_sha256"] != prior["model"]["weights_sha256"])):
        raise ValueError("dataset EIDs/features or model weights disagree with Milestone 5")
    result["milestone5_identity_verified"] = True
    with torch.inference_mode():
        ref = reference_stages(g, x, layer)
        variants = {f"{op}/{name}": Variant(fixed, Kernel(lib, store, op))
                    for op in OPERATIONS for name, store in stores.items()}
        for name, plan in variants.items():
            result["correctness"][name] = plan.validate(x, layer, ref, dst)
        forwards = {key: (lambda p=p: p.forward(x, layer)) for key, p in variants.items()}
        forwards["stock"] = lambda: layer(g, x)
        isolated = {key: (lambda p=p: p.isolated(ref, layer)) for key, p in variants.items()}
        for mode, calls in (("forward", forwards), ("kernel", isolated)):
            result[mode+"_warmups"] = blocks(calls, args.warmup, rng)
            result[mode] = blocks(calls, args.repetitions, rng)
        result["conversion"] = blocks(builders, 5, rng)
    save(args.output, result)
    print(f"{args.dataset} round {args.worker+1}: 36 validated substitutions", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repetitions", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--order-seed", type=int, default=20261113)
    p.add_argument("--data-dir", type=Path, default=Path("results/datasets"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cpu", type=int)
    p.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    p.add_argument("--dataset", choices=DATASETS, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.rounds < 2 or args.warmup < 1 or args.repetitions < 2 or not 0 <= args.seed < 2**31:
        p.error("require rounds>=2, warmup>=1, repetitions>=2 and a valid seed")
    if args.output.exists():
        p.error("output already exists")
    if args.worker is not None:
        return worker(args)
    build_library()
    scratch = Path("results/sparse_formats_workers") / uuid4().hex
    scratch.mkdir(parents=True)
    data = dict(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
                settings=dict(rounds=args.rounds, warmup=args.warmup, repetitions=args.repetitions,
                              seed=args.seed, order_seed=args.order_seed, threads=1, heads=4,
                              channels=8, dtype="float32"),
                sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in HERE.iterdir() if p.suffix in (".py", ".cpp") and p.name != "report.py"},
                datasets={d: [] for d in args.datasets})
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               OMP_DYNAMIC="FALSE", PYTHONUNBUFFERED="1")
    rng = random.Random(args.order_seed)
    for r in range(args.rounds):
        datasets = list(data["datasets"])
        rng.shuffle(datasets)
        for d in datasets:
            output = scratch / f"{r}-{d}.json"
            cmd = [sys.executable, str(HERE/"run.py"), "--dataset", d, "--worker", str(r),
                   "--output", str(output), "--data-dir", str(args.data_dir),
                   "--warmup", str(args.warmup), "--repetitions", str(args.repetitions),
                   "--seed", str(args.seed), "--order-seed", str(args.order_seed)]
            if args.cpu is not None:
                cmd += ["--cpu", str(args.cpu)]
            subprocess.run(cmd, env=env, check=True)
            data["datasets"][d].append(json.loads(output.read_text()))
            save(args.output, data)
    data.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
    save(args.output, data)


if __name__ == "__main__":
    main()
