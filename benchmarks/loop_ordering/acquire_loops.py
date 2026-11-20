"""Collect serial fresh-process randomized matched timing blocks."""
import argparse
from functools import partial
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

from pipeline import State, baseline, bitwise, dgl, fingerprint, prepared, save, setup, stock_profile, torch
from compiled import HERE, compile_library, sha
from design import DATASETS, FLOOR, SEED, STAGES, SWEEPS, SYMBOLS, columns, design


def empty():
    pass


def timer_six():
    result = {}
    for stage in STAGES:
        start = perf_counter_ns()
        result[stage] = perf_counter_ns()-start
    return result


def collect(sweep,calls,profiles,rng,count):
    schedule = columns(sweep)
    assert set(schedule)==set(calls)
    blocks = []
    for _ in range(count):
        order = schedule.copy()
        rng.shuffle(order)
        observations = []
        for key in order:
            call = calls[key]
            start = perf_counter_ns()
            output = call()
            elapsed = perf_counter_ns()-start
            del output
            row = [key,elapsed]
            if key.startswith("profile/"):
                stages = [profiles[key][stage] for stage in STAGES]
                assert min(stages)>0 and sum(stages)<=elapsed
                row.append(stages)
            observations.append(row)
        blocks.append(observations)
    return blocks


def worker(args):
    g,x,layer,src,dst,meta = setup(args.dataset,args.data_dir,args.cpu)
    official = dgl.nn.pytorch.GATConv
    assert type(layer) is official and layer.forward.__func__ is official.forward
    assert os.sched_getaffinity(0)=={meta["cpu"]}
    state = State(g,args.binary)
    assert state.index_sha256["src"]==meta["edge_sha256"][0]
    assert state.index_sha256["dst"]==meta["edge_sha256"][1]
    assert int((state.src==state.dst).sum())==state.n
    forward_source = inspect.getsource(official.forward)
    seed = SEED+1009*args.round+DATASETS.index(args.dataset)
    rng = random.Random(seed)
    sweep_order = [*SWEEPS,"stock"]
    rng.shuffle(sweep_order)
    blocks,checks,routes = {},{},{}
    with torch.inference_mode():
        ref,shared = baseline(state,x,layer)
        profiled,record = stock_profile(layer)
        bitwise(profiled(g,x),ref["output"])
        bitwise(layer(g,x),ref["output"])
        for sweep in sweep_order:
            calls = {"timer/empty":empty,"timer/six":timer_six}
            profiles = {}
            if sweep=="stock":
                calls.update({"plain/DGL":partial(layer,g,x),"profile/DGL":partial(profiled,g,x)})
                profiles["profile/DGL"] = record
            else:
                checks[sweep],routes[sweep] = {},{}
                calls["allocation"] = getattr(state,"alloc_"+sweep)
                for code in SWEEPS[sweep]:
                    item = prepared(sweep,code,state,x,layer,ref,shared)
                    assert item["routes"]=={s:SYMBOLS[v] for s,v in design(sweep,code).items()}
                    profiles[f"profile/{code}"] = item["profile"]
                    checks[sweep][code],routes[sweep][code] = item["check"],item["routes"]
                    calls.update(item["calls"])
            blocks[sweep] = collect(sweep,calls,profiles,rng,args.warmup+args.repetitions)
        assert {k:fingerprint(t) for k,t in layer.state_dict().items()}==meta["weights"]
        assert meta["graph_formats"]==g.formats()
    save(args.output,dict(**meta,round=args.round,order_seed=seed,sweep_order=sweep_order,
        blocks=blocks,checks=checks,routes=routes,stage_payload=state.meta,index_sha256=state.index_sha256,
        self_loops=state.n,parallel_edges=state.e-len(np_unique_edges(src,dst,state.n)),
        binary_sha256=sha(args.binary),loaded_offsets=state.lib.offsets(),stock_profile_bitwise=True,
        stock_forward_sha256=hashlib.sha256(forward_source.encode()).hexdigest()))
    print(f"{args.dataset} round {args.round+1}: all loop-order sweeps validated and timed",flush=True)


def np_unique_edges(src,dst,num_nodes):
    import numpy as np
    return np.unique(src*num_nodes+dst)


def acquire(args):
    binary,build = compile_library()
    dependencies = ("native_sparse_formats/acquire.py","tuning_axes/common.py","tuning_axes/orchestration.py",
                    "dgl_baseline/common.py","dgl_baseline/baseline.py","sparse_formats/run.py",
                    "sparse_format_validation/stage_profile.py","gat.py")
    own = ("acquire_loops.py","design.py","kernels.cpp","compiled.py","pipeline.py")
    files = [*(f"loop_ordering/{name}" for name in own),*dependencies]
    result = dict(status="running",settings=dict(rounds=args.rounds,warmup=args.warmup,
                  repetitions=args.repetitions,order_seed=SEED,threads=1,model_seed=0,
                  dtype="float32",heads=4,channels=8,practical_speedup_floor=FLOOR),
                  build=build,sources={f:sha(HERE.parent/f,text=True) for f in files},
                  lock_sha256=sha(HERE.parents[1]/"uv.lock",text=True),
                  datasets={d:[] for d in args.datasets},dataset_orders=[])
    offsets = {code:v["address"]-build["symbols"]["A1"]["address"] for code,v in build["symbols"].items()}
    parent_rng = random.Random(SEED)
    with TemporaryDirectory() as temp:
        for i in range(args.rounds):
            names = list(args.datasets)
            parent_rng.shuffle(names)
            result["dataset_orders"].append(names)
            for dataset in names:
                output = Path(temp)/f"{dataset}-{i}.json"
                command = [sys.executable,__file__,"--dataset",dataset,"--round",str(i),
                           "--binary",str(binary),"--output",str(output),"--data-dir",str(args.data_dir),
                           "--warmup",str(args.warmup),"--repetitions",str(args.repetitions)]
                if args.cpu is not None:
                    command += ["--cpu",str(args.cpu)]
                subprocess.run(command,check=True)
                row = json.loads(output.read_text())
                assert row["binary_sha256"]==build["binary_sha256"] and row["loaded_offsets"]==offsets
                result["datasets"][dataset].append(row)
                save(args.output,result)
    result["status"] = "complete"
    save(args.output,result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--datasets",nargs="+",choices=DATASETS,default=DATASETS)
    parser.add_argument("--rounds",type=int,default=8)
    parser.add_argument("--warmup",type=int,default=10)
    parser.add_argument("--repetitions",type=int,default=30)
    parser.add_argument("--cpu",type=int)
    parser.add_argument("--round",type=int,help=argparse.SUPPRESS)
    parser.add_argument("--dataset",choices=DATASETS,help=argparse.SUPPRESS)
    parser.add_argument("--binary",type=Path,help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.output.exists() or min(args.rounds,args.repetitions)<2 or args.warmup<1
            or len(args.datasets)!=len(set(args.datasets))):
        parser.error("require a fresh output, unique datasets, rounds/repetitions >=2, warmup >=1")
    if args.round is not None:
        if args.dataset is None or args.binary is None:
            parser.error("internal worker requires a dataset and compiled binary")
        return worker(args)
    if args.dataset is not None or args.binary is not None:
        parser.error("internal worker options require a process round")
    return acquire(args)


if __name__=="__main__":
    main()
