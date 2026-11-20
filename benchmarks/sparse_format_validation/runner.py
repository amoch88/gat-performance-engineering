"""Matched single-thread forwards, in-forward stages, and isolated calls."""
import argparse
import ctypes
from functools import partial
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from time import perf_counter_ns
from uuid import uuid4

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"comprehensive_sparse_formats"))
from benchmark import (Forward, OPS, SIZES, FALLBACK, SUPPORTED, Native, Pipeline, view,
                       dgl, torch, np, H, F, RTOL, ATOL, DATASETS, setup, save, validate,
                       validate_view, view_info, reference_stages, stock_stages, isolated,
                       build_library, build_blocked, Kernel, build_padded, PaddedKernel, library)
from storage import build
from stage_profile import STAGES, instrument, stock_profile

NATIVE = ("coo","csr","csc")
CUSTOM = [f"custom_{f}" for f in NATIVE]+[f"{f}{s}" for f in ("csb","sell") for s in SIZES]
MATCHED = [f"matched_{f}" for f in NATIVE]
CONFIGS = ["stock",*NATIVE,*CUSTOM,*MATCHED]
SEED = 2026111900
TARGETS = dict(score="logits",softmax="attention",aggregation="aggregated")


def timed(call):
    start = perf_counter_ns()
    output = call()
    return perf_counter_ns()-start,output


def timer_bookkeeping():
    record = {}
    for _ in STAGES:
        start = perf_counter_ns()
        record[_] = perf_counter_ns()-start
    return record


def setup_cases(g,x,layer,src,dst,ref):
    fixed = Pipeline(g)
    views = {f:view(g,f) for f in NATIVE}
    for graph in views.values():
        validate_view(graph,g)
    natives = {(op,fmt):Native(views[fmt],op,fmt) for op in OPS for fmt in SUPPORTED[op]}
    args = (g.num_nodes(),src,dst)
    builders = {f"custom_{f}":partial(build,*args,f,0) for f in NATIVE}
    builders.update({f"csb{s}":partial(build_blocked,*args,"csb",s) for s in SIZES})
    builders.update({f"sell{s}":partial(build_padded,*args,"sell",s,1<<28) for s in SIZES})
    stores = {name:create() for name,create in builders.items()}
    for store in stores.values():
        store.validate(src,dst)
    storage = {f:dict(bytes=view_info(graph)["unique_sparse_storage_payload_bytes"],index_bits=64,
                      masks=graph.formats()) for f,graph in views.items()}
    storage.update({f:dict(bytes=s.bytes,index_bits=32,slots=getattr(s,"slots",len(src)))
                    for f,s in stores.items()})
    binary,compiler = build_library()
    blocked_lib = ctypes.CDLL(str(binary.resolve()))
    sell_lib,sell_compiler = library()
    routes,checks,forwards,profiles = {},{},{},{}
    stock,stock_record = stock_profile(layer)
    forwards["stock"] = partial(layer,g,x)
    profiles["stock"] = (partial(stock,g,x),stock_record)
    calls = {f"isolated/{op}/stock":fn for op,fn in stock_stages(g,layer,ref).items()}
    checks["stock"] = {"output":0.0}
    torch.testing.assert_close(stock(g,x),ref["output"],rtol=0,atol=0)
    for f in CONFIGS[1:]:
        if f in NATIVE:
            routes[f] = {op:f if f in SUPPORTED[op] else FALLBACK[op] for op in OPS}
            kernels = {op:natives[op,fmt] for op,fmt in routes[f].items()}
        elif f in MATCHED:
            selected = f.removeprefix("matched_")
            routes[f] = {op:f"custom_{selected if selected in SUPPORTED[op] else FALLBACK[op]}"
                         for op in OPS}
            kernels = {op:Kernel(blocked_lib,stores[fmt],op) for op,fmt in routes[f].items()}
        else:
            routes[f] = dict.fromkeys(OPS,f)
            kernel,lib = (PaddedKernel,sell_lib) if f.startswith("sell") else (Kernel,blocked_lib)
            kernels = {op:kernel(lib,stores[f],op) for op in OPS}
        forward = Forward(fixed,kernels)
        checks[f] = validate(forward,x,layer,ref,dst)["errors"]
        forwards[f] = partial(forward,x,layer)
        func,record = instrument(Forward.__call__,stock=False)
        profile = partial(func,forward,x,layer)
        torch.testing.assert_close(profile(),forwards[f](),rtol=0,atol=0)
        profiles[f] = (profile,record)
        for op in OPS:
            if routes[f][op]==f:  # Stage table leaves unsupported native cells absent.
                calls[f"isolated/{op}/{f}"] = isolated(kernels[op],op,ref,layer)
    for key,call in calls.items():
        target = TARGETS[key.split("/")[1]]
        torch.testing.assert_close(call(),ref[target],rtol=RTOL,atol=ATOL)
    project = lambda:fixed.project(layer.feat_drop(x),layer)
    calls.update({"isolated/projection/shared":project,
                  "isolated/terms/shared":partial(fixed.node_terms,ref["projected"],layer),
                  "isolated/output/shared":partial(fixed.bias,ref["aggregated"],layer),
                  "timer/empty":lambda:None,"timer/six_regions":timer_bookkeeping})
    # These untimed preparations change execution context only. They are
    # diagnostics for the projection discrepancy, not a format comparison.
    def after_graph_guard():
        fixed.guard()
        return timed(project)[0]

    def after_sparse():
        # Same native stages and already prepared tensors, with no linear read.
        for op in OPS:
            calls[f"isolated/{op}/stock"]()
        return timed(project)[0]

    def after_forward():
        forwards["stock"]()
        return timed(project)[0]

    context = {"context/guard":after_graph_guard,"context/sparse":after_sparse,
               "context/forward":after_forward}
    return forwards,profiles,calls,context,storage,builders,routes,checks,(compiler,sell_compiler)


def worker(a):
    g,x,layer,src,dst,meta = setup(a.dataset,a.data_dir,a.cpu)
    n = g.num_nodes()
    seed = SEED+1009*a.round+DATASETS.index(a.dataset)
    rng = random.Random(seed)
    with torch.inference_mode():
        ref = reference_stages(g,x,layer)
        forwards,profiles,calls,context,storage,builders,routes,checks,compilers = setup_cases(g,x,layer,src,dst,ref)
        decisions = [*(f"forward/{f}" for f in CONFIGS),*(f"profile/{f}" for f in CONFIGS),*calls,*context]
        raw = []
        for block in range(a.warmup+a.repetitions):
            order = decisions.copy()
            rng.shuffle(order)
            rows = []
            for key in order:
                if key.startswith("forward/"):
                    elapsed,output = timed(forwards[key.split("/")[1]])
                    del output
                    rows.append([key,elapsed])
                elif key.startswith("profile/"):
                    func,regions = profiles[key.split("/")[1]]
                    elapsed,output = timed(func)
                    del output
                    stages = [regions[name] for name in STAGES]
                    assert sum(stages)<=elapsed and all(ms>0 for ms in stages)
                    rows.append([key,elapsed,stages])
                elif key.startswith("context/"):
                    rows.append([key,context[key]()])
                else:
                    elapsed,output = timed(calls[key])
                    del output
                    rows.append([key,elapsed])
            raw.append(rows)
    construction = []
    for _ in range(a.build_repetitions):
        order = list(storage)
        rng.shuffle(order)
        rows = []
        for f in order:
            if f in NATIVE:
                cold = dgl.graph((src,dst),num_nodes=n)
                elapsed,store = timed(partial(view,cold,f))
            else:
                elapsed,store = timed(builders[f])
            rows.append([f,elapsed])
            del store
        construction.append(rows)
    save(a.output,dict(**meta,round=a.round,order_seed=seed,schedule_columns=decisions,
                       storage=storage,routes=routes,checks=checks,compilers=compilers,
                       blocks=raw,construction=construction))
    print(f"{a.dataset} round {a.round+1} complete",flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--data-dir",type=Path,required=True)
    p.add_argument("--datasets",nargs="+",choices=DATASETS,default=DATASETS)
    p.add_argument("--rounds",type=int,default=8)
    p.add_argument("--warmup",type=int,default=10)
    p.add_argument("--repetitions",type=int,default=30)
    p.add_argument("--build-repetitions",type=int,default=5)
    p.add_argument("--cpu",type=int)
    p.add_argument("--round",type=int,help=argparse.SUPPRESS)
    p.add_argument("--dataset",choices=DATASETS,help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.output.exists() or min(a.rounds,a.repetitions,a.build_repetitions)<2 or a.warmup<1:
        p.error("require fresh output, rounds/repetitions/build repetitions >=2, warmup >=1")
    if a.round is not None:
        return worker(a)
    scratch = Path("results/milestone9_workers")/uuid4().hex
    scratch.mkdir(parents=True)
    # Include the actual reused code and the independent reference in provenance.
    dependencies = ("comprehensive_sparse_formats/benchmark.py","comprehensive_sparse_formats/padded.py",
                    "comprehensive_sparse_formats/padded.cpp","sparse_formats/kernels.cpp",
                    "sparse_formats/storage.py","sparse_formats/experiment.py","native_sparse_formats/sparse.py",
                    "native_sparse_formats/acquire.py","tuning_axes/orchestration.py","tuning_axes/common.py",
                    "dgl_baseline/common.py","dgl_baseline/baseline.py")
    sources = [HERE/"runner.py",HERE/"stage_profile.py",*(HERE.parent/f for f in dependencies)]
    def fingerprint(path):
        return hashlib.sha256(path.read_bytes().replace(b"\r\n",b"\n")).hexdigest()
    result = dict(status="running",settings=dict(rounds=a.rounds,warmup=a.warmup,repetitions=a.repetitions,
                  build_repetitions=a.build_repetitions,seed=SEED,model_seed=0,heads=H,channels=F,
                  threads=1,dtype="float32"),
                  sources={str(f.relative_to(HERE.parent)):fingerprint(f) for f in sources},
                  previous_raw_sha256=fingerprint(HERE.parent/"comprehensive_sparse_formats/RESULTS.json"),
                  datasets={d:[] for d in a.datasets})
    rng = random.Random(SEED)
    for round_index in range(a.rounds):
        order = list(result["datasets"])
        rng.shuffle(order)
        for dataset in order:
            target = scratch/f"{round_index}-{dataset}.json"
            cmd = [sys.executable,__file__,"--dataset",dataset,"--round",str(round_index),
                   "--output",str(target),"--data-dir",str(a.data_dir),"--warmup",str(a.warmup),
                   "--repetitions",str(a.repetitions),"--build-repetitions",str(a.build_repetitions)]
            if a.cpu is not None:
                cmd += ["--cpu",str(a.cpu)]
            subprocess.run(cmd,check=True)
            result["datasets"][dataset].append(json.loads(target.read_text()))
            save(a.output,result)
    result["status"] = "complete"
    save(a.output,result)


if __name__=="__main__":
    main()
