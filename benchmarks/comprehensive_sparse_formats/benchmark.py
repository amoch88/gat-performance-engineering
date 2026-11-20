"""Run each format through every supported GAT stage."""
import argparse
import ctypes
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from time import perf_counter_ns
from uuid import uuid4

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"native_sparse_formats"))
from acquire import DATASETS, setup, blocks, old, save, torch, dgl
from common import H, F, RTOL, ATOL, reference_stages, validate_view, view_info
from orchestration import Pipeline
from sparse import SUPPORTED, BLOCKED, Native, build_blocked, view, SOURCE
from experiment import Kernel, build_library
from padded import PaddedKernel, StorageLimit, build_padded, library, np

OPS = tuple(SUPPORTED)
SIZES = (2,4,8,16,32,64)
PADDED = {f"{kind}{size}": (kind,size) for kind in ("dense","sell") for size in SIZES}
FORMATS = ("coo","csr","csc",*BLOCKED,*PADDED)
FALLBACK = dict(score="coo",softmax="csc",aggregation="csc")
ORDER_SEED = 2026111800


class Forward:
    def __init__(self, fixed, kernels):
        self.fixed = fixed
        self.score,self.softmax,self.aggregation = (kernels[op] for op in OPS)

    def __call__(self,x,layer,details=False):
        p = self.fixed
        p.guard()
        z = p.project(layer.feat_drop(x),layer)
        left,right = p.node_terms(z,layer)
        scores = self.score(left,right,layer.leaky_relu.negative_slope)
        alpha = self.softmax(scores)
        aggregated = self.aggregation(z,alpha)
        result = p.bias(aggregated,layer)
        if details:
            return dict(projected=z, source_contribution=left, destination_contribution=right,
                        logits=scores,attention=alpha,aggregated=aggregated,output=result)
        return result


def stock_stages(g,layer,ref):
    """Replay the sparse calls of the homogeneous stock forward in isolation."""
    import dgl.function as fn
    local = g.local_var()
    local.srcdata.update({"el":ref["source_contribution"],"ft":ref["projected"]})
    local.dstdata["er"] = ref["destination_contribution"]
    local.edata["a"] = ref["attention"]

    def score():
        local.apply_edges(fn.u_add_v("el","er","e"))
        return layer.leaky_relu(local.edata.pop("e"))

    def aggregate():
        local.srcdata["ft"] = ref["projected"]
        local.update_all(fn.u_mul_e("ft","a","m"),fn.sum("m","ft"))
        return local.dstdata["ft"]

    return dict(score=score,softmax=lambda:dgl.ops.edge_softmax(local,ref["logits"]),
                aggregation=aggregate)


def isolated(kernel,op,ref,layer):
    if op == "score":
        return partial(kernel,ref["source_contribution"],ref["destination_contribution"],
                       layer.leaky_relu.negative_slope)
    if op == "softmax":
        return partial(kernel,ref["logits"])
    return partial(kernel,ref["projected"],ref["attention"])


def validate(call,x,layer,ref,dst):
    stages = call(x,layer,details=True)
    errors = {}
    for name,output in stages.items():
        torch.testing.assert_close(output,ref[name],atol=ATOL,rtol=RTOL)
        errors[name] = float((output-ref[name]).abs().max())
    sums = np.zeros((len(x),H),dtype=np.float64)
    np.add.at(sums,dst,stages["attention"].numpy().reshape(-1,H))
    active = np.bincount(dst,minlength=len(x))>0
    np.testing.assert_allclose(sums[active],1,atol=ATOL,rtol=0)
    torch.testing.assert_close(call(x,layer),stages["output"],atol=0,rtol=0)
    return dict(errors=errors,normalization_error=float(np.max(np.abs(sums[active]-1),initial=0)),
                bitwise_repeatable=True)


def worker(args):
    g,x,layer,src,dst,record = setup(args.dataset,args.data_dir,args.cpu)
    n = g.num_nodes()
    path, compiler = build_library()
    blocked_lib = ctypes.CDLL(str(path.resolve()))
    padded_lib,padded_compiler = library()
    fixed = Pipeline(g)
    views = {fmt:view(g,fmt) for fmt in FORMATS[:3]}
    for graph in views.values():
        validate_view(graph,g)
    native = {(op,fmt):Native(views[fmt],op,fmt) for op in OPS for fmt in SUPPORTED[op]}
    storage = {fmt:dict(bytes=view_info(graph)["unique_sparse_storage_payload_bytes"],
                        index_bits=64,mask=graph.formats()) for fmt,graph in views.items()}
    builders = {name:partial(build_blocked,n,src,dst,kind,size)
                for name,(kind,size) in BLOCKED.items()}
    builders.update({name:partial(build_padded,n,src,dst,kind,size,args.layout_budget)
                     for name,(kind,size) in PADDED.items()})
    stores,guards = {},{}
    resident = 0
    for name,build in builders.items():
        try:
            if name in PADDED:
                kind,size = PADDED[name]
                effective = min(args.layout_budget,args.resident_budget-resident)
                store = build_padded(n,src,dst,kind,size,effective)
            else:
                store = build()
        except StorageLimit as exc:
            guards[name] = dict(required_bytes=exc.required,limit_bytes=exc.limit,reason=str(exc))
            continue
        store.validate(src,dst)
        stores[name] = store
        resident += store.bytes
        storage[name] = dict(bytes=store.bytes,index_bits=32)
        if name in PADDED:
            storage[name].update(groups=store.groups,slots=store.slots,
                                 parallel_sidecar_edges=len(store.arrays[4]))
        else:
            storage[name]["groups"] = store.tiles
    assert resident <= args.resident_budget
    routes,checks,sparse_calls = {},{},{}
    forwards = {"stock":partial(layer,g,x)}
    with torch.inference_mode():
        ref = reference_stages(g,x,layer)
        sparse_calls.update({f"{op}/stock":fn for op,fn in stock_stages(g,layer,ref).items()})
        for name in FORMATS:
            if name in guards:
                continue
            if name in views:
                routes[name] = {op:name if name in SUPPORTED[op] else FALLBACK[op] for op in OPS}
                kernels = {op:native[op,fmt] for op,fmt in routes[name].items()}
            else:
                cls,lib = (PaddedKernel,padded_lib) if name in PADDED else (Kernel,blocked_lib)
                kernels = {op:cls(lib,stores[name],op) for op in OPS}
                routes[name] = dict.fromkeys(OPS,name)
            forward = Forward(fixed,kernels)
            checks[name] = validate(forward,x,layer,ref,dst)
            forwards[name] = partial(forward,x,layer)
            for op,kernel in kernels.items():
                if routes[name][op] == name:
                    sparse_calls[f"{op}/{name}"] = isolated(kernel,op,ref,layer)
        targets = dict(score="logits",softmax="attention",aggregation="aggregated")
        isolated_errors = {}
        for key,call in sparse_calls.items():
            op = key.split("/")[0]
            observed = call()
            expected = ref[targets[op]]
            torch.testing.assert_close(observed,expected,atol=ATOL,rtol=RTOL)
            isolated_errors[key] = float((observed-expected).abs().max())
        # Dense stages are identical across formats. Time the shared calls once
        # per block so noise cannot create a fictitious format-specific rank.
        stage_calls = dict(projection=lambda:fixed.project(layer.feat_drop(x),layer),
                           terms=partial(fixed.node_terms,ref["projected"],layer))
        stage_calls.update(sparse_calls)
        seed = ORDER_SEED+1009*args.worker+DATASETS.index(args.dataset)
        rng = random.Random(seed)
        for mode,calls in (("forward",forwards),("stage",stage_calls)):
            record[mode+"_warmups"] = blocks(calls,args.warmup,rng)
            record[mode] = blocks(calls,args.repetitions,rng)
    construction = []
    for _ in range(args.build_repetitions):
        order = list(storage)
        rng.shuffle(order)
        observations = []
        for name in order:
            if name in views:
                cold = dgl.graph((src,dst),num_nodes=n)
                start = perf_counter_ns()
                built = view(cold,name)
            else:
                start = perf_counter_ns()
                built = builders[name]()
            observations.append([name,(perf_counter_ns()-start)/1e6])
            del built
        construction.append(observations)
    assert all(v.formats()==storage[name]["mask"] for name,v in views.items())
    save(args.output,dict(**record,round=args.worker,order_seed=seed,storage=storage,
         guards=guards,routes=routes,checks=checks,isolated_errors=isolated_errors,
         construction=construction,compiler=compiler,padded_compiler=padded_compiler,
         resident_custom_index_bytes=resident,official_dispatch_source=SOURCE))
    print(f"{args.dataset} round {args.worker+1}: {len(routes)} format pipelines validated, "
          f"{len(guards)} memory exclusions",flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--data-dir",type=Path,required=True)
    p.add_argument("--datasets",nargs="+",choices=DATASETS,default=DATASETS)
    p.add_argument("--rounds",type=int,default=8)
    p.add_argument("--warmup",type=int,default=10)
    p.add_argument("--repetitions",type=int,default=30)
    p.add_argument("--build-repetitions",type=int,default=5)
    p.add_argument("--worker",type=int,help=argparse.SUPPRESS)
    p.add_argument("--dataset",choices=DATASETS,help=argparse.SUPPRESS)
    p.add_argument("--layout-budget",type=int,help=argparse.SUPPRESS)
    p.add_argument("--resident-budget",type=int,help=argparse.SUPPRESS)
    p.add_argument("--cpu",type=int)
    args = p.parse_args()
    if args.output.exists() or min(args.rounds,args.repetitions,args.build_repetitions)<2 or args.warmup<1:
        p.error("require a new output, rounds/repetitions/build repetitions >=2 and warmup >=1")
    if args.worker is not None:
        return worker(args)
    available = old.available_memory()
    layout_budget = min(1<<30,available//8)
    resident_budget = min(2<<30,available//3)
    scratch = Path("results/milestone8_workers")/uuid4().hex
    scratch.mkdir(parents=True)
    paths = list(HERE.glob("*.cpp")) + [HERE/name for name in ("benchmark.py","padded.py")]
    for directory,names in (
        ("native_sparse_formats",("acquire.py","sparse.py")),
        ("sparse_formats",("experiment.py","kernels.cpp","run.py","storage.py")),
        ("tuning_axes",("common.py","orchestration.py","native.py")),
        ("dgl_baseline",("common.py","baseline.py")),
    ):
        paths.extend(HERE.parent/directory/name for name in names)
    data = dict(status="running",settings=dict(rounds=args.rounds,warmup=args.warmup,
                repetitions=args.repetitions,build_repetitions=args.build_repetitions,
                available_memory_bytes=available,layout_budget=layout_budget,resident_budget=resident_budget,
                threads=1,heads=H,channels=F,model_seed=0,dtype="float32",order_seed=ORDER_SEED),
                sources={str(path.relative_to(HERE.parent)):hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in paths},datasets={d:[] for d in args.datasets})
    rng = random.Random(ORDER_SEED)
    for i in range(args.rounds):
        order = list(data["datasets"])
        rng.shuffle(order)
        for d in order:
            target = scratch/f"{i}-{d}.json"
            cmd = [sys.executable,__file__,"--dataset",d,"--worker",str(i),"--data-dir",str(args.data_dir),
                   "--output",str(target),"--repetitions",str(args.repetitions),
                   "--build-repetitions",str(args.build_repetitions),"--warmup",str(args.warmup),
                   "--layout-budget",str(layout_budget),"--resident-budget",str(resident_budget)]
            if args.cpu is not None:
                cmd += ["--cpu",str(args.cpu)]
            subprocess.run(cmd,check=True)
            data["datasets"][d].append(json.loads(target.read_text()))
            save(args.output,data)
    data["status"] = "complete"
    save(args.output,data)


if __name__=="__main__":
    main()
