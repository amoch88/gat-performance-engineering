"""Direct stock GATConv benchmarks for independent node and edge sweeps."""
import argparse
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
sys.path[:0] = [str(HERE.parent/name) for name in ("native_sparse_formats","sparse_format_validation")]
from acquire import setup, save, torch, dgl
from common import ATOL, RTOL, fingerprint
from stage_profile import STAGES, stock_profile
from policies import (AXES, DATASETS, SEED, build, case_specs, digest, graph_diagnostics,
                      np, sparse_diagnostics, validate_graph)


def sha(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n",b"\n")).hexdigest()


def timer_calibration():
    record = {}
    for stage in STAGES:
        start = perf_counter_ns()
        record[stage] = perf_counter_ns()-start
    return record


def validate_output(g,x,layer,profiled,nodes,edges,expected,attention):
    result,alpha = layer(g,x,get_attention=True)
    prof,prof_alpha = profiled(g,x,get_attention=True)
    torch.testing.assert_close(prof,result,rtol=0,atol=0)
    torch.testing.assert_close(prof_alpha,alpha,rtol=0,atol=0)
    restored = torch.empty_like(result)
    restored[torch.from_numpy(nodes)] = result
    restored_alpha = torch.empty_like(alpha)
    restored_alpha[torch.from_numpy(edges)] = alpha
    assert restored.isfinite().all() and restored_alpha.isfinite().all()
    torch.testing.assert_close(restored,expected,rtol=RTOL,atol=ATOL)
    torch.testing.assert_close(restored_alpha,attention,rtol=RTOL,atol=ATOL)
    dst = g.edges(order="eid")[1].numpy()
    heads = alpha.shape[1]
    sums = np.zeros((len(nodes),heads),dtype=np.float64)
    np.add.at(sums,dst,alpha.numpy().reshape(-1,heads))
    np.testing.assert_allclose(sums,1.,rtol=0,atol=ATOL)
    return dict(output_abs=float((restored-expected).abs().max()),
                attention_abs=float((restored_alpha-attention).abs().max()),
                normalization_abs=float(abs(sums-1).max()),
                bitwise_profile_match=True,finite=True,atol=ATOL,rtol=RTOL)


def encode(ids):
    if np.array_equal(ids,np.arange(len(ids))):
        return {"identity":len(ids)}
    return {"new_to_original":ids.tolist()}


def run_axis(candidates,layer,profiled,regions,rng,count):
    keys = list(candidates)
    schedule = [(mode,f) for mode in ("plain","profile") for f in keys]
    schedule += [("timer","empty"),("timer","six")]
    calls = {}
    for f,(graph,x) in candidates.items():
        calls["plain",f] = partial(layer,graph,x)
        calls["profile",f] = partial(profiled,graph,x)
    calls["timer","empty"] = lambda:None
    calls["timer","six"] = timer_calibration
    result = []
    for _ in range(count):
        order = schedule.copy()
        rng.shuffle(order)
        rows = []
        for mode,f in order:
            invoke = calls[mode,f]
            start = perf_counter_ns()
            output = invoke()
            elapsed = perf_counter_ns()-start
            del output
            if mode=="profile":
                stage_ns = [regions[s] for s in STAGES]
                assert min(stage_ns)>0 and sum(stage_ns)<=elapsed
                rows.append([mode,f,elapsed,stage_ns])
            else:
                rows.append([mode,f,elapsed])
        result.append(rows)
    return result


def worker(a):
    base,x,layer,src,dst,meta = setup(a.dataset,a.data_dir,a.cpu)
    stock_type = dgl.nn.pytorch.GATConv
    assert type(layer) is stock_type and layer.forward.__func__ is stock_type.forward
    assert os.sched_getaffinity(0)=={meta["cpu"]}
    profiled,regions = stock_profile(layer)
    seed = SEED+1009*a.round+DATASETS.index(a.dataset)
    rng = random.Random(seed)
    axes = list(AXES)
    rng.shuffle(axes)
    cases,timings,maps = {},{},{}
    loader_sparse = sparse_diagnostics(base,np.arange(len(x)),np.arange(len(src)),src,dst)
    with torch.inference_mode():
        expected,alpha = layer(base,x,get_attention=True)
        for axis in axes:
            cases[axis],maps[axis] = {},{}
            candidates = {}
            for f,policy,perm_seed in case_specs(a.dataset,axis):
                g,nodes,edges,_,before = build(base,src,dst,axis,policy,perm_seed)
                representations = validate_graph(base,g,nodes,edges,src,dst,axis)
                errors = validate_output(g,g.ndata["feat"],layer,profiled,nodes,edges,expected,alpha)
                cases[axis][f] = dict(policy=policy,permutation_seed=perm_seed,maps_sha256=digest(nodes,edges),
                    graph=graph_diagnostics(g),sparse=representations,before_materialization=before,
                    after_materialization=g.formats(),correctness=errors)
                maps[axis][f] = dict(nodes=encode(nodes),edges=encode(edges),seed=perm_seed)
                candidates[f] = g,g.ndata["feat"]
            timings[axis] = run_axis(candidates,layer,profiled,regions,rng,a.warmup+a.repetitions)
            for f,(g,_) in candidates.items():
                assert g.formats()==cases[axis][f]["after_materialization"]
            del candidates
        assert {k:fingerprint(t) for k,t in layer.state_dict().items()}==meta["weights"]
        assert {k:fingerprint(t) for k,t in profiled.state_dict().items()}==meta["weights"]
        assert cases["node"]["original"]==cases["edge"]["original"]
        construction = []
        builds = [(axis,f,policy,seed) for axis in axes for f,policy,seed in case_specs(a.dataset,axis)]
        for _ in range(a.build_repetitions):
            order = builds.copy()
            rng.shuffle(order)
            block = []
            for axis,f,policy,perm_seed in order:
                g,nodes,edges,elapsed,before = build(base,src,dst,axis,policy,perm_seed)
                assert before==cases[axis][f]["before_materialization"]
                assert g.formats()==cases[axis][f]["after_materialization"]
                block.append([axis,f,*elapsed])
                del g
            construction.append(block)
    save(a.output,dict(**meta,round=a.round,order_seed=seed,axis_order=axes,cases=cases,
                       maps=maps,loader_sparse=loader_sparse,blocks=timings,construction=construction))
    print(f"{a.dataset} process {a.round+1}: both sweeps validated and timed",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--maps",type=Path,default=None)
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--datasets",choices=DATASETS,nargs="+",default=DATASETS)
    parser.add_argument("--rounds",type=int,default=8)
    parser.add_argument("--warmup",type=int,default=10)
    parser.add_argument("--repetitions",type=int,default=30)
    parser.add_argument("--build-repetitions",type=int,default=5)
    parser.add_argument("--cpu",type=int)
    parser.add_argument("--round",type=int,help=argparse.SUPPRESS)
    parser.add_argument("--dataset",choices=DATASETS,help=argparse.SUPPRESS)
    a = parser.parse_args()
    maps_path = a.maps or a.output.with_name(a.output.stem+"_PERMUTATIONS.json")
    if (a.output.exists() or maps_path.exists() or maps_path==a.output or
            min(a.rounds,a.repetitions,a.build_repetitions)<2 or a.warmup<1):
        parser.error("require distinct fresh outputs, rounds/repetitions/build repetitions >=2, warmup >=1")
    if a.round is not None:
        return worker(a)
    scratch = Path("results/milestone10_workers")/uuid4().hex
    scratch.mkdir(parents=True)
    sources = ["ordering_baselines/benchmark.py","ordering_baselines/policies.py",
               "native_sparse_formats/acquire.py","sparse_format_validation/stage_profile.py",
               "tuning_axes/common.py","sparse_formats/run.py","dgl_baseline/common.py",
               "dgl_baseline/baseline.py","gat.py"]
    data = dict(status="running",settings=dict(rounds=a.rounds,warmup=a.warmup,repetitions=a.repetitions,
                build_repetitions=a.build_repetitions,order_seed=SEED,model_seed=0,threads=1,heads=4,channels=8,
                practical_speedup_floor=1.01),
                sources={f:sha(HERE.parent/f) for f in sources},
                lock_sha256=sha(HERE.parents[1]/"uv.lock"),datasets={d:[] for d in a.datasets})
    maps = {}
    rng = random.Random(SEED)
    for i in range(a.rounds):
        order = list(data["datasets"])
        rng.shuffle(order)
        for d in order:
            output = scratch/f"{d}-{i}.json"
            cmd = [sys.executable,__file__,"--dataset",d,"--round",str(i),"--output",str(output),
                   "--data-dir",str(a.data_dir),"--warmup",str(a.warmup),"--repetitions",str(a.repetitions),
                   "--build-repetitions",str(a.build_repetitions)]
            if a.cpu is not None:
                cmd += ["--cpu",str(a.cpu)]
            subprocess.run(cmd,check=True)
            row = json.loads(output.read_text())
            permutation_maps = row.pop("maps")
            if d in maps:
                assert maps[d]==permutation_maps
            else:
                maps[d] = permutation_maps
                save(maps_path,maps)
            data["datasets"][d].append(row)
            save(a.output,data)
    data.update(status="complete",maps_sha256=sha(maps_path))
    save(a.output,data)


if __name__=="__main__":
    main()
