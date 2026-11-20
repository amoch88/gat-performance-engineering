"""Milestone 7 acquisition: preliminary stock equivalence, then sparse OPAT."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from uuid import uuid4

os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                  OMP_DYNAMIC="FALSE", DGLBACKEND="pytorch")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sparse_formats"))
from run import blocks, DATASETS
from common import configure, fingerprint, model_for, old, provenance, save, torch, dgl

ORDER_SEED = 2026111700
EQUIVALENCE = (0.97, 1.03)  # annotated / official latency; fixed before acquisition


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup(dataset, data_dir, cpu):
    allowed = os.sched_getaffinity(0)
    cpu = min(allowed) if cpu is None else cpu
    if cpu not in allowed:
        raise ValueError("CPU outside affinity mask")
    os.sched_setaffinity(0, {cpu})
    configure(1)
    assert torch.get_num_threads() == 1 and torch.get_num_interop_threads() == 1
    dgl.utils.set_num_threads(1)
    g, x, graph = old.load_dataset(dataset, data_dir)
    g.create_formats_()
    src, dst = (t.long().numpy() for t in g.edges(order="eid"))
    layer = model_for(x.shape[1], 0)
    prior = json.loads((HERE.parent/"tuning_axes/RESULTS.json").read_text())["datasets"][dataset]["rounds"][0]
    meta = dict(dataset=dataset, cpu=cpu, pid=os.getpid(), graph=graph, environment=provenance(),
                edge_sha256=[hashlib.sha256(a.tobytes()).hexdigest() for a in (src,dst)],
                weights={k:fingerprint(v) for k,v in layer.state_dict().items()},
                graph_formats=g.formats())
    assert meta["weights"] == prior["model"]["weights_sha256"]
    assert graph["feature_sha256"] == prior["dataset_info"]["feature_sha256"]
    assert meta["edge_sha256"] == prior["axes"]["score_traversal"]["cases"]["coo"]["implementation"]["views"]["score"]["sparse_hashes"]["coo"]["indices"]
    return g, x, layer, src, dst, meta


def preliminary(g, x, official, args, rng):
    old.verify_source()
    annotated = old.GATConv(x.shape[1], 8, 4, feat_drop=0., attn_drop=0.,
                            residual=False, activation=None, bias=True,
                            allow_zero_in_degree=False).float().cpu().eval()
    annotated.load_state_dict(official.state_dict())
    for name, tensor in official.state_dict().items():
        torch.testing.assert_close(tensor, annotated.state_dict()[name], rtol=0, atol=0)
    globals_used = ("th", "fn", "edge_softmax", "DGLError", "expand_as_pair")
    assert all(type(annotated).forward.__globals__[key] is type(official).forward.__globals__[key]
               for key in globals_used)
    expected, ea = official(g, x, get_attention=True)
    output, a = annotated(g, x, get_attention=True)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(a, ea, rtol=0, atol=0)
    formats = g.formats()
    calls = {"official": lambda: official(g,x), "annotated": lambda: annotated(g,x)}
    warmups = blocks(calls, args.warmup, rng)
    measured = blocks(calls, args.repetitions, rng)
    assert g.formats() == formats
    return dict(warmups=warmups, forward=measured, executable_class_ast_equal=True,
                same_forward_dependencies=True, bitwise_output=True, bitwise_attention=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("preliminary", "sparse"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--datasets", choices=DATASETS, nargs="+", default=DATASETS)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repetitions", type=int)
    p.add_argument("--cpu", type=int)
    p.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.repetitions is None:
        args.repetitions = 60 if args.phase == "preliminary" else 30
    if args.output.exists() or args.rounds < 2 or args.warmup < 1 or args.repetitions < 2:
        p.error("require a new output, rounds>=2, warmup>=1 and repetitions>=2")
    if args.worker is not None:
        dataset, = args.datasets
        g,x,layer,src,dst,meta = setup(dataset,args.data_dir,args.cpu)
        seed = ORDER_SEED + (0 if args.phase == "preliminary" else 100000) + args.worker*1009 + DATASETS.index(dataset)
        rng = random.Random(seed)
        with torch.inference_mode():
            if args.phase == "preliminary":
                result = preliminary(g,x,layer,args,rng)
            else:
                from sparse import acquire
                result = acquire(g,x,layer,src,dst,args,rng)
        save(args.output, dict(**meta, **result, round=args.worker, order_seed=seed))
        print(f"{args.phase}: {dataset} round {args.worker+1} passed",flush=True)
        return
    scratch = Path("results/milestone7_workers") / uuid4().hex
    scratch.mkdir(parents=True)
    result = dict(phase=args.phase, status="running",
                  settings=dict(rounds=args.rounds,warmup=args.warmup,repetitions=args.repetitions,
                                tolerance=EQUIVALENCE, model_seed=0, threads=1, heads=4, channels=8, dtype="float32"),
                  sources={str(path.relative_to(HERE.parent)):sha(path) for path in (
                      HERE/"acquire.py", HERE.parent/"sparse_formats/run.py", HERE.parent/"dgl_baseline/baseline.py",
                      *( [HERE/"sparse.py", HERE.parent/"sparse_formats/experiment.py",
                           HERE.parent/"sparse_formats/storage.py",HERE.parent/"sparse_formats/kernels.cpp"]
                         if args.phase=="sparse" else []))},
                  datasets={d:[] for d in args.datasets})
    rng=random.Random(ORDER_SEED)
    for r in range(args.rounds):
        order=list(result["datasets"]); rng.shuffle(order)
        for d in order:
            target=scratch/f"{d}-{r}.json"
            cmd=[sys.executable,__file__,args.phase,"--output",str(target),"--data-dir",str(args.data_dir),
                 "--datasets",d,"--worker",str(r),"--warmup",str(args.warmup),"--repetitions",str(args.repetitions)]
            if args.cpu is not None:
                cmd += ["--cpu",str(args.cpu)]
            subprocess.run(cmd,check=True)
            result["datasets"][d].append(json.loads(target.read_text()))
            save(args.output,result)
    result["status"]="complete"
    save(args.output,result)


if __name__ == "__main__":
    main()
