"""Milestone 7 native DGL comparisons and additional blocked tile sizes."""
import ctypes
import numpy as np
from experiment import Kernel, OPERATIONS, Variant, build_library
from common import dgl, reference_stages, torch, validate_view, view_info
from orchestration import Pipeline
from storage import Storage, build, packed, ptr
from run import blocks

SUPPORTED = {"score": ("coo", "csr"), "softmax": ("csc",), "aggregation": ("coo", "csc")}
BLOCKED = {f"{f}{b}": (f,b) for f in ("bcsr","bcsc","csb") for b in (2,4,8,16,32,64)}
SOURCE = "https://raw.githubusercontent.com/dmlc/dgl/v1.1.3/src/array/kernel.cc"


def build_blocked(n, src, dst, name, block):
    if f"{name}{block}" not in BLOCKED:
        raise ValueError("unsupported blocked format or tile size")
    if block >= 16:
        return build(n, src, dst, name, block)
    # Extend the Milestone 6 encoding without changing its registry/artifacts.
    # Validate dimensions, integer bounds and endpoints with the existing builder.
    coo = build(n, src, dst, "coo", 0)
    e = coo.e
    u,v = (dst,src) if name == "bcsc" else (src,dst)
    row,col = u//block,v//block
    ids = np.lexsort((u%block,col,row))
    row,col = row[ids],col[ids]
    lr,lc = u[ids]%block,v[ids]%block
    starts = (np.flatnonzero(np.r_[True,(row[1:]!=row[:-1]) | (col[1:]!=col[:-1])])
              if e else np.empty(0,dtype=np.int64))
    tiles = len(starts)
    offsets = np.r_[starts,e]
    if name == "csb":
        arrays = (packed(offsets),packed(row[starts]),packed(col[starts]),packed(lr),packed(lc),packed(ids))
    else:
        if tiles*block >= 2**31:
            raise ValueError("tile pointer table too large")
        local_rows = np.repeat(np.arange(tiles),np.diff(offsets))*block+lr
        arrays = (ptr(row[starts],(n+block-1)//block),packed(col[starts]),packed([]),
                  ptr(local_rows,tiles*block),packed(lc),packed(ids))
    return Storage(n,e,name,block,tiles,arrays)


def view(g, fmt):
    result = g.formats(fmt)
    result.create_formats_()
    assert result.formats() == {"created":[fmt],"not created":[]}
    return result


class Native:
    def __init__(self, g, op, fmt):
        if fmt not in SUPPORTED[op]:
            raise ValueError(f"no native {op}/{fmt} traversal in DGL 1.1.3 CPU")
        self.g,self.op = g,op

    def __call__(self,x,y=None,slope=.2):
        if self.op == "score":
            return torch.nn.functional.leaky_relu(dgl.ops.gsddmm(self.g,"add",x,y),slope)
        if self.op == "softmax":
            return dgl.ops.edge_softmax(self.g,x,norm_by="dst")
        return dgl.ops.gspmm(self.g,"mul","sum",x,y)


def acquire(g,x,layer,src,dst,args,rng):
    path,compiler = build_library()
    library = ctypes.CDLL(str(path.resolve()))
    fixed = Pipeline(g,score="coo",aggregation="csc")
    builders = {fmt:(lambda f=fmt:view(dgl.graph((src,dst),num_nodes=g.num_nodes()),f))
                for fmt in ("coo","csr","csc")}
    # Time restriction/materialization from an already created fresh COO graph.
    # Graph construction itself is not part of conversion.
    builders.update({key:(lambda f=f,b=b:build_blocked(g.num_nodes(),src,dst,f,b))
                     for key,(f,b) in BLOCKED.items()})
    stores = {key:builders[key]() for key in BLOCKED}
    views = {fmt:view(g,fmt) for fmt in ("coo","csr","csc")}
    storage = {}
    for key,v in views.items():
        validate_view(v,g)
        storage[key] = dict(bytes=view_info(v)["unique_sparse_storage_payload_bytes"], index_bits=64,
                            nonempty_tiles=0,mask=v.formats())
    for key,s in stores.items():
        s.validate(src,dst)
        storage[key] = dict(bytes=s.bytes,index_bits=32,nonempty_tiles=s.tiles)
    variants = {}
    for op in OPERATIONS:
        for fmt in SUPPORTED[op]:
            variants[f"{op}/{fmt}"] = Variant(fixed,Native(views[fmt],op,fmt))
        for key,s in stores.items():
            variants[f"{op}/{key}"] = Variant(fixed,Kernel(library,s,op))
    ref = reference_stages(g,x,layer)
    checks = {k:v.validate(x,layer,ref,dst) for k,v in variants.items()}
    forwards = {k:(lambda v=v:v.forward(x,layer)) for k,v in variants.items()}
    forwards["stock"] = lambda:layer(g,x)
    isolated = {k:(lambda v=v:v.isolated(ref,layer)) for k,v in variants.items()}
    result = dict(storage=storage,compiler=compiler,correctness=checks,source=SOURCE,
                  unsupported=[f"{op}/{fmt}" for op in OPERATIONS for fmt in ("coo","csr","csc")
                               if fmt not in SUPPORTED[op]])
    for mode,calls in (("forward",forwards),("kernel",isolated)):
        result[mode+"_warmups"] = blocks(calls,args.warmup,rng)
        result[mode] = blocks(calls,args.repetitions,rng)
    result["conversion"] = []
    from time import perf_counter_ns
    for _ in range(5):
        order = list(builders); rng.shuffle(order)
        observations = []
        for key in order:
            if key in views:
                cold = dgl.graph((src,dst),num_nodes=g.num_nodes())
                start = perf_counter_ns()
                built = view(cold,key)
            else:
                start = perf_counter_ns()
                built = builders[key]()
            observations.append([key,(perf_counter_ns()-start)/1e6])
            del built
        result["conversion"].append(observations)
    for fmt,v in views.items():
        assert v.formats() == storage[fmt]["mask"]
    return result
