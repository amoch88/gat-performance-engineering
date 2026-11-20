"""Matched storage, allocation, and projection/output around the loop kernels."""
from functools import partial
import sys

from compiled import Compiled, HERE
from design import DEFAULT, design, memory

sys.path[:0] = [str(HERE.parent / name) for name in ("native_sparse_formats","sparse_format_validation")]
from acquire import DATASETS, dgl, fingerprint, save, setup, torch
from common import ATOL, RTOL, H, F, reference_stages
from orchestration import Pipeline
from stage_profile import instrument, stock_profile
import numpy as np


class State:
    def __init__(self, graph, binary):
        self.graph, self.n, self.e = graph, graph.num_nodes(), graph.num_edges()
        self.lib = Compiled(binary)
        self.src,self.dst = graph.adj_tensors("coo")
        self.ptr,self.in_src,self.eid = graph.adj_tensors("csc")
        if not self.eid.numel():
            self.eid = torch.arange(self.e,dtype=torch.int64)
        for index in (self.src,self.dst,self.ptr,self.in_src,self.eid):
            assert index.dtype==torch.int64 and index.is_contiguous() and index.device.type=="cpu"
        u,v = graph.edges(order="eid")
        assert torch.equal(u,self.src) and torch.equal(v,self.dst)
        assert torch.equal(self.eid.sort().values,torch.arange(self.e))
        assert self.ptr[0]==0 and self.ptr[-1]==self.e
        destinations = np.repeat(np.arange(self.n),np.diff(self.ptr.numpy()))
        np.testing.assert_array_equal(destinations,self.dst[self.eid].numpy())
        assert torch.equal(self.src[self.eid],self.in_src)
        self.meta = memory(self.n,self.e)
        self.index_sha256 = {k:fingerprint(getattr(self,k)) for k in ("src","dst","ptr","in_src","eid")}
        self.shapes = {"terms":(self.n,H), "edge":(self.e,H), "node":(self.n,H,F)}

    def alloc_terms(self):
        left = torch.empty(self.shapes["terms"],dtype=torch.float32)
        return left,torch.empty_like(left)

    def alloc_score(self):
        return torch.empty(self.shapes["edge"],dtype=torch.float32)

    def alloc_softmax(self):
        return self.alloc_score(),torch.empty((2,H),dtype=torch.float32)

    def alloc_aggregation(self):
        return torch.empty(self.shapes["node"],dtype=torch.float32)

    def terms(self, call, z, layer):
        l,r = self.alloc_terms()
        call(*pointers(z,layer.attn_l,layer.attn_r,l,r),self.n)
        return l,r

    def score(self, call, l, r, slope):
        s = self.alloc_score()
        call(*pointers(l,r,self.src,self.dst,s),self.e,slope)
        return s

    def softmax(self, call, s):
        a,workspace = self.alloc_softmax()
        call(*pointers(self.ptr,self.eid,s,a,workspace),self.n)
        return a

    def aggregation(self, call, z, a):
        y = self.alloc_aggregation()
        call(*pointers(self.ptr,self.in_src,self.eid,z,a,y),self.n)
        return y


def pointers(*tensors):
    return (a.data_ptr() for a in tensors)


class Forward:
    guard, project, bias = Pipeline.guard, Pipeline.project, Pipeline.bias

    def __init__(self,state,orders):
        self.g = state.graph
        self.orders = orders
        self.node_terms = state.lib.bind(state.terms,orders["terms"])
        self.score = state.lib.bind(state.score,orders["score"])
        self.softmax = state.lib.bind(state.softmax,orders["softmax"])
        self.aggregation = state.lib.bind(state.aggregation,orders["aggregation"])

    def __call__(self,x,layer,details=False):
        p = self
        p.guard()
        z = p.project(layer.feat_drop(x), layer)
        left, right = p.node_terms(z, layer)
        scores = self.score(left, right, layer.leaky_relu.negative_slope)
        alpha = self.softmax(scores)
        aggregated = self.aggregation(z, alpha)
        result = p.bias(aggregated, layer)
        if details:
            return dict(projected=z,source_contribution=left,destination_contribution=right,
                        logits=scores,attention=alpha,aggregated=aggregated,output=result)
        return result

    def routes(self):
        return {stage: getattr(self,"node_terms" if stage=="terms" else stage).args[0].__name__
                for stage in self.orders}

    def isolated(self,stage,inputs,layer):
        z,l,r,s,a = (inputs[key] for key in ("projected","source_contribution",
                                             "destination_contribution","logits","attention"))
        if stage=="terms":
            return partial(self.node_terms,z,layer), "source_contribution", "destination_contribution"
        if stage=="score":
            return partial(self.score,l,r,layer.leaky_relu.negative_slope), "logits"
        if stage=="softmax":
            return partial(self.softmax,s), "attention"
        if stage=="aggregation":
            return partial(self.aggregation,z,a), "aggregated"
        raise ValueError(stage)


def bitwise(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype==torch.float32
    assert torch.equal(a.view(torch.int32),b.view(torch.int32))


def errors(value,target):
    torch.testing.assert_close(value,target,atol=ATOL,rtol=RTOL)
    assert torch.isfinite(value).all()
    ref = target.double().numpy()
    err = np.abs(value.double().numpy()-ref)
    nonzero = ref!=0.
    return dict(max_absolute=float(err.max(initial=0.)),
                max_relative_nonzero=float(np.max(err[nonzero]/np.abs(ref[nonzero]),initial=0.)),
                max_absolute_at_reference_zero=float(err[~nonzero].max(initial=0.)),
                max_tolerance_fraction=float(np.max(err/(ATOL+RTOL*np.abs(ref)),initial=0.)))


def validate(call, profiled, isolate, keys, state, layer, x, ref, shared):
    data = call(x,layer,details=True)
    result = {}
    for key,value in data.items():
        assert value.is_contiguous()
        target = ref[key].squeeze(-1) if key in (
            "source_contribution","destination_contribution","logits","attention") else ref[key]
        result[key] = errors(value,target)
        bitwise(value,shared[key])
    bitwise(call(x,layer),data["output"])
    bitwise(call(x,layer),data["output"])
    bitwise(profiled(),data["output"])
    got = isolate()
    got = got if isinstance(got,tuple) else (got,)
    for key,tensor in zip(keys,got,strict=True):
        bitwise(tensor,shared[key])
    assert (data["attention"]>=0).all()
    sums = np.zeros((state.n,H),dtype=np.float64)
    np.add.at(sums,state.dst.numpy(),data["attention"].numpy())
    incoming = np.diff(state.ptr.numpy())>0
    np.testing.assert_allclose(sums[incoming],1.,atol=ATOL,rtol=0)
    np.testing.assert_allclose(sums[~incoming],0.,atol=0,rtol=0)
    return dict(errors=result,max_normalization_error=float(np.max(np.abs(sums[incoming]-1.),initial=0.)),
                custom_bitwise=True,repeat_bitwise=True,profile_bitwise=True,isolated_bitwise=True)


def baseline(state,x,layer):
    ref = reference_stages(state.graph,x,layer)
    shared = Forward(state,DEFAULT)(x,layer,details=True)
    return ref,shared


def prepared(sweep,code,state,x,layer,ref,shared):
    forward = Forward(state,design(sweep,code))
    function,record = instrument(Forward.__call__,stock=False)
    profiled = partial(function,forward,x,layer)
    isolate,*keys = forward.isolated(sweep,shared,layer)
    check = validate(forward,profiled,isolate,keys,state,layer,x,ref,shared)
    return dict(calls={f"plain/{code}":partial(forward,x,layer),f"profile/{code}":profiled,
                       f"isolated/{code}":isolate},profile=record,check=check,routes=forward.routes())
