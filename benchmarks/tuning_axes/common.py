"""Shared CPU correctness/setup utilities; previous benchmark files are read-only."""
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import statistics
import sys
from time import perf_counter_ns

os.environ['DGLBACKEND']='pytorch'
# Fixed control: avoid idle spinning by the Torch and native OpenMP teams.
# Set before importing either runtime; never vary this inside a comparison.
os.environ['OMP_WAIT_POLICY']='PASSIVE'
os.environ['KMP_BLOCKTIME']='0'
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'dgl_baseline'))
# Import by file path: do not confuse this module with historical common.py.
import importlib.util
spec=importlib.util.spec_from_file_location('historical_dgl_utilities',HERE.parent/'dgl_baseline/common.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
import dgl
import torch
from dgl.nn.pytorch import GATConv
H,F=4,8
RTOL,ATOL=1e-4,1e-5


def configure(threads):
    assert dgl.__version__=='1.1.3' and torch.version.cuda is None
    torch.set_default_dtype(torch.float32);torch.set_num_threads(threads);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);old.verify_source()


def model_for(width,seed=0):
    torch.manual_seed(seed);dgl.seed(seed)
    return GATConv(width,F,H,feat_drop=0.,attn_drop=0.,residual=False,
        activation=None,bias=True,allow_zero_in_degree=False).float().cpu().eval()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp.json');temp.write_text(json.dumps(value,allow_nan=False)+'\n');temp.replace(path)


def stats(values):
    return {'samples':values,'median':statistics.median(values),'mean':statistics.mean(values),
            'stddev':statistics.stdev(values),'min':min(values)}


def fingerprint(t):return hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()


def tensor_info(t):
    return {'logical_shape':list(t.shape),'strides':list(t.stride()),'contiguous':t.is_contiguous(),
            'storage_bytes':t.untyped_storage().nbytes(),'dtype':str(t.dtype)}


def view_info(g):
    state=old.representation(g)
    state['idtype']=str(g.idtype)
    tensors=[]
    # Report tensor payload, deduplicating shared storage, not Python object size.
    for fmt in g.formats()['created']:
        tensors.extend(g.adj_tensors(fmt))
    unique={}
    for t in tensors:
        if t.numel():unique[(t.untyped_storage().data_ptr(),t.untyped_storage().nbytes())]=t.untyped_storage().nbytes()
    state['unique_sparse_storage_payload_bytes']=sum(unique.values())
    state['storage_measurement']='DGL exposed sparse tensor storage only; not allocator/RSS/whole-graph peak'
    return state


def check_bounds(g):
    n,e=g.num_nodes(),g.num_edges()
    bounds={'node_count':n,'edge_count':e,'row_pointer_max':e,'eid_max':max(0,e-1),
            'softmax_flat_EH_max':max(0,e*H-1),'node_feature_ND_max':max(0,n*g.ndata['feat'].shape[1]-1),
            'projected_NHF_max':max(0,n*H*F-1)}
    for fmt in g.formats()['created']:
        for i,t in enumerate(g.adj_tensors(fmt)):
            if t.numel():bounds[f'{fmt}_array_{i}_minmax']=[int(t.min()),int(t.max())]
    legal=all(v<2**31 for v in bounds.values() if isinstance(v,int))
    legal=legal and all(v[0]>=0 and v[1]<2**31 for v in bounds.values() if isinstance(v,list))
    return {'values':bounds,'int32_legal':legal,'checked':'graph counts/pointers/EIDs/flattened actual intermediate offsets'}


def validate_view(g,base):
    n,e=base.num_nodes(),base.num_edges();bu,bv=base.edges(order='eid')
    assert g.num_nodes()==n and g.num_edges()==e
    for fmt in g.formats()['created']:
        a=g.adj_tensors(fmt)
        if fmt=='coo':
            u,v=a;ids=torch.arange(e)
        else:
            ptr,idx,eids=a
            ids=eids.long() if eids.numel() else torch.arange(e)
            rows=torch.repeat_interleave(torch.arange(n),torch.diff(ptr).long())
            u,v=(rows,idx.long()) if fmt=='csr' else (idx.long(),rows)
        assert torch.equal(torch.sort(ids).values,torch.arange(e))
        assert torch.equal(u.long(),bu[ids].long()) and torch.equal(v.long(),bv[ids].long())
    for key in base.ndata:assert torch.equal(g.ndata[key],base.ndata[key])
    for key in base.edata:assert torch.equal(g.edata[key],base.edata[key])
    return True


def row_partition(ptr,threads,policy):
    start=perf_counter_ns();degree=torch.diff(ptr).long();n=len(degree)
    if policy=='static':
        rows=torch.arange(n);parts=torch.tensor([n*i//threads for i in range(threads+1)])
    elif policy=='degree_balanced':
        # EXPERIMENTAL CHANGE: largest-degree-first greedy row ownership. Each
        # row stays intact, and its neighbor iteration order never changes.
        bins=[[] for _ in range(threads)];heap=[(0,t) for t in range(threads)];heapq.heapify(heap)
        for node in sorted(range(n),key=lambda v:(-int(degree[v]),v)):
            load,thread=heapq.heappop(heap);bins[thread].append(node)
            heapq.heappush(heap,(load+int(degree[node])+1,thread))
        flat=[];offset=[0]
        for bucket in bins:flat.extend(sorted(bucket));offset.append(len(flat))
        rows=torch.tensor(flat,dtype=torch.int64);parts=torch.tensor(offset,dtype=torch.int64)
    else:raise ValueError(policy)
    assert torch.equal(torch.sort(rows).values,torch.arange(n))
    work=[int(degree[rows[parts[t]:parts[t+1]]].sum()) for t in range(threads)]
    return {'order':rows,'parts':parts,'policy':policy,'preparation_ms':(perf_counter_ns()-start)/1e6,
            'row_counts':[int(parts[t+1]-parts[t]) for t in range(threads)],'edge_counts':work,
            'row_order_sha256':fingerprint(rows),'ownership':'one complete destination per thread; no atomics or merges'}


def reference_stages(g,x,layer):
    z=layer.fc(x).view(-1,H,F)
    l=(z*layer.attn_l).sum(-1).unsqueeze(-1);r=(z*layer.attn_r).sum(-1).unsqueeze(-1)
    coo=g.formats('coo');coo.create_formats_();csc=g.formats('csc');csc.create_formats_()
    q=dgl.ops.gsddmm(coo,'add',l,r)
    s=layer.leaky_relu(q);a=dgl.ops.edge_softmax(csc,s)
    agg=dgl.ops.gspmm(csc,'mul','sum',z,a)
    y,oracle_a=layer(g,x,get_attention=True)
    torch.testing.assert_close(a,oracle_a,rtol=RTOL,atol=ATOL)
    torch.testing.assert_close(agg+layer.bias.view(1,H,F),y,rtol=RTOL,atol=ATOL)
    return {'projected':z,'source_contribution':l,'destination_contribution':r,
            'raw_logits':q,'logits':s,'attention':a,'aggregated':agg,'output':y}


def validate(case,g,x,layer):
    ref=reference_stages(g,x,layer)
    got,detail=case.forward(x,layer,details=True)
    torch.testing.assert_close(got,ref['output'],rtol=RTOL,atol=ATOL)
    errors={}
    for name,value in detail['intermediates'].items():
        target=ref[name]
        torch.testing.assert_close(value,target,rtol=RTOL,atol=ATOL)
        errors[name]={'max_absolute':float((value-target).abs().max()),
                     'max_scaled':float(((value-target).abs()/(ATOL+RTOL*target.abs())).max())}
    second=case.forward(x,layer)
    torch.testing.assert_close(second,got,rtol=0,atol=0)
    assert torch.isfinite(got).all()
    return {'status':'passed','rtol':RTOL,'atol':ATOL,'errors':errors,
            'output_max_absolute':float((got-ref['output']).abs().max()),'bitwise_repeatable':True,
            'tensors':detail.get('tensors',{}),'intermediate_payload_bytes':detail.get('intermediate_payload_bytes'),
            'coverage':list(detail['intermediates'])}


def provenance():
    info=old.metadata()
    info.update(source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob('*')) if p.suffix in ['.py','.cpp']},
        lock_sha256=hashlib.sha256((HERE.parents[1]/'uv.lock').read_bytes()).hexdigest(),
        runtime_thread_controls={k:os.environ.get(k) for k in ['OMP_WAIT_POLICY','KMP_BLOCKTIME']},
        loaded_openmp_libraries=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libgomp' in line or 'libiomp' in line}),
        cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip())
    return info
