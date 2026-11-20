"""Public DGL graph transformations and untouched official CPU GATConv only."""
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
from time import perf_counter_ns

os.environ['DGLBACKEND'] = 'pytorch'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'dgl_baseline'))
from common import (available_memory, digest, load_dataset, metadata, reddit_memory_estimate,
                    representation, verify_source, RTOL, ATOL)
import dgl
import torch
from dgl.nn.pytorch import GATConv

NODES = ['original', 'rcmk', 'degree']
EDGES = ['original', 'src', 'dst']
FORMATS = {'all': ['coo','csr','csc'], 'coo+csc': ['coo','csc'], 'csr+csc': ['csr','csc']}
BASE = 'original/original/all'


def save(path, data, pretty=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2 if pretty else None, allow_nan=False) + '\n')
    temporary.replace(path)


def configure(threads=1):
    if dgl.__version__ != '1.1.3' or torch.version.cuda is not None:
        raise RuntimeError('DGL 1.1.3 and CPU-only PyTorch required')
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    verify_source()  # The preserved annotated baseline is not edited.


def model_for(width, seed=0):
    torch.manual_seed(seed)
    dgl.seed(seed)
    return GATConv(width, 8, 4, feat_drop=0., attn_drop=0., residual=False,
                   bias=True, activation=None, allow_zero_in_degree=False).float().cpu().eval()


def stats(samples):
    return {'samples': samples, 'median': statistics.median(samples),
            'mean': statistics.mean(samples), 'stddev': statistics.stdev(samples), 'min': min(samples)}


def attach_ids(graph):
    graph.ndata['joint_nid'] = torch.arange(graph.num_nodes())
    graph.edata['joint_eid'] = torch.arange(graph.num_edges())


def prepare(base, node, edge, fmt):
    """No layout/kernel code: use official reorder_graph and formats APIs."""
    start = perf_counter_ns()
    if node == 'original' and edge == 'original':
        changed = base
        reorder_ms = 0.
    else:
        # EXPERIMENTAL CHANGE: vary only public node/edge permutation settings.
        # DGL 1.1.3's custom edge path explicitly undoes edge reordering caused by
        # node_subgraph before applying edges_perm. Identity edges_perm therefore
        # preserves ORIGINAL canonical EIDs even after RCMK/custom node relabeling.
        # No nonexistent edge_permute_algo=None option is used.
        options = {}
        algo = None if node == 'original' else node
        if node == 'degree':
            algo = 'custom'
            options['nodes_perm'] = torch.argsort(base.in_degrees() + base.out_degrees(), stable=True)
        if edge == 'original':
            options['edges_perm'] = torch.arange(base.num_edges())
        with base.local_scope():
            changed = dgl.reorder_graph(base, node_permute_algo=algo,
                edge_permute_algo='custom' if edge == 'original' else edge,
                permute_config=options, store_ids=True)
        reorder_ms = (perf_counter_ns() - start) / 1e6
    start = perf_counter_ns()
    # EXPERIMENTAL CHANGE: restrict permissions and fully materialize exactly
    # the pilot-approved set. This does not specify per-operator kernel dispatch.
    if changed is base and fmt == 'all':
        formatted = changed
        format_ms = 0.
    else:
        formatted = changed.formats(FORMATS[fmt])
        formatted.create_formats_()
        format_ms = (perf_counter_ns() - start) / 1e6
    ids = formatted.ndata['joint_nid']
    start = perf_counter_ns()
    if node == 'original':
        inverse = None
        map_ms = 0.
    else:
        identity = torch.equal(ids, torch.arange(base.num_nodes()))
        inverse = None if identity else torch.argsort(ids)
        map_ms = (perf_counter_ns() - start) / 1e6
    return {'graph': formatted, 'x': formatted.ndata['feat'], 'ids': ids, 'inverse': inverse,
            'key': f'{node}/{edge}/{fmt}', 'node':node, 'edge':edge, 'format':fmt,
            'preparation_ms': {'reorder_and_feature_permutation': reorder_ms,
                               'format_restriction_materialization': format_ms,
                               'restoration_map':map_ms, 'total': reorder_ms + format_ms + map_ms}}


def restore(output, inverse):
    return output if inverse is None else output.index_select(0, inverse)


def verify_prepared(case, base, model, reference):
    graph, ids = case['graph'], case['ids']
    eids = graph.edata['joint_eid']
    n, e = base.num_nodes(), base.num_edges()
    assert graph.num_nodes() == n and graph.num_edges() == e
    assert torch.equal(torch.sort(ids).values, torch.arange(n))
    assert torch.equal(torch.sort(eids).values, torch.arange(e))
    assert torch.equal(case['x'], base.ndata['feat'][ids])
    if dgl.NID in graph.ndata:
        assert torch.equal(ids, graph.ndata[dgl.NID])
    if dgl.EID in graph.edata:
        assert torch.equal(eids, graph.edata[dgl.EID])
    u, v = graph.edges(order='eid')
    bu, bv = base.edges(order='eid')
    # Exact EID-indexed multigraph equality, including parallel edges and loops.
    assert torch.equal(ids[u], bu[eids]) and torch.equal(ids[v], bv[eids])
    if case['edge'] == 'original':
        assert torch.equal(eids, torch.arange(e))
    elif case['edge'] == 'src':
        assert bool((u[1:] >= u[:-1]).all())
    else:
        assert bool((v[1:] >= v[:-1]).all())
    if case['node'] == 'degree':
        expected = torch.argsort(base.in_degrees() + base.out_degrees(), stable=True)
        assert torch.equal(ids, expected)
    if case['node'] == 'original':
        assert torch.equal(ids, torch.arange(n))
    before = representation(graph)
    with torch.inference_mode():
        output = model(graph, case['x'])
        restored = restore(output, case['inverse'])
        torch.testing.assert_close(restored, reference, rtol=RTOL, atol=ATOL)
        assert torch.isfinite(restored).all().item()
    after = representation(graph)
    allowed = sorted(FORMATS[case['format']])
    assert after['allowed_formats'] == allowed
    assert sorted(after['formats']['created']) == allowed and not after['formats']['not created']
    signature_data = {'representations':after, 'features':digest(case['x']),
                      'node_ids':digest(ids), 'edge_ids':digest(eids)}
    signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
    return {'topology_and_eids':True, 'feature_and_node_maps':True, 'numerical_equivalence':True,
            'max_absolute_error':(restored-reference).abs().max().item(), 'rtol':RTOL, 'atol':ATOL,
            'representations_before_forward':before, 'representations_at_execution':after,
            'signature':signature, 'input_hashes':signature_data,
            'restoration':'identity/no-copy' if case['inverse'] is None else 'index_select/inverse_nid'}


def pilot(seed=0):
    torch.manual_seed(seed)
    graph = dgl.graph(([3,0,2,1,0,2,2], [0,2,0,3,1,3,0]), num_nodes=6)
    graph = dgl.add_self_loop(dgl.remove_self_loop(graph))
    graph.create_formats_()
    x = torch.randn(6, 11)
    model = model_for(11, seed)
    with torch.inference_mode():
        reference = model(graph, x)
    records=[]
    for count in (1,2,3):
        for formats in itertools.combinations(['coo','csr','csc'], count):
            changed = graph.formats(list(formats))
            changed.create_formats_()
            record={'policy':'+'.join(formats), 'allowed':list(formats),
                    'before':representation(changed), 'status':'pending'}
            try:
                with torch.inference_mode():
                    output=model(changed, x)
                    torch.testing.assert_close(output, reference, rtol=RTOL, atol=ATOL)
                record.update(status='supported',numerical_equivalence=True,
                              max_absolute_error=(output-reference).abs().max().item())
            except (dgl.DGLError, AssertionError) as exc:
                record.update(status='unsupported' if isinstance(exc,dgl.DGLError) else 'incorrect',
                              error_type=type(exc).__name__,error=str(exc),numerical_equivalence=False)
            record['after']=representation(changed)
            assert record['after']['allowed_formats']==sorted(formats)
            records.append(record)
    return {'kind':'synthetic compatibility pilot, NOT dataset performance',
            'model':{'input_features':11,'heads':4,'out_features':8,'seed':seed},
            'environment':metadata(), 'policies':records}


def valid_formats(pilot_result):
    supported={tuple(sorted(r['allowed'])) for r in pilot_result['policies'] if r['status']=='supported'}
    return [name for name,formats in FORMATS.items() if tuple(sorted(formats)) in supported]


def configurations(formats):
    return list(itertools.product(NODES, EDGES, formats))


def timed_block(case, model, warmup, repetitions):
    graph,x,inverse=case['graph'],case['x'],case['inverse']
    forwards=[];restorations=[];combined=[]
    with torch.inference_mode():
        for _ in range(warmup):
            restore(model(graph,x),inverse)
        for _ in range(repetitions):
            start=perf_counter_ns()
            output=model(graph,x)
            middle=perf_counter_ns()
            restored=restore(output,inverse)
            end=perf_counter_ns()
            forwards.append((middle-start)/1e6)
            restorations.append((end-middle)/1e6)
            combined.append((end-start)/1e6)
        assert torch.isfinite(restored).all().item()
    return {'forward_ms':stats(forwards),'restoration_ms':stats(restorations),'combined_ms':stats(combined)}


def provenance():
    result=metadata()
    result['source_sha256']={str(p.relative_to(HERE.parent.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
        for directory in [HERE, HERE.parent/'dgl_baseline'] for p in sorted(directory.glob('*.py'))}
    result['lock_sha256']=hashlib.sha256((HERE.parent.parent/'uv.lock').read_bytes()).hexdigest()
    result['cgroup']={name:(Path('/sys/fs/cgroup')/name).read_text().strip()
        for name in ['cpu.max','memory.max'] if (Path('/sys/fs/cgroup')/name).exists()}
    return result
