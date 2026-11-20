"""Dependency-free correctness checks; synthetic inputs are NEVER dataset results."""
import ast
import inspect
import torch
from common import (dgl, GATConv, OfficialGATConv, measure_case, verify_source,
                    representation, RTOL, ATOL)


def verify_experiment(name, cases, args):
    verify_source()
    torch.manual_seed(42)
    # Deliberately unsorted edges, heterogeneous degrees, one isolated node before loops.
    graph = dgl.graph(([2, 0, 3, 1, 0, 2, 3], [0, 2, 1, 0, 1, 3, 2]), num_nodes=5)
    graph = dgl.add_self_loop(dgl.remove_self_loop(graph))
    features = torch.randn(5, 11)
    graph.ndata['feat'] = features
    graph.create_formats_()
    defaults = {'heads':4, 'out_features':8}
    canonical = graph
    if name == 'node_ordering':
        canonical = dgl.reorder_graph(graph, edge_permute_algo='src')
        canonical.create_formats_()
    outcomes = []
    for label, overrides, changed, x, ids in cases(graph, features, defaults):
        try:
            record = measure_case(label, overrides, changed, x, ids, canonical, features, defaults, args)
            assert len(record['runtime_ms']['samples']) == args.repetitions
            assert min(record['runtime_ms']['samples']) > 0
            if ids is not None:
                u, v = changed.edges()
                u, v = ids[u], ids[v]
            else:
                u, v = changed.edges()
            a, b = graph.edges()
            assert torch.equal(torch.sort(u * 5 + v).values, torch.sort(a * 5 + b).values)
            outcomes.append(label + ':passed')
        except dgl.DGLError as exc:
            if name != 'sparse_format':
                raise
            outcomes.append(label + ':unsupported:' + str(exc).splitlines()[0])
    assert outcomes, 'No experiment cases tested'
    print('Synthetic verification:', outcomes)
    # Cover the preserved non-default branches with official/annotated pairs.
    for residual in [False, True]:
        for bias in [False, True]:
            for width in [8, 11]:  # Identity and linear residual branches.
                x = features[:, :width] if width == 8 else features
                model = GATConv(width, 4, 2, residual=residual, bias=bias,
                                activation=torch.relu).eval()
                official = OfficialGATConv(width, 4, 2, residual=residual, bias=bias,
                                          activation=torch.relu).eval()
                official.load_state_dict(model.state_dict())
                with torch.inference_mode():
                    out, attn = model(graph, x, get_attention=True)
                    ref, weights = official(graph, x, get_attention=True)
                    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)
                    torch.testing.assert_close(attn, weights, rtol=RTOL, atol=ATOL)

    return outcomes


def verify_pipeline():
    """Test JSON and report generation with an explicitly labeled temporary fixture."""
    from argparse import Namespace
    import json
    import statistics
    import sys
    import tempfile
    from pathlib import Path
    from unittest.mock import patch
    import common
    from report import render

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        graph = dgl.add_self_loop(dgl.graph(([0,2,1], [1,0,2]), num_nodes=3))
        x = torch.tensor([[1.,2.,3.],[4.,5.,6.],[2.,1.,3.]])
        graph.ndata['feat'] = x
        graph.create_formats_()
        info = {'nodes':3,'dataset_edges':3,'edges_with_self_loops':6,'original_features':3}
        argv=['input_features.py','--datasets','synthetic-test-fixture','--output-dir',str(root),
              '--warmup','1','--repetitions','2']
        from input_features import cases
        with patch.dict(common.DATASETS, {'synthetic-test-fixture':(None,None)}), \
             patch.object(common,'load_dataset',return_value=(graph,x,info)), \
             patch.object(sys,'argv',argv):
            common.run_experiment('input_features',cases)
        record=json.loads((root/'synthetic-test-fixture/input_features.json').read_text())
        assert record['status']=='completed' and len(record['cases'])==5
        assert record['baseline']['correctness']['annotated_vs_official']=='passed'
        for case in record['cases']:
            samples=case['runtime_ms']['samples']
            assert len(samples)==2
            assert case['runtime_ms']['median']==statistics.median(samples)
            assert case['runtime_ms']['mean']==statistics.mean(samples)
            assert case['runtime_ms']['stddev']==statistics.stdev(samples)
        render(root,root/'TEST-REPORT.md')
        text=(root/'TEST-REPORT.md').read_text()
        assert 'synthetic-test-fixture' in text and 'different workloads' in text
        # The fixture is temporary and never contributes to the research report.
    print('Temporary synthetic fixture: end-to-end JSON/statistics/report checks passed')


def verify_skipped_baseline():
    """Regression: a safe case must serialize when its default baseline is skipped."""
    import json
    import sys
    import tempfile
    from pathlib import Path
    from unittest.mock import patch
    import common
    from attention_heads import cases
    with tempfile.TemporaryDirectory() as temporary:
        graph = dgl.add_self_loop(dgl.graph(([0,1,2],[1,2,0]), num_nodes=3))
        x = torch.randn(3,602)
        graph.ndata['feat'] = x
        graph.create_formats_()
        info = {'nodes':3,'dataset_edges':3,'edges_with_self_loops':6,'original_features':602,
                'source':'SYNTHETIC regression fixture; never a real Reddit measurement'}
        argv = ['attention_heads.py','--datasets','reddit','--output-dir',temporary,
                '--warmup','1','--repetitions','2']
        print('Temporary SYNTHETIC fixture testing Reddit memory-control serialization')
        with patch.object(common,'load_dataset',return_value=(graph,x,info)), \
             patch.object(common,'available_memory',return_value=32*2**30), \
             patch.object(torch,'set_num_interop_threads'), \
             patch.object(sys,'argv',argv):
            common.run_experiment('attention_heads',cases)
        record=json.loads((Path(temporary)/'reddit/attention_heads.json').read_text())
        assert record['baseline']['status']=='skipped_memory'
        assert [c['status'] for c in record['cases']]==['passed']+['skipped_memory']*3
        assert len(record['cases'][0]['runtime_ms']['samples'])==2
        assert record['status']=='completed_with_skips'
    print('Skipped-baseline serialization and per-configuration memory regression passed')


if __name__ == '__main__':
    verify_pipeline()
    verify_skipped_baseline()
