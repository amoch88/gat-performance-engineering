"""Shared full-forward measurement protocol. No stage/kernel timing or tuning."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
from time import perf_counter_ns

os.environ['DGLBACKEND'] = 'pytorch'
# Support independently running a file or importing through benchmarks.*.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import dgl
import numpy as np
import psutil
import requests
import torch
from dgl.nn.pytorch import GATConv as OfficialGATConv
from baseline import GATConv
from gat import metadata, smoke_test

DATASETS = {
    'cora': (dgl.data.CoraGraphDataset, 'cora_v2'),
    'citeseer': (dgl.data.CiteseerGraphDataset, 'citeseer'),
    'pubmed': (dgl.data.PubmedGraphDataset, 'pubmed'),
    'reddit': (dgl.data.RedditDataset, 'reddit'),
}
RTOL, ATOL = 1e-4, 1e-5


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def digest(tensor):
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()


def available_memory():
    available = psutil.virtual_memory().available
    try:
        limit = Path('/sys/fs/cgroup/memory.max').read_text().strip()
        used = int(Path('/sys/fs/cgroup/memory.current').read_text())
        # Dataset file-cache pages are reclaimable; don't treat inactive file
        # cache as anonymous working memory (Docker working-set convention).
        stats = dict(line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
        reclaimable = int(stats.get('inactive_file', 0))
        if limit != 'max':
            available = min(available, max(0, int(limit) - used + reclaimable))
    except (OSError, ValueError):
        pass
    return available


def reddit_memory_estimate(heads=4, input_features=602, out_features=8):
    # Conservative full-graph baseline: simultaneous graph copies/formats,
    # attention/logit temporaries (4 heads), model outputs and reference output.
    # Not a measured peak or an assertion that a smaller configuration cannot fit.
    n, e, f, h, o = 232965, 114615892, input_features, heads, out_features
    return e * (8 * 16 + 4 * h * 8) + n * (f * 4 * 4 + h * o * 4 * 4)


def reddit_configurations(name):
    """Existing sweep dimensions only; used before loading any full graph."""
    labels = {
        'baseline': ['original'], 'input_features': [16, 32, 64, 128, 602],
        'output_features': [8, 16, 32, 64], 'attention_heads': [1, 2, 4, 8],
        'sparse_format': ['coo', 'csr', 'csc'], 'edge_ordering': ['original', 'src', 'dst'],
        'node_ordering': ['original', 'rcmk', 'degree'],
    }[name]
    return [(str(value), {
        'heads': value if name == 'attention_heads' else 4,
        'input_features': value if name == 'input_features' else 602,
        'out_features': value if name == 'output_features' else 8,
    }) for value in labels]


def memory_skip(label, config, budget):
    return {'case': label, 'status': 'skipped_memory', 'config': config,
            'estimated_required_bytes': reddit_memory_estimate(**config),
            'memory_budget_bytes': budget,
            'reason': 'Conservative configuration working set exceeds 70% of pre-load available memory'}


def load_dataset(name, data_dir):
    cls, archive = DATASETS[name]
    # Fail fast on access denial rather than DGL's repeated failed downloads.
    # Only probe when neither raw extracted data nor the archive is present.
    if not any(data_dir.glob(archive + '_*')) and not (data_dir / (archive + '.zip')).exists():
        response = requests.head(f'https://data.dgl.ai/dataset/{archive}.zip', timeout=20)
        response.raise_for_status()
    kwargs = dict(raw_dir=str(data_dir), verbose=False)
    if name != 'reddit':
        # Official citation loaders otherwise default to an RCM reordering.
        # Explicitly disable it so "original" means loader output without RCM.
        kwargs['reorder'] = False
    start = perf_counter_ns()
    graph = cls(**kwargs)[0].to('cpu')
    load_ms = (perf_counter_ns() - start) / 1e6
    info = {'nodes': graph.num_nodes(), 'dataset_edges': graph.num_edges(),
            'original_features': graph.ndata['feat'].shape[1],
            'source': f'DGL 1.1.3 {cls.__name__}',
            'loading_ms': load_ms, 'loader_reorder': False if name != 'reddit' else None}
    start = perf_counter_ns()
    graph = dgl.add_self_loop(dgl.remove_self_loop(graph))
    features = graph.ndata.pop('feat').to(dtype=torch.float32).contiguous()
    # Labels and train/test masks are irrelevant to inference; avoid copying them
    # during reordering. Do not otherwise alter the topology or edge order.
    for key in list(graph.ndata):
        graph.ndata.pop(key)
    for key in list(graph.edata):
        graph.edata.pop(key)
    graph.ndata['feat'] = features
    graph.create_formats_()
    info.update(edges_with_self_loops=graph.num_edges(),
                self_loop_policy='remove all loops; append exactly one per node',
                preprocessing_ms=(perf_counter_ns() - start) / 1e6,
                feature_sha256=digest(features))
    return graph, features, info


def resize_features(features, width):
    """Deterministic prefix truncation; zero-pad beyond original width."""
    if width <= features.shape[1]:
        return features[:, :width].contiguous()
    return torch.nn.functional.pad(features, (0, width - features.shape[1])).contiguous()


def representation(graph):
    formats = graph.formats()
    result = {'formats': formats, 'allowed_formats': sorted(formats['created'] + formats['not created']), 'sparse_hashes': {}}
    for fmt in formats['created']:
        arrays = graph.adj_tensors(fmt)
        # Storage/traversal index hashes distinct from original EID maps.
        result['sparse_hashes'][fmt] = {
            'indices': [digest(a) for a in arrays[:2]],
            'eid_map': digest(arrays[2]) if len(arrays) > 2 else None}
    return result


def seeded_model(width, out_features, heads, seed, official=False):
    torch.manual_seed(seed)
    cls = OfficialGATConv if official else GATConv
    return cls(width, out_features, heads, feat_drop=0, attn_drop=0,
               residual=False, bias=True, activation=None).float().cpu().eval()


def verify_source():
    """Executable class AST must match installed code, including all branches."""
    import ast
    import inspect
    def cls_ast(cls):
        return ast.dump(ast.parse(inspect.getsource(cls)), include_attributes=False)
    if dgl.__version__ != '1.1.3':
        raise RuntimeError('This reference is specifically DGL 1.1.3')
    assert cls_ast(GATConv) == cls_ast(OfficialGATConv), 'Annotated source changed execution'


def measure_case(label, overrides, graph, features, restore_ids, canonical_graph,
                 original_features, defaults, args):
    width = features.shape[1]
    config = dict(defaults, **overrides, input_features=width)
    config['total_output_features'] = config['heads'] * config['out_features']
    start = perf_counter_ns()
    model = seeded_model(width, config['out_features'], config['heads'], args.seed)
    official = seeded_model(width, config['out_features'], config['heads'], args.seed, official=True)
    official.load_state_dict(model.state_dict())
    model_ms = (perf_counter_ns() - start) / 1e6
    before = representation(graph)
    with torch.inference_mode():
        # Same-graph, same-weights check: annotated copy vs untouched official.
        output = model(graph, features)
        untouched = official(graph, features)
        torch.testing.assert_close(output, untouched, rtol=RTOL, atol=ATOL)
        reference = official(canonical_graph, resize_features(original_features, width))
        restored = torch.empty_like(output) if restore_ids is not None else output
        if restore_ids is not None:
            restored[restore_ids] = output
        torch.testing.assert_close(restored, reference, rtol=RTOL, atol=ATOL)
        assert output.shape == (graph.num_nodes(), config['heads'], config['out_features'])
        assert output.device.type == 'cpu' and output.dtype == torch.float32
        assert torch.isfinite(output).all().item()
        max_abs_error = (restored - reference).abs().max().item()
        # Validation above may reveal operator-induced format creation. Record it
        # before measuring so conversions are never hidden in timed iterations.
        after_validation = representation(graph)
        for _ in range(args.warmup):
            model(graph, features)
        samples = []
        for _ in range(args.repetitions):
            start = perf_counter_ns()
            output = model(graph, features)
            samples.append((perf_counter_ns() - start) / 1e6)
    after = representation(graph)
    assert after == after_validation, 'Representation changed during timing'
    return {'case': label, 'status': 'passed', 'config': config,
            'model_initialization_ms': model_ms, 'feature_sha256': digest(features),
            'state_dict_sha256': {key: digest(value) for key, value in model.state_dict().items()},
            'representation_pre_validation': before, 'representation_before_timing': after_validation,
            'representation_after_timing': after,
            'node_permutation_sha256': digest(restore_ids) if restore_ids is not None else None,
            'correctness': {'annotated_vs_official': 'passed', 'canonical_graph': 'passed',
                            'rtol': RTOL, 'atol': ATOL, 'max_absolute_error': max_abs_error},
            'runtime_ms': {'samples': samples, 'median': statistics.median(samples),
                           'mean': statistics.mean(samples), 'stddev': statistics.stdev(samples),
                           'min': min(samples)}}


def run_experiment(name, cases):
    parser = argparse.ArgumentParser(description=f'CPU stock DGL {name} experiment')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=['cora','citeseer','pubmed'])
    parser.add_argument('--data-dir', type=Path, default=Path('results/datasets'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/dgl_baseline'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repetitions', type=int, default=20)
    parser.add_argument('--verify-only', action='store_true', help='Source and synthetic correctness checks only; no dataset claims')
    args = parser.parse_args()
    if args.threads < 1 or args.warmup < 1 or args.repetitions < 2 or not 0 <= args.seed < 2**31:
        parser.error('Require threads/warmup >=1, repetitions >=2, seed in [0,2**31)')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(args.seed); np.random.seed(args.seed); dgl.seed(args.seed)
    if torch.version.cuda is not None:
        raise RuntimeError('CPU-only PyTorch required')
    verify_source()
    smoke_test()
    if args.verify_only:
        from tests import verify_experiment
        outcomes = verify_experiment(name, cases, args)
        write_json(args.output_dir / f'verification-{name}.json', {
            'kind': 'synthetic_correctness_only', 'experiment': name,
            'environment': metadata(), 'source_ast_matches_installed': True,
            'outcomes': outcomes, 'residual_bias_activation_attention_checks': 'passed',
            'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}})
        print(f'{name}: source and synthetic checks completed; see per-case outcomes above')
        return
    environment = metadata()
    environment.update(memory_available_bytes=available_memory(),
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')})
    for fname in ('pyproject.toml', 'uv.lock'):
        environment[fname + '_sha256'] = hashlib.sha256((HERE.parents[1] / fname).read_bytes()).hexdigest()
    defaults = {'heads': 4, 'out_features': 8}
    failed = False
    for dataset in dict.fromkeys(args.datasets):
        result = {'run_id': os.environ.get('DGL_EXPERIMENT_RUN_ID'), 'schema_version': 2, 'experiment': name, 'dataset': dataset,
                  'environment': environment, 'seed': args.seed, 'warmup': args.warmup,
                  'repetitions': args.repetitions, 'device': 'cpu', 'dtype': 'float32',
                  'timing': 'perf_counter_ns; full forward only; no preprocessing',
                  'argv': sys.argv[1:], 'cases': [], 'status': 'pending'}
        path = args.output_dir / dataset / f'{name}.json'
        memory_budget = available_memory() * 0.7
        if dataset == 'reddit':
            specs = reddit_configurations(name)
            result['memory_budget_bytes'] = memory_budget
            result['memory_policy'] = 'Conservative total working set; 70% of pre-load available host/cgroup memory'
            if all(reddit_memory_estimate(**config) > memory_budget for _, config in specs):
                result.update(status='skipped_memory',
                              reason='Every requested configuration exceeds the memory safety budget',
                              estimated_required_bytes=reddit_memory_estimate(),
                              documented_dataset={'nodes':232965, 'edges':114615892, 'features':602,
                                                  'source':'DGL 1.1.3 documentation; observed statistics are in measured Reddit records'},
                              not_run_cases='all requested cases in this experiment',
                              cases=[memory_skip(label, config, memory_budget) for label, config in specs])
                write_json(path, result)
                print(f'{dataset}/{name}: skipped_memory', flush=True)
                continue
        try:
            graph, features, info = load_dataset(dataset, args.data_dir)
            result['dataset_info'] = info
        except (requests.RequestException, OSError) as exc:
            result.update(status='blocked_dataset', reason=f'{type(exc).__name__}: {exc}',
                          not_run_cases='all cases in this experiment')
            write_json(path, result)
            print(f'{dataset}/{name}: blocked_dataset: {exc}', flush=True)
            failed = True
            continue
        # The node-order sweep uses source-grouped edges in EVERY case, including
        # its own original-node baseline, to hold edge-ordering policy constant.
        canonical = graph
        if name == 'node_ordering':
            canonical = dgl.reorder_graph(graph, node_permute_algo=None, edge_permute_algo='src')
            canonical.create_formats_()
        result['baseline_policy'] = 'original node IDs; src-grouped edges' if name == 'node_ordering' else 'original loader edge order after self-loop policy; all formats'
        if dataset == 'reddit' and reddit_memory_estimate() > memory_budget:
            baseline = memory_skip('baseline', {'heads':4, 'input_features':features.shape[1], 'out_features':8}, memory_budget)
        else:
            start = perf_counter_ns()
            baseline = measure_case('baseline', {}, canonical, features, None, canonical, features, defaults, args)
            baseline['setup_validation_warmup_ms'] = (perf_counter_ns() - start) / 1e6 - sum(baseline['runtime_ms']['samples'])
        result['baseline'] = baseline
        write_json(path, result)
        iterator = iter(cases(graph, features, defaults))
        while True:
            start = perf_counter_ns()
            try:
                label, overrides, changed_graph, changed_features, restore_ids = next(iterator)
            except StopIteration:
                break
            prep_ms = (perf_counter_ns() - start) / 1e6
            config = dict(defaults, **overrides, input_features=changed_features.shape[1])
            if dataset == 'reddit' and reddit_memory_estimate(**config) > memory_budget:
                result['cases'].append(memory_skip(label, config, memory_budget))
                write_json(path, result)
                print(f'{dataset}/{name}/{label}: skipped_memory', flush=True)
                continue
            try:
                record = measure_case(label, overrides, changed_graph, changed_features, restore_ids,
                                      canonical, features, defaults, args)
            except (dgl.DGLError, AssertionError) as exc:
                # An official single-format restriction can be unsupported by
                # an operator. Keep this as an explicit failure, never a speedup.
                record = {'case':label, 'status':'unsupported_format' if name == 'sparse_format' and isinstance(exc, dgl.DGLError) and any(text in str(exc) for text in ('restricted sparse format', 'In degree cannot', 'SDDMM only supports')) else 'failed',
                          'reason':str(exc), 'representation':representation(changed_graph)}
                if record['status'] == 'failed': failed = True
            if record['status'] == 'passed' and baseline['status'] == 'passed':
                base_hashes = baseline['representation_before_timing']['sparse_hashes']
                case_hashes = record['representation_before_timing']['sparse_hashes']
                record['storage_distinct_from_baseline'] = {
                    fmt: {'indices': base_hashes[fmt]['indices'] != case_hashes[fmt]['indices'],
                          'eid_map': base_hashes[fmt]['eid_map'] != case_hashes[fmt]['eid_map']}
                    for fmt in base_hashes.keys() & case_hashes.keys()}
            record['case_preprocessing_ms'] = prep_ms
            result['cases'].append(record)
            write_json(path, result)
            print(f'{dataset}/{name}/{label}: {record["status"]}', flush=True)
        statuses = {case['status'] for case in result['cases']}
        result['status'] = ('completed_with_failures' if 'failed' in statuses else
                            'completed_with_unsupported_cases' if 'unsupported_format' in statuses else
                            'completed_with_skips' if 'skipped_memory' in statuses else 'completed')
        write_json(path, result)
    if failed:
        raise SystemExit(1)
