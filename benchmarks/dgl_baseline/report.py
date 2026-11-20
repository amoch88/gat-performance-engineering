"""Render only observed benchmark records; never fabricate missing measurements."""
import argparse
import json
from pathlib import Path

SWEEPS = ['input_features', 'output_features', 'attention_heads', 'sparse_format',
          'edge_ordering', 'node_ordering']
OBJECTIVES = {
    'input_features': 'Input widths 16, 32, 64, 128 and original; prefix-truncate or zero-pad original features before timing.',
    'output_features': 'Output features per head 8, 16, 32, 64; four heads fixed.',
    'attention_heads': 'Heads 1, 2, 4, 8; eight output features per head fixed. Total output dimension changes.',
    'sparse_format': 'Restrict allowed storage to COO, CSR or CSC; materialize before timing and record operator compatibility.',
    'edge_ordering': 'Original loader edge-ID order, source-grouped and destination-grouped; original node IDs retained.',
    'node_ordering': 'Original node IDs, official RCM, ascending total degree with node-ID ties; source-grouped edges throughout.',
}


def table(headers, rows):
    def clean(value):
        return str(value).replace('|','/').replace('\n',' ')
    return ['| ' + ' | '.join(headers) + ' |',
            '| ' + ' | '.join(['---'] * len(headers)) + ' |'] + [
            '| ' + ' | '.join(map(clean,row)) + ' |' for row in rows] + ['']


def render(root, output):
    paths = sorted(root.glob('*/*.json'))
    records = [json.loads(path.read_text()) for path in paths]
    datasets = sorted({r['dataset'] for r in records})
    verification = [json.loads(p.read_text()) for p in sorted(root.glob('verification-*.json'))]
    access_path = root.parent / 'dataset-access.json'
    access = json.loads(access_path.read_text()) if access_path.exists() else []
    bundle = output.with_name('MEASUREMENTS.json')
    bundle.parent.mkdir(parents=True, exist_ok=True)
    bundle.write_text(json.dumps({'schema_version': 1, 'dataset_access': access,
                                 'measurements': records, 'verification': verification},
                                indent=2, allow_nan=False) + '\n')
    passed = sum(case.get('status') == 'passed' for r in records for case in r.get('cases', []) + ([r['baseline']] if 'baseline' in r else []))
    lines = ['# DGL GAT CPU baseline parameter experiments', '',
        '## Objective and scope', '',
        f'This report contains **{passed} measured real-dataset case/control records**. Read skipped and unsupported entries separately: they contain no performance measurements.', '',
        'Measure how public GATConv constructor arguments and public graph representation APIs affect full-forward CPU latency. '
        'No prediction, fusion, custom attention algorithm, optimized kernel or training is introduced. '
        'Changed dimensions are different workloads; they are not equivalent-computation speedups.', '',
        'All raw samples and configuration/representation/provenance records are retained in [MEASUREMENTS.json](MEASUREMENTS.json), alongside this report for review.', '',
        '## Implementation reference and environment', '',
        'DGL **1.1.3**, [official annotated source reference](https://www.dgl.ai/dgl_docs/en/1.1.x/_modules/dgl/nn/pytorch/conv/gatconv.html). '
        '`baseline.py` copies the full installed implementation, cross-checked byte-for-byte against the v1.1.3 upstream source before annotation. '
        'The executable class AST is checked against the untouched installed class. Imports/backend/CLI and comments are the only adaptations. '
        'The Apache-2.0 license is retained in `DGL-LICENSE`. The five labeled logical stages are not a claim about five compiled CPU kernels.', '']
    if records:
        env = records[0]['environment']
        lines += table(['Field','Observed value'], [[k,env.get(k)] for k in
            ['cpu_model','logical_cpus','cpu_affinity','os','architecture','python','dgl','torch','numpy',
             'torch_threads','torch_interop_threads','thread_environment','memory_available_bytes','docker_image','git_commit','git_dirty']])
        lines += ['Raw JSON includes per-run environment and source/lockfile SHA-256 hashes. '
                  'An unset Docker identity means these records came from the native locked environment, not a claimed container measurement. '
                  'The repository still has no commit when `git_commit` is null; source hashes identify the measured files.', '']
    settings = sorted({(r.get('seed'), r.get('warmup'), r.get('repetitions')) for r in records})
    lines += [f'Observed (seed, warmups, timed repetitions): **{settings}**. Fixed CPU/OpenMP/MKL/OpenBLAS settings are recorded above.', '']
    lines += ['## Datasets', '',
        'Official DGL dataset classes are used with real topology and features. Citation loader `reorder=False` explicitly disables its default RCM preprocessing. '
        '“Original edge order” means the official loader’s resulting edge-ID order, not an asserted order from the original paper. '
        'Remove all self-loops, then append exactly one per node; preserve non-loop edges and retain real node features. '
        'Loading, feature transforms, permutations and sparse materialization are outside timing.', '']
    rows=[]
    for dataset in datasets:
        group=[r for r in records if r['dataset']==dataset]
        observed=next((r.get('dataset_info') for r in group if r.get('dataset_info')),None)
        if observed:
            rows.append([dataset,observed['nodes'],observed['dataset_edges'],observed['edges_with_self_loops'],observed['original_features'],'observed'])
        else:
            doc=group[0].get('documented_dataset',{})
            rows.append([dataset,doc.get('nodes','not loaded'),doc.get('edges','not loaded'),'not measured',doc.get('features','not loaded'),group[0]['status'] + ('; counts are documentation only' if doc else '')])
    lines += table(['Dataset','Nodes','Dataset edges','Edges after loops','Original width','Evidence'],rows)
    if access:
        lines += table(['Dataset archive','HTTP status','Redirects','Official loader'], [
            [r['url'], r.get('status', 'error'), r.get('redirects', []), r.get('loader_status', 'not verified')]
            for r in access])
    lines += ['## Standalone baseline controls', '',
              'These controls use original dataset features and node/edge order after self-loop preprocessing; four heads, eight output features per head. Each sweep also measures its own in-process control for comparisons.', '']
    controls=[]
    for r in records:
        if r['experiment'] != 'baseline':
            continue
        timing = r.get('baseline', {}).get('runtime_ms')
        controls.append([r['dataset'], *([f'{timing[k]:.6f}' for k in ('median','mean','stddev')] if timing else ['—']*3),
                         'measured' if timing else r['status']])
    lines += table(['Dataset','Median ms','Mean ms','Sample SD ms','Status'],controls)

    lines += ['## Default configuration and measurement protocol', '',
        'One stock GATConv: dataset-original input width, **4 heads × 8 output features/head**, float32, CPU, bias on, residual off, activation none, '
        'zero feature/attention dropout, eval/inference mode. All three formats are allowed/materialized in the ordinary baseline. '
        'Each sweep measures its own matching baseline in the same process. Node-ordering uses source-grouped edges for its baseline and every case. '
        'Seeded model construction is reset for every case; identical shapes have identical weight hashes. Changed shapes use the same seed but necessarily different weights/shapes.', '',
        'Each record stores seed, warmups and repetitions. `perf_counter_ns` measures synchronous full forwards. '
        'Raw samples, median, mean, sample standard deviation and minimum are saved. Preprocessing/model initialization and setup/check/warmup costs are separate. '
        'The representation is checked before validation, before timing and after timing, including compressed index hashes and EID maps. '
        'No per-stage timing or compiled-kernel selection claim is made. Numerical comparisons use rtol=1e-4, atol=1e-5 and restore original node IDs.', '']
    for sweep in SWEEPS:
        lines += [f'## {sweep.replace("_"," ").title()}', '',OBJECTIVES[sweep],'']
        subset=[r for r in records if r['experiment']==sweep]
        if not subset:
            lines += ['Not run: no records supplied.','']
            continue
        rows=[]
        for record in subset:
            dataset=record['dataset']
            if 'baseline' not in record:
                for case in record.get('cases') or [{'case':'all requested cases','status':record['status']}]:
                    rows.append([dataset,case['case'],'—','—','—','—',case['status']])
                continue
            base=record['baseline'].get('runtime_ms',{}).get('median')
            for case in [record['baseline']]+record['cases']:
                if case['status']!='passed':
                    rows.append([dataset,case['case'],'—','—','—','—',case['status']])
                    lines += [f"- {dataset}/{case['case']}: {case.get('reason','').splitlines()[0]}"]
                    continue
                timing=case['runtime_ms'];c=case['config']
                compare = 'different workloads'
                if sweep in ['sparse_format','edge_ordering','node_ordering']:
                    if base is None:
                        compare = 'baseline not measured'
                    elif case['case'] == 'baseline':
                        compare = '1.000× (control)'
                    else:
                        distinct = case['representation_before_timing'] != record['baseline']['representation_before_timing']
                        compare = f'{base/timing["median"]:.3f}×' + ('' if distinct else ' (repeat control; same storage)')
                rows.append([dataset,f"{case['case']}: in={c['input_features']}, heads={c['heads']}, out/head={c['out_features']}",
                    f'{timing["median"]:.6f}',f'{timing["mean"]:.6f}',f'{timing["stddev"]:.6f}',compare,'passed'])
            if sweep=='edge_ordering':
                for case in record['cases']:
                    if case['status']=='passed':
                        before=record['baseline']['representation_before_timing']['sparse_hashes']
                        after=case['representation_before_timing']['sparse_hashes']
                        for fmt in sorted(before.keys() & after.keys()):
                            lines += [f"- {dataset}/{case['case']}/{fmt}: traversal indices distinct from baseline = **{before[fmt]['indices'] != after[fmt]['indices']}**; EID map distinct = **{before[fmt]['eid_map'] != after[fmt]['eid_map']}**."]
            if sweep in SWEEPS:
                measured=[c for c in record['cases'] if c['status']=='passed']
                if measured:
                    fastest=min(measured,key=lambda c:c['runtime_ms']['median'])
                    slowest=max(measured,key=lambda c:c['runtime_ms']['median'])
                    lines += [f"- {dataset}: among measured cases, {fastest['case']} has the smallest median ({fastest['runtime_ms']['median']:.6f} ms) and {slowest['case']} the largest ({slowest['runtime_ms']['median']:.6f} ms). This is descriptive, not proof of monotonic scaling or a causal optimization benefit."]
                    cvs = [case['runtime_ms']['stddev']/case['runtime_ms']['mean']*100 for case in measured]
                    lines += [f"- {dataset}: sample coefficient of variation spans {min(cvs):.1f}%–{max(cvs):.1f}%. Ratios are exploratory; this is one process trial per sweep."]
        lines += ['']+table(['Dataset','Configuration','Median ms','Mean ms','Sample SD ms','Baseline / case','Status'],rows)
        if all('baseline' not in r for r in subset):
            lines += ['No real-dataset timings exist for this sweep, so runtime trends and speedups cannot be concluded.','']
    lines += ['## Correctness, blocked work and limitations','',
        'Every measured case must pass annotated-versus-official output checks with identical weights and an official canonical-graph check. '
        'Node-order experiments restore output using new-to-old DGL NID mappings and verify feature association. '
        'Independent `--verify-only` checks use explicitly synthetic test graphs; their timing samples are discarded and are never reported as dataset performance.', '',
        'Single-format restrictions can be incompatible with different operators in the complete stock forward. '
        'No zero-degree guard or format restriction is disabled to make a case run. '
        'A format restriction never establishes that every operator would use that representation.', '']
    if verification:
        lines += table(['Synthetic-only verification', 'Source AST', 'Case outcomes'], [
            [v['experiment'], v['source_ast_matches_installed'], '; '.join(v['outcomes'])] for v in verification])
        lines += ['The per-script verification also checks residual (identity and linear), bias, activation and returned attention against the untouched official class. '
                  'Unsupported format cases have no timing or claimed output equivalence. These test-graph outcomes do not replace real-dataset verification.', '']
    else:
        lines += ['No synthetic verification artifact was supplied to this renderer.', '']
    reasons=set()
    for record in records:
        if record.get('status') in ('blocked_dataset','skipped_memory'):
            reasons.add((record['dataset'],record['status'],record.get('reason','')))
    for dataset,status,reason in sorted(reasons):
        affected = [r['experiment'] for r in records if r['dataset']==dataset and r['status']==status and r.get('reason','')==reason]
        lines += [f'- **{dataset}: {status}** — {reason}. Affected experiments: {", ".join(affected)}. Their requested cases were not measured; other experiments may have successful measurements.']
    reddit=next((r for r in records if r.get('estimated_required_bytes')),None)
    if reddit:
        lines += ['',f"Reddit conservative baseline working-set estimate: **{reddit['estimated_required_bytes']/2**30:.2f} GiB**, compared against 70% of available host/cgroup memory. "
            'The estimate budgets simultaneous graph copies/formats, logits/attention temporaries and input/output buffers. '
            'It is a safety policy, not a measured peak. Each existing configuration is checked separately; lower-head-count cases can run while the default baseline is skipped. A missing Reddit baseline prevents a same-work speedup comparison.']
    lines += ['', 'Timing results are exploratory within one shared cloud CPU allocation, without repeated independent process trials, pinned CPU frequency or isolated hosts. '
        'No training/accuracy conclusions follow from dimension changes. Graph reordering can change reduction order, hence tolerances. '
        'Docker does not normalize physical CPU behavior. Tables retain all samples through linked raw JSON; no plots are needed to distinguish missing data.', '',
        '## Reproduction', '',
        'From the repository root, using the unchanged uv lockfile and Dockerfile:', '',
        '```sh',
        'export UV_CACHE_DIR=/workspace/.cache/uv  # writable cache in this cloud; optional elsewhere',
        'export DGLBACKEND=pytorch OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1',
        'uv sync --frozen',
        'uv run --frozen python benchmarks/dgl_baseline/run_all.py \\',
        '  --datasets cora citeseer pubmed reddit --threads 1 --seed 0 --warmup 5 --repetitions 20',
        'uv run --frozen python benchmarks/dgl_baseline/report.py',
        '```','',
        'Each script also supports `--verify-only --warmup 1 --repetitions 2`. Dataset acquisition requires HTTPS access to `data.dgl.ai`; '
        'download outcomes and redirect chains are recorded in the dataset-access table when available. '
        'JSON is under `results/dgl_baseline/<dataset>/<experiment>.json`; generated data and raw samples remain ignored by Git. '
        'The adjacent MEASUREMENTS.json bundles the raw records for GitHub review. Retain the dataset cache for reruns. Docker commands and additional methodology are in the root README.', '']
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text('\n'.join(lines))
    print(f'Wrote {output} from {len(records)} records')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=Path('results/dgl_baseline'))
    parser.add_argument('--output',type=Path,default=Path('benchmarks/dgl_baseline/REPORT.md'))
    args=parser.parse_args()
    render(args.input_dir,args.output)
