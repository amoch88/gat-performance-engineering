"""Render real joint measurements, including selection uncertainty and costs."""
import argparse
import json
from pathlib import Path
from analysis import analyze, BASE


def table(head,rows):
    clean=lambda x:str(x).replace('|','/').replace('\n',' ')
    return ['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+[
        '| '+' | '.join(clean(v) for v in row)+' |' for row in rows]+['']


def interval(summary):
    lo,hi=summary['ci95']
    return f"{summary['geomean']:.3f}× [{lo:.3f}, {hi:.3f}]"


def render(source,output):
    data=json.loads(source.read_text());analysis=analyze(data)
    (output.parent/'ANALYSIS.json').write_text(json.dumps(analysis,indent=2,allow_nan=False)+'\n')
    options=data['options'];environment=data['environment'];cut=options['discovery_rounds']
    lines=['# Joint DGL GAT CPU tuning — milestone 3','',
        '## Question and experimental design','',
        'Does jointly selecting node order, edge order and a compatible sparse policy outperform combining independently selected settings? '
        'The primary objective is the **combined interface time** returning output rows in original logical node order. '
        'Raw full-forward latency is also ranked separately. This is a measured factorial experiment; no custom kernel, fusion, layout or predictive model is introduced.','',
        f"Each dataset has **{options['rounds']} fresh-process rounds**, {options['warmup']} warmups per block, and {options['repetitions']} timed repetitions per block. "
        f"Rounds 1–{cut} select configurations; the remaining {options['rounds']-cut} rounds evaluate those choices without re-selection. "
        'All candidates are also measured in held-out rounds to provide exploratory rankings; the all-round best is explicitly separate from the discovery-selected winner.','',
        'Each round randomizes candidate order with a recorded seed. Every candidate has a warmed baseline control immediately before and after it. '
        'Relative latency divides the candidate median by the arithmetic mean of its bracketing control medians. Speedup is the reciprocal. '
        'Each control uses the same graph, features, weights and untouched official layer as the original/original/all configuration. '
        'Round-level geometric means of ratios determine rankings. Timed repetitions within a process are **not** treated as independent statistical trials.','',
        '## Environment and invariant workload','']
    headline=['## Findings for review','',
        'The exploratory winners and independently combined choices are listed below. '
        'Use the held-out intervals to assess whether different selections provide a repeatable advantage; selecting the smallest observed time alone is not enough.', '']
    headline+=table(['Dataset',f"Exploratory {options['rounds']}-round joint winner",'Mean combined ms','Independent axes (all rounds)'],[
        [name,r['rankings']['combined_ms'][0]['key'],f"{r['rankings']['combined_ms'][0]['mean_round_median_ms']:.4f}",
         r['selection']['combined_ms']['exploratory_all_rounds']['independent']]
         for name,r in analysis['datasets'].items() if r['status']=='completed'])
    headline+=['Held-out test below uses choices selected ONLY on the discovery rounds; these need not be the exploratory all-round winners.', '']
    headline+=table(['Dataset','Discovery joint choice','Discovery independent choice','Held-out joint advantage [95% CI]'],[
        [name,r['selection']['combined_ms']['discovery']['joint'],r['selection']['combined_ms']['discovery']['independent'],
         interval(r['selection']['combined_ms']['discovery']['held_out_joint_over_independent_speedup'])]
         for name,r in analysis['datasets'].items() if r['status']=='completed'])
    insert=lines.index('## Environment and invariant workload')
    lines[insert:insert]=headline
    lines+=table(['Field','Value'],[[k,environment.get(k)] for k in ['cpu_model','logical_cpus','cpu_affinity','os','python','dgl','torch','torch_threads','torch_interop_threads','thread_environment','git_commit','git_dirty','cgroup']])
    lines += ['The commit above is the base commit; new module source hashes are recorded for every worker in RESULTS.json. '
        'The original annotated baseline, previous scripts, REPORT.md and MEASUREMENTS.json remain unchanged. '
        'Runtime dependencies, Dockerfile and uv.lock are unchanged. All performance measurements use the native locked environment; separate container checks validate reproducibility, not identical physical-CPU timing.','',
        'Official dataset topology and native feature width are fixed. Citation loaders use reorder=False. Remove all self-loops, then append exactly one loop per node. '
        'Canonical EIDs refer to this preprocessed graph. Four heads, eight output features/head, float32, bias enabled, no residual/activation, dropout zero, evaluation/inference mode, seed '
        f"{options['seed']} and identical weight hashes across all cases/rounds per dataset. Sparse formats are fully materialized before validation/warmup/timing.",'',
        'An exact EID-indexed topology check maps every reordered endpoint back to the original node IDs, preserving multiplicities and loops. '
        'Feature rows, new-to-old NIDs, EIDs and requested edge grouping are checked. Original EID order after node permutation uses official reorder_graph with '
        'edge_permute_algo="custom" and an identity edge permutation; DGL explicitly undoes node-induced EID reordering in this API. '
        'Outputs are restored with a precomputed inverse NID index_select, or a no-copy identity path when nodes are unchanged. Numerical equivalence uses untouched official GATConv, rtol=1e-4/atol=1e-5.','',
        '## Sparse compatibility pilot','',
        'All seven nonempty policies were tested on a small graph with isolated nodes before self-loop preprocessing and parallel edges. This is a compatibility test, not dataset performance. '
        'Allowed/materialized states before and after the full forward and exact errors are preserved in [PILOT.json](PILOT.json) and RESULTS.json.','']
    lines+=table(['Policy','Status','Numerically equivalent','Observed failure'],[
        [r['policy'],r['status'],r['numerical_equivalence'] if r['status']!='unsupported' else 'not evaluated (no output)',r.get('error','').splitlines()[0] if r.get('error') else '—'] for r in data['pilot']['policies']])
    lines += ['CSR+CSC is excluded if the pilot fails; no extra COO permission or bypass of the stock in-degree check is introduced. '
        'Format permissions and available sparse tensors do not prove which compiled kernel or representation each operator dispatches to.','',
        '## Valid configuration matrix','',
        'Cross original/RCMK/ascending total-degree nodes, original-EID/src/dst edges and only pilot-approved policies. '
        'Degree ties use original node IDs. Exact duplicates are identified only when allowed/materialized representations (including sparse index and EID-map hashes), input features, NIDs and EIDs are all identical. '
        'These aliases reuse a representative measurement. Different permissions are retained even when shared compressed tensors match: similar timings or permissions alone do not establish ineffective kernel dispatch.','']
    matrix=[]
    for n in ['original','rcmk','degree']:
        for e in ['original','src','dst']:
            matrix.append([n,e,*['valid' if f in data['valid_formats'] else 'excluded by pilot' for f in ['all','coo+csc','csr+csc']]])
    lines+=table(['Node','Edge','All','COO+CSC','CSR+CSC'],matrix)
    lines += ['## Timing and amortization definitions','',
        'A single continuous sample records t0 → complete synchronous official forward → t1 → restore original output order → t2. '
        'Forward=t1−t0, restoration=t2−t1, combined=t2−t0. No graph setup, feature generation, serialization, correctness assertions or inverse-map construction is timed. '
        'Clock reads, Python dispatch, allocations and ordinary object lifetimes remain part of the respective interface costs. '
        'The identity restoration path includes its small Python/clock overhead and does not copy data. Every block warms both forward and restoration.','',
        'One-time preparation measures official node/edge reordering (including feature permutation), format restriction/materialization and inverse-map construction separately in every fresh process. '
        'Dataset loading/normalization and model initialization are shared setup and excluded from differential amortization. Validation/hashing is research overhead and excluded. '
        'Break-even calls = ceil(mean incremental preparation ms / mean bracketing-control combined-time saving ms), only if mean saving is positive. '
        'Positive-round count and per-round break-even values reveal unstable savings; point estimates are not guarantees.','',
        'Intervals use Student-t on the mean log ratio across independent rounds and then exponentiate. They are exploratory small-sample intervals, not multiple-comparison-adjusted significance claims. '
        'Held-out joint-vs-independent ratios use matching rounds and their local control normalization. A ratio above one favors the joint choice.','']
    for dataset,result in analysis['datasets'].items():
        lines += [f'## {dataset.title()}','']
        if result['status']!='completed':
            lines += [f"**{result['status']}**: {result['reason']}",
                f"Single-case estimate: {result['single_case_estimate_bytes']/2**30:.2f} GiB; resident joint-matrix sizing estimate: {result['joint_matrix_lower_bound_bytes']/2**30:.2f} GiB; safety budget: {result['budget_bytes']/2**30:.2f} GiB.",
                'These are conservative allocation estimates, not measured peaks; storage sharing can reduce the matrix footprint. The single-case estimate independently exceeds the safety budget. All joint configurations were skipped before graph loading. The fixed four-head workload was not replaced with a smaller-head or synthetic workload.','']
            continue
        info=result['dataset_info'];drift=result['control_drift_ratio']
        lines += [f"Nodes: {info['nodes']}; dataset edges: {info['dataset_edges']}; edges after self-loops: {info['edges_with_self_loops']}; native feature width: {info['original_features']}.",
            f"Maximum absolute reference error: {result['max_absolute_reference_error']:.3g}. Control after/before ratio: median {drift['median']:.3f}, range {drift['min']:.3f}–{drift['max']:.3f}.",'',
            '### Joint versus independent selection','',
            'Independent selection minimizes each one-factor axis while holding the other two factors at original/original/all, then combines the selected values. '
            'Joint selection searches the complete valid grid. Exact aliases are resolved before comparing choices.','']
        rows=[]
        for metric,selection in result['selection'].items():
            chosen=selection['discovery'];explore=selection['exploratory_all_rounds']
            rows.append([metric,chosen['joint'],chosen['independent'],chosen['different'],interval(chosen['held_out_joint_over_independent_speedup'])])
            lines += [f"- Exploratory all-round {metric}: fastest {explore['joint']}; independent-axis combination {explore['independent']}; different={explore['different']}. This comparison uses all rounds for selection and is not held-out confirmation."]
        lines += ['']+table(['Objective','Discovery joint choice','Discovery independent combination','Different?','Held-out joint/independent speedup [95% CI]'],rows)
        first_round=data['datasets'][dataset]['rounds'][0]
        lines += ['**Recorded representation evidence** (original node and edge order):','']
        rep_rows=[]
        full=first_round['configurations'][BASE]['representations_at_execution']
        for fmt in data['valid_formats']:
            key=f'original/original/{fmt}'
            rep=first_round['configurations'][key]['representations_at_execution']
            shared=full['sparse_hashes'].keys() & rep['sparse_hashes'].keys()
            matches={name:full['sparse_hashes'][name]==rep['sparse_hashes'][name] for name in shared}
            rep_rows.append([fmt,rep['allowed_formats'],rep['formats']['created'],matches])
        lines+=table(['Policy','Allowed','Materialized','Shared sparse tensor/EID-map hashes match all-policy'],rep_rows)
        lines+=['Shared tensors can match while permissions differ; without dispatch traces this is not proof of equal kernel execution, so both policies remain measured candidates.','']
        chosen=result['selection']['combined_ms']['discovery'];lo,hi=chosen['held_out_joint_over_independent_speedup']['ci95']
        conclusion='Selections resolve to the same configuration.' if not chosen['different'] else (
            'The held-out interval favors the jointly selected configuration.' if lo>1 else
            'The held-out interval favors the independent-axis combination.' if hi<1 else
            'The selections differ, but the held-out interval includes one; this run does not establish a reliable speed advantage.')
        lines += [conclusion,'','### Combined-interface ranking (all rounds; exploratory)','']
        ranking=result['rankings']['combined_ms']
        lines+=table(['Rank','Node / edge / policy','Forward ms','Restore ms','Combined ms','Between-round SD ms','Speedup [95% CI]'],[
            [i+1,r['key'],f"{next(f for f in result['rankings']['forward_ms'] if f['key']==r['key'])['mean_round_median_ms']:.4f}",
             f"{r['mean_restoration_ms']:.4f}",f"{r['mean_round_median_ms']:.4f}",f"{r['between_round_sd_ms']:.4f}",interval(r['speedup'])]
             for i,r in enumerate(ranking)])
        lines += ['Times above are means of round medians; ranks use geometric mean normalized ratios. Complete per-round medians, means, SDs and raw samples are in RESULTS.json. '
            'ANALYSIS.json also contains the independent forward-only ranking.','',f'![{dataset} combined ranking](figures/{dataset}-ranking.svg)','',
            '### Pairwise interactions','',
            'Each table fixes the third factor at baseline and reports I(a,b)=T(a,b)×T(0,0)/(T(a,0)×T(0,b)), using normalized combined times within each round. '
            'One means multiplicative separability; below one means the pair costs less than the product of its isolated ratios. '
            'These are descriptive interaction contrasts, not a fitted performance model or kernel-cause claim. They do not capture all three-way interactions.','']
        for interaction in result['interactions']:
            lines += [f"**{interaction['pair']}**",'']
            lines+=table(['Setting']+interaction['columns'],[
                [name]+[f"{c['geomean']:.3f}" for c in interaction['cells'][i]] for i,name in enumerate(interaction['rows'])])
        lines += [f'![{dataset} pairwise interactions](figures/{dataset}-interactions.svg)','','### One-time cost and amortization','']
        lines+=table(['Configuration','Mean preparation ms','Mean combined saving ms','Positive rounds','Point-estimate break-even calls'],[
            [r['key'],f"{r['preparation_ms']['mean']:.3f}",f"{r['amortization']['mean_combined_saving_ms']:.4f}",
             f"{r['amortization']['positive_rounds']}/{options['rounds']}",
             'control' if r['key']==BASE else r['amortization']['point_estimate_calls'] if r['amortization']['point_estimate_calls'] is not None else 'never (mean saving ≤0)'] for r in ranking])
        duplicates={k:v for k,v in result['aliases'].items() if k!=v}
        lines += ['',f'Exact representation/input duplicates: **{duplicates or "none"}**. '
            'All nonidentical policies remain candidates; a similar latency is not grounds for claiming equivalent dispatch.','']
    lines += ['## Limitations and reproduction','',
        'Only these graph datasets, one fixed model shape/seed and one shared CPU allocation are studied. Small round counts and shared-host scheduling/frequency variation limit generalization. '
        'Randomization/interleaved controls reduce but cannot eliminate drift or cache/carryover effects. Only one process measures at a time; configurations/graphs remain resident within a round. '
        'Controls use one original graph, so cache footprints need not match reordered graphs. Full-forward timings cannot attribute changes to linear projection, SDDMM, softmax or aggregation separately. '
        'No profiler establishes kernel dispatch. Held-out comparisons avoid selecting winners on their own validation results; exploratory full-grid rankings still have selection bias. '
        'Amortization assumes repeated inference on a fixed graph and stable average savings; topology changes, allocator effects, preprocessing memory and online conversion can change the tradeoff. Preparation order is fixed rather than randomized; its cost estimates also depend on allocator/cache state. '
        'Docker reproduces software dependencies, not identical hardware performance.','',
        'Run from the repository root. Dependencies and versions stay in the existing uv.lock. Plot rendering uses Matplotlib 3.10.8 separately from the benchmark environment.','',
        '```sh','export DGLBACKEND=pytorch OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1',
        'export UV_CACHE_DIR=/workspace/.cache/uv',
        'uv run --frozen python benchmarks/joint_tuning/tests.py',
        'uv run --frozen python benchmarks/joint_tuning/run.py --datasets cora citeseer pubmed reddit \\',
        '  --rounds 6 --discovery-rounds 3 --warmup 5 --repetitions 20 --threads 1 --seed 0 --order-seed 20261104',
        'uv run --frozen python benchmarks/joint_tuning/report.py',
        'MPLCONFIGDIR=/tmp/dgl-joint-mpl python benchmarks/joint_tuning/plots.py',
        '```','',
        'PILOT.json contains exact compatibility errors and materialized states. RESULTS.json preserves all worker records, controls, raw forward/restoration/combined samples, permutations/representations as hashes and provenance. '
        'ANALYSIS.json contains rankings, selection decisions, uncertainty and algebraic interaction/amortization summaries. No dataset downloads or virtual environments are included in this module.','']
    output.write_text('\n'.join(lines))
    print(f'Wrote {output}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('benchmarks/joint_tuning/RESULTS.json'))
    parser.add_argument('--output',type=Path,default=Path('benchmarks/joint_tuning/REPORT.md'))
    args=parser.parse_args();render(args.input,args.output)
