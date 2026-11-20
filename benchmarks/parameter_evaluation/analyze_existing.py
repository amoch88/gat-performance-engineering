"""Read-only reanalysis of stored timings. No DGL execution, tuning or model fit."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
BASE='original/original/all'


def ratio_ci(values):
    logs=[math.log(v) for v in values]
    mean=statistics.mean(logs)
    delta=float(t.ppf(.975,len(logs)-1))*statistics.stdev(logs)/math.sqrt(len(logs))
    return {'geomean':math.exp(mean),'ci95':[math.exp(mean-delta),math.exp(mean+delta)],'round_ratios':values}


def extract():
    files=['benchmarks/dgl_baseline/MEASUREMENTS.json','benchmarks/joint_tuning/RESULTS.json',
           'benchmarks/joint_tuning/ANALYSIS.json','benchmarks/joint_tuning/PILOT.json']
    baseline=json.loads((ROOT/files[0]).read_text())
    joint=json.loads((ROOT/files[1]).read_text())
    assert joint['status']=='completed'
    def validate_block(block):
        samples=block['samples']
        assert len(samples)==20 and all(math.isfinite(v) and v>0 for v in samples)
        for name,func in [('median',statistics.median),('mean',statistics.mean),('stddev',statistics.stdev),('min',min)]:
            assert block[name]==func(samples)
    baseline_blocks=joint_blocks=0
    for record in baseline['measurements']:
        for case in record['cases']+([record['baseline']] if 'baseline' in record else []):
            if case['status']=='passed':validate_block(case['runtime_ms']);baseline_blocks+=1
    for group in joint['datasets'].values():
        for rr in group.get('rounds',[]):
            for trial in rr['trials']:
                for name in ['before_control','measurement','after_control']:
                    for metric in ['forward_ms','restoration_ms','combined_ms']:
                        validate_block(trial[name][metric])
                    joint_blocks+=1
    out={'raw_statistics_verified':{'baseline_forward_blocks':baseline_blocks,'joint_interface_blocks':joint_blocks},
         'scope':'Offline analysis only; no new measurements; no performance model',
         'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
         'datasets':{},'opat':[],'restricted_axis_slices':[],'pilot':joint['pilot']['policies']}
    for r in baseline['measurements']:
        if 'dataset_info' in r:out['datasets'][r['dataset']]=r['dataset_info']
        if r['experiment'] not in ['node_ordering','edge_ordering','sparse_format']:continue
        record={'dataset':r['dataset'],'parameter':r['experiment'],'status':r['status'],
                'baseline_policy':r.get('baseline_policy'),'cases':[]}
        control=r.get('baseline',{}).get('runtime_ms')
        if control:record['baseline_ms']={k:v for k,v in control.items() if k!='samples'}
        for c in r['cases']:
            row={'case':c['case'],'status':c['status']}
            if c['status']=='passed':
                data=c['runtime_ms'];samples=data['samples']
                assert len(samples)==20
                assert statistics.median(samples)==data['median'] and statistics.mean(samples)==data['mean']
                row.update({k:v for k,v in data.items() if k!='samples'})
                row.update(speedup_vs_control=control['median']/data['median'],
                           n_samples=len(samples),within_block_cv=data['stddev']/data['mean'],
                           preparation_ms=c.get('case_preprocessing_ms'),
                           distinct_storage=c.get('storage_distinct_from_baseline'),
                           correctness=c['correctness'])
            else:row['reason']=c.get('reason',r.get('reason'))
            record['cases'].append(row)
        passed=[c for c in record['cases'] if c['status']=='passed']
        if passed:record['fastest_candidate_block']=min(passed,key=lambda c:c['median'])['case']
        out['opat'].append(record)
    specs=[('edge: node=original, format=all',['original','src','dst'],lambda v:f'original/{v}/all','original'),
           ('node: edge=original EIDs, format=all',['original','rcmk','degree'],lambda v:f'{v}/original/all','original'),
           ('node: edge=src grouped, format=all',['original','rcmk','degree'],lambda v:f'{v}/src/all','original'),
           ('format: node=original, edge=original',['all','coo+csc'],lambda v:f'original/original/{v}','all')]
    for dataset,group in joint['datasets'].items():
        if group['status']!='completed':continue
        rounds=group['rounds']
        def trial(rr,key):return next(b for b in rr['trials'] if b['key']==rr['aliases'][key])
        for label,values,key,reference in specs:
            record={'dataset':dataset,'axis':label,'context':'post-hoc single-axis slice from existing Milestone 3 data; NOT a new independent experiment',
                    'reference_option':reference,'n_independent_process_rounds':len(rounds),'cases':[],'pairwise':[]}
            for value in values:
                blocks=[trial(rr,key(value)) for rr in rounds]
                row={'case':value}
                for metric in ['forward_ms','combined_ms']:
                    medians=[b['measurement'][metric]['median'] for b in blocks]
                    speedups=[trial(rr,key(reference))['normalized'][metric]['relative_latency']/b['normalized'][metric]['relative_latency'] for rr,b in zip(rounds,blocks)]
                    row[metric]={'mean_round_median':statistics.mean(medians),'sd_round_median':statistics.stdev(medians),
                                 'round_medians':medians,'speedup_vs_axis_reference':ratio_ci(speedups)}
                row['mean_restoration_ms']=statistics.mean(b['measurement']['restoration_ms']['median'] for b in blocks)
                row['mean_preparation_ms']=statistics.mean(rr['configurations'][key(value)]['preparation_ms']['total'] for rr in rounds)
                record['cases'].append(row)
            for a,b in itertools.combinations(values,2):
                ratios=[trial(rr,key(a))['normalized']['combined_ms']['relative_latency']/trial(rr,key(b))['normalized']['combined_ms']['relative_latency'] for rr in rounds]
                record['pairwise'].append({'a':a,'b':b,'a_over_b_combined_relative_latency':ratio_ci(ratios)})
            record['fastest_combined_normalized_option']=min(record['cases'],key=lambda c:c['combined_ms']['speedup_vs_axis_reference']['geomean']**-1)['case']
            out['restricted_axis_slices'].append(record)
    out['format_representation_evidence']={}
    for name,group in joint['datasets'].items():
        if group['status']!='completed':continue
        configs=group['rounds'][0]['configurations']
        a=configs['original/original/all']['representations_at_execution']
        b=configs['original/original/coo+csc']['representations_at_execution']
        out['format_representation_evidence'][name]={
            'all_allowed':a['allowed_formats'],'coo_csc_allowed':b['allowed_formats'],
            'shared_hashes_equal':{fmt:a['sparse_hashes'][fmt]==b['sparse_hashes'][fmt] for fmt in ('coo','csc')}}
    out['analytical_sizes_not_measurements']={}
    for name,g in out['datasets'].items():
        n,e=g['nodes'],g['edges_with_self_loops']
        out['analytical_sizes_not_measurements'][name]={
            'mean_in_degree_after_loops':e/n,'fixed_four_head_EH':e*4,
            'int32_saves_bytes_for_nonsharing_coo_csr_csc_explicit_eid':4*(6*e+2*(n+1)),
            'one_float32_edge_tensor_bytes_at_four_heads':e*4*4}
    return out


def table(head,rows):
    return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+[
        '| '+' | '.join(str(v).replace('|','/').replace('\n',' ') for v in row)+' |' for row in rows])+'\n'


def evidence_markdown(data):
    lines=['### 4.1 Original one-parameter-at-a-time measurements','',
        'These are the original Milestone 2 forward-only blocks (20 timed repetitions in one process per dataset/sweep). '
        'SD describes repetitions within that block, **not** uncertainty across independent trials. '
        'Speedup is the separately measured matching control median divided by the candidate median. '
        'The original-option repeat block is not forcibly set to one and is not cherry-picked against the separate control.','',
        'Edge sweep: original nodes, all formats. Node sweep: source-grouped edges for every value, all formats. '
        'Both use native input width, four heads and eight output features/head. Node restoration is checked but **excluded** from these timings.','']
    rows=[]
    for r in data['opat']:
        for c in r['cases']:
            if c['status']=='passed':rows.append([r['dataset'],r['parameter'],c['case'],f"{c['median']:.6f}",f"{c['mean']:.6f}",f"{c['stddev']:.6f}",f"{c['speedup_vs_control']:.4f}×"])
    lines+=[table(['Dataset','Axis','Option','Median ms','Mean ms','Within-block SD ms','Control/case'],rows)]
    lines+=['Fastest *candidate blocks* (not a statement of statistical superiority):','']
    rows=[]
    for r in data['opat']:
        if r['parameter']=='sparse_format':continue
        if 'fastest_candidate_block' not in r:rows.append([r['dataset'],r['parameter'],'not measured','—']);continue
        c=next(c for c in r['cases'] if c['case']==r['fastest_candidate_block'])
        rows.append([r['dataset'],r['parameter'],c['case'],f"{c['speedup_vs_control']:.4f}×"])
    lines+=[table(['Dataset','Axis','Fastest block','Control/case'],rows),
        'Sparse single-format cases failed on Cora/Citeseer/PubMed; there is no numerical format ranking from Milestone 2. '
        'All Reddit node/edge/format sweeps were memory-skipped. Its one-head result is a different workload and cannot fill the four-head evidence gap.','',
        '### 4.2 Restricted existing-data slices: other factors fixed','',
        'These tables re-read ONLY one-axis rows from the existing six-round Milestone 3 data. '
        'They are post-hoc corroboration, not a new benchmark or joint-winner argument. '
        'For each round, relative times first use the recorded bracketing controls; the axis-reference normalized time is then divided by the option normalized time. '
        'Intervals are unadjusted Student-t 95% intervals of the mean log ratio over **six process rounds**, exponentiated. '
        'The six round medians provide the between-process SD. They do not turn 120 within-process repetitions into 120 independent trials. '
        'Comparisons are exploratory, selected after inspecting these datasets and not multiple-comparison corrected.','']
    for axis in dict.fromkeys(r['axis'] for r in data['restricted_axis_slices']):
        lines += [f'**{axis}**','']
        rows=[]
        for r in data['restricted_axis_slices']:
            if r['axis']!=axis:continue
            for c in r['cases']:
                m=c['combined_ms'];s=m['speedup_vs_axis_reference'];f=c['forward_ms']
                rows.append([r['dataset'],c['case'],f"{f['mean_round_median']:.4f}",f"{m['mean_round_median']:.4f}",f"{m['sd_round_median']:.4f}",f"{s['geomean']:.4f} [{s['ci95'][0]:.4f}, {s['ci95'][1]:.4f}]"])
        lines += [table(['Dataset','Option','Forward ms','Combined ms','Between-round SD ms','Combined reference/case [95% CI]'],rows)]
    lines += ['Reference/case=1 for the reference by construction, not an interval claiming zero timing noise. '
        'Combined time includes restoring original node order; forward excludes restoration. '
        'The node/src slice matches Milestone 2\'s edge-policy context; the node/original-EID slice does not. '
        'Do not pool these two contexts or compare their absolute times as repeated measurements of a single treatment.','']
    return '\n'.join(lines)


if __name__=='__main__':
    data=extract()
    (OUT/'EVIDENCE.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    fragment=evidence_markdown(data)
    (OUT/'EVIDENCE.md').write_text(fragment)
    report=OUT/'REPORT.md'
    if report.exists():
        text=report.read_text();start='<!-- BEGIN RECORDED EVIDENCE -->';end='<!-- END RECORDED EVIDENCE -->'
        if start in text and end in text:
            prefix,tail=text.split(start,1);_,suffix=tail.split(end,1)
            report.write_text(prefix+start+'\n\n'+fragment+'\n'+end+suffix)
    print('Reanalyzed existing JSON only; no model or benchmark executed')
