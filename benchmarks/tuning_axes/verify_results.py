"""Audit retained real measurements without executing any GAT forwards."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
CORE=['common.py','native.py','worker.py','run.py','index_dtype.py','traversal.py',
      'layouts.py','scheduling.py','registry.py','orchestration.py','kernels.cpp']
AXES=['index_dtype','score_traversal','aggregation_traversal','projected_layout',
      'attention_layout','softmax_schedule','aggregation_schedule']


def audit(path):
    raw=json.loads(path.read_text());assert raw['status']=='completed'
    expected_rounds=raw['options']['rounds'];reps=raw['options']['repetitions']
    workers=[];blocks=0;samples=0;profile_blocks=0;validated=0;drifts=[];orders={};largest_error=0.

    def block(value):
        nonlocal blocks,samples
        data=value['samples'];assert len(data)==reps
        assert all(math.isfinite(x) and x>0 for x in data)
        for key,fn in [('median',statistics.median),('mean',statistics.mean),('stddev',statistics.stdev),('min',min)]:
            assert math.isclose(value[key],fn(data),rel_tol=1e-12,abs_tol=1e-12)
        blocks+=1;samples+=len(data)

    for dataset,record in raw['datasets'].items():
        if record['status']=='skipped_memory':
            assert dataset=='reddit' and record['estimate_bytes']>record['budget_bytes'];continue
        assert record['status']=='completed' and len(record['rounds'])==expected_rounds*2
        cohort=record['rounds'];weights=cohort[0]['model'];feature=cohort[0]['dataset_info']['feature_sha256']
        for threads in [1,4]:assert sorted(w['round'] for w in cohort if w['threads']==threads)==list(range(expected_rounds))
        for worker in cohort:
            workers.append(worker['pid']);assert worker['status']=='completed'
            assert worker['model']==weights and worker['dataset_info']['feature_sha256']==feature
            assert worker['environment']['torch_threads']==worker['threads']
            assert worker['environment']['torch_interop_threads']==1
            assert all(worker['environment']['thread_environment'][k]==str(worker['threads']) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])
            assert worker['environment']['runtime_thread_controls']=={'OMP_WAIT_POLICY':'PASSIVE','KMP_BLOCKTIME':'0'}
            for name in CORE:
                assert worker['environment']['source_sha256'][name]==hashlib.sha256((HERE/name).read_bytes()).hexdigest(),name
        for axis in AXES:
            cohort_axes=[w['axes'][axis] for w in cohort if axis in w['axes']]
            assert len(cohort_axes)==expected_rounds
            orders[f'{dataset}/{axis}']=[a['randomized_case_order'] for a in cohort_axes]
            for entry in cohort_axes:
                assert entry['status']=='completed' and len(entry['cases'])==2
                block(entry['official_before']);block(entry['official_after'])
                for value,case in entry['cases'].items():
                    assert case['status']=='measured' and case['correctness']['status']=='passed'
                    assert case['correctness']['bitwise_repeatable']
                    validated+=1;largest_error=max(largest_error,case['correctness']['output_max_absolute'])
                    assert all(x['max_scaled']<=1.000001 for x in case['correctness']['errors'].values())
                    for label in ['before_control','measurement','after_control']:block(case[label])
                    expected=(case['before_control']['median']+case['after_control']['median'])/2/case['measurement']['median']
                    assert math.isclose(expected,case['normalized_speedup'],rel_tol=1e-12)
                    drifts.append(case['control_drift_ratio'])
                    if axis.endswith('_schedule'):assert case['implementation']['actual_openmp_team']==4
                    if axis=='index_dtype':
                        assert case['implementation']['bounds']['int32_legal']
                        assert case['implementation']['graph']['idtype']=='torch.'+value
                        assert set(case['implementation']['graph']['allowed_formats'])=={'coo','csr','csc'}
                        assert set(case['implementation']['graph']['formats']['created'])=={'coo','csr','csc'}
                    else:
                        stages=case['profile']['stages_ms'];assert len(stages)==9
                        for stage in stages.values():block(stage);profile_blocks+=1
                        for stage,expected in [('score',value if axis=='score_traversal' else 'coo'),('softmax','csc'),('aggregation',value if axis=='aggregation_traversal' else 'csc')]:
                            view=case['implementation']['views'][stage]
                            assert view['allowed_formats']==[expected] and view['formats']['created']==[expected]
                    if axis=='projected_layout' and value=='hnf':
                        assert case['correctness']['tensors']['projected_logical']['strides']==[8,record['rounds'][0]['dataset_info']['nodes']*8,1]
                    if axis=='attention_layout' and value=='he':
                        assert case['correctness']['tensors']['attention_logical']['strides'][:2]==[1,record['rounds'][0]['dataset_info']['edges_with_self_loops']]
    assert len(set(workers))==len(workers)
    return {'status':'passed','results_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'independent_worker_processes':len(workers),'validated_candidate_process_cases':validated,
            'ordinary_forward_samples':(blocks-profile_blocks)*reps,'instrumented_stage_samples':profile_blocks*reps,
            'timing_and_stage_blocks_audited':blocks,'timing_and_stage_samples_audited':samples,
            'largest_output_absolute_error':largest_error,'randomized_orders':orders,
            'control_drift_ratio_range':[min(drifts),max(drifts)],
            'control_brackets_exceeding_10_percent_drift':sum(abs(x-1)>.1 for x in drifts),
            'control_bracket_count':len(drifts),'data_filtering':'None; all declared rounds and samples retained',
            'source_scope':'All measured implementation/driver sources match. Reporting/analysis/audit code is postprocessing, not a timed implementation.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,default=HERE/'RESULTS.json')
    p.add_argument('--output',type=Path);args=p.parse_args();result=audit(args.input)
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='randomized_orders'},indent=2))
