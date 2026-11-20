"""Paired independent-round summaries; no predictive model or factorial analysis."""
import math
import statistics
from scipy.stats import t


def interval(values):
    logs=[math.log(x) for x in values];mean=statistics.mean(logs)
    half=float(t.ppf(.975,len(logs)-1))*statistics.stdev(logs)/math.sqrt(len(logs)) if len(logs)>1 else 0.
    return {'values':values,'geomean':math.exp(mean),'ci95':[math.exp(mean-half),math.exp(mean+half)]}


def analyze(raw):
    if raw['status']!='completed':raise ValueError('Incomplete experiment: do not report partial data as complete')
    result={'datasets':{},'axes':{},'notes':'Within-family contrasts only. Intervals use round log ratios, not repetitions as independent observations.'}
    for dataset,record in raw['datasets'].items():
        if record['status']!='completed':result['datasets'][dataset]=record;continue
        info=record['rounds'][0]['dataset_info'];out={'dataset_info':info,'axes':{}}
        names=list(dict.fromkeys(a for rr in record['rounds'] for a in rr['axes']))
        for axis in names:
            rounds=[rr for rr in record['rounds'] if axis in rr['axes']]
            entries=[rr['axes'][axis] for rr in rounds]
            reference=next((a['reference'] for a in entries if 'reference' in a),None)
            labels=list(dict.fromkeys(name for a in entries for name in a['cases']))
            summary={'reference':reference,'threads':rounds[0]['threads'],'cases':{},'n_process_rounds':len(rounds)}
            for label in labels:
                cases=[a['cases'].get(label,{'status':'unavailable'}) for a in entries]
                if any(c['status']!='measured' for c in cases):
                    summary['cases'][label]={'status':'excluded','records':cases};continue
                medians=[c['measurement']['median'] for c in cases]
                means=[c['measurement']['mean'] for c in cases];sds=[c['measurement']['stddev'] for c in cases]
                effects=[c['normalized_speedup'] for c in cases] if label!=reference else [1.]*len(cases)
                official=[(a['official_before']['median']+a['official_after']['median'])/2/c['measurement']['median'] for a,c in zip(entries,cases)]
                stage_names=list(cases[0]['profile'].get('stages_ms',{}))
                stages={name:{'round_medians':[c['profile']['stages_ms'][name]['median'] for c in cases],
                              'mean_median_ms':statistics.mean(c['profile']['stages_ms'][name]['median'] for c in cases)} for name in stage_names}
                if axis=='index_dtype':
                    preparation=[c['implementation']['cast_ms']+c['implementation']['materialization_ms'] for c in cases]
                elif axis in ['softmax_schedule','aggregation_schedule']:
                    key='softmax_partition' if axis=='softmax_schedule' else 'aggregation_partition'
                    preparation=[c['implementation'][key]['preparation_ms'] for c in cases]
                elif axis in ['score_traversal','aggregation_traversal']:
                    preparation=[c['implementation']['view_construction_ms'] for c in cases]
                else:preparation=[0.]*len(cases) # Layout packing is per forward, not a hidden setup charge.
                summary['cases'][label]={'status':'measured','round_medians_ms':medians,
                    'median_round_median_ms':statistics.median(medians),'mean_round_median_ms':statistics.mean(medians),
                    'mean_round_mean_ms':statistics.mean(means),'mean_within_round_sd_ms':statistics.mean(sds),
                    'between_round_sd_ms':statistics.stdev(medians),'speedup_vs_reference':interval(effects),
                    'official_over_case_speedup':interval(official),'profile_stages':stages,
                    'one_time_parameter_preparation_ms':{'values':preparation,'mean':statistics.mean(preparation)},
                    'output_max_absolute_error':max(c['correctness']['output_max_absolute'] for c in cases),
                    'max_scaled_intermediate_error':max((v['max_scaled'] for c in cases for v in c['correctness']['errors'].values()),default=0.),
                    'bitwise_repeatable':all(c['correctness']['bitwise_repeatable'] for c in cases),
                    'representative_implementation':cases[0]['implementation'],
                    'representative_tensors':cases[0]['correctness']['tensors'],
                    'representative_intermediate_payload_bytes':cases[0]['correctness'].get('intermediate_payload_bytes')}
            valid={k:v for k,v in summary['cases'].items() if v['status']=='measured'}
            summary['best_observed']=max(valid,key=lambda k:valid[k]['speedup_vs_reference']['geomean']) if valid else None
            out['axes'][axis]=summary
        result['datasets'][dataset]=out
    names=list(dict.fromkeys(a for g in result['datasets'].values() for a in g.get('axes',{})))
    for axis in names:
        panels={name:g['axes'][axis] for name,g in result['datasets'].items() if axis in g.get('axes',{})}
        directions={};magnitudes=[]
        for name,panel in panels.items():
            alt=next((v for k,v in panel['cases'].items() if k!=panel['reference'] and v['status']=='measured'),None)
            if not alt:directions[name]='unavailable';continue
            ratio=alt['speedup_vs_reference'];lo,hi=ratio['ci95'];magnitudes.append(max(ratio['geomean'],1/ratio['geomean'])-1)
            directions[name]='alternative' if lo>1 else 'reference' if hi<1 else 'inconclusive'
        if 'unavailable' in directions.values():category='C';reason='At least one required alternative lacks complete valid measurements'
        elif 'alternative' in directions.values() and 'reference' in directions.values() and min(magnitudes or [0])>=.03:category='A';reason='Exploratory intervals support opposite preferences with at least 3% effects across datasets; confirm with new cohorts'
        elif len(set(directions.values()))==1 and 'inconclusive' not in directions.values() and min(magnitudes or [0])>=.03:
            category='E';reason='Same option credibly dominates by at least 3% on all three tested workloads; collapse only within this envelope'
        elif any(v!='inconclusive' for v in directions.values()) and max(magnitudes or [0])>=.03:
            category='D';reason='Useful measured effect, but conditional on this implementation/resource/workload family; no demonstrated reversal'
        else:category='B';reason='Real implementations, but effects are small or process-round intervals are inconclusive'
        result['axes'][axis]={'classification':category,'reason':reason,'credible_preference':directions,
                            'best_observed':{name:p['best_observed'] for name,p in panels.items()},
                            'scope':'Exploratory, unadjusted 95% intervals; not multiple-comparison-adjusted; 3% practical threshold declared before full run'}
    result['axes']['edge_processing_schedule']={'classification':'E','reason':'Reuse score/aggregation traversal evidence; no duplicate schedule labels or extra experiments'}
    return result
