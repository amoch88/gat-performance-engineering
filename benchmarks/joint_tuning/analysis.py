"""Factorial summaries and held-out comparisons, NOT a predictive model."""
import math
import statistics
from scipy.stats import t

BASE='original/original/all'
NODES=['original','rcmk','degree']
EDGES=['original','src','dst']


def geometric(values):
    return math.exp(statistics.mean(math.log(v) for v in values))


def ratio_summary(values):
    logs=[math.log(v) for v in values]
    mean=statistics.mean(logs)
    sd=statistics.stdev(logs) if len(logs)>1 else 0.
    half=float(t.ppf(.975,len(logs)-1))*sd/math.sqrt(len(logs)) if len(logs)>1 else 0.
    return {'values':values,'geomean':math.exp(mean),
            'ci95':[math.exp(mean-half),math.exp(mean+half)],'n_rounds':len(values)}


def trial(round_record,key):
    resolved=round_record['aliases'][key]
    return next(r for r in round_record['trials'] if r['key']==resolved)


def relative(round_record,key,metric):
    return trial(round_record,key)['normalized'][metric]['relative_latency']


def choose(rounds,keys,metric):
    return min(keys,key=lambda key:geometric([relative(r,key,metric) for r in rounds]))


def select(rounds,formats,metric):
    axis_node=choose(rounds,[f'{n}/original/all' for n in NODES],metric).split('/')[0]
    axis_edge=choose(rounds,[f'original/{e}/all' for e in EDGES],metric).split('/')[1]
    axis_format=choose(rounds,[f'original/original/{f}' for f in formats],metric).split('/')[2]
    independent=f'{axis_node}/{axis_edge}/{axis_format}'
    all_keys=[r['key'] for r in rounds[0]['trials']]
    joint=choose(rounds,all_keys,metric)
    independent=rounds[0]['aliases'][independent]
    return {'joint':joint,'independent':independent,'different':joint!=independent,
            'axis_choices':{'node':axis_node,'edge':axis_edge,'format':axis_format}}


def interactions(rounds,formats,metric):
    """Baseline-anchored cross-ratios with the third factor fixed to baseline.

    I(a,b) = T(a,b)*T(0,0)/(T(a,0)*T(0,b)). This is a descriptive
    interaction contrast, not fitted timing prediction or kernel attribution.
    """
    specs=[('node × edge',NODES,EDGES,lambda a,b:f'{a}/{b}/all'),
           ('node × format',NODES,formats,lambda a,b:f'{a}/original/{b}'),
           ('edge × format',EDGES,formats,lambda a,b:f'original/{a}/{b}')]
    output=[]
    for title,rows,columns,key in specs:
        cells=[]
        for a in rows:
            row=[]
            for b in columns:
                values=[relative(r,key(a,b),metric)*relative(r,BASE,metric)/
                        (relative(r,key(a,columns[0]),metric)*relative(r,key(rows[0],b),metric))
                        for r in rounds]
                row.append(ratio_summary(values))
            cells.append(row)
        output.append({'pair':title,'rows':rows,'columns':columns,'cells':cells})
    return output


def analyze(data):
    if data['status']!='completed':
        raise ValueError('Incomplete measurement run; do not report partial data as complete')
    cut=data['options']['discovery_rounds']
    formats=data['valid_formats']
    output={'schema_version':1,'run_id':data['run_id'],'datasets':{},
            'selection_metric':'combined_ms','split':{'discovery':cut,'held_out':data['options']['rounds']-cut},
            'interaction_formula':'T(a,b)*T(0,0)/(T(a,0)*T(0,b)); third factor fixed at baseline'}
    for dataset,record in data['datasets'].items():
        if record['status']!='completed':
            output['datasets'][dataset]=record
            continue
        rounds=record['rounds'];keys=[r['key'] for r in rounds[0]['trials']]
        rankings={}
        for metric in ['forward_ms','combined_ms']:
            ranking=[]
            for key in keys:
                blocks=[trial(r,key) for r in rounds]
                medians=[b['measurement'][metric]['median'] for b in blocks]
                means=[b['measurement'][metric]['mean'] for b in blocks]
                within_sd=[b['measurement'][metric]['stddev'] for b in blocks]
                ratios=[b['normalized'][metric]['speedup'] for b in blocks]
                preparation=[r['configurations'][key]['preparation_ms']['total'] for r in rounds]
                savings=[b['normalized']['combined_ms']['control_median']-b['measurement']['combined_ms']['median'] for b in blocks]
                point=statistics.mean(savings)
                per_round_break_even=[math.ceil(p/s) if s>0 else None for p,s in zip(preparation,savings)]
                ranking.append({'key':key,'mean_round_median_ms':statistics.mean(medians),
                    'mean_round_mean_ms':statistics.mean(means),'mean_within_round_sd_ms':statistics.mean(within_sd),
                    'between_round_sd_ms':statistics.stdev(medians),'round_medians_ms':medians,
                    'speedup':ratio_summary(ratios),
                    'mean_restoration_ms':statistics.mean(b['measurement']['restoration_ms']['median'] for b in blocks),
                    'preparation_ms':{'values':preparation,'mean':statistics.mean(preparation)},
                    'amortization':{'mean_combined_saving_ms':point,'positive_rounds':sum(s>0 for s in savings),
                        'per_round_calls':per_round_break_even,
                        'point_estimate_calls':math.ceil(statistics.mean(preparation)/point) if point>0 else None}})
            rankings[metric]=sorted(ranking,key=lambda r:r['speedup']['geomean'],reverse=True)
        selections={}
        for metric in ['forward_ms','combined_ms']:
            selected=select(rounds[:cut],formats,metric)
            values=[relative(r,selected['independent'],metric)/relative(r,selected['joint'],metric) for r in rounds[cut:]]
            selected['held_out_joint_over_independent_speedup']=ratio_summary(values)
            selected['held_out_raw_independent_over_joint']=[
                trial(r,selected['independent'])['measurement'][metric]['median']/
                trial(r,selected['joint'])['measurement'][metric]['median'] for r in rounds[cut:]]
            exploratory=select(rounds,formats,metric)
            selections[metric]={'discovery':selected,'exploratory_all_rounds':exploratory}
        drift=[b['normalized']['combined_ms']['control_drift_ratio'] for r in rounds for b in r['trials']]
        output['datasets'][dataset]={'status':'completed','dataset_info':rounds[0]['dataset_info'],
            'rankings':rankings,'selection':selections,
            'interactions':interactions(rounds,formats,'combined_ms'),
            'matrix':{k:{'status':v['status'],'equivalent_to':v.get('equivalent_to'),
                         'restoration':v['restoration']} for k,v in rounds[0]['configurations'].items()},
            'aliases':rounds[0]['aliases'],
            'control_drift_ratio':{'min':min(drift),'max':max(drift),'median':statistics.median(drift)},
            'max_absolute_reference_error':max(v['max_absolute_error'] for r in rounds for v in r['configurations'].values())}
    return output
