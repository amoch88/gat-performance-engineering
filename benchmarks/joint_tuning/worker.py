"""One independent process per dataset/round. All preprocessing is outside timing."""
import argparse
import hashlib
import json
import random
from pathlib import Path
from core import *


def execute(args):
    configure(args.threads)
    base,x,info=load_dataset(args.dataset,Path(args.data_dir))
    attach_ids(base)
    model=model_for(x.shape[1],args.seed)
    with torch.inference_mode():
        reference=model(base,x)
    approved=valid_formats(json.loads(Path(args.pilot).read_text()))
    if 'all' not in approved:
        raise RuntimeError('Control policy is not pilot-approved')
    prepared={};audit={};aliases={};signatures={}
    # Control is first in this deterministic construction order, so exact
    # duplicate aliases always retain it rather than silently discarding it.
    for node,edge,fmt in configurations(approved):
        case=prepare(base,node,edge,fmt)
        checked=verify_prepared(case,base,model,reference)
        key=case['key']
        audit[key]={'node':node,'edge':edge,'format':fmt,
                    'preparation_ms':case['preparation_ms'],**checked}
        signature=checked['signature']
        if signature in signatures:
            aliases[key]=signatures[signature]
            audit[key]['status']='exact_duplicate'
            audit[key]['equivalent_to']=signatures[signature]
        else:
            signatures[signature]=key
            prepared[key]=case
            aliases[key]=key
            audit[key]['status']='validated'
    order=list(prepared)
    # Stable dataset-specific seed, never Python's randomized hash().
    order_seed=args.order_seed + args.round*1009 + int(hashlib.sha256(args.dataset.encode()).hexdigest()[:8],16)
    random.Random(order_seed).shuffle(order)
    control=prepared[BASE]
    trials=[]
    for key in order:
        before=timed_block(control,model,args.warmup,args.repetitions)
        measured=timed_block(prepared[key],model,args.warmup,args.repetitions)
        after=timed_block(control,model,args.warmup,args.repetitions)
        normalized={}
        for metric in ['forward_ms','combined_ms']:
            bracket=(before[metric]['median']+after[metric]['median'])/2
            normalized[metric]={'control_median':bracket,
                'relative_latency':measured[metric]['median']/bracket,
                'speedup':bracket/measured[metric]['median'],
                'control_drift_ratio':after[metric]['median']/before[metric]['median']}
        trials.append({'key':key,'before_control':before,'measurement':measured,
                       'after_control':after,'normalized':normalized})
    for key,case in prepared.items():
        assert representation(case['graph'])==audit[key]['representations_at_execution'], 'Timing changed formats'
    result={'dataset':args.dataset,'round':args.round,'pid':os.getpid(),
        'dataset_info':info,'model':{'input_features':x.shape[1],'heads':4,'out_features':8,
            'dtype':'float32','seed':args.seed,'weight_sha256':{k:digest(v) for k,v in model.state_dict().items()}},
        'canonical_topology_sha256':[digest(a) for a in base.edges(order='eid')],
        'environment':provenance(),'order_seed':order_seed,'randomized_order':order,
        'warmup':args.warmup,'repetitions':args.repetitions,'configurations':audit,'aliases':aliases,
        'trials':trials,'status':'completed'}
    save(args.output,result,pretty=False)
    print(f'{args.dataset} round {args.round+1}: {len(trials)} configurations; all invariants passed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['cora','citeseer','pubmed','reddit'],required=True)
    parser.add_argument('--round',type=int,required=True)
    parser.add_argument('--pilot',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--data-dir',default='results/datasets')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--order-seed',type=int,default=20261104)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--warmup',type=int,default=5)
    parser.add_argument('--repetitions',type=int,default=20)
    execute(parser.parse_args())
