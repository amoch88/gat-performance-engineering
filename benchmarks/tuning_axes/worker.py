"""One fresh process per dataset/round/thread group; axes remain OPAT contrasts."""
import argparse
import json
import os
from pathlib import Path
import random
import resource
from time import perf_counter_ns
from common import *
from native import Native
from orchestration import Official
from registry import AXES


def block(case,x,model,warmup,repetitions):
    with torch.inference_mode():
        for _ in range(warmup):case.forward(x,model)
        values=[]
        for _ in range(repetitions):
            start=perf_counter_ns();y=case.forward(x,model);values.append((perf_counter_ns()-start)/1e6)
        assert torch.isfinite(y).all()
    return stats(values)


def profile(case,x,model,repetitions):
    if isinstance(case,Official):return {'available':False,'reason':'Untouched official layer; no substituted stage timing'}
    values={}
    with torch.inference_mode():
        for _ in range(repetitions):
            y,detail=case.forward(x,model,profile=True)
            for k,v in detail['stages'].items():values.setdefault(k,[]).append(v)
    return {'available':True,'stages_ms':{k:stats(v) for k,v in values.items()},
            'note':'Separate instrumented passes, not included in ordinary full-forward timing; do not sum to predict ordinary latency'}


def execute(args):
    configure(args.threads)
    g,x,info=old.load_dataset(args.dataset,Path(args.data_dir))
    if g.idtype!=torch.int64:g=g.long();g.create_formats_()
    model=model_for(x.shape[1],args.seed);native=Native();official=Official(g)
    assert native.lib.axes_team_size(args.threads)==args.threads
    degrees=g.in_degrees().long()
    info['incoming_degree_summary']={'mean':float(degrees.float().mean()),'max':int(degrees.max()),
        'p50':float(torch.quantile(degrees.float(),.5)),'p95':float(torch.quantile(degrees.float(),.95))}
    cold_formats={}
    cu,cv=g.edges(order='eid')
    for fmt in ['coo','csr','csc']:
        cold=dgl.graph((cu.clone(),cv.clone()),num_nodes=g.num_nodes())
        begin=perf_counter_ns();view=cold.formats(fmt);view.create_formats_()
        cold_formats[fmt]={'restriction_and_materialization_ms':(perf_counter_ns()-begin)/1e6,
                           'state':view_info(view),'note':'Fresh COO graph source tensors constructed before this format timer'}
    axes=[k for k,v in AXES.items() if v['threads']==args.threads]
    order_seed=args.order_seed+args.round*1009+int(hashlib.sha256(args.dataset.encode()).hexdigest()[:8],16)+args.threads
    randomizer=random.Random(order_seed);randomizer.shuffle(axes)
    result={'dataset':args.dataset,'round':args.round,'threads':args.threads,'pid':os.getpid(),
        'environment':provenance(),'dataset_info':info,'compiler':native.build_info,
        'cold_sparse_format_construction':cold_formats,'order_seed':order_seed,'axis_order':axes,'warmup':args.warmup,'repetitions':args.repetitions,
        'model':{'heads':H,'out_features':F,'input_features':x.shape[1],'seed':args.seed,'dtype':'float32',
                 'weights_sha256':{k:fingerprint(v) for k,v in model.state_dict().items()}},'axes':{}}
    for axis in axes:
        setting=AXES[axis];begin=perf_counter_ns()
        try:
            cases=setting['builder'](g,native,args.threads)
        except (dgl.DGLError,RuntimeError,ValueError,AssertionError) as exc:
            result['axes'][axis]={'status':'setup_unavailable','cases':{},'error_type':type(exc).__name__,'error':str(exc)}
            print(f'{args.dataset} {axis}: setup_unavailable: {exc}',flush=True)
            continue
        construction=(perf_counter_ns()-begin)/1e6
        record={'reference':setting['reference'],'family':setting['family'],'shared_case_setup_validation_ms':construction,
                'cases':{},'status':'pending'}
        supported={}
        # Diagnose alternatives independently. Unsupported DGL primitives retain
        # exact errors; incorrect implementations never enter performance ranks.
        for name,case in cases.items():
            if isinstance(case,dict):record['cases'][name]={'status':'unsupported',**case};continue
            try:
                with torch.inference_mode():correctness=validate(case,g,x,model)
                supported[name]=case
                record['cases'][name]={'status':'validated','implementation':case.info,'correctness':correctness}
            except (dgl.DGLError,AssertionError,RuntimeError) as exc:
                record['cases'][name]={'status':'unsupported' if isinstance(exc,dgl.DGLError) else 'failed_correctness',
                                       'error_type':type(exc).__name__,'error':str(exc),'implementation':case.info}
        reference=supported.get(setting['reference'])
        if reference is None:
            record['status']='reference_unavailable';result['axes'][axis]=record;continue
        order=list(supported);randomizer.shuffle(order);record['randomized_case_order']=order
        record['official_before']=block(official,x,model,args.warmup,args.repetitions)
        for name in order:
            before=block(reference,x,model,args.warmup,args.repetitions)
            measurement=block(supported[name],x,model,args.warmup,args.repetitions)
            after=block(reference,x,model,args.warmup,args.repetitions)
            control=(before['median']+after['median'])/2
            record['cases'][name].update(status='measured',before_control=before,measurement=measurement,
                after_control=after,normalized_speedup=control/measurement['median'],
                control_drift_ratio=after['median']/before['median'])
        record['official_after']=block(official,x,model,args.warmup,args.repetitions)
        # Profiles and per-thread diagnostic clock calls occur AFTER all ordinary
        # measured blocks for this axis, never inside an end-to-end sample.
        for name in order:
            case=supported[name]
            record['cases'][name]['profile']=profile(case,x,model,args.repetitions)
            with torch.inference_mode():
                if hasattr(case,'diagnostics'):record['cases'][name]['scheduling_diagnostics']=case.diagnostics(x,model)
            if hasattr(case,'score_g'):
                for label,view in [('score',case.score_g),('softmax',case.softmax_g),('aggregation',case.aggregation_g)]:
                    assert old.representation(view)['formats']==case.info['views'][label]['formats']
        record['status']='completed' if all(c['status']=='measured' for c in record['cases'].values()) else 'completed_with_exclusions'
        result['axes'][axis]=record
        print(f'{args.dataset} round {args.round+1} {axis}: '+', '.join(k+"="+v['status'] for k,v in record['cases'].items()),flush=True)
    result['process_peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['peak_memory_note']='Process peak across dataset load, all axes, profiles and validation; not a per-case allocation measurement'
    result['status']='completed'
    save(args.output,result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['cora','citeseer','pubmed','reddit'],required=True)
    p.add_argument('--round',type=int,required=True);p.add_argument('--threads',type=int,choices=[1,4],required=True)
    p.add_argument('--data-dir',default='results/datasets');p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--order-seed',type=int,default=20261105)
    p.add_argument('--warmup',type=int,default=5);p.add_argument('--repetitions',type=int,default=20)
    execute(p.parse_args())
