"""Execute all four families sequentially, six independent process rounds/axis."""
import argparse
from datetime import datetime,timezone
import subprocess
import uuid
from common import *
from native import build


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets',nargs='+',choices=['cora','citeseer','pubmed','reddit'],default=['cora','citeseer','pubmed','reddit'])
    p.add_argument('--rounds',type=int,default=6);p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--repetitions',type=int,default=20);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--order-seed',type=int,default=20261105);p.add_argument('--data-dir',default='results/datasets')
    p.add_argument('--output-dir',type=Path,default=HERE);p.add_argument('--work-dir',type=Path,default=Path('results/tuning_axes'))
    args=p.parse_args()
    if args.rounds<2 or args.warmup<1 or args.repetitions<2 or not 0<=args.seed<2**31:p.error('Require rounds>=2, warmup>=1, repetitions>=2, valid seed')
    _,compile_info=build();run_id=uuid.uuid4().hex;work=args.work_dir/run_id;work.mkdir(parents=True)
    result={'schema_version':1,'run_id':run_id,'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),
        'options':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'compiler_initial_build':compile_info,'datasets':{},'scope':'Independent one-parameter comparisons; no factorial search or cost model'}
    target=args.output_dir/'RESULTS.json';save(target,result)
    for dataset in dict.fromkeys(args.datasets):
        available=old.available_memory()
        if dataset=='reddit' and old.reddit_memory_estimate()>available*.7:
            result['datasets'][dataset]={'status':'skipped_memory','available_bytes':available,'budget_bytes':available*.7,
                'estimate_bytes':old.reddit_memory_estimate(),'reason':'Retained conservative four-head full-forward guard; no shape/topology reduction; all four families unrun'}
            save(target,result);print('reddit: fixed-shape memory safety skip',flush=True);continue
        record={'status':'running','rounds':[]};result['datasets'][dataset]=record
        for round_index in range(args.rounds):
            for threads in [1,4]:
                output=work/f'{dataset}-{round_index}-t{threads}.json'
                env=dict(os.environ,DGLBACKEND='pytorch',OMP_NUM_THREADS=str(threads),
                         MKL_NUM_THREADS=str(threads),OPENBLAS_NUM_THREADS=str(threads))
                command=[sys.executable,str(HERE/'worker.py'),'--dataset',dataset,'--round',str(round_index),
                    '--threads',str(threads),'--output',str(output),'--data-dir',args.data_dir,
                    '--warmup',str(args.warmup),'--repetitions',str(args.repetitions),'--seed',str(args.seed),'--order-seed',str(args.order_seed)]
                subprocess.run(command,env=env,check=True)
                data=json.loads(output.read_text())
                if record['rounds']:assert data['model']==record['rounds'][0]['model']
                record['rounds'].append(data);save(target,result)
        record['status']='completed'
    result.update(status='completed',finished_utc=datetime.now(timezone.utc).isoformat());save(target,result)
    print(f'Saved {target}',flush=True)


if __name__=='__main__':main()
