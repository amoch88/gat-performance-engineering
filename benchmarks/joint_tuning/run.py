"""Sequential independent worker processes with randomized complete-block rounds."""
import argparse
from datetime import datetime,timezone
import subprocess
import uuid
from core import *


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',nargs='+',choices=['cora','citeseer','pubmed','reddit'],default=['cora','citeseer','pubmed','reddit'])
    parser.add_argument('--rounds',type=int,default=6)
    parser.add_argument('--discovery-rounds',type=int,default=3)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--order-seed',type=int,default=20261104)
    parser.add_argument('--warmup',type=int,default=5)
    parser.add_argument('--repetitions',type=int,default=20)
    parser.add_argument('--data-dir',default='results/datasets')
    parser.add_argument('--output-dir',type=Path,default=HERE)
    parser.add_argument('--work-dir',type=Path,default=Path('results/joint_tuning'))
    args=parser.parse_args()
    if args.rounds < 4 or not 2 <= args.discovery_rounds <= args.rounds-2:
        parser.error('Need >=2 discovery and >=2 held-out rounds')
    if args.threads<1 or args.warmup<1 or args.repetitions<2 or not 0<=args.seed<2**31:
        parser.error('Require positive threads/warmup, repetitions>=2, seed in [0,2**31)')
    configure(args.threads)
    env=dict(os.environ, DGLBACKEND='pytorch',OMP_NUM_THREADS=str(args.threads),
             MKL_NUM_THREADS=str(args.threads),OPENBLAS_NUM_THREADS=str(args.threads))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    run_id=uuid.uuid4().hex
    work=args.work_dir/run_id
    work.mkdir(parents=True)
    pilot_result=pilot(args.seed)
    pilot_path=args.output_dir/'PILOT.json'
    save(pilot_path,pilot_result)
    result={'schema_version':1,'run_id':run_id,'started_utc':datetime.now(timezone.utc).isoformat(),
            'options':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            'pilot':pilot_result,'valid_formats':valid_formats(pilot_result),
            'requested_matrix':[dict(node=n,edge=e,format=f,key=f'{n}/{e}/{f}')
                for n,e,f in configurations(list(FORMATS))],
            'environment':provenance(),'datasets':{},'status':'running'}
    path=args.output_dir/'RESULTS.json'
    save(path,result,pretty=False)
    for dataset in dict.fromkeys(args.datasets):
        available=available_memory()
        # Existing 4-head lower bound plus resident joint graph matrix. Unlike
        # milestone 2, feature/head widths must NEVER be reduced to fit Reddit.
        n,e,f=232965,114615892,602
        matrix_lower_bound=len(configurations(result['valid_formats']))*e*8*2 + n*f*4*9
        if dataset=='reddit' and max(reddit_memory_estimate(),matrix_lower_bound) > available*.7:
            result['datasets'][dataset]={'status':'skipped_memory','available_bytes':available,
                'budget_bytes':available*.7,'single_case_estimate_bytes':reddit_memory_estimate(),
                'joint_matrix_lower_bound_bytes':matrix_lower_bound,
                'skipped':list(configurations(result['valid_formats'])),
                'reason':'Fixed four-head full-graph workload and joint graph matrix exceed 70% safety budget; no feature/head reduction or synthetic substitution'}
            save(path,result,pretty=False)
            print('reddit: fixed workload skipped by memory preflight',flush=True)
            continue
        record={'status':'running','rounds':[]}
        result['datasets'][dataset]=record
        for index in range(args.rounds):
            outfile=work/f'{dataset}-{index}.json'
            command=[sys.executable,str(HERE/'worker.py'),'--dataset',dataset,'--round',str(index),
                '--pilot',str(pilot_path),'--output',str(outfile),'--data-dir',args.data_dir,
                '--seed',str(args.seed),'--order-seed',str(args.order_seed),'--threads',str(args.threads),
                '--warmup',str(args.warmup),'--repetitions',str(args.repetitions)]
            subprocess.run(command,env=env,check=True)
            round_result=json.loads(outfile.read_text())
            assert round_result['round']==index and round_result['status']=='completed'
            if index:
                first=record['rounds'][0]
                assert first['model']==round_result['model']
                assert first['canonical_topology_sha256']==round_result['canonical_topology_sha256']
                assert first['aliases']==round_result['aliases']
                assert {k:v['signature'] for k,v in first['configurations'].items()}=={
                    k:v['signature'] for k,v in round_result['configurations'].items()}
            record['rounds'].append(round_result)
            save(path,result,pretty=False)
        record['status']='completed'
    result.update(status='completed',finished_utc=datetime.now(timezone.utc).isoformat())
    save(path,result,pretty=False)
    print(f'Saved {path}',flush=True)


if __name__=='__main__':
    main()
