"""Run sequentially; reuse a confirmed dataset blocker without repeated downloads."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from time import time_ns
import uuid

SWEEPS = ['input_features','output_features','attention_heads','sparse_format','edge_ordering','node_ordering']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',nargs='+',choices=['cora','citeseer','pubmed','reddit'],default=['cora','citeseer','pubmed','reddit'])
    parser.add_argument('--data-dir',type=Path,default=Path('results/datasets'))
    parser.add_argument('--output-dir',type=Path,default=Path('results/dgl_baseline'))
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--warmup',type=int,default=5)
    parser.add_argument('--repetitions',type=int,default=20)
    args=parser.parse_args()
    env=dict(os.environ, DGL_EXPERIMENT_RUN_ID=str(uuid.uuid4()))
    here=Path(__file__).resolve().parent
    failures=False
    for dataset in dict.fromkeys(args.datasets):
        for experiment in ['baseline']+SWEEPS:
            path=args.output_dir/dataset/f'{experiment}.json'
            command=[sys.executable,str(here/f'{experiment}.py'),'--datasets',dataset,
                     '--data-dir',str(args.data_dir),'--output-dir',str(args.output_dir),
                     '--threads',str(args.threads),'--seed',str(args.seed),
                     '--warmup',str(args.warmup),'--repetitions',str(args.repetitions)]
            before=time_ns()
            process=subprocess.run(command,env=env)
            if not path.exists() or path.stat().st_mtime_ns < before:
                raise RuntimeError(f'{experiment}/{dataset} produced no fresh result; exit {process.returncode}')
            record=json.loads(path.read_text())
            if record['status'] == 'blocked_dataset':
                # Carry forward a confirmed shared prerequisite failure. These
                # cases are explicitly NOT RUN, not executed/failed/passed cases.
                for skipped in SWEEPS:
                    copy=dict(record,experiment=skipped,derived_from=str(path),cases=[],
                              not_run_cases='all requested cases; shared dataset prerequisite not met')
                    target=args.output_dir/dataset/f'{skipped}.json'
                    target.write_text(json.dumps(copy,indent=2)+'\n')
                failures |= record['status']=='blocked_dataset'
                break
            failures |= process.returncode != 0
    subprocess.run([sys.executable,str(here/'report.py'),'--input-dir',str(args.output_dir)],check=True)
    if failures:
        raise SystemExit(1)


if __name__=='__main__':
    main()
