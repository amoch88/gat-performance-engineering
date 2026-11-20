"""Seven-policy compatibility pilot using the complete untouched official layer."""
import argparse
from core import configure, pilot, save

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='benchmarks/joint_tuning/PILOT.json')
    args=parser.parse_args()
    configure()
    result=pilot()
    save(args.output,result)
    for r in result['policies']:
        print(r['policy'],r['status'],r.get('error','').splitlines()[:1])
