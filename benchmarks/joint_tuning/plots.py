"""Standalone Matplotlib figure rendering; no benchmark-runtime imports."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform

os.environ.setdefault('MPLCONFIGDIR','/tmp/dgl-joint-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/dgl-joint-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render(source,directory):
    data=json.loads(source.read_text())
    directory.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'svg.hashsalt':'dgl-joint-milestone-3','font.family':'DejaVu Sans',
                         'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    generated=[]
    for dataset,result in data['datasets'].items():
        if result['status']!='completed': continue
        ranks=result['rankings']['combined_ms'];y=np.arange(len(ranks))
        centers=np.array([r['speedup']['geomean'] for r in ranks])
        lows=np.array([r['speedup']['ci95'][0] for r in ranks])
        highs=np.array([r['speedup']['ci95'][1] for r in ranks])
        fig,ax=plt.subplots(figsize=(9,max(5,len(ranks)*.31)))
        ax.errorbar(centers,y,xerr=np.vstack([centers-lows,highs-centers]),fmt='o',
                    color='#146b7b',ecolor='#8fadb4',capsize=3)
        ax.axvline(1,color='#555',linestyle='--',linewidth=1)
        ax.set_yticks(y,[r['key'] for r in ranks]);ax.invert_yaxis()
        ax.set_xlabel('Combined-interface speedup vs interleaved control (geometric mean; 95% t interval)')
        ax.set_title(f'{dataset.title()}: all-round exploratory joint ranking')
        ax.grid(axis='x',alpha=.2);fig.tight_layout()
        name=f'{dataset}-ranking.svg';fig.savefig(directory/name,metadata={'Date':None});plt.close(fig);generated.append(name)
        fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
        maximum=max(abs(math.log2(c['geomean'])) for pair in result['interactions'] for row in pair['cells'] for c in row)
        maximum=max(maximum,.01)
        for ax,pair in zip(axes,result['interactions']):
            values=np.array([[c['geomean'] for c in row] for row in pair['cells']])
            picture=ax.imshow(np.log2(values),cmap='RdBu_r',vmin=-maximum,vmax=maximum,aspect='auto')
            ax.set_xticks(np.arange(len(pair['columns'])),pair['columns'],rotation=25,ha='right')
            ax.set_yticks(np.arange(len(pair['rows'])),pair['rows']);ax.set_title(pair['pair'])
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):ax.text(j,i,f'{values[i,j]:.3f}',ha='center',va='center',color='black')
        fig.colorbar(picture,ax=list(axes),label='log₂ baseline-anchored interaction ratio',shrink=.7)
        fig.suptitle(f'{dataset.title()}: pairwise combined-time interactions; third factor fixed at baseline')
        name=f'{dataset}-interactions.svg';fig.savefig(directory/name,metadata={'Date':None});plt.close(fig);generated.append(name)
    manifest={'matplotlib':matplotlib.__version__,'python':platform.python_version(),
              'analysis_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'files':generated}
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Rendered {len(generated)} SVG figures with Matplotlib {matplotlib.__version__}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('benchmarks/joint_tuning/ANALYSIS.json'))
    p.add_argument('--output-dir',type=Path,default=Path('benchmarks/joint_tuning/figures'))
    args=p.parse_args();render(args.input,args.output_dir)
