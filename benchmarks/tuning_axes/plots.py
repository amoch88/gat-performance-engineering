"""Scientific SVG summaries of independent axes, rendered outside benchmark timing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/tuning-axes-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/tuning-axes-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render(source,directory):
    data=json.loads(source.read_text());directory.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'svg.hashsalt':'dgl-independent-axes','font.family':'DejaVu Sans','font.size':9})
    axes=list(k for k in data['axes'] if k!='edge_processing_schedule')
    fig,panels=plt.subplots(len(axes),1,figsize=(10,len(axes)*2.1),layout='constrained')
    colors=['#146b7b','#b46123','#72549b']
    for ax,name in zip(panels,axes):
        labels=[];comparison=name
        for i,(dataset,g) in enumerate((n,g) for n,g in data['datasets'].items() if 'axes' in g):
            a=g['axes'].get(name,{})
            alt=next(((v,c) for v,c in a.get('cases',{}).items() if v!=a['reference'] and c['status']=='measured'),None)
            if not alt:continue
            value,c=alt;comparison=f"{name}: {value} versus {a['reference']} (reference)"
            r=c['speedup_vs_reference'];center=r['geomean'];lo,hi=r['ci95']
            ax.errorbar(center,i,xerr=[[center-lo],[hi-center]],fmt='o',capsize=3,color=colors[i%3])
            labels.append(dataset)
        ax.set_yticks(range(len(labels)),labels);ax.axvline(1,color='#555',linestyle='--')
        ax.set_title(comparison);ax.grid(axis='x',alpha=.2)
        ax.set_xlabel('Within-family reference/case speedup; round-level geometric mean and unadjusted 95% interval')
    fig.savefig(directory/'axis-effects.svg',metadata={'Date':None});plt.close(fig)
    svg=directory/'axis-effects.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    manifest={'matplotlib':matplotlib.__version__,'analysis_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'files':['axis-effects.svg']}
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Rendered independent-axis effect plot')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path('benchmarks/tuning_axes/ANALYSIS.json'))
    p.add_argument('--output-dir',type=Path,default=Path('benchmarks/tuning_axes/figures'));a=p.parse_args();render(a.input,a.output_dir)
