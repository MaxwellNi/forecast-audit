"""Plot all peer-sampling null comparisons from the released count table."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/'results/design_validation/peer_sampling.csv'


def wilson(count, trials):
    z=1.959963984540054
    p=count/trials
    denominator=1+z*z/trials
    center=(p+z*z/(2*trials))/denominator
    half=z*math.sqrt(p*(1-p)/trials+z*z/(4*trials*trials))/denominator
    return max(0.,center-half),min(1.,center+half)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=DEFAULT)
    ap.add_argument('--output-dir',type=Path,default=DEFAULT.parent)
    args=ap.parse_args()
    with args.source.open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==18
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,
       'axes.labelsize':8,'axes.titlesize':8.5,'legend.fontsize':8,
       'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,
       'axes.spines.right':False,'axes.linewidth':.6,'xtick.major.size':3,
       'ytick.major.size':3,'svg.hashsalt':'peer-sampling-figure'})
    fig,axes=plt.subplots(1,3,figsize=(7.04,2.2),sharey=True)
    fig.subplots_adjust(left=.082,right=.992,bottom=.235,top=.73,wspace=.18)
    panels=[('resampled_peers','(a) Resampled peers'),
            ('fixed_peers_aligned_effects','(b) Fixed peers, aligned effects'),
            ('fixed_peers_opposed_effects','(c) Fixed peers, opposed effects')]
    methods=[('shared_references','Shared reference','#222222','o','white'),
             ('distinct_references','Distinct references','#0072B2','s','#0072B2')]
    coordinates=[]
    for panel,(regime,title) in zip(axes,panels):
        for method,label,color,marker,fill in methods:
            select=sorted([r for r in rows if r['peer_regime']==regime and r['method']==method],key=lambda r:int(r['periods']))
            assert [int(r['periods']) for r in select]==[25,100,400]
            x=np.array([int(r['periods']) for r in select],float)
            y=np.array([int(r['rejections'])/int(r['replications']) for r in select])
            bounds=np.array([wilson(int(r['rejections']),int(r['replications'])) for r in select])
            for r,point,bound in zip(select,y,bounds):
                assert abs(point-float(r['rate']))<1e-14
                assert np.max(np.abs(bound-np.array([float(r['wilson_low']),float(r['wilson_high'])])))<1e-12
            panel.errorbar(x,y,yerr=np.maximum(0,np.vstack((y-bounds[:,0],bounds[:,1]-y))),
                marker=marker,markerfacecolor=fill,markeredgecolor=color,markersize=4,
                color=color,linewidth=1,elinewidth=.7,capsize=2,label=label)
            for r,xx,yy,bb in zip(select,x,y,bounds):
                coordinates.append({'peer_regime':regime,'method':method,'periods':int(r['periods']),
                                    'replications':int(r['replications']),'rejections':int(r['rejections']),
                                    'plotted_x':float(xx),'rejection_rate':float(yy),'wilson95':bb.tolist()})
        panel.set_xscale('log',base=4)
        panel.set_xticks([25,100,400],['25','100','400'])
        panel.minorticks_off()
        panel.set_xlim(19,520)
        panel.set_ylim(-.025,1.025)
        panel.set_title(title,pad=6)
        panel.set_xlabel('Independent groups, $M$')
        panel.axhline(.05,color='.4',linestyle=':',linewidth=.8,zorder=0)
        panel.grid(axis='y',color='.9',linewidth=.5)
        panel.set_axisbelow(True)
    axes[0].set_ylabel('Positive rejection rate')
    axes[0].set_yticks([0,.25,.5,.75,1],['0','.25','.50','.75','1'])
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.53,.994),ncol=2,
               frameon=False,handlelength=2,columnspacing=2.5)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output_dir/'peer_sampling.pdf',metadata={'Creator':'Matplotlib','CreationDate':None,'ModDate':None})
    fig.savefig(args.output_dir/'peer_sampling.svg',metadata={'Date':None})
    fig.savefig(args.output_dir/'peer_sampling.png',dpi=240)
    plt.close(fig)
    record={'source_file':args.source.name,'source_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),
            'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':'Conditional-null simulations with 32 entities per period, 300 replications per cell and exact conditional means. Intervals are pointwise 95% Wilson intervals. Fixed effects are aligned or opposed between forecast and outcome.',
            'reference_line':.05,'points':coordinates,'size_inches':[7.04,2.2]}
    (args.output_dir/'peer_sampling_coordinates.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
