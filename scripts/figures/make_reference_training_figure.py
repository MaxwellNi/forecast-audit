"""Plot reference reuse, estimated means and fixed-peer counterexamples."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    peer_path = ROOT/'results/design_validation/peer_sampling.csv'
    learned_path = ROOT/'results/learned_references/summary.csv'
    peers = pd.read_csv(peer_path)
    learned = pd.read_csv(learned_path)
    plt.rcParams.update({'font.family':'Liberation Serif','font.size':8,
        'axes.titlesize':8,'axes.labelsize':8,'legend.fontsize':8,
        'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.6,'xtick.major.size':3,
        'ytick.major.size':3,'svg.hashsalt':'reference-training-study'})
    fig, axes = plt.subplots(1,4,figsize=(7.15,2.18),sharey=True)
    fig.subplots_adjust(left=.059,right=.991,bottom=.23,top=.73,wspace=.20)
    panels = [('resampled_peers','(a) Resampled, exact means'),
        ('learned','(b) Resampled, fitted means'),
        ('fixed_peers_aligned_effects','(c) Fixed, aligned effects'),
        ('fixed_peers_opposed_effects','(d) Fixed, opposed effects')]
    coordinates = []
    for ax,(regime,title) in zip(axes,panels):
        for short,long,label,color,marker,fill in [
            ('shared','shared_references','Shared references','#222222','o','white'),
            ('distinct','distinct_references','Distinct references','#0072B2','s','#0072B2')]:
            if regime == 'learned':
                frame = learned[(learned.entities==64)&(learned.groups==400)&
                    (learned.training_rows>0)&(learned.method==short)].sort_values('training_rows')
                x = frame.training_rows.to_numpy(); expected = [32,512,8192]
            else:
                frame = peers[(peers.peer_regime==regime)&(peers.method==long)].sort_values('periods')
                x = frame.periods.to_numpy(); expected = [25,100,400]
            assert list(x)==expected
            y = frame.rejections.to_numpy()/frame.replications.to_numpy()
            np.testing.assert_allclose(y,frame.rate.to_numpy(),atol=1e-14,rtol=0)
            z = norm.ppf(.975); n = frame.replications.to_numpy(); denominator=1+z*z/n
            middle=(y+z*z/(2*n))/denominator
            half=z*np.sqrt(y*(1-y)/n+z*z/(4*n*n))/denominator
            lo,hi = np.maximum(0,middle-half),np.minimum(1,middle+half)
            np.testing.assert_allclose(lo,frame.wilson_low,atol=1e-12,rtol=0)
            np.testing.assert_allclose(hi,frame.wilson_high,atol=1e-12,rtol=0)
            ax.errorbar(x,y,yerr=np.maximum(0,np.vstack((y-lo,hi-y))),
                marker=marker,markerfacecolor=fill,markeredgecolor=color,markersize=3.7,
                color=color,linewidth=.9,elinewidth=.65,capsize=2,label=label)
            coordinates.extend(dict(panel=regime,method=short,x=int(xx),rate=float(yy),
                rejections=int(rr),replications=int(nn),wilson_low=float(ll),wilson_high=float(hh))
                for xx,yy,rr,nn,ll,hh in zip(x,y,frame.rejections,n,lo,hi))
        if regime=='learned':
            ax.set_xscale('log',base=16);ax.set_xlim(20,13100)
            ax.set_xticks([32,512,8192],['32','512','8,192'])
            ax.set_xlabel('Training observations, $m$')
        else:
            ax.set_xscale('log',base=4);ax.set_xlim(17,588)
            ax.set_xticks([25,100,400],['25','100','400'])
            ax.set_xlabel('Independent groups, $M$')
        ax.minorticks_off();ax.set_ylim(-.025,1.025)
        ax.set_title(title,pad=6)
        ax.axhline(.05,color='.4',linestyle=':',linewidth=.8,zorder=0)
        ax.grid(axis='y',color='.9',linewidth=.5);ax.set_axisbelow(True)
    axes[0].set_ylabel('Positive rejection rate')
    axes[0].set_yticks([0,.25,.5,.75,1],['0','.25','.50','.75','1'])
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.53,.995),
        ncol=2,frameon=False,handlelength=2,columnspacing=2.5)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output_dir/'reference_training.pdf',metadata={'Creator':None,'Producer':None,'CreationDate':None,'ModDate':None})
    fig.savefig(args.output_dir/'reference_training.svg',metadata={'Creator':None,'Date':None})
    fig.savefig(args.output_dir/'reference_training.png',dpi=240)
    plt.close(fig)
    receipt = {'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (peer_path,learned_path)},
        'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'points':coordinates,'reference_line':.05,'intervals':'pointwise Wilson 95%',
        'sampling':'Panels a,c,d: N=32,300 replications and exact means. Panel b: N=64,M=400,1000 replications, empirically estimated means from independent training.'}
    (args.output_dir/'reference_training_coordinates.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__':main()
