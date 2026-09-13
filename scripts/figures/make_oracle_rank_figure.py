"""Vector figure and table from all frozen oracle-calibration cells."""
from pathlib import Path
import hashlib
import json
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'rerun_oracle_figure'
DATA.mkdir(exist_ok=False)
shutil.copyfile(ROOT/'results/oracle_rank_calibration/summary.csv', DATA/'summary.csv')
df = pd.read_csv(DATA/'summary.csv', keep_default_na=False)
methods = ['gcm_coarse','gcm_fine','gcm_spline','extrapolated','extrapolated_beta2','kci_gamma']
names = ['GCM, 8 bins','GCM, 32 bins','GCM, splines','Extrapolation, exponent 1','Extrapolation, exponent 2','KCI, gamma']
regimes = ['linear','smooth','nonlinear','heavy_tail']
titles = ['Linear','Smooth','Quadratic','Heavy tails']
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'axes.titlesize':9,
                     'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
                     'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
                     'axes.spines.top':False,'axes.spines.right':False})
fig, axes = plt.subplots(2,4,figsize=(7.15,3.9),sharey=True)
coordinates=[]
for j,(regime,title) in enumerate(zip(regimes,titles)):
    for i,alternative in enumerate(['null','positive_covariance']):
        ax=axes[i,j]
        for k,method in enumerate(methods):
            row=df[(df.regime==regime)&(df.alternative==alternative)&(df.method==method)].iloc[0]
            color = '#0072B2' if method=='extrapolated' else '#D55E00' if method=='extrapolated_beta2' else '#444444'
            marker = 'D' if method=='extrapolated_beta2' else 'o'
            lo,hi=row.conditional_bank_wilson95_low,row.conditional_bank_wilson95_high
            ax.errorbar(row.rate,5-k,xerr=np.array([[row.rate-lo],[hi-row.rate]]),fmt=marker,
                        ms=3.2,lw=.7,capsize=1.4,color=color)
            coordinates.append({'regime':regime,'alternative':alternative,'method':method,
                                'x':float(row.rate),'y':5-k,'low':float(lo),'high':float(hi)})
        ax.set_yticks(range(6));ax.set_yticklabels(names[::-1])
        ax.grid(axis='x',color='#E0E0E0',lw=.5)
        ax.set_axisbelow(True)
        ax.set_ylim(-.65,5.65)
        if i==0:
            ax.set_title(title,pad=6);ax.set_xlim(0,.16);ax.set_xticks([0,.05,.10,.15])
            ax.axvline(.05,color='black',ls=':',lw=.8)
            ax.set_xlabel('Null rejection')
        else:
            ax.set_xlim(.50,1.01);ax.set_xticks([.5,.75,1.]);ax.set_xlabel('Detection')
fig.subplots_adjust(left=.26,right=.99,bottom=.12,top=.91,wspace=.20,hspace=.52)
for ext in ['pdf','svg']:
    fig.savefig(DATA/f'fig_oracle_rank.{ext}',metadata={'Creator':'Matplotlib'})
fig.savefig(DATA/'fig_oracle_rank.png',dpi=180)
plt.close(fig)
pd.DataFrame(coordinates).to_csv(DATA/'figure_coordinates.csv',index=False)
df.to_csv(DATA/'oracle_rank_table.csv',index=False)
receipt={'source_sha256':hashlib.sha256((DATA/'summary.csv').read_bytes()).hexdigest(),
         'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'points':len(coordinates),'manual_point_movement':False,
         'interval_scope':'Conditional on each realized calibration bank, not the marginal Monte Carlo size.',
         'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DATA.glob('fig*'))}}
(DATA/'figure_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
