"""Compact rendering of protocol-declared focus settings; no new analysis grid."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();OUTPUT=args.output;OUTPUT.mkdir(parents=True,exist_ok=True)
ORDER=['absolute_range','signed_range','absolute_variance','signed_variance']
STYLE={
 'absolute_range':dict(color='#b35a00',linestyle='--',marker='x',markersize=7),
 'signed_range':dict(color='#176b9d',linestyle='--',marker='D',markersize=7,markerfacecolor='none'),
 'absolute_variance':dict(color='#b35a00',linestyle='-',marker='s',markersize=6,markerfacecolor='none'),
 'signed_variance':dict(color='#176b9d',linestyle='-',marker='o',markersize=3.5)}
LABEL=['Absolute + range','Signed + range','Absolute + variance','Signed + variance']
s=pd.read_csv(HERE/'summary.csv');p=pd.read_csv(HERE/'paired_differences.csv')
protocol=json.loads((HERE/'provenance/protocol.json').read_text())
a=s[(s.design=='eight_rare')&(s.signal==.25)&(s.fit=='estimated')&(s.validation_pairs==8192)&s.groups.isin([100,400])].copy()
c=s[(s.design=='two_balanced')&(s.signal==.5)&(s.fit=='opposed_shifts')&(s.validation_pairs==8192)&s.groups.isin([100,400])].copy()
for frame,key in [(a,'figure_settings'),(c,'persistent_table_settings')]:
 for _,r in frame.iterrows():
  assert any(all(r[k]==v for k,v in setting.items()) for setting in protocol[key])
 assert len(frame)==8
plt.rcParams.update({'font.size':8,'axes.titlesize':8,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42})
fig,axs=plt.subplots(1,3,figsize=(7,2.05),gridspec_kw={'width_ratios':[1,1.12,1]})
fig.subplots_adjust(left=.055,right=.987,bottom=.235,top=.735,wspace=.38)
handles=[]
for ax,frame,title in [(axs[0],a,r'(a) Eight categories, $\vartheta=0.01366$'),(axs[2],c,r'(c) Opposed fits, $\vartheta=0.02$')]:
 for m,label in zip(ORDER,LABEL):
  q=frame[frame.method==m].sort_values('groups')
  line,=ax.plot(q.total_observations/1000,q.power,linewidth=1.25,label=label,clip_on=False,**STYLE[m])
  if ax is axs[0]:handles.append(line)
 ax.set_ylim(-.055,1.055);ax.set_yticks([0,.5,1]);ax.set_ylabel('Rejection rate',labelpad=1)
 ax.set_title(title,pad=5)
axs[0].annotate('0.435',xy=(30.976,.435),xytext=(6,3),textcoords='offset points',color='#176b9d',fontsize=8)
axs[0].annotate('0.274',xy=(30.976,.274),xytext=(6,-9),textcoords='offset points',color='#b35a00',fontsize=8)
axs[2].annotate('0.070',xy=(50.176,.07),xytext=(-30,6),textcoords='offset points',color='#176b9d',fontsize=8)
q=a[a.method=='absolute_range'].sort_values('groups');v=a[a.method=='absolute_variance'].sort_values('groups')
ax=axs[1]
ax.plot(q.total_observations/1000,1000*q.absolute_allowance,color='#b35a00',marker='s',markerfacecolor='none',markersize=5,linewidth=1.2,label=r'$\widehat B_{\rm abs}$')
ax.plot(q.total_observations/1000,1000*q.signed_allowance,color='#176b9d',marker='o',markersize=3,linewidth=1.2,label=r'$\widehat B_+$')
ax.plot(q.total_observations/1000,1000*q.radius,color='#555555',marker='x',markersize=5,linestyle='--',linewidth=1.2,label=r'$r_{\rm range}$')
ax.plot(v.total_observations/1000,1000*v.radius,color='#555555',marker='D',markerfacecolor='none',markersize=4,linewidth=1.2,label=r'$r_{\rm var}$')
ax.set_title('(b) Mean components in (a)',pad=5);ax.set_ylabel(r'Allowance ($\times 10^{-3}$)',labelpad=1)
ax.set_ylim(0,20);ax.set_yticks([0,10,20]);ax.legend(loc='upper right',ncol=2,frameon=False,bbox_to_anchor=(1.08,1.12),columnspacing=.45,handlelength=1,handletextpad=.3,borderpad=0,labelspacing=.2)
for ax in axs:
 ax.set_xlim(29,52);ax.set_xticks([30.976,50.176]);ax.set_xticklabels(['30,976','50,176'])
 ax.grid(axis='y',color='.9',linewidth=.5);ax.set_axisbelow(True)
 ax.spines[['top','right']].set_visible(False)
 ax.set_xlabel('Total observations',labelpad=2)
fig.legend(handles,LABEL,loc='upper center',bbox_to_anchor=(.5,1.02),ncol=4,frameon=False,handlelength=1.9,columnspacing=1.1,handletextpad=.4)
for extension in ['pdf','png']:fig.savefig(OUTPUT/f'fig_certificate_factorial.{extension}',dpi=300,metadata={'CreationDate':None,'ModDate':None} if extension=='pdf' else None)
plt.close(fig)
meta={'scope':'Post-exposure ablation; only previously declared focus settings; all100original settings retained separately.',
 'size_inches':[7,2.05],'minimum_font_points':8,'panels':{'a':a.to_dict('records'),'b':{'absolute_allowance':q.absolute_allowance.tolist(),'signed_allowance':q.signed_allowance.tolist(),'range_radius':q.radius.tolist(),'variance_radius':v.radius.tolist(),'total_observations':q.total_observations.tolist()},'c':c.to_dict('records')},
 'plotting':'No horizontal/vertical point jitter. Distinct marker sizes/shapes show coincident methods. Lines join only the two displayed budgets and do not estimate intervening power.',
 'input_sha256':{name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ['summary.csv','paired_differences.csv','provenance/protocol.json']},
 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUTPUT/'fig_certificate_factorial.json').write_text(json.dumps(meta,indent=2)+'\n')
# All predeclared table cells, with the incremental signed effect under variance.
rows=[]
for d in protocol['figure_settings']+protocol['persistent_table_settings']:
 mask=np.ones(len(s),dtype=bool);pmask=np.ones(len(p),dtype=bool)
 for k,value in d.items():mask &= s[k].eq(value);pmask &= p[k].eq(value)
 g=s[mask].set_index('method');pair=p[pmask & p.contrast.eq('learning_at_variance')].iloc[0]
 rows.append(dict(**d,total_observations=int(g.total_observations.iloc[0]),target=float(g.target.iloc[0]),**{m:int(g.loc[m,'rejections']) for m in ORDER},replications=1000,signed_increment=float(pair.delta),paired_ci_lower=float(pair.ci_lower),paired_ci_upper=float(pair.ci_upper)))
t=pd.DataFrame(rows);t.to_csv(OUTPUT/'compact_focus_table.csv',index=False,float_format='%.17g')
lines=['| Design / fit | θ | Cost | Absolute/range | Signed/range | Absolute/variance | Signed/variance | Signed increment at variance (paired 95% CI) |','|---|---:|---:|---:|---:|---:|---:|---|']
for r in rows:
 label=('Two' if r['design']=='two_balanced' else 'Eight')+(' / opposed' if r['fit']=='opposed_shifts' else ' / estimated')
 vals=[label,f"{r['target']:.5f}",str(r['total_observations'])]+[str(r[m]) for m in ORDER]+[f"{r['signed_increment']:.3f} [{r['paired_ci_lower']:.3f}, {r['paired_ci_upper']:.3f}]"]
 lines.append('| '+' | '.join(vals)+' |')
(OUTPUT/'COMPACT_TABLE.md').write_text('All counts are out of 1,000. Confidence intervals are conservative exact pointwise paired intervals; no simultaneous-grid coverage is claimed.\n\n'+'\n'.join(lines)+'\n')
print('Rendered fixed 7 × 2.05 inch figure and all12 predeclared focus table settings.')
