"""Plot same-budget certificate comparisons from the complete summary table."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, NullFormatter
import pandas as pd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--summary',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(args.summary)
    plt.rcParams.update({'font.family':'Liberation Serif','font.size':8,'mathtext.fontset':'stix','axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(7.15,2.28),sharey=True)
    records=[]
    for ax,design,title in zip(axes,['two_balanced','eight_rare'],[r'(a) Two balanced categories, $\vartheta=0.010$',r'(b) Eight categories with a rare cell, $\vartheta=0.0137$']):
        sub=frame[(frame.design==design)&(frame.signal==.25)&(frame.fit=='estimated')&(frame.validation_pairs==8192)]
        table=sub.pivot(index='total_observations',columns='method',values='power').sort_index(); x=table.index.to_numpy()/1000
        ax.plot(x,table.absolute_range,'--s',color='.48',lw=1,ms=4,mfc='white',label='Absolute allowance + range')
        ax.plot(x,table.signed_range,'-.^',color='#b56720',lw=1,ms=4,mfc='white',label='Signed allowance + range')
        ax.plot(x,table.signed_variance,'-o',color='#2166ac',lw=1.3,ms=3.5,label='Signed + grouped-U variance',zorder=4)
        ax.plot(x,table.independent_triples,':s',color='.35',lw=1,ms=7,mfc='none',mew=.8,label='Signed + independent triples',zorder=5)
        ax.plot(x,table.pooled_variance,'--x',color='.15',lw=.8,ms=4,mew=.8,label='Signed + pooled-U variance',zorder=6)
        ax.set_xscale('log');ax.set_xticks(x,[f'{v:.1f}' for v in x]);ax.set_ylim(-.04,1.04);ax.set_yticks([0,.5,1]);ax.set_title(title,loc='left',pad=5)
        ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_formatter(NullFormatter()); ax.set_xlabel('Total observations (thousands)');ax.grid(axis='y',color='.9',lw=.5);ax.set_axisbelow(True)
        records.extend(sub.to_dict('records'))
    axes[0].set_ylabel('Detection probability')
    handles,labels=axes[0].get_legend_handles_labels();order=[0,1,2,3,4]
    fig.legend([handles[i] for i in order],[labels[i] for i in order],ncol=3,loc='upper center',bbox_to_anchor=(.52,1.015),frameon=False,columnspacing=1.7,handlelength=2.7)
    fig.subplots_adjust(left=.075,right=.99,bottom=.20,top=.69,wspace=.19)
    for ext in ['pdf','svg','png']:
        kw={'dpi':240} if ext=='png' else {'metadata':{'CreationDate':None,'ModDate':None,'Creator':None,'Producer':None}} if ext=='pdf' else {'metadata':{'Date':None,'Creator':None}}
        fig.savefig(args.output_dir/f'fig_certificate_efficiency.{ext}',**kw)
    (args.output_dir/'fig_certificate_efficiency.json').write_text(json.dumps({'records':records,'method_markers':{'absolute_range':'open square','signed_range':'open triangle','signed_variance':'filled circle','independent_triples':'large open square','pooled_variance':'cross'},'coincident_markers_are_not_offset':True,'replications_per_point':1000,'canvas_inches':[7.15,2.28],'minimum_font_points':8,'zero_observed_null_rejections_is_not_zero_risk':True},indent=2)+'\n')
    plt.close(fig)
if __name__=='__main__': main()
