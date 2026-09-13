#!/usr/bin/env python3
"""Plot all public forecasts on matched evaluation observations.

Read the recorded four-configuration model table. Export a lossless 82-row
companion for the two middle-fold configurations, with support and fallback
fractions derived from the recorded counts. No observations or decisions are
selected by their results. This is a sensitivity comparison, not calibration.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DISPLAY = {
    'dlinear': 'DLinear', 'drift': 'Drift',
    'histgbr_lag': 'Histogram gradient boosting',
    'holt_winters': 'Holt-Winters', 'lasso_lag': 'Lasso lag regression',
    'linear_lag': 'Linear lag regression', 'naive_1': 'One-day persistence',
    'ridge_lag': 'Ridge lag regression', 'seasonal_naive_7': 'Seven-day persistence',
    'ses': 'Exponential smoothing', 'theta': 'Theta',
    'Item nearest neighbours': 'Item nearest neighbors',
    'Past price calendar and history': 'Price, calendar and history',
    'Past price and calendar': 'Price and calendar',
    'Exponential twelve-month mean': '12-month exponential mean',
    'Seasonal twelve-month lag': '12-month seasonal lag',
    'Trailing three-month return': 'Trailing 3-month return',
    'Trailing twelve-month return': 'Trailing 12-month return',
}
PANELS = [
    ('electricity', 'Electricity: 438 days', (-4., 12.), [0,5,10]),
    ('ratings', 'MovieLens: 168 user clusters', (-.5, 14.), [0,5,10]),
    ('retail', 'M5 retail: 17 weeks', (-21., 15.), [-20,-10,0,10]),
    ('portfolios', 'OSAP forecasts: 79 months', (-3., 2.1), [-2,-1,0,1,2]),
]
CONFIGS = ('complementary_middle', 'directional_middle')

def finite(value):
    if value == '': return None
    out = float(value)
    if not math.isfinite(out): raise ValueError('Unexpected nonfinite recorded value')
    return out

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.input.open(newline='')))
    assert len(rows)==164
    by_key={(r['domain'],r['model'],r['configuration']):r for r in rows}
    assert len(by_key)==164
    keys=sorted({(r['domain'],r['model']) for r in rows})
    assert len(keys)==41
    paired=[]
    for domain,model in keys:
        a,b=(by_key[domain,model,c] for c in CONFIGS)
        for col in ['evaluation_rows','observed_groups','cohort_sha256','original_policy_abstain','original_guard_reason','family_size','beta','frequency','lag']:
            assert a[col]==b[col], (domain,model,col)
        all_n=int(by_key[domain,model,'complementary_all']['evaluation_rows'])
        for r in [a,b]:
            out=dict(r)
            out['display_name']=DISPLAY.get(model,model)
            out['full_evaluation_rows']=str(all_n)
            out['support_fraction']=int(r['evaluation_rows'])/all_n
            for col in ['forecast_pooled_fallback_rows','outcome_pooled_fallback_rows','forecast_unseen_entity_rows','outcome_unseen_entity_rows']:
                out[col.replace('_rows','_fraction')]=int(r[col])/int(r['evaluation_rows'])
            mean,se,t=(finite(r[c]) for c in ['mean_product','standard_error','raw_statistic'])
            if t is not None: assert math.isclose(mean/se,t,rel_tol=2e-14,abs_tol=2e-14)
            paired.append(out)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    csv_path=args.output_dir/'public_directional_models.csv'
    with csv_path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
    plt.rcParams.update({'font.family':'Liberation Serif','font.size':7.6,
        'mathtext.fontset':'stix','axes.titlesize':8.4,'xtick.labelsize':7.4,
        'ytick.labelsize':7.4,'pdf.fonttype':42,'ps.fonttype':42,
        'axes.linewidth':.45,'xtick.major.width':.45,'ytick.major.width':0,
        'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,
        'svg.hashsalt':'public-directional-models'})
    fig,axes=plt.subplots(2,2,figsize=(7.15,4.35))
    fig.subplots_adjust(left=.24,right=.995,top=.91,bottom=.10,hspace=.34,wspace=1.48)
    lines=[]
    for ax,(domain,title,limits,ticks) in zip(axes.flat,PANELS):
        models=sorted(m for d,m in keys if d==domain)
        assert len(models)==int(by_key[domain,models[0],CONFIGS[0]]['family_size'])
        ax.set_xlim(*limits); ax.set_xticks(ticks); ax.set_ylim(len(models)-.5,-.75)
        ax.axvline(0,color='#CCCCCC',linewidth=.5,zorder=0)
        ax.axvline(NormalDist().inv_cdf(.95),color='#777777',linewidth=.55,linestyle=':',zorder=0)
        labels=[]
        for y,model in enumerate(models):
            a,b=(by_key[domain,model,c] for c in CONFIGS)
            x1,x2=finite(a['raw_statistic']),finite(b['raw_statistic'])
            assert (x1 is None)==(x2 is None), (domain,model)
            labels.append(DISPLAY.get(model,model)+('$^{\\dagger}$' if a['original_policy_abstain']=='True' else ''))
            if x1 is not None and x2 is not None:
                assert limits[0]<min(x1,x2)<=max(x1,x2)<limits[1]
                ax.plot([x1,x2],[y,y],color='#AAAAAA',linewidth=.6,zorder=1)
            for x,color,marker,fill in [(x1,'#333333','o','white'),(x2,'#0072B2','s','#0072B2')]:
                if x is not None:
                    ax.plot(x,y,marker=marker,markersize=3.25,color=color,
                            markerfacecolor=fill,markeredgewidth=.65,zorder=2)
                else:
                    ax.text(.045,y,'undefined',transform=ax.get_yaxis_transform(),
                            va='center',fontsize=6.9,color='#666666',fontstyle='italic')
                    break
            lines.append({'domain':domain,'model':model,'complementary':x1,'directional':x2,
                          'guarded':a['original_policy_abstain']=='True'})
        ax.set_yticks(range(len(models)),labels);ax.tick_params(axis='y',length=0,pad=4)
        ax.set_title(title,pad=5)
    handles=[Line2D([],[],color='#333333',marker='o',markerfacecolor='white',
                    linestyle='none',markersize=4,label='Complementary fitting'),
             Line2D([],[],color='#0072B2',marker='s',linestyle='none',markersize=4,label='Directional fitting')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.55,1.0),ncol=2,
               frameon=False,handletextpad=.4,columnspacing=1.8)
    fig.text(.53,.02,r'Raw residual statistic $T$; dotted line: nominal one-sided 5% normal threshold.',
             ha='center',fontsize=7.5)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    # All text is checked against the figure boundary, including long model names.
    for text in fig.findobj(match=matplotlib.text.Text):
        if not text.get_visible() or not text.get_text(): continue
        box=text.get_window_extent(renderer)
        assert box.x0>=-1 and box.y0>=-1 and box.x1<=fig.bbox.width+1 and box.y1<=fig.bbox.height+1,(text.get_text(),box)
    for ext in ['pdf','png','svg']:
        kwargs={'dpi':240} if ext=='png' else {}
        if ext=='pdf': kwargs['metadata']={'Creator':'','Producer':'','CreationDate':None,'ModDate':None}
        if ext=='svg': kwargs['metadata']={'Creator':'','Date':None}
        fig.savefig(args.output_dir/f'fig_public_directional.{ext}',**kwargs)
    plt.close(fig)
    finite_pairs=sum(x['complementary'] is not None and x['directional'] is not None for x in lines)
    assert len(lines)==41 and finite_pairs==39
    guard_count=sum(x['guarded'] for x in lines)
    receipt={'input_sha256':sha(args.input),'models':41,'rows':len(paired),
             'finite_pairs':finite_pairs,'undefined_pairs':41-finite_pairs,
             'guarded_models':guard_count,'configurations':list(CONFIGS),
             'all_support_and_cohort_checks_passed':True,
             'plotted_values':lines,'files':{p.name:sha(p) for p in args.output_dir.iterdir() if p.suffix in ['csv','pdf','png','svg']}}
    (args.output_dir/'figure_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ['models','rows','finite_pairs','undefined_pairs','guarded_models']}))
if __name__=='__main__': main()
