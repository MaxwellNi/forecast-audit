"""Build a vector sensitivity figure and companion text from recounted decisions."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PACKAGE = Path(__file__).resolve().parents[2]
ROOT = PACKAGE/'rerun_historical_figure'
ROOT.mkdir(exist_ok=False)
(ROOT/'recount.json').write_bytes((PACKAGE/'results/additional_baseline_results/recount.json').read_bytes())
data = json.loads((ROOT/'recount.json').read_text())
assert not data['historical_rate_mismatches'] and not data['native_driver_rate_mismatches']
cells = data['cells']
regimes = ['smooth', 'nonlinear', 'large_n', 'heavy_tail']
regime_names = ['Smooth', 'Nonlinear', 'Larger panel', 'Heavy tails']
settings = [x['setting'] for x in data['ablation_summary']]
labels = ['Default', 'Short ladder', 'Long ladder', r'$\beta=0.5$',
          r'$\beta=1.5$', r'$\beta=2$', 'Equal-width bins', '4 fit sweeps', '24 fit sweeps']


def cell(suite, setting, regime, metric):
    return next(x for x in cells if x['suite'] == suite and x['setting'] == setting
                and x['regime'] == regime and x['legacy_metric'] == metric)


def count(x):
    return f"{x['rejections']}/{x['replicates']}"


def rate_ci(x):
    return f"{x['rate']:.3f} [{x['wilson95_low']:.3f}, {x['wilson95_high']:.3f}]"


plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'DejaVu Sans'})
fig = plt.figure(figsize=(6.9, 5.9), constrained_layout=True)
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.7])
ax = fig.add_subplot(gs[0, :])
for method, shift, color, label in [('fixed_beta1', -.09, '#22577A', r'Fixed $\beta=1$'),
                                    ('adaptive_heuristic', .09, '#C8553D', 'Adaptive heuristic')]:
    rows = [cell('adaptive_beta', method, r, 'typeI') for r in regimes]
    y = np.asarray([r['rate'] for r in rows])
    errs = np.asarray([[r['rate']-r['wilson95_low'] for r in rows],
                       [r['wilson95_high']-r['rate'] for r in rows]])
    ax.errorbar(np.arange(4)+shift, y, yerr=errs, fmt='o', ms=4,
                capsize=3, color=color, label=label)
ax.set(xticks=np.arange(4), xticklabels=regime_names, ylim=(0,1.04),
       ylabel='Zero-signal rejection rate', title='(a) Fixed and adaptive exponents: 50 paired draws per regime')
ax.legend(frameon=False, loc='upper left')
ax.grid(axis='y', color='.9', linewidth=.6)
ax.spines[['top','right']].set_visible(False)
for j, (metric,title) in enumerate([('typeI', '(b) Zero-signal rejection'), ('power', '(c) Positive-signal rejection')]):
    ax = fig.add_subplot(gs[1,j])
    array = np.asarray([[cell('ablation', s, r, metric)['rate'] for r in regimes] for s in settings])
    ax.pcolormesh(np.arange(5)-.5, np.arange(10)-.5, array, vmin=0, vmax=1,
                  cmap='Blues', shading='flat', rasterized=False)
    ax.set_ylim(8.5,-.5)
    ax.set(xticks=np.arange(4), xticklabels=['Smooth', 'Nonlin.', 'Larger', 'Heavy'],
           yticks=np.arange(9), yticklabels=labels if j == 0 else ['']*9, title=title)
    ax.tick_params(length=0)
    for a in range(9):
        for b in range(4):
            ax.text(b,a,f'{array[a,b]:.3f}',ha='center',va='center',fontsize=7,
                    color='white' if array[a,b] > .62 else '#14213D')
    for spine in ax.spines.values():
        spine.set_visible(False)
fig.savefig(ROOT/'historical_sensitivity.pdf', metadata={'CreationDate':None,'ModDate':None})
fig.savefig(ROOT/'historical_sensitivity.svg', metadata={'Date':None})
fig.savefig(ROOT/'historical_sensitivity.png', dpi=180)
plt.close(fig)
