#!/usr/bin/env python3
"""Deterministic publication figures and the complete 41-model companion.

Run from any directory. All measurements are loaded from recorded sources; no
measured value is jittered, rounded before plotting, dropped, or simulated.
Use --output-dir to select a separate output directory. The default writes
into results/publication_figure_assets and replaces its generated files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
from pathlib import Path
from statistics import NormalDist

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    'prior_plot_inputs': 'results/figure_evidence/plot_inputs.json',
    'canonical_summary': 'results/canonical_beta2/summary.json',
    'public_scores': 'results/public_score_link/public_all_41_models.csv',
    'spline_panel': 'results/spline_panel/public_all_models.csv',
}
BLUE, ORANGE = '#0072B2', '#D55E00'
DISPLAY_NAMES = {
    'Item nearest neighbours': 'Item nearest neighbors',
    'dlinear': 'DLinear', 'drift': 'Drift', 'histgbr_lag': 'Histogram gradient boosting',
    'holt_winters': 'Holt-Winters', 'lasso_lag': 'Lasso lag regression',
    'linear_lag': 'Linear lag regression', 'naive_1': 'One-day persistence',
    'ridge_lag': 'Ridge lag regression', 'seasonal_naive_7': 'Seven-day persistence',
    'ses': 'Exponential smoothing', 'theta': 'Theta',
}
DOMAINS = [
    ('electricity', 'Electricity', (-6.5, 10.7), [-5, 0, 5, 10]),
    ('ratings', 'MovieLens ratings', (-.8, 15.2), [0, 5, 10, 15]),
    ('retail', 'M5 retail', (-14.8, 11.8), [-10, 0, 10]),
    ('portfolios', 'OSAP forecasts', (-2.6, 2.15), [-2, -1, 0, 1, 2]),
]
METHODS = [
    ('gcm_coarse', 'GCM: 8 bins'), ('gcm_fine', 'GCM: 32 bins'),
    ('gcm_spline', 'GCM: splines'), ('extrapolated', r'Extrapolation ($\beta=1$)'),
    ('extrapolated_beta2', r'Extrapolation ($\beta=2$)'), ('kci_gamma', 'KCI'),
]
CONDITIONS = [
    ('linear', 'null', 'Linear', BLUE, 'o', 'white'),
    ('smooth', 'null', 'Smooth', BLUE, 'o', BLUE),
    ('nonlinear', 'null', 'Quadratic', ORANGE, 's', 'white'),
    ('heavy_tail', 'null', r'Heavy tail ($t_5$)', ORANGE, 's', ORANGE),
    ('smooth', 'zero_covariance_dependence', 'Zero covariance', '#333333', 'D', 'white'),
]
plt.rcParams.update({
    'font.family': 'Liberation Serif', 'font.size': 8.5,
    'mathtext.fontset': 'stix', 'axes.labelsize': 8.5,
    'axes.titlesize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .5, 'xtick.major.width': .5, 'ytick.major.width': .5,
    'svg.hashsalt': 'icdm-publication-20260907b',
})


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    # csv deliberately preserves literal "null" decisions and empty cells.
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def number(value):
    if value is None or value == '':
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'Unexpected non-finite source value: {value!r}')
    return result


def equal(a, b):
    a, b = number(a), number(b)
    return a == b or (a is not None and b is not None and math.isclose(a, b, rel_tol=2e-15, abs_tol=2e-15))


def wilson(k, n):
    z = NormalDist().inv_cdf(.975)
    p, zz = k / n, z*z
    center = (p + zz / (2*n)) / (1 + zz/n)
    half = z * math.sqrt(p*(1-p)/n + zz/(4*n*n)) / (1 + zz/n)
    return center-half, center+half


def sources_and_data(source_root, source_paths):
    assert set(source_paths) == set(SOURCES), "Source map must have exactly the four documented keys"
    paths = {name: source_root / value for name, value in source_paths.items()}
    old = json.loads(paths['prior_plot_inputs'].read_text())
    canonical = json.loads(paths['canonical_summary'].read_text())
    scores, spline = read_csv(paths['public_scores']), read_csv(paths['spline_panel'])
    assert canonical['method_cells'] == len(canonical['rows']) == 54
    assert len(scores) == 41 and len(spline) == 123
    key = lambda row: (row['domain'], row['model'])
    scores_by_key = {key(row): row for row in scores}
    prior_by_key = {key(row): row for row in old['public_comparison_rows']}
    spline_by_key = {key(row): row for row in spline if row['method'] == 'local_spline_GCM'}
    beta_by_key = {key(row): row for row in spline if row['method'] == 'extrapolated_beta_2'}
    assert len(scores_by_key) == len(spline_by_key) == len(beta_by_key) == len(prior_by_key) == 41
    assert set(scores_by_key) == set(spline_by_key) == set(beta_by_key) == set(prior_by_key)
    combined = []
    for score in scores:
        k = key(score)
        sp, beta, prior = spline_by_key[k], beta_by_key[k], prior_by_key[k]
        t_beta, t_spline = number(score['audit_raw_T']), number(sp['raw_statistic'])
        assert equal(t_beta, beta['raw_statistic']) and equal(t_beta, prior['extrapolated_T']), k
        assert equal(t_spline, prior['spline_T']), k
        assert equal(score['audit_guarded_BY_p'], beta['policy_by_adjusted_p']), k
        assert (t_beta is None and t_spline is None) == prior['undefined'], k
        assert (t_beta is None) == (t_spline is None), k
        guard = beta['redundancy_abstain'] == 'True'
        # Prior figure also classified zero forecast-rank variation as a guard;
        # new dagger explicitly identifies finite copy/weak-order guards only.
        if t_beta is not None:
            assert guard == prior['copy_or_order_guard'], k
        combined.append({
            'domain': score['domain'], 'model': score['model'],
            'display_name': DISPLAY_NAMES.get(score['model'], score['model']),
            'descriptive_score': score['mean_cluster_spearman'],
            'mae': score['original_scale_mae'], 'mae_status': score['mae_status'],
            'mae_units': score['mae_units'], 'T_beta2': score['audit_raw_T'],
            'T_spline': sp['raw_statistic'], 'guarded_BY_p': score['audit_guarded_BY_p'],
            'final_decision': score['audit_final_label'],
            'abstention_reason': score['audit_abstention_reason'],
            'finite_copy_or_order_guard': guard and t_beta is not None,
            'undefined': t_beta is None,
        })
    assert sum(row['undefined'] for row in combined) == 3
    assert sum(row['finite_copy_or_order_guard'] for row in combined) == 5
    assert sum(row['final_decision'] == 'null' for row in combined) == 10
    svd = scores_by_key['ratings', 'SVD interaction']
    assert svd['original_scale_mae'] == '' and svd['mae_status'] == 'interaction_only_score_not_rating_prediction'
    canonical_by_key = {(row['regime'], row['alternative'], row['method']): row for row in canonical['rows']}
    assert len(canonical_by_key) == 54
    for row in old['canonical_rows']:
        new = canonical_by_key[row['regime'], row['alternative'], row['method']]
        for name in ('n', 'replications', 'rejections', 'rate', 'wilson95'):
            assert new[name] == row[name], (name, row)
    for row in canonical['rows']:
        assert row['rate'] == row['rejections'] / row['replications']
        assert all(abs(a-b) < 3e-16 for a, b in zip(wilson(row['rejections'], row['replications']), row['wilson95']))
    receipts = {name: {'path': source_paths[name], 'sha256': digest(path), 'bytes': path.stat().st_size}
                for name, path in paths.items()}
    return canonical['rows'], combined, receipts


def save_figure(fig, out, name, data):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    min_size = 999
    for artist in fig.findobj(match=matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        box = artist.get_window_extent(renderer)
        min_size = min(min_size, artist.get_fontsize())
        if box.width and box.height and (box.x0 < -.5 or box.y0 < -.5 or box.x1 > fig.bbox.width + .5 or box.y1 > fig.bbox.height + .5):
            outside.append(artist.get_text())
    assert not outside, (name, outside)
    assert min_size >= 8
    fig.savefig(out / f'{name}.pdf', metadata={'CreationDate': None, 'ModDate': None, 'Creator': 'Matplotlib'})
    fig.savefig(out / f'{name}.svg', metadata={'Date': None})
    fig.savefig(out / f'{name}.png', dpi=180)
    data.update({'canvas_inches': list(fig.get_size_inches()), 'minimum_font_pt': min_size,
                 'text_outside_canvas': outside})
    plt.close(fig)
    return data


def canonical_plot(rows, out):
    lookup = {(row['regime'], row['alternative'], row['method']): row for row in rows}
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    fig.subplots_adjust(left=.365, right=.98, bottom=.19, top=.81)
    plotted = []
    for j, (regime, alternative, label, color, marker, fill) in enumerate(CONDITIONS):
        for i, (method, _) in enumerate(METHODS):
            row = lookup[regime, alternative, method]
            rate, (lo, hi), y = row['rate'], row['wilson95'], i + (j-2)*.15
            ax.errorbar(rate, y, xerr=[[max(0, rate-lo)], [max(0, hi-rate)]],
                        fmt=marker, ms=3.0, mfc=fill, mec=color, mew=.65,
                        color=color, elinewidth=.65, capsize=0, zorder=3)
            plotted.append({**{name: row[name] for name in ('regime', 'alternative', 'method', 'rejections', 'replications', 'rate', 'wilson95')},
                            'x': rate, 'y_categorical': y})
    for y in np.arange(.5, 5.5):
        ax.axhline(y, color='.9', lw=.5, zorder=0)
    ax.axvline(.05, color='.35', ls=':', lw=.65, zorder=1)
    ax.set(xlim=(-.012, 1.025), ylim=(5.5, -.5), xlabel='Rejection rate')
    ax.set_yticks(range(6), [label for _, label in METHODS])
    ax.set_xticks([0, .25, .5, .75, 1], ['0', '0.25', '0.50', '0.75', '1'])
    ax.tick_params(axis='y', length=0, pad=4)
    ax.tick_params(axis='x', length=2.5, pad=2)
    handles = [Line2D([0], [0], marker=m, ms=3.7, color=c, mfc=f, lw=.65, mew=.65, label=label)
               for _, _, label, c, m, f in CONDITIONS]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .994), ncol=3,
               frameon=False, handlelength=1.0, handletextpad=.4, columnspacing=.8, labelspacing=.2)
    fig.text(.51, .018, 'Zero covariance is an alternative for KCI.', ha='center', fontsize=8)
    return save_figure(fig, out, 'fig_canonical_comparison', {
        'plotted_cells': plotted, 'all_source_cells': len(rows), 'reference_line': .05,
        'uncertainty': 'Unrounded recorded pointwise 95% Wilson intervals; no simultaneous-coverage claim.',
        'categorical_positions': 'Five conditions occupy separate subrows within each method; no measured-axis jitter.',
    })


def public_plot(rows, out):
    # Each name owns its own categorical row; coincident measurements remain
    # individually visible without changing either statistic.
    fig = plt.figure(figsize=(7.0, 5.0))
    panels = []
    plot_boxes = [(.242, .575, .244, .33), (.747, .575, .241, .33),
                  (.242, .135, .244, .33), (.747, .135, .241, .33)]
    threshold = NormalDist().inv_cdf(.95)
    for (domain, title, limits, ticks), box in zip(DOMAINS, plot_boxes):
        selected = [row for row in rows if row['domain'] == domain]
        ax = fig.add_axes(box)
        ax.axvline(0, color='.82', lw=.55, zorder=0)
        ax.axvline(threshold, color='.40', lw=.65, ls=':', zorder=0)
        labels, plotted = [], []
        for i, row in enumerate(selected):
            labels.append(row['display_name'] + (r'$^{\dagger}$' if row['finite_copy_or_order_guard'] else ''))
            a, b = number(row['T_beta2']), number(row['T_spline'])
            if a is None:
                # Axes-fraction annotation is not a zero-valued observation.
                ax.text(.52, i, 'undefined (both)', transform=ax.get_yaxis_transform(),
                        va='center', ha='center', fontsize=8, color='.35', style='italic')
            else:
                assert limits[0] < min(a, b) and max(a, b) < limits[1], row
                ax.plot([a, b], [i, i], color='.64', lw=.7, zorder=1)
                ax.plot(b, i, 'o', ms=5.2, mfc='white', mec=ORANGE, mew=.85, zorder=2)
                ax.plot(a, i, 'o', ms=2.8, mfc=BLUE, mec=BLUE, mew=.45, zorder=3)
            plotted.append({'domain': domain, 'model': row['model'], 'display_name': row['display_name'], 'x_beta2': a, 'x_spline': b,
                            'y_categorical': i, 'undefined': row['undefined'],
                            'finite_copy_or_order_guard': row['finite_copy_or_order_guard']})
        ax.set(xlim=limits, ylim=(len(selected)-.5, -.5))
        ax.set_xticks(ticks)
        ax.set_yticks(range(len(selected)), labels)
        ax.tick_params(axis='y', length=0, pad=5)
        ax.tick_params(axis='x', length=2.5, pad=2)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('.55')
        ax.set_title(f'{title} ({len(selected)} models)', loc='left', pad=8)
        # Measure adjacent model-name boxes; no two named rows may collide.
        fig.canvas.draw()
        label_boxes = [label.get_window_extent(fig.canvas.get_renderer()) for label in ax.get_yticklabels()]
        assert all(label_boxes[i].y0 >= label_boxes[i+1].y1 - .1 for i in range(len(label_boxes)-1)), domain
        panels.append({'domain': domain, 'x_limits': limits, 'x_ticks': ticks, 'models': plotted})
    handles = [Line2D([0], [0], marker='o', linestyle='none', ms=3.2, mfc=BLUE, mec=BLUE, label=r'Extrapolation ($\beta=2$)'),
               Line2D([0], [0], marker='o', linestyle='none', ms=5.2, mfc='white', mec=ORANGE, mew=.85, label='Spline GCM')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.52, 1.0), ncol=2,
               frameon=False, handletextpad=.5, columnspacing=1.7)
    fig.text(.52, .068, r'Raw residual statistic $T$ (domain scales differ); dotted line: one-sided 5% threshold.',
             ha='center', fontsize=8.5)
    fig.text(.52, .025, r'$\dagger$ Copy / weak-order guard: abstain. All 41 models shown; three pairs are undefined.',
             ha='center', fontsize=8.5)
    return save_figure(fig, out, 'fig_public_named_comparison', {
        'panels': panels, 'reference_line': threshold, 'total_models': 41,
        'finite_pairs': 38, 'undefined_pairs': 3, 'finite_copy_or_order_guards': 5,
        'uncertainty': 'Observed raw statistics only; no confidence intervals implied.',
        'categorical_positions': 'One original model per named row; exact coordinates and no numeric jitter.',
    })


def formatted(value, digits=4, p_value=False):
    if value == '': return 'undefined'
    v = float(value)
    return ('1' if v == 1 else f'{v:.3e}') if p_value else f'{v:.{digits}f}'


def table_companion(rows, out):
    from matplotlib.backends.backend_pdf import PdfPages
    import textwrap
    columns = ['domain', 'model', 'display_name', 'descriptive_score', 'mae', 'mae_status', 'mae_units', 'T_beta2', 'T_spline',
               'guarded_BY_p', 'final_decision', 'abstention_reason', 'finite_copy_or_order_guard', 'undefined']
    with (out / 'public_41_complete.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
    display = []
    for domain in ('electricity','ratings','retail','portfolios'):
        for row in rows:
            if row['domain'] != domain: continue
            values = [row['display_name'], formatted(row['descriptive_score']), formatted(row['mae']),
                      formatted(row['T_beta2']), formatted(row['T_spline']),
                      formatted(row['guarded_BY_p'], p_value=True), ('NOT_RETAINED' if row['final_decision'] == 'null' else row['final_decision'].upper())]
            display.append({'domain':domain,'model':row['model'],'values':values,
                            'finite_guard':row['finite_copy_or_order_guard']})
    (out/'public_41_complete.display.json').write_text(json.dumps(display,indent=2)+'\n')
    headings=['Model','Score','MAE','T (beta=2)','T (spline)','Guarded BY p','Decision']
    markdown=['# Complete public-model comparison','','All 41 models under the original complementary-fold protocol. Decisions use the within-domain guarded beta=2 BY diagnostic screen. See ../../ERRATA.md for subsequent directional sensitivity.','']
    descriptions = [
      'Score: mean within-cluster Spearman correlation over clusters where it is defined; descriptive only. MAE: original-scale mean absolute error, with domain-specific units. The CSV retains full stored precision.',
      'RETAIN is the operational diagnostic screen at 0.05; real-panel calibration is not established. NOT_RETAINED means non-rejection; original CSV null codes denote the same decision. ABSTAIN denotes a guard or an undefined audit. An asterisk marks the five finite copy or weak-order guards.',
      'Global mean and User mean have undefined descriptive scores and statistics. SVD interaction is an interaction-only score and has no rating-scale MAE. Training series mean has an undefined audit scale. All three undefined statistic pairs remain in their model families.',
    ]
    titles={'electricity':'Electricity (MAE: original UCI load units)', 'ratings':'Ratings (MAE: rating points)',
            'retail':'Retail (MAE: weekly sales units)', 'portfolios':'Portfolios (MAE: percentage points of return)'}
    with PdfPages(out/'public_41_complete.pdf',metadata={'Title':'Complete public-model comparison','Creator':'Matplotlib','CreationDate':None,'ModDate':None}) as pdf:
        for page, domains in enumerate([('electricity','ratings'),('retail','portfolios')],1):
            fig=plt.figure(figsize=(11.7,8.3)); ax=fig.add_axes([.035,.30,.93,.60]);ax.axis('off')
            fig.text(.035,.955,'Complete public-model comparison',fontsize=16,weight='bold')
            fig.text(.965,.955,f'{page} / 2',ha='right',fontsize=10)
            fig.text(.035,.92,'All 41 scored models; complementary-fold fits and guarded within-domain beta=2 BY decisions.',fontsize=10)
            cells=[]; section_indices=[]
            for domain in domains:
                section_indices.append(len(cells)+1);cells.append([titles[domain]]+['']*6)
                markdown += ['## '+titles[domain], '', '| '+' | '.join(headings)+' |','| '+' | '.join(['---']*7)+' |']
                for row in display:
                    if row['domain']!=domain:continue
                    values=list(row['values']);values[0]+=' *' if row['finite_guard'] else ''
                    cells.append(values);markdown.append('| '+' | '.join(values)+' |')
                markdown.append('')
            table=ax.table(cellText=cells,colLabels=headings,cellLoc='right',colLoc='right',
                           colWidths=[.30,.10,.11,.12,.12,.14,.11],bbox=[0,0,1,1])
            table.auto_set_font_size(False);table.set_fontsize(8.7)
            for (r,c),cell in table.get_celld().items():
                cell.set_edgecolor('#D4D4D4');cell.set_linewidth(.35)
                if c==0:cell.get_text().set_ha('left')
                if r==0:cell.set_facecolor('#E7EDF2');cell.get_text().set_weight('bold')
                if r in section_indices:
                    cell.set_facecolor('#F0F2F4');cell.get_text().set_weight('bold');cell.get_text().set_fontsize(8)
            y=.255
            for paragraph in descriptions:
                wrapped=textwrap.fill(paragraph,width=157)
                fig.text(.035,y,wrapped,fontsize=8.4,va='top',linespacing=1.3)
                y-=.024*(len(wrapped.splitlines())+1)
            pdf.savefig(fig);plt.close(fig)
    (out/'public_41_complete.md').write_text('\n'.join(markdown+descriptions)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=ROOT, help='Root used to resolve source-map paths')
    parser.add_argument('--sources-json', type=Path, help='JSON object mapping the four documented source names to paths')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/publication_figure_assets')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source_root = args.source_root.resolve()
    source_paths = SOURCES if args.sources_json is None else json.loads((source_root / args.sources_json).read_text())
    rows, public, receipts = sources_and_data(source_root, source_paths)
    figures = {'canonical': canonical_plot(rows, out), 'public_named': public_plot(public, out)}
    table_companion(public, out)
    import fitz
    companion = fitz.open(out / 'public_41_complete.pdf')
    assert len(companion) == 2
    for i, page in enumerate(companion, 1):
        page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(out / f'public_41_complete_page{i}.png')
    (out / 'plot_coordinates.json').write_text(json.dumps(figures, indent=2) + '\n')
    provenance = {
        'generator': str(Path(__file__).resolve().relative_to(source_root)),
        'generator_sha256': digest(Path(__file__).resolve()), 'sources': receipts,
        'runtime': {'python': platform.python_version(), 'matplotlib': matplotlib.__version__, 'numpy': np.__version__},
        'checks': {'all_54_canonical_rates_and_wilson_intervals_recomputed': True,
                   'all_45_original_canonical_cells_unchanged': True,
                   'all_41_names_exactly_matched_across_three_sources': True,
                   'all_38_beta2_and_spline_pairs_matched_to_prior_plot_inputs': True,
                   'all_41_beta2_statistics_and_guarded_BY_p_matched_to_spline_panel': True,
                   'three_undefined_pairs_retained': True, 'ten_literal_null_decisions_retained': True,
                   'svd_interaction_mae_explicitly_undefined': True,
                   'minimum_figure_font_pt': 8, 'adjacent_model_labels_do_not_overlap': True},
        'outputs': {name + suffix: digest(out / (name + suffix))
                    for name in ('fig_canonical_comparison', 'fig_public_named_comparison')
                    for suffix in ('.pdf', '.svg', '.png')},
    }
    provenance['outputs']['public_41_complete.csv'] = digest(out / 'public_41_complete.csv')
    provenance['outputs']['public_41_complete.md'] = digest(out / 'public_41_complete.md')
    provenance['outputs']['public_41_complete.display.json'] = digest(out / 'public_41_complete.display.json')
    provenance['outputs']['public_41_complete.pdf'] = digest(out / 'public_41_complete.pdf')
    provenance['outputs']['plot_coordinates.json'] = digest(out / 'plot_coordinates.json')
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'output_directory': str(out), 'figure_sizes_inches': {k: v['canvas_inches'] for k, v in figures.items()},
                      'public_models': len(public), 'canonical_plotted_cells': len(figures['canonical']['plotted_cells'])}, indent=2))


if __name__ == '__main__':
    main()
