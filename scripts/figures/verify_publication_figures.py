#!/usr/bin/env python3
"""Independent source-to-SVG/table audit; does not import the figure generator."""
import argparse
import csv
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/publication_figure_assets'
NS = {'s': 'http://www.w3.org/2000/svg'}


def csv_rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def svg_axes(name):
    svg = ET.parse(OUT / name).getroot()
    return [g for g in svg.findall('.//s:g', NS) if re.fullmatch(r'axes_\d+', g.get('id', ''))]


def box(axis):
    path = next(g for g in axis.findall('s:g', NS) if g.get('id', '').startswith('patch_')).find('s:path', NS)
    v = [float(x) for x in re.findall(r'[-+]?\d+(?:\.\d+)?', path.get('d'))]
    return min(v[::2]), min(v[1::2]), max(v[::2]), max(v[1::2])


def markers(axis):
    return [(float(u.get('x')), float(u.get('y')), u.get('style'))
            for u in axis.findall('.//s:use', NS)
            if u.get('x') is not None and ('stroke: #0072b2' in u.get('style', '')
                                        or 'stroke: #d55e00' in u.get('style', '')
                                        or 'stroke: #333333; stroke-width: 0.65' in u.get('style', ''))]


def point_error(actual, expected):
    e = max(abs(a-b) for a,b in zip(actual, expected))
    assert e <= 0.000001, (actual, expected, e)
    return e


def main():
    global ROOT, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args = parser.parse_args()
    ROOT, OUT = args.source_root.resolve(), args.output_dir.resolve()
    provenance = json.loads((OUT / 'provenance.json').read_text())
    source_paths = {name: ROOT / receipt['path'] for name, receipt in provenance['sources'].items()}
    scores = csv_rows(source_paths['public_scores'])
    panel = csv_rows(source_paths['spline_panel'])
    spline = {(r['domain'], r['model']): r for r in panel if r['method']=='local_spline_GCM'}
    canonical = json.loads(source_paths['canonical_summary'].read_text())['rows']
    keyed = {(r['regime'], r['alternative'], r['method']):r for r in canonical}
    max_wilson = 0
    for row in canonical:
        n, k, z = row['replications'], row['rejections'], norm.ppf(.975)
        center = (k + z*z/2) / (n+z*z)
        half = z / (n+z*z) * math.sqrt(k*(n-k)/n + z*z/4)
        max_wilson = max(max_wilson, abs(center-half-row['wilson95'][0]), abs(center+half-row['wilson95'][1]))
        assert row['rate'] == k/n
    assert max_wilson < 5e-16

    # Directly inspect the delivered SVG marker positions, not coordinate JSON.
    domain_limits = [('electricity',(-6.5,10.7)),('ratings',(-.8,15.2)),
                     ('retail',(-14.8,11.8)),('portfolios',(-2.6,2.15))]
    max_public_error, public_count = 0, 0
    for axis, (domain, limits) in zip(svg_axes('fig_public_named_comparison.svg'), domain_limits):
        selected = [r for r in scores if r['domain']==domain]
        actual = markers(axis)
        finite = [r for r in selected if r['audit_raw_T']!='']
        assert len(actual) == 2*len(finite)
        x0,y0,x1,y1 = box(axis)
        for i, row in enumerate(selected):
            target_y = y0 + (i+.5)/len(selected)*(y1-y0)
            at_row = [point for point in actual if abs(point[1]-target_y)<1e-6]
            if row['audit_raw_T']=='':
                assert not at_row
                continue
            assert len(at_row)==2
            values = {'#0072b2': float(row['audit_raw_T']),
                      '#d55e00': float(spline[domain,row['model']]['raw_statistic'])}
            for color, value in values.items():
                point = next(p for p in at_row if color in p[2])
                target_x = x0+(value-limits[0])/(limits[1]-limits[0])*(x1-x0)
                max_public_error = max(max_public_error, point_error(point[:2], (target_x,target_y)))
                public_count += 1

    axis = svg_axes('fig_canonical_comparison.svg')[0]
    actual = markers(axis)
    assert len(actual)==30
    bars = [g.find('s:path',NS) for g in axis.findall('s:g',NS) if g.get('id','').startswith('LineCollection_')]
    assert len(bars)==30
    bar_coords = [[float(x) for x in re.findall(r'[-+]?\d+(?:\.\d+)?',p.get('d'))] for p in bars]
    x0,y0,x1,y1 = box(axis)
    methods = ['gcm_coarse','gcm_fine','gcm_spline','extrapolated','extrapolated_beta2','kci_gamma']
    conditions = [('linear','null'),('smooth','null'),('nonlinear','null'),('heavy_tail','null'),('smooth','zero_covariance_dependence')]
    max_canonical_error = 0
    for j, (regime, alternative) in enumerate(conditions):
        for i, method in enumerate(methods):
            row = keyed[regime,alternative,method]
            target_y = y0+(i+(j-2)*.15+.5)/6*(y1-y0)
            convert_x = lambda v: x0+(v+.012)/1.037*(x1-x0)
            matches = [p for p in actual if abs(p[1]-target_y)<1e-6]
            assert len(matches)==1
            max_canonical_error = max(max_canonical_error, point_error(matches[0][:2], (convert_x(row['rate']),target_y)))
            interval = [v for v in bar_coords if abs(v[1]-target_y)<1e-6]
            assert len(interval)==1
            expected = (convert_x(row['wilson95'][0]),target_y,convert_x(row['wilson95'][1]),target_y)
            max_canonical_error = max(max_canonical_error, point_error(interval[0],expected))

    complete = csv_rows(OUT / 'public_41_complete.csv')
    source_keys = {(r['domain'],r['model']) for r in scores}
    assert len(complete)==41 and {(r['domain'],r['model']) for r in complete}==source_keys
    score_map = {(r['domain'],r['model']):r for r in scores}
    for row in complete:
        source = score_map[row['domain'],row['model']]
        fields = {'descriptive_score':'mean_cluster_spearman','mae':'original_scale_mae','mae_status':'mae_status',
                  'mae_units':'mae_units','T_beta2':'audit_raw_T','guarded_BY_p':'audit_guarded_BY_p',
                  'final_decision':'audit_final_label','abstention_reason':'audit_abstention_reason'}
        for output_key, source_key in fields.items():
            assert row[output_key] == source[source_key], (row['model'],output_key)
        assert row['T_spline'] == spline[row['domain'],row['model']]['raw_statistic']
    assert sum(r['final_decision']=='null' for r in complete)==10
    assert sum(r['T_beta2']==r['T_spline']=='' for r in complete)==3
    assert next(r for r in complete if r['model']=='SVD interaction')['mae']==''
    assert [(r['T_beta2'],r['T_spline']) for r in complete if r['model']=='Item mean'] == [(r['T_beta2'],r['T_spline']) for r in complete if r['model']=='User and item bias']
    display = json.loads((OUT / 'public_41_complete.display.json').read_text())
    assert len(display)==41
    table_order = [r for d in ('electricity','ratings','retail','portfolios') for r in complete if r['domain']==d]
    for shown,row in zip(display,table_order):
        fields=shown['values']
        assert shown['domain']==row['domain'] and shown['model']==row['model']
        assert fields[0]==row['display_name']
        for pos,key in [(1,'descriptive_score'),(2,'mae'),(3,'T_beta2'),(4,'T_spline')]:
            assert fields[pos]==('undefined' if row[key]=='' else f'{float(row[key]):.4f}')
        p_source=float(row['guarded_BY_p'])
        assert fields[5]==('1' if p_source==1 else f'{p_source:.3e}')
        assert fields[6]==('NOT_RETAINED' if row['final_decision']=='null' else row['final_decision'].upper())
    import fitz
    with fitz.open(OUT/'public_41_complete.pdf') as document:
        assert len(document)==2
        rendered=' '.join(page.get_text() for page in document)
        assert 'RETAIN' in rendered and 'ABSTAIN' in rendered and 'NOT_RETAINED' in rendered
    provenance = json.loads((OUT/'provenance.json').read_text())
    for receipt in provenance['sources'].values():
        assert hashlib.sha256((ROOT/receipt['path']).read_bytes()).hexdigest()==receipt['sha256']
    for name, digest in provenance['outputs'].items():
        assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==digest, name
    report = {
        'status':'PASS', 'independent_checker': str(Path(__file__).resolve().relative_to(ROOT)),
        'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'checked_all_54_wilson_intervals_max_error':max_wilson,
        'actual_svg_public_markers_checked':public_count, 'actual_svg_canonical_markers_checked':30,
        'actual_svg_canonical_error_bars_checked':30,
        'maximum_svg_public_coordinate_error_pt':max_public_error,
        'maximum_svg_canonical_coordinate_error_pt':max_canonical_error,
        'svg_serialization_tolerance_pt':1e-6,
        'all_41_table_rows_source_matched':True, 'all_41_display_rows_and_rounded_measurements_checked':True,
        'ten_null_strings_preserved':True,'three_undefined_pairs_have_no_numeric_marker':True,
        'coincident_models_have_separate_named_rows':True,'svd_interaction_mae_remains_undefined':True,
        'all_source_and_output_receipts_verified':True,
    }
    (OUT/'independent_coordinate_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
