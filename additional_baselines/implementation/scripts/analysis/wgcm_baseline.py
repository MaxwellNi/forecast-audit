"""Supplementary synthetic comparison; see additional_baselines/README.md
from the package root for assumptions and interpretation limits."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
from scipy.stats import norm
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))
from clustered_calibration import panel, _cont_bin, _fast_resid
OUT = ROOT / 'outputs/theory_simulation/audit_wgcm'
ALPHA = 0.05
ZA = float(norm.ppf(1 - ALPHA))
QF = 32

def _fe_resid(v, firm, month, N, T, folds):
    return _fast_resid(v, [firm, month], [N, T], folds)

def _gcm_residual_products(fr, yr, firm, month, momr, N, T, folds, q=QF):
    idxs = [firm, month, _cont_bin(momr, q)]
    ks = [N, T, int(q)]
    e_f = _fast_resid(fr, idxs, ks, folds)
    e_y = _fast_resid(yr, idxs, ks, folds)
    return e_f * e_y

def _held_out_bin_means(R, mom_bin, folds, q):
    n = len(R)
    mbar_oof = np.zeros(n)
    for f in (0, 1):
        ho = folds != f
        te = folds == f
        bins_ho = mom_bin[ho]
        R_ho = R[ho]
        ssum = np.bincount(bins_ho, weights=R_ho, minlength=q)
        scnt = np.bincount(bins_ho, minlength=q).astype(float)
        with np.errstate(invalid='ignore', divide='ignore'):
            mbar = np.where(scnt > 0, ssum / np.maximum(scnt, 1.0), 0.0)
        mbar_oof[te] = mbar[mom_bin[te]]
    return mbar_oof

def _studentised(S):
    n = len(S)
    return float(np.sqrt(n) * S.mean() / (S.std() + 1e-12))

def wgcm_sign(fr, yr, firm, month, momr, N, T, rng):
    folds = rng.integers(0, 2, size=len(fr))
    R = _gcm_residual_products(fr, yr, firm, month, momr, N, T, folds, QF)
    mom_bin = _cont_bin(momr, QF)
    mbar_oof = _held_out_bin_means(R, mom_bin, folds, QF)
    w = np.sign(mbar_oof)
    S = w * R
    return int(_studentised(S) > ZA)

def wgcm_reg(fr, yr, firm, month, momr, N, T, rng):
    folds = rng.integers(0, 2, size=len(fr))
    R = _gcm_residual_products(fr, yr, firm, month, momr, N, T, folds, QF)
    mom_bin = _cont_bin(momr, QF)
    w = _held_out_bin_means(R, mom_bin, folds, QF)
    S = w * R
    return int(_studentised(S) > ZA)

def gcm_fine(fr, yr, firm, month, momr, N, T, rng):
    folds = rng.integers(0, 2, size=len(fr))
    R = _gcm_residual_products(fr, yr, firm, month, momr, N, T, folds, QF)
    return int(_studentised(R) > ZA)

def main(n_seeds=200, seed0=505, w_overlap=0.8):
    METHODS = {'gcm_fine': gcm_fine, 'wgcm_sign': wgcm_sign, 'wgcm_reg': wgcm_reg}
    order = ['gcm_fine', 'wgcm_sign', 'wgcm_reg']
    rows = []
    times = {m: 0.0 for m in order}
    for regime in ('smooth', 'nonlinear', 'large_n', 'heavy_tail'):
        ns = n_seeds if regime != 'large_n' else 100
        for kind, delta in (('typeI', 0.0), ('power', 0.15)):
            acc = {m: 0 for m in order}
            for s in range(ns):
                rng = np.random.default_rng(seed0 + s)
                fr, yr, firm, month, momr, N, T = panel(regime, rng, delta, w_overlap)
                for m in order:
                    t = time.time()
                    acc[m] += METHODS[m](fr, yr, firm, month, momr, N, T, rng)
                    times[m] += time.time() - t
            row = dict(regime=regime, metric=kind, n_seeds=ns, **{m: round(acc[m] / ns, 3) for m in order})
            rows.append(row)
            print(f'{regime:10s} {kind:6s} (n={ns:3d}) | ' + '  '.join((f'{m}={acc[m] / ns:.3f}' for m in order)), flush=True)
    tI = {r['regime']: r for r in rows if r['metric'] == 'typeI'}
    pw = {r['regime']: r for r in rows if r['metric'] == 'power'}
    summary = {'method': 'WGCM (Weighted Generalised Covariance Measure; Scheidegger-Hoerrmann-Buehlmann, JMLR 2022)', 'headline_variant': 'wgcm_sign (sign of held-out per-momentum-bin product mean)', 'alpha': ALPHA, 'ZA': round(ZA, 4), 'q_resolution': QF, 'conditioning': 'firm + month + momentum(q=32), == GCM-fine nuisance', 'weight': 'function of momentum only, cross-fitted on the held-out fold', 'regimes': list(tI.keys()), 'rows': rows, 'typeI': {m: {r: tI[r][m] for r in tI} for m in order}, 'power': {m: {r: pw[r][m] for r in pw} for m in order}, 'max_typeI': {m: round(max((tI[r][m] for r in tI)), 3) for m in order}, 'min_power': {m: round(min((pw[r][m] for r in pw)), 3) for m in order}, 'total_seconds': {m: round(times[m], 1) for m in order}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'audit_wgcm_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT / 'audit_wgcm.csv', index=False)
    print('\n=== WGCM vs GCM-fine (same q=32 conditioning) ===')
    for m in order:
        print(f"  {m:10s} maxTypeI={summary['max_typeI'][m]:.3f}  minPower={summary['min_power'][m]:.3f}  time={times[m]:.0f}s")
    print(f'\nartifacts -> {OUT}')
if __name__ == '__main__':
    main()
