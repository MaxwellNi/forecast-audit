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
OUT = ROOT / 'outputs/theory_simulation/pcm'
ALPHA = 0.05
ZA = float(norm.ppf(1 - ALPHA))
QR = 24
QZ = 8
QX = 3

def _fe_resid(v, firm, month, N, T, folds):
    return _fast_resid(v, [firm, month], [N, T], folds)

def _clustered_se(s, month):
    order = np.argsort(month, kind='stable')
    ms = month[order]
    bnd = np.r_[0, np.flatnonzero(np.diff(ms)) + 1]
    csum = np.add.reduceat(s[order], bnd)
    G = len(csum)
    cm = csum.mean()
    var_total = G / max(G - 1, 1) * np.sum((csum - cm) ** 2)
    return float(np.sqrt(var_total + 1e-30))

def _learn_projection(e_f_tr, momr_tr, e_y_tr, qz, qx):
    ze = np.quantile(momr_tr, np.linspace(0, 1, qz + 1)[1:-1]) if qz > 1 else np.array([])
    xe = np.quantile(e_f_tr, np.linspace(0, 1, qx + 1)[1:-1]) if qx > 1 else np.array([])
    zb = np.digitize(momr_tr, ze)
    xb = np.digitize(e_f_tr, xe)
    cell = zb * qx + xb
    K = qz * qx
    cnt = np.bincount(cell, minlength=K).astype(float)
    ssum = np.bincount(cell, weights=e_y_tr, minlength=K)
    table = np.where(cnt > 0, ssum / np.maximum(cnt, 1), 0.0)
    tab2d = table.reshape(qz, qx)
    cnt2d = cnt.reshape(qz, qx)
    zmean = (tab2d * cnt2d).sum(1) / np.maximum(cnt2d.sum(1), 1)
    table = (tab2d - zmean[:, None]).reshape(-1)
    return (ze, xe, table)

def _apply_projection(ze, xe, table, e_f, momr, qx):
    zb = np.digitize(momr, ze)
    xb = np.digitize(e_f, xe)
    return table[zb * qx + xb]

def pcm(fr, yr, firm, month, momr, N, T, rng):
    n = len(fr)
    nuis_folds = rng.integers(0, 2, size=n)
    mq = _cont_bin(momr, QR)
    idxs = [firm, month, mq]
    ks = [N, T, QR]
    e_f = _fast_resid(fr, idxs, ks, nuis_folds)
    e_y = _fast_resid(yr, idxs, ks, nuis_folds)
    perm = rng.permutation(T)
    half_A_months = np.zeros(T, dtype=bool)
    half_A_months[perm[:T // 2]] = True
    in_A = half_A_months[month]
    idx_A = np.flatnonzero(in_A)
    idx_B = np.flatnonzero(~in_A)
    S_total = 0.0
    V_total = 0.0
    for train, test in ((idx_A, idx_B), (idx_B, idx_A)):
        ze, xe, table = _learn_projection(e_f[train], momr[train], e_y[train], QZ, QX)
        f_hat_te = _apply_projection(ze, xe, table, e_f[test], momr[test], QX)
        zb_te = np.digitize(momr[test], ze)
        zmean_te = np.bincount(zb_te, weights=f_hat_te, minlength=QZ) / np.maximum(np.bincount(zb_te, minlength=QZ), 1)
        f_hat_te = f_hat_te - zmean_te[zb_te]
        g = f_hat_te * e_y[test]
        S_half = g.sum()
        se_half = _clustered_se(g, month[test])
        S_total += S_half
        V_total += se_half ** 2
    T_stat = S_total / np.sqrt(V_total + 1e-30)
    return int(T_stat > ZA)

def main(n_seeds=200, n_seeds_large=100, seed0=505, w_overlap=0.8):
    regimes = ('smooth', 'nonlinear', 'large_n', 'heavy_tail')
    rows = []
    t0 = time.time()
    for regime in regimes:
        ns = n_seeds_large if regime == 'large_n' else n_seeds
        for kind, delta in (('typeI', 0.0), ('power', 0.15)):
            rej = 0
            for s in range(ns):
                rng = np.random.default_rng(seed0 + s)
                fr, yr, firm, month, momr, N, T = panel(regime, rng, delta, w_overlap)
                rej += pcm(fr, yr, firm, month, momr, N, T, rng)
            rate = rej / ns
            rows.append(dict(regime=regime, metric=kind, n_seeds=ns, rate=round(rate, 3)))
            print(f'{regime:10s} {kind:6s} (n={ns:3d}, delta={delta:.2f}) -> reject_rate = {rate:.3f}', flush=True)
    tI = {r['regime']: r['rate'] for r in rows if r['metric'] == 'typeI'}
    pw = {r['regime']: r['rate'] for r in rows if r['metric'] == 'power'}
    summary = {'method': 'Projected Covariance Measure (PCM, Lundborg-Kim-Shah-Samworth, AoS 2024)', 'alpha': ALPHA, 'ZA': round(ZA, 4), 'QR_nuisance_momentum_bins': QR, 'QZ_projection_momentum_bins': QZ, 'QX_projection_efresid_bins': QX, 'split': 'month-level (whole-month) two-halves, cross-fitted A<->B', 'regimes': list(regimes), 'n_seeds': {r: n_seeds_large if r == 'large_n' else n_seeds for r in regimes}, 'typeI_delta0': tI, 'power_delta0.15': pw, 'max_typeI': round(max(tI.values()), 3), 'min_power': round(min(pw.values()), 3), 'rows': rows, 'elapsed_seconds': round(time.time() - t0, 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'pcm_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('\n=== PCM summary ===')
    print(json.dumps(summary, indent=2))
    print(f'\nartifacts -> {OUT}')
    return summary
if __name__ == '__main__':
    main()
