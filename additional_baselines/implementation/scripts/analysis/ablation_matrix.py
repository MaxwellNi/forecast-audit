"""Supplementary synthetic comparison; see additional_baselines/README.md
from the package root for assumptions and interpretation limits."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))
from clustered_calibration import panel, ZA, _intercept_weights, _cont_bin, _fast_resid, _clustered_se
REGIMES = ['smooth', 'nonlinear', 'large_n', 'heavy_tail']
W_OVERLAP = 0.8

def _equalwidth_bin(v, q):
    lo, hi = (v.min(), v.max())
    e = np.linspace(lo, hi + 1e-09, int(q) + 1)
    return np.clip(np.digitize(v, e[1:-1]), 0, int(q) - 1).astype(np.int64)

def cert(fr, yr, firm, month, momr, N, T, rng, ladder, beta, binning, nsweep):
    ladder = np.asarray(ladder, float)
    Wv = _intercept_weights(ladder, beta)
    folds = rng.integers(0, 2, size=len(fr))
    P = np.empty((len(fr), len(ladder)))
    binf = _cont_bin if binning == 'equal-mass' else _equalwidth_bin
    for j, q in enumerate(ladder):
        idxs = [firm, month, binf(momr, q)]
        ks = [N, T, int(q)]
        P[:, j] = _fast_resid(fr, idxs, ks, folds, n_sweep=nsweep) * _fast_resid(yr, idxs, ks, folds, n_sweep=nsweep)
    s = P @ Wv
    return int(s.mean() / (_clustered_se(s, month) + 1e-18) > ZA)
DEPLOYED = ('deployed', [8, 12, 16, 24, 32], 1.0, 'equal-mass', 12)
ABLATIONS = [DEPLOYED, ('ladder: short {8,16,32}', [8, 16, 32], 1.0, 'equal-mass', 12), ('ladder: long {8..40}', [8, 12, 16, 20, 24, 28, 32, 40], 1.0, 'equal-mass', 12), ('beta=0.5', [8, 12, 16, 24, 32], 0.5, 'equal-mass', 12), ('beta=1.5', [8, 12, 16, 24, 32], 1.5, 'equal-mass', 12), ('beta=2.0', [8, 12, 16, 24, 32], 2.0, 'equal-mass', 12), ('binning: equal-width', [8, 12, 16, 24, 32], 1.0, 'equal-width', 12), ('nuisance: light fit (4 sweeps)', [8, 12, 16, 24, 32], 1.0, 'equal-mass', 4), ('nuisance: heavy fit (24 sweeps)', [8, 12, 16, 24, 32], 1.0, 'equal-mass', 24)]

def _rate(cfg, regime, delta, n_seeds, seed0):
    _, ladder, beta, binning, nsweep = cfg
    rej = 0
    for s in range(n_seeds):
        rng = np.random.default_rng(seed0 + s)
        d = panel(regime, rng, delta, W_OVERLAP)
        rej += cert(*d, rng, ladder, beta, binning, nsweep)
    return rej / n_seeds

def main(n_seeds=150, seed0=70707):
    t0 = time.time()
    rows = {}
    for cfg in ABLATIONS:
        label = cfg[0]
        rows[label] = {}
        for reg in REGIMES:
            ns = n_seeds if reg != 'large_n' else max(80, n_seeds // 2)
            t1 = _rate(cfg, reg, 0.0, ns, seed0)
            pw = _rate(cfg, reg, 0.15, ns, seed0 + 5000)
            rows[label][reg] = {'typeI': round(t1, 3), 'power': round(pw, 2)}
        wt = max((rows[label][r]['typeI'] for r in REGIMES))
        mp = min((rows[label][r]['power'] for r in REGIMES))
        rows[label]['worst_typeI'] = wt
        rows[label]['min_power'] = mp
        rows[label]['robust'] = bool(wt <= 0.18 and mp >= 0.9)
        print(f"{label:32s} worst typeI={wt:.3f} min power={mp:.2f} {('ROBUST' if rows[label]['robust'] else 'breaks')}", flush=True)
    OUT = ROOT / 'outputs/theory_simulation/audit_ablation'
    OUT.mkdir(parents=True, exist_ok=True)
    robust = [k for k, v in rows.items() if v['robust']]
    out = {'overlap': W_OVERLAP, 'n_seeds': n_seeds, 'regimes': REGIMES, 'ablations': rows, 'robust_choices': robust, 'n_robust': len(robust), 'n_total': len(ABLATIONS), 'elapsed_sec': round(time.time() - t0, 1)}
    (OUT / 'ablation_matrix_summary.json').write_text(json.dumps(out, indent=2) + '\n')
    print(f'\nrobust under {len(robust)}/{len(ABLATIONS)} design choices: {robust}')
if __name__ == '__main__':
    main()
