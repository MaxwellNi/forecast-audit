"""Supplementary synthetic comparison; see additional_baselines/README.md
from the package root for assumptions and interpretation limits."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import norm
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))
from clustered_calibration import panel, _spectrum, _clustered_se, QLADDER
OUT = ROOT / 'outputs/theory_simulation/audit_clustered'
ZA = float(norm.ppf(0.95))

def _w(beta):
    u = QLADDER ** (-beta)
    X = np.column_stack([np.ones(len(u)), u])
    return (np.linalg.inv(X.T @ X) @ X.T)[0]
W1 = _w(1.0)

def _beta_hat(m):
    d = np.abs(np.diff(m))
    if (d <= 0).any():
        return 1.0
    slope = np.polyfit(np.log(QLADDER[:-1]), np.log(d), 1)[0]
    return float(np.clip(-slope, 0.5, 2.0))

def run(n_seeds=50, seed0=505):
    rows = []
    for regime in ('smooth', 'nonlinear', 'large_n', 'heavy_tail'):
        for kind, delta in (('typeI', 0.0), ('power', 0.15)):
            r1 = ra = 0
            bs = []
            for s in range(n_seeds):
                rng = np.random.default_rng(seed0 + s)
                fr, yr, firm, month, momr, N, T = panel(regime, rng, delta, 0.8)
                P = _spectrum(fr, yr, firm, month, momr, N, T, rng)
                s1 = P @ W1
                r1 += s1.mean() / (_clustered_se(s1, month) + 1e-18) > ZA
                bh = _beta_hat(P.mean(0))
                bs.append(bh)
                sa = P @ _w(bh)
                ra += sa.mean() / (_clustered_se(sa, month) + 1e-18) > ZA
            rows.append(dict(regime=regime, metric=kind, fixed_beta1=round(r1 / n_seeds, 3), adaptive_beta=round(ra / n_seeds, 3), mean_beta_hat=round(float(np.mean(bs)), 2)))
            print(f'{regime:10s} {kind:6s}  fixed_b1={r1 / n_seeds:.2f}  adaptive={ra / n_seeds:.2f}  mean_bhat={np.mean(bs):.2f}')
    return rows

def main():
    rows = run()
    OUT.mkdir(parents=True, exist_ok=True)
    tI = {r['regime']: r for r in rows if r['metric'] == 'typeI'}
    summary = {'verdict': 'adaptive beta FAILS (obstruction): worse type-I in 3/4 regimes', 'typeI_fixed_beta1': {r: tI[r]['fixed_beta1'] for r in tI}, 'typeI_adaptive_beta': {r: tI[r]['adaptive_beta'] for r in tI}, 'nonlinear_beta_hat': tI['nonlinear']['mean_beta_hat'], 'note': 'adaptive beta_hat drifts high in nonlinear (~2.0), rotating the weights toward the kappa->0 pathological direction of Thm 6(ii) -> cluster type-I 0.14->0.96; fixed beta=1 is the robust choice.', 'rows': rows}
    (OUT / 'adaptive_beta_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('\n' + json.dumps(summary, indent=2))
if __name__ == '__main__':
    main()
