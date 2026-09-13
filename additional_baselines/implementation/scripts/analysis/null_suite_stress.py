"""Supplementary synthetic comparison; see additional_baselines/README.md
from the package root for assumptions and interpretation limits."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))
from gcm_residual_audit import _colrank_z
from clustered_calibration import panel, _spectrum, _clustered_se, W
from baseline_implementations import extrapolated_covariance
OUT = ROOT / 'outputs/theory_simulation/null_suite_stress'
ALPHA = 0.05
ZA = float(norm.ppf(1 - ALPHA))
DELTA_ALT = 0.15
W_OVERLAP = 0.8

def panel_high_dim_fe(rng, delta, w, N=600, T=60):
    mu = rng.normal(0, 1.0, N)
    lam = rng.normal(0, 0.5, T)
    Z = rng.normal(0, 1, (N, T))
    mom = rng.normal(0, 1, (N, T))
    eps = rng.normal(0, 1.0, (N, T))
    g = np.tanh(1.5 * mom)
    r = mu[:, None] + lam[None, :] + 0.8 * g + delta * Z + eps
    f = w * g + delta * Z + rng.normal(0, 0.5, (N, T))
    firm = np.repeat(np.arange(N), T)
    month = np.tile(np.arange(T), N)
    return (_colrank_z(f).ravel(), _colrank_z(r).ravel(), firm, month, _colrank_z(mom).ravel(), N, T)

def panel_serial(rng, delta, w, N=120, T=100, rho_t=0.8, rho_i=0.8):
    mu = rng.normal(0, 1.0, N)
    lam = np.empty(T)
    lam[0] = rng.normal(0, 0.5)
    innov = rng.normal(0, 0.5 * np.sqrt(1 - rho_t ** 2), T)
    for t in range(1, T):
        lam[t] = rho_t * lam[t - 1] + innov[t]
    Z = rng.normal(0, 1, (N, T))
    mom = rng.normal(0, 1, (N, T))
    eps = np.empty((N, T))
    eps[:, 0] = rng.normal(0, 1.0, N)
    e_innov = rng.normal(0, np.sqrt(1 - rho_i ** 2), (N, T))
    for t in range(1, T):
        eps[:, t] = rho_i * eps[:, t - 1] + e_innov[:, t]
    g = np.tanh(1.5 * mom)
    r = mu[:, None] + lam[None, :] + 0.8 * g + delta * Z + eps
    f = w * g + delta * Z + rng.normal(0, 0.5, (N, T))
    firm = np.repeat(np.arange(N), T)
    month = np.tile(np.arange(T), N)
    return (_colrank_z(f).ravel(), _colrank_z(r).ravel(), firm, month, _colrank_z(mom).ravel(), N, T)

def panel_overlap(rng, delta, w, N=120, T=100, K=3):
    mu = rng.normal(0, 1.0, N)
    lam = rng.normal(0, 0.5, T)
    Z = rng.normal(0, 1, (N, T))
    mom = rng.normal(0, 1, (N, T))
    eps = rng.normal(0, 1.0, (N, T))
    g = np.tanh(1.5 * mom)
    r = mu[:, None] + lam[None, :] + 0.8 * g + delta * Z + eps
    csum = np.cumsum(r, axis=1)
    Yfwd = np.empty((N, T - K + 1))
    Yfwd[:, 0] = csum[:, K - 1]
    Yfwd[:, 1:] = csum[:, K:] - csum[:, :T - K]
    Tk = T - K + 1
    f = (w * g + delta * Z + rng.normal(0, 0.5, (N, T)))[:, :Tk]
    mom_k = mom[:, :Tk]
    firm = np.repeat(np.arange(N), Tk)
    month = np.tile(np.arange(Tk), N)
    return (_colrank_z(f).ravel(), _colrank_z(Yfwd).ravel(), firm, month, _colrank_z(mom_k).ravel(), N, Tk)

def panel_small_T(rng, delta, w, N=120, T=24):
    mu = rng.normal(0, 1.0, N)
    lam = rng.normal(0, 0.5, T)
    Z = rng.normal(0, 1, (N, T))
    mom = rng.normal(0, 1, (N, T))
    eps = rng.normal(0, 1.0, (N, T))
    g = np.tanh(1.5 * mom)
    r = mu[:, None] + lam[None, :] + 0.8 * g + delta * Z + eps
    f = w * g + delta * Z + rng.normal(0, 0.5, (N, T))
    firm = np.repeat(np.arange(N), T)
    month = np.tile(np.arange(T), N)
    return (_colrank_z(f).ravel(), _colrank_z(r).ravel(), firm, month, _colrank_z(mom).ravel(), N, T)
STRESS = {'high_dim_fe': panel_high_dim_fe, 'serial': panel_serial, 'overlap': panel_overlap, 'small_T': panel_small_T}
SEEDS = {'high_dim_fe': 100, 'serial': 200, 'overlap': 200, 'small_T': 200}

def run(seed0=505):
    rows = []
    for regime, dgp in STRESS.items():
        ns = SEEDS[regime]
        out = {}
        for kind, delta in (('typeI', 0.0), ('power', DELTA_ALT)):
            rej = 0
            t0 = time.time()
            N = T = None
            for s in range(ns):
                rng = np.random.default_rng(seed0 + s)
                fr, yr, firm, month, momr, N, T = dgp(rng, delta, W_OVERLAP)
                rej += extrapolated_covariance(fr, yr, firm, month, momr, N, T, rng)
            out[kind] = rej / ns
            out['N'], out['T'] = (int(N), int(T))
            out[f'{kind}_time_s'] = round(time.time() - t0, 1)
        row = dict(regime=regime, n_seeds=ns, N=out['N'], T=out['T'], typeI=round(out['typeI'], 3), power=round(out['power'], 3), time_s=round(out['typeI_time_s'] + out['power_time_s'], 1))
        rows.append(row)
        print(f"{regime:12s} N={out['N']:3d} T={out['T']:3d} seeds={ns:3d} | typeI={out['typeI']:.3f}  power={out['power']:.3f}  ({row['time_s']:.0f}s)", flush=True)
    return rows

def main():
    print(f'alpha={ALPHA}  ZA={ZA:.4f}  delta_alt={DELTA_ALT}  method=extrapolated residual-covariance diagnostic\n')
    rows = run()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / 'null_suite_stress.csv', index=False)

    def mc_hw(n):
        return 1.96 * np.sqrt(0.05 * 0.95 / n)
    verdict = {}
    for r in rows:
        hw = mc_hw(r['n_seeds'])
        if r['typeI'] <= 0.05 + 2 * hw:
            tag = 'HOLDS (<= nominal+2MC)'
        elif r['typeI'] <= 0.1:
            tag = 'MILD over-rejection (0.05-0.10)'
        else:
            tag = 'DEGRADES (> 0.10)'
        verdict[r['regime']] = dict(typeI=r['typeI'], power=r['power'], mc_halfwidth=round(float(hw), 3), tag=tag)
    summary = {'alpha': ALPHA, 'ZA': round(ZA, 4), 'delta_alt': DELTA_ALT, 'w_overlap': W_OVERLAP, 'method': 'extrapolated residual-covariance diagnostic with cluster variance', 'regimes_new': list(STRESS.keys()), 'rows': rows, 'typeI_by_regime': {r['regime']: r['typeI'] for r in rows}, 'power_by_regime': {r['regime']: r['power'] for r in rows}, 'verdict': verdict, 'max_typeI': max((r['typeI'] for r in rows)), 'min_power': min((r['power'] for r in rows))}
    (OUT / 'null_suite_stress_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('\n=== verdict (calibration under stress) ===')
    for reg, v in verdict.items():
        print(f"  {reg:12s} typeI={v['typeI']:.3f} (+-{v['mc_halfwidth']:.3f} MC) power={v['power']:.3f} -> {v['tag']}")
    print(f"\nmax typeI = {summary['max_typeI']:.3f}   min power = {summary['min_power']:.3f}")
    print(f'\nartifacts -> {OUT}')
if __name__ == '__main__':
    main()
