"""Supplementary synthetic comparison; see additional_baselines/README.md
from the package root for assumptions and interpretation limits."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import norm
OUT = Path('outputs/theory_simulation/audit_gcm')

def _colrank_z(m):
    r = np.argsort(np.argsort(m, axis=0), axis=0).astype(float)
    return (r - r.mean(0)) / (r.std(0) + 1e-12)

def simulate(N, T, delta, theta_mom=0.6, sigma_mu=1.0, sigma_lam=0.5, sigma_eps=1.0, rng=None):
    rng = rng or np.random.default_rng()
    mu = rng.normal(0, sigma_mu, N)
    lam = rng.normal(0, sigma_lam, T)
    mom = rng.normal(0, 1, (N, T))
    Z = rng.normal(0, 1, (N, T))
    eps = rng.normal(0, sigma_eps, (N, T))
    r = mu[:, None] + lam[None, :] + theta_mom * mom + delta * Z + eps
    return dict(r=r, mu=mu, lam=lam, mom=mom, Z=Z)

def forecaster(kind, p, rng, noise=0.5):
    N, T = p['mom'].shape
    nz = rng.normal(0, noise, (N, T))
    if kind == 'genuine':
        return p['Z'] + nz
    if kind == 'momentum':
        return p['mom'] + nz
    if kind == 'firm':
        return p['mu'][:, None] * np.ones((1, T)) + nz
    if kind == 'firm_plus_momentum':
        return 0.7 * p['mu'][:, None] + 0.7 * p['mom'] + nz
    raise ValueError(kind)

def _design(N, T, mom_rank, q=10):
    firm_idx = np.repeat(np.arange(N), T)
    month_idx = np.tile(np.arange(T), N)
    n = N * T
    Zf = np.zeros((n, N))
    Zf[np.arange(n), firm_idx] = 1.0
    Zm = np.zeros((n, T))
    Zm[np.arange(n), month_idx] = 1.0
    mr = mom_rank.ravel()
    edges = np.quantile(mr, np.linspace(0, 1, q + 1)[1:-1])
    dec = np.digitize(mr, edges)
    Zq = np.zeros((n, q))
    Zq[np.arange(n), dec] = 1.0
    return np.column_stack([Zf, Zm, Zq])

def _crossfit_resid(target_flat, Z, folds, lam=1.0):
    n, d = Z.shape
    resid = np.empty(n)
    for f in np.unique(folds):
        tr = folds != f
        te = folds == f
        Ztr = Z[tr]
        A = Ztr.T @ Ztr + lam * np.eye(d)
        b = Ztr.T @ target_flat[tr]
        beta = np.linalg.solve(A, b)
        resid[te] = target_flat[te] - Z[te] @ beta
    return resid

def gcm_test(pred, y, mom, rng, K=2, lam=1.0):
    N, T = pred.shape
    fr = _colrank_z(pred).ravel()
    yr = _colrank_z(y).ravel()
    Z = _design(N, T, _colrank_z(mom))
    folds = rng.integers(0, K, size=N * T)
    e_f = _crossfit_resid(fr, Z, folds, lam)
    e_y = _crossfit_resid(yr, Z, folds, lam)
    prod = e_f * e_y
    n = len(prod)
    Tstat = np.sqrt(n) * prod.mean() / (prod.std() + 1e-12)
    pval = float(norm.sf(Tstat))
    r2_pred = float(1.0 - e_f.var() / (fr.var() + 1e-12))
    return (float(Tstat), pval, r2_pred)

def run(n_draws=60, N=150, T=120, alpha=0.05, tau=0.5, seed=7):
    rng = np.random.default_rng(seed)
    rows = []

    def eval_cond(name, kind, delta):
        decl = abst = 0
        Ts, R2 = ([], [])
        for _ in range(n_draws):
            p = simulate(N, T, delta=delta, rng=rng)
            f = forecaster(kind, p, rng)
            Tstat, pv, r2 = gcm_test(f, p['r'], p['mom'], rng)
            Ts.append(Tstat)
            R2.append(r2)
            if r2 > tau:
                abst += 1
            elif pv < alpha:
                decl += 1
        return {'test': name, 'delta': delta, 'declare_genuine_rate': decl / n_draws, 'abstain_rate': abst / n_draws, 'mean_T': float(np.mean(Ts)), 'mean_nuisance_R2': float(np.mean(R2))}
    for delta in (0.0, 0.05, 0.1, 0.2, 0.4):
        rows.append(eval_cond('genuine', 'genuine', delta))
    for kind in ('firm', 'momentum', 'firm_plus_momentum'):
        rows.append(eval_cond(f'nuisance:{kind}', kind, 0.0))
    import pandas as pd
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / 'gcm_residual_audit.csv', index=False)
    nui = df[df.test.str.startswith('nuisance')]
    gen = df[df.test == 'genuine']
    typeI = float(nui['declare_genuine_rate'].max())
    summary = {'method': 'cross-fitted GCM (Shah-Peters 2020) + nuisance-separation abstain pre-test', 'nominal_alpha': alpha, 'separation_tau': tau, 'selective_typeI_max_on_nuisance': typeI, 'typeI_controlled': bool(typeI <= 2 * alpha), 'nuisance_abstain_min': float(nui['abstain_rate'].min()), 'genuine_abstain_max': float(gen['abstain_rate'].max()), 'power_at_delta_0.40': float(gen[gen.delta == 0.4]['declare_genuine_rate'].iloc[0]), 'power_at_delta_0.10': float(gen[gen.delta == 0.1]['declare_genuine_rate'].iloc[0]), 'declare_rate_genuine_delta0': float(gen[gen.delta == 0.0]['declare_genuine_rate'].iloc[0]), 'power_monotone': bool(gen.sort_values('delta')['declare_genuine_rate'].is_monotonic_increasing), 'genuine_mean_R2': float(gen['mean_nuisance_R2'].mean()), 'nuisance_mean_R2': float(nui['mean_nuisance_R2'].mean()), 'separation_clean': bool(gen['mean_nuisance_R2'].max() < tau < nui['mean_nuisance_R2'].min())}
    with open(OUT / 'gcm_residual_audit_summary.json', 'w') as fjs:
        json.dump(summary, fjs, indent=2)
    print(df.to_string(index=False))
    print('\n=== GCM with abstention diagnostic summary ===')
    print(json.dumps(summary, indent=2))
    print(f'\nartifacts -> {OUT}')
if __name__ == '__main__':
    run()
