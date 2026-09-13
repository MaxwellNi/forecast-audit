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
from clustered_calibration import panel, _spectrum, _clustered_se, W, _cont_bin, _fast_resid
OUT = ROOT / 'outputs/theory_simulation/synthetic_baselines'
ALPHA = 0.05
ZA = float(norm.ppf(1 - ALPHA))
QF = 32
SUB = 800

def _fe_resid(v, firm, month, N, T, folds):
    return _fast_resid(v, [firm, month], [N, T], folds)

def extrapolated_covariance(fr, yr, firm, month, momr, N, T, rng):
    P = _spectrum(fr, yr, firm, month, momr, N, T, rng)
    s = P @ W
    return int(s.mean() / (_clustered_se(s, month) + 1e-18) > ZA)

def _gcm_at_q(fr, yr, firm, month, momr, N, T, rng, q):
    folds = rng.integers(0, 2, size=len(fr))
    idxs = [firm, month, _cont_bin(momr, q)]
    ks = [N, T, int(q)]
    e_f = _fast_resid(fr, idxs, ks, folds)
    e_y = _fast_resid(yr, idxs, ks, folds)
    p = e_f * e_y
    return int(np.sqrt(len(p)) * p.mean() / (p.std() + 1e-12) > ZA)

def gcm_fine(fr, yr, firm, month, momr, N, T, rng):
    return _gcm_at_q(fr, yr, firm, month, momr, N, T, rng, QF)

def gcm_coarse(fr, yr, firm, month, momr, N, T, rng):
    return _gcm_at_q(fr, yr, firm, month, momr, N, T, rng, 8)

def partial_linear(fr, yr, firm, month, momr, N, T, rng):
    folds = rng.integers(0, 2, size=len(fr))
    mc = momr - momr.mean()
    rf = _fast_resid(fr, [firm, month], [N, T], folds)
    ry = _fast_resid(yr, [firm, month], [N, T], folds)
    rf = rf - mc * (rf @ mc) / (mc @ mc + 1e-12)
    ry = ry - mc * (ry @ mc) / (mc @ mc + 1e-12)
    p = rf * ry
    return int(np.sqrt(len(p)) * p.mean() / (p.std() + 1e-12) > ZA)

def naive_perm(fr, yr, firm, month, momr, N, T, rng, B=99):
    folds = rng.integers(0, 2, size=len(fr))
    idxs = [firm, month, _cont_bin(momr, 8)]
    ks = [N, T, 8]
    e_f = _fast_resid(fr, idxs, ks, folds)
    e_y = _fast_resid(yr, idxs, ks, folds)
    obs = (e_f * e_y).mean()
    perm = np.array([(e_f * e_y[rng.permutation(len(e_y))]).mean() for _ in range(B)])
    return int((1 + (perm >= obs).sum()) / (B + 1) < ALPHA)

def _deep_resid(target, firm, month, momr, N, T, folds, device, torch, nn):
    n = len(target)
    fi = torch.tensor(firm, dtype=torch.long, device=device)
    mo = torch.tensor(month, dtype=torch.long, device=device)
    mm = torch.tensor(momr, dtype=torch.float32, device=device).unsqueeze(1)
    yt = torch.tensor(target, dtype=torch.float32, device=device).unsqueeze(1)
    resid = np.empty(n)
    for f in (0, 1):
        tr = torch.tensor(folds != f, device=device)
        te = ~tr
        ef, em, d = (8, 8, 32)
        emb_f = nn.Embedding(N, ef).to(device)
        emb_m = nn.Embedding(T, em).to(device)
        net = nn.Sequential(nn.Linear(ef + em + 1, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1)).to(device)
        params = list(emb_f.parameters()) + list(emb_m.parameters()) + list(net.parameters())
        opt = torch.optim.Adam(params, lr=0.005, weight_decay=0.0001)
        Xtr = (fi[tr], mo[tr], mm[tr])
        ytr = yt[tr]
        for _ in range(80):
            opt.zero_grad()
            h = torch.cat([emb_f(Xtr[0]), emb_m(Xtr[1]), Xtr[2]], 1)
            loss = ((net(h) - ytr) ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            h = torch.cat([emb_f(fi[te]), emb_m(mo[te]), mm[te]], 1)
            pred = net(h)
        r = (yt[te] - pred).squeeze(1).cpu().numpy()
        resid[folds == f] = r
    return resid

def deep_gcm(fr, yr, firm, month, momr, N, T, rng, device, torch, nn):
    folds = rng.integers(0, 2, size=len(fr))
    e_f = _deep_resid(fr, firm, month, momr, N, T, folds, device, torch, nn)
    e_y = _deep_resid(yr, firm, month, momr, N, T, folds, device, torch, nn)
    p = e_f * e_y
    return int(np.sqrt(len(p)) * p.mean() / (p.std() + 1e-12) > ZA)

def kcit_rff(fr, yr, firm, month, momr, N, T, rng, D=256):
    folds = rng.integers(0, 2, size=len(fr))
    e_f = _fe_resid(fr, firm, month, N, T, folds)
    e_y = _fe_resid(yr, firm, month, N, T, folds)
    m = momr.reshape(-1, 1)
    gamma = 1.0 / (2 * np.median(np.abs(m - m[rng.integers(0, len(m), 1)])) ** 2 + 1e-09)
    w = rng.normal(0, np.sqrt(2 * gamma), (1, D))
    b = rng.uniform(0, 2 * np.pi, D)
    phi = np.sqrt(2.0 / D) * np.cos(m @ w + b)
    A = phi.T @ phi + 0.001 * np.eye(D)
    rf = e_f - phi @ np.linalg.solve(A, phi.T @ e_f)
    ry = e_y - phi @ np.linalg.solve(A, phi.T @ e_y)
    p = rf * ry
    return int(np.sqrt(len(p)) * p.mean() / (p.std() + 1e-12) > ZA)

def _dcov(a, b):
    A = np.abs(a[:, None] - a[None, :])
    B = np.abs(b[:, None] - b[None, :])
    A = A - A.mean(0)[None, :] - A.mean(1)[:, None] + A.mean()
    B = B - B.mean(0)[None, :] - B.mean(1)[:, None] + B.mean()
    return (A * B).mean()

def dcor_part(fr, yr, firm, month, momr, N, T, rng, B=49):
    folds = rng.integers(0, 2, size=len(fr))
    e_f = _fe_resid(fr, firm, month, N, T, folds)
    e_y = _fe_resid(yr, firm, month, N, T, folds)
    idx = rng.choice(len(fr), size=min(SUB, len(fr)), replace=False)
    a, b, z = (e_f[idx], e_y[idx], momr[idx])
    Da = np.abs(a[:, None] - a[None, :])
    Db = np.abs(b[:, None] - b[None, :])
    Z = np.abs(z[:, None] - z[None, :])
    zc = Z - Z.mean()
    zz = (zc * zc).sum() + 1e-12

    def part(D):
        return D - (D * zc).sum() / zz * Z
    Pa = part(Da)
    obs = (Pa * part(Db)).mean()
    perm = np.empty(B)
    for k in range(B):
        pi = rng.permutation(len(b))
        perm[k] = (Pa * part(Db[np.ix_(pi, pi)])).mean()
    return int((1 + (perm >= obs).sum()) / (B + 1) < ALPHA)

def main(n_seeds=300, seed0=505, W_overlap=0.8):
    import torch
    import torch.nn as nn
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    FNS = {'gcm_coarse': gcm_coarse, 'gcm_fine': gcm_fine, 'partial_linear': partial_linear, 'naive_perm': naive_perm, 'kcit_rff': kcit_rff, 'dcor_part': dcor_part, 'ours': extrapolated_covariance}
    METHODS = ['gcm_coarse', 'gcm_fine', 'partial_linear', 'naive_perm', 'deep_gcm', 'kcit_rff', 'dcor_part', 'ours']
    rows = []
    times = {m: 0.0 for m in METHODS}
    for regime in ('smooth', 'nonlinear', 'large_n', 'heavy_tail'):
        for kind, delta in (('typeI', 0.0), ('power', 0.15)):
            acc = {m: 0 for m in METHODS}
            ns = n_seeds if regime != 'large_n' else max(150, n_seeds // 2)
            for s in range(ns):
                rng = np.random.default_rng(seed0 + s)
                fr, yr, firm, month, momr, N, T = panel(regime, rng, delta, W_overlap)
                for m in ('gcm_coarse', 'gcm_fine', 'partial_linear', 'naive_perm'):
                    t = time.time()
                    acc[m] += FNS[m](fr, yr, firm, month, momr, N, T, rng)
                    times[m] += time.time() - t
                t = time.time()
                acc['deep_gcm'] += deep_gcm(fr, yr, firm, month, momr, N, T, rng, device, torch, nn)
                times['deep_gcm'] += time.time() - t
                for m in ('kcit_rff', 'dcor_part', 'ours'):
                    t = time.time()
                    acc[m] += FNS[m](fr, yr, firm, month, momr, N, T, rng)
                    times[m] += time.time() - t
            rows.append(dict(regime=regime, metric=kind, n_seeds=ns, **{m: round(acc[m] / ns, 3) for m in METHODS}))
            print(f'{regime:10s} {kind:6s} | ' + '  '.join((f'{m}={acc[m] / ns:.2f}' for m in METHODS)), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tI = {r['regime']: r for r in rows if r['metric'] == 'typeI'}
    pw = {r['regime']: r for r in rows if r['metric'] == 'power'}
    robust = {m: dict(worst_typeI=round(max((tI[r][m] for r in tI)), 3), worst_power=round(min((pw[r][m] for r in pw)), 3), non_catastrophic=bool(max((tI[r][m] for r in tI)) <= 0.2 and min((pw[r][m] for r in pw)) >= 0.9)) for m in METHODS}
    summary = {'alpha': ALPHA, 'regimes': list(tI.keys()), 'rows': rows, 'per_method': robust, 'champion': [m for m in METHODS if robust[m]['non_catastrophic']], 'total_seconds': {m: round(times[m], 1) for m in METHODS}}
    (OUT / 'synthetic_baselines_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT / 'synthetic_baselines.csv', index=False)
    print('\n=== champion (non-catastrophic on type-I<=0.2 AND power>=0.9 in ALL regimes) ===')
    for m in METHODS:
        v = robust[m]
        print(f"  {m:10s} worstTypeI={v['worst_typeI']:.2f} worstPower={v['worst_power']:.2f} time={times[m]:.0f}s -> {('CHAMPION' if v['non_catastrophic'] else 'fails')}")
    print(f'\nartifacts -> {OUT}')
if __name__ == '__main__':
    main()
