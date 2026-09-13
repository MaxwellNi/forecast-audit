"""Rating algorithms fitted only on the training frame supplied by the chronological task."""
from __future__ import annotations

from pathlib import Path

import numpy as np

import pandas as pd

from scipy.sparse import csr_matrix

from scipy.sparse.linalg import svds

NFAC = 40

SEED = 20260821

def fit_forecasters(tr):
    mu = tr['rating'].mean()
    item = tr.groupby('movieId')['rating'].mean()
    user = tr.groupby('userId')['rating'].mean()
    item_b = tr.groupby('movieId')['rating'].mean() - mu
    user_b = tr.assign(r=tr['rating'] - mu - tr['movieId'].map(item_b).fillna(0)).groupby('userId')['r'].mean()
    uids = {u: i for (i, u) in enumerate(tr['userId'].unique())}
    mids = {m: i for (i, m) in enumerate(tr['movieId'].unique())}
    r = (tr['rating'] - mu - tr['movieId'].map(item_b).fillna(0) - tr['userId'].map(user_b).fillna(0)).to_numpy()
    M = csr_matrix((r, (tr['userId'].map(uids), tr['movieId'].map(mids))), shape=(len(uids), len(mids)))
    k = min(NFAC, min(M.shape) - 1)
    (U, s, Vt) = svds(M, k=k)
    P = U * s @ Vt
    return dict(mu=mu, item=item, user=user, item_b=item_b, user_b=user_b, P=P, uids=uids, mids=mids)

def predict(te, f, kind):
    (mu, item, user) = (f['mu'], f['item'], f['user'])
    if kind == 'item_mean':
        return te['movieId'].map(item).fillna(mu).to_numpy()
    if kind == 'user_mean':
        return te['userId'].map(user).fillna(mu).to_numpy()
    if kind == 'baseline':
        return (mu + te['movieId'].map(f['item_b']).fillna(0) + te['userId'].map(f['user_b']).fillna(0)).to_numpy()
    if kind == 'svd_interaction':
        ui = te['userId'].map(f['uids'])
        mi = te['movieId'].map(f['mids'])
        ok = ui.notna() & mi.notna()
        out = np.zeros(len(te))
        ii = np.where(ok.to_numpy())[0]
        out[ii] = f['P'][ui[ok].astype(int).to_numpy(), mi[ok].astype(int).to_numpy()]
        return out
    if kind == 'svd_cf':
        ui = te['userId'].map(f['uids'])
        mi = te['movieId'].map(f['mids'])
        base = (mu + te['movieId'].map(f['item_b']).fillna(0) + te['userId'].map(f['user_b']).fillna(0)).to_numpy()
        ok = ui.notna() & mi.notna()
        out = base.copy()
        ii = np.where(ok.to_numpy())[0]
        out[ii] = base[ii] + f['P'][ui[ok].astype(int).to_numpy(), mi[ok].astype(int).to_numpy()]
        return out
    raise ValueError(kind)

def dense_matrices(tr, uids, mids):
    (nu, nm) = (len(uids), len(mids))
    ui = tr['userId'].map(uids).to_numpy(int)
    mi = tr['movieId'].map(mids).to_numpy(int)
    r = tr['rating'].to_numpy(np.float32)
    M = np.zeros((nu, nm), np.float32)
    M[ui, mi] = r
    B = np.zeros((nu, nm), np.float32)
    B[ui, mi] = 1.0
    return (M, B, ui, mi, r)

def pred_slope_one(tr, te, f, M, B):
    (uids, mids) = (f['uids'], f['mids'])
    S = M.T @ B - B.T @ M
    C = B.T @ B
    dev = np.divide(S, C, out=np.zeros_like(S), where=C > 0)
    user_items = {}
    for (u, g) in tr.groupby('userId'):
        user_items[u] = (g['movieId'].map(mids).to_numpy(int), g['rating'].to_numpy(np.float32))
    user_mean = f['user']
    mu = f['mu']
    out = np.empty(len(te))
    k = 0
    for (u, g) in te.groupby('userId', sort=False):
        fallback = float(user_mean.get(u, mu))
        (items, ratings) = user_items.get(u, (np.empty(0, int), np.empty(0, np.float32)))
        jm = g['movieId'].map(mids)
        for j in jm:
            if len(items) == 0 or pd.isna(j):
                out[k] = fallback
            else:
                jj = int(j)
                c = C[jj, items]
                den = c.sum()
                out[k] = (c * (dev[jj, items] + ratings)).sum() / den if den > 0 else fallback
            k += 1
    idx = np.concatenate([g.index.to_numpy() for (_, g) in te.groupby('userId', sort=False)])
    res = pd.Series(out, index=idx).reindex(te.index)
    return res.to_numpy(float)

def pred_knn_item(tr, te, f, M, B, k_nn=20, shrink=10.0):
    (uids, mids) = (f['uids'], f['mids'])
    umean = tr.groupby('userId')['rating'].mean()
    Xc = M.copy()
    urow = np.zeros(M.shape[0], np.float32)
    for (u, i) in uids.items():
        urow[i] = umean.get(u, f['mu'])
    Xc[B > 0] -= np.repeat(urow, (B > 0).sum(1))
    norms = np.sqrt((Xc ** 2).sum(0)) + 1e-09
    SIM = Xc.T @ Xc / np.outer(norms, norms)
    C = B.T @ B
    SIM *= C / (C + shrink)
    np.fill_diagonal(SIM, 0.0)
    item_mean = f['item']
    mu = f['mu']
    imean_vec = np.full(M.shape[1], mu, np.float32)
    for (m, i) in mids.items():
        imean_vec[i] = item_mean.get(m, mu)
    user_items = {u: (g['movieId'].map(mids).to_numpy(int), g['rating'].to_numpy(np.float32)) for (u, g) in tr.groupby('userId')}
    out = np.empty(len(te))
    k = 0
    order_idx = []
    for (u, g) in te.groupby('userId', sort=False):
        (items, ratings) = user_items.get(u, (np.empty(0, int), np.empty(0, np.float32)))
        centered = ratings - imean_vec[items] if len(items) else ratings
        for j in g['movieId'].map(mids):
            if pd.isna(j) or len(items) == 0:
                out[k] = mu
                k += 1
                continue
            jj = int(j)
            s = SIM[jj, items]
            top = np.argsort(-s)[:k_nn]
            s_top = s[top]
            pos = s_top > 0
            denom = np.abs(s_top[pos]).sum()
            base = imean_vec[jj]
            out[k] = base + (s_top[pos] * centered[top][pos]).sum() / denom if denom > 0 else base
            k += 1
        order_idx.append(g.index.to_numpy())
    idx = np.concatenate(order_idx)
    return pd.Series(out, index=idx).reindex(te.index).to_numpy(float)

def pred_als(tr, te, f, seed=SEED, k=16, reg=0.1, iters=15):
    (uids, mids) = (f['uids'], f['mids'])
    ui = tr['userId'].map(uids).to_numpy(int)
    mi = tr['movieId'].map(mids).to_numpy(int)
    resid = (tr['rating'] - f['mu'] - tr['movieId'].map(f['item_b']).fillna(0) - tr['userId'].map(f['user_b']).fillna(0)).to_numpy(np.float64)
    (nu, nm) = (len(uids), len(mids))
    rng = np.random.default_rng(seed)
    U = rng.normal(0, 0.1, (nu, k))
    V = rng.normal(0, 0.1, (nm, k))
    ord_u = np.argsort(ui, kind='stable')
    ord_m = np.argsort(mi, kind='stable')
    (uu, um, ur) = (ui[ord_u], mi[ord_u], resid[ord_u])
    (mu_, mm_, mr_) = (ui[ord_m], mi[ord_m], resid[ord_m])
    ubnd = np.searchsorted(uu, np.arange(nu + 1))
    mbnd = np.searchsorted(mm_, np.arange(nm + 1))
    eye = reg * np.eye(k)
    for _ in range(iters):
        for a in range(nu):
            (lo, hi) = (ubnd[a], ubnd[a + 1])
            if hi <= lo:
                continue
            Vs = V[um[lo:hi]]
            U[a] = np.linalg.solve(Vs.T @ Vs + eye, Vs.T @ ur[lo:hi])
        for b in range(nm):
            (lo, hi) = (mbnd[b], mbnd[b + 1])
            if hi <= lo:
                continue
            Us = U[mu_[lo:hi]]
            V[b] = np.linalg.solve(Us.T @ Us + eye, Us.T @ mr_[lo:hi])
    base = (f['mu'] + te['movieId'].map(f['item_b']).fillna(0) + te['userId'].map(f['user_b']).fillna(0)).to_numpy(float)
    tui = te['userId'].map(uids)
    tmi = te['movieId'].map(mids)
    ok = tui.notna() & tmi.notna()
    out = base.copy()
    ii = np.where(ok.to_numpy())[0]
    out[ii] = base[ii] + np.einsum('ij,ij->i', U[tui[ok].astype(int).to_numpy()], V[tmi[ok].astype(int).to_numpy()])
    return out

def pred_svd_k32(tr, te, f):
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import svds
    (uids, mids) = (f['uids'], f['mids'])
    r = (tr['rating'] - f['mu'] - tr['movieId'].map(f['item_b']).fillna(0) - tr['userId'].map(f['user_b']).fillna(0)).to_numpy()
    M = csr_matrix((r, (tr['userId'].map(uids), tr['movieId'].map(mids))), shape=(len(uids), len(mids)))
    (U, s, Vt) = svds(M, k=32)
    P = U * s @ Vt
    base = (f['mu'] + te['movieId'].map(f['item_b']).fillna(0) + te['userId'].map(f['user_b']).fillna(0)).to_numpy(float)
    tui = te['userId'].map(uids)
    tmi = te['movieId'].map(mids)
    ok = tui.notna() & tmi.notna()
    out = base.copy()
    ii = np.where(ok.to_numpy())[0]
    out[ii] = base[ii] + P[tui[ok].astype(int).to_numpy(), tmi[ok].astype(int).to_numpy()]
    return out

MODEL_NAMES = {'item_mean': 'Item mean', 'user_mean': 'User mean', 'baseline': 'User and item bias', 'svd_cf': 'SVD (40 factors)', 'svd_interaction': 'SVD interaction', 'global_mean': 'Global mean', 'slope_one': 'Slope One', 'knn_item': 'Item nearest neighbours', 'als_mf': 'Alternating least squares', 'svd_k32': 'SVD (32 factors)'}
