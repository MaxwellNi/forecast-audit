#!/usr/bin/env python3
"""Independent checks for the general distinct-reference projection.

No project implementation is imported. Re-run from any working directory:
  python reference_covariance_checks.py --repetitions 3000
Results are written next to this file. Requires numpy and scipy.
"""
from __future__ import annotations

import argparse
from itertools import permutations, product
import json
from math import comb, sqrt
from pathlib import Path

import numpy as np
from scipy.special import gammaln
from scipy.stats import norm


def comparison(x: np.ndarray) -> np.ndarray:
    """Rows are focal values; columns are reference values, including ties."""
    return (x[:, None] > x[None, :]).astype(float) + 0.5 * (
        x[:, None] == x[None, :]
    )


def population(support, p, f=None, g=None):
    """Exact finite-support population formula; Z values may repeat."""
    support, p = np.asarray(support), np.asarray(p, dtype=float)
    av, aw = comparison(support[:, 0]), comparison(support[:, 1])
    rv, rw = av @ p, aw @ p
    mv, mw = np.zeros(len(p)), np.zeros(len(p))
    for z in np.unique(support[:, 2]):
        mask = support[:, 2] == z
        mv[mask] = np.dot(p[mask], rv[mask]) / p[mask].sum()
        mw[mask] = np.dot(p[mask], rw[mask]) / p[mask].sum()
    f = mv if f is None else np.asarray(f)
    g = mw if g is None else np.asarray(g)
    q = (rv - f) * (rw - g)
    mu = np.dot(p, q)
    sv = (av - f[:, None]).T @ (p * (rw - g))
    sw = (aw - g[:, None]).T @ (p * (rv - f))
    psi = q + sv + sw - 3 * mu
    theta = np.dot(p, (rv - mv) * (rw - mw))
    bias = np.dot(p, (mv - f) * (mw - g))
    gamma = np.einsum("i,ij,j->", p, av * aw, p) - np.dot(p, rv * rw)
    return dict(av=av, aw=aw, rv=rv, rw=rw, mv=mv, mw=mw,
                f=f, g=g, q=q, mu=mu, theta=theta, bias=bias, sv=sv,
                sw=sw, psi=psi, tau2=np.dot(p, psi**2),
                focal_variance=np.dot(p, (q-mu)**2), gamma=gamma)


def count_scores(counts, pop):
    """All group scores from joint category counts; raw data are unnecessary."""
    counts = np.asarray(counts)
    n = counts.sum(axis=-1, keepdims=True)
    av, aw = pop["av"], pop["aw"]
    uv = (counts @ av.T - .5) / (n - 1)
    uw = (counts @ aw.T - .5) / (n - 1)
    joint = (counts @ (av * aw).T - .25) / (n - 1)
    ci = (uv - pop["f"]) * (uw - pop["g"])
    di = ci + (uv * uw - joint) / (n - 2)
    return (counts * ci).sum(axis=-1) / n[..., 0], (counts * di).sum(axis=-1) / n[..., 0]


def projection_checks():
    # Generic nonzero target, nonconstant controls, imperfect fitted means.
    support = np.array(list(product(range(3), range(2), range(2))))
    weights = np.arange(1, len(support)+1, dtype=float) ** 2
    p = weights / weights.sum()
    f = np.where(support[:, 2] == 0, .35, .62)
    g = np.where(support[:, 2] == 0, .46, .71)
    pop = population(support, p, f, g)
    h = (pop["av"][:, :, None] - f[:, None, None]) * (
        pop["aw"][:, None, :] - g[:, None, None])
    hs = sum(h.transpose(perm) for perm in permutations(range(3))) / 6
    centered_projection = np.einsum("ijk,j,k->i", hs, p, p) - pop["mu"]
    errors = dict(
        mean_decomposition=abs(pop["mu"] - pop["theta"] - pop["bias"]),
        influence_centering=abs(np.dot(p, pop["psi"])),
        symmetrized_projection=float(np.max(abs(3 * centered_projection - pop["psi"]))),
        reference_v_mean=abs(np.dot(p, pop["sv"]) - pop["mu"]),
        reference_w_mean=abs(np.dot(p, pop["sw"]) - pop["mu"]),
    )
    # Compare the comparison-sum implementation with literal ordered triples.
    ids = np.array([0, 1, 1, 4, 8, 10])
    counts = np.bincount(ids, minlength=len(p))[None, :]
    _, d = count_scores(counts, pop)
    direct = np.mean([h[ids[i], ids[j], ids[k]]
                      for i, j, k in permutations(range(len(ids)), 3)])
    errors["score_vs_ordered_triples"] = abs(d[0] - direct)
    assert max(errors.values()) < 1e-13, errors
    return errors


def base_distribution(local_weight=0.):
    pi = [1, 3, 0, 2]
    table = np.zeros((4, 4))
    for v in range(4):
        table[v, pi[v]] += (1-local_weight)/4
        table[v, v] += local_weight/4
    v, w = np.nonzero(table)
    support = np.column_stack([v, w, np.zeros(len(v), dtype=int)])
    p = table[v, w]
    return support, p, population(support, p)


def finite_group_moments():
    _, p, pop = base_distribution()
    rows = []
    for n in [3, 4, 8, 16, 32, 64]:
        counts = np.array([(a,b,c,n-a-b-c) for a in range(n+1)
                          for b in range(n-a+1) for c in range(n-a-b+1)])
        prob = np.exp(gammaln(n+1)-gammaln(counts+1).sum(axis=1)-n*np.log(4))
        c, d = count_scores(counts, pop)
        mean_d = np.dot(prob, d)
        variance_d = np.dot(prob, (d-mean_d)**2)
        assert abs(prob.sum()-1) < 2e-12
        assert abs(mean_d-pop["theta"]) < 2e-14
        rows.append(dict(N=n, configurations=comb(n+3, 3), mean_D=mean_d,
                         mean_C=float(np.dot(prob, c)), N_var_D=n*variance_d,
                         limiting_tau2=pop["tau2"]))
    return rows


def wilson(k, n):
    z = norm.ppf(.975)
    center = (k/n+z*z/(2*n))/(1+z*z/n)
    half = z*sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/(1+z*z/n)
    return [center-half, center+half]


def simulations(repetitions, seed):
    rng = np.random.default_rng(seed)
    results = []
    cutoff = norm.ppf(.95)
    for m, n in [(100,16), (400,64), (1000,256)]:
        for local_c in [0., 1.5]:
            t = local_c / sqrt(m*n)
            _, p, pop = base_distribution(t)
            ts, wrong_ts, tauhats = [], [], []
            max_jackknife_difference = 0.
            for start in range(0, repetitions, 100):
                b = min(100, repetitions-start)
                counts = rng.multinomial(n, p, size=(b, m))
                _, d = count_scores(counts, pop)
                mean_d = d.mean(axis=1)
                variance_d = d.var(axis=1, ddof=1)
                se = np.sqrt(variance_d/m)
                ts.extend(mean_d / se)
                # Deliberately wrong variance using the oracle focal product only.
                wrong_ts.extend(mean_d * sqrt(m*n/pop["focal_variance"]))
                tauhats.extend(n*variance_d)
                deleted = (m*mean_d[:, None]-d)/(m-1)
                jack_var = (m-1)/m * ((deleted-mean_d[:, None])**2).sum(axis=1)
                max_jackknife_difference = max(max_jackknife_difference,
                                              np.max(abs(jack_var-se**2)))
            ts, wrong_ts, tauhats = map(np.asarray, (ts, wrong_ts, tauhats))
            reject = int((ts > cutoff).sum())
            noncentrality = sqrt(m*n)*pop["theta"]/sqrt(pop["tau2"])
            results.append(dict(M=m, N=n, local_c=local_c, mixture_weight=t,
                                theta=pop["theta"], tau2=pop["tau2"],
                                focal_variance=pop["focal_variance"],
                                theoretical_mean_finite_law=noncentrality,
                                asymptotic_mean=local_c,
                                mean_T=float(ts.mean()), sd_T=float(ts.std(ddof=1)),
                                rejection_rate=reject/repetitions,
                                rejection_wilson_95=wilson(reject, repetitions),
                                predicted_rejection=float(norm.sf(cutoff-noncentrality)),
                                wrong_focal_rejection_rate=float((wrong_ts>cutoff).mean()),
                                wrong_focal_asymptotic_null_rejection=float(norm.sf(cutoff*3/5)),
                                mean_N_vhat=float(tauhats.mean()),
                                max_group_jackknife_difference=max_jackknife_difference))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=202609120541)
    args = parser.parse_args()
    support, p, pop = base_distribution()
    assert abs(pop["theta"]) < 1e-15
    assert abs(pop["tau2"]-25/4096) < 1e-15
    assert abs(pop["focal_variance"]-9/4096) < 1e-15
    output = dict(seed=args.seed, repetitions=args.repetitions,
                  projection_errors=projection_checks(),
                  counterexample=dict(support=support.tolist(), probabilities=p.tolist(),
                                      q=pop["q"].tolist(), sv=pop["sv"].tolist(),
                                      sw=pop["sw"].tolist(), psi=pop["psi"].tolist(),
                                      theta=pop["theta"], gamma=pop["gamma"],
                                      tau2=pop["tau2"], focal_variance=pop["focal_variance"],
                                      variance_ratio=pop["tau2"]/pop["focal_variance"]),
                  finite_group_moments=finite_group_moments(),
                  simulations=simulations(args.repetitions, args.seed))
    target = Path(__file__).with_name("results.json")
    target.write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
