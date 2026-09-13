#!/usr/bin/env python3
"""Prespecified betting extension on exposed original synthetic streams."""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import beta, rankdata
from classical_reference_comparators import mixture_betting_pvalue

HERE = Path(__file__).resolve().parent
GENERATOR = HERE.parents[1] / 'scripts/analysis/category_reference_certificate.py'
spec = importlib.util.spec_from_file_location('original_generator', GENERATOR)
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)
SEED, REPS = 202609121457, 300
DESIGNS = [('two_balanced', np.array([.5, .5]), np.array([.2, .8])),
           ('eight_rare', np.array([.005] + [.995/7]*7), np.linspace(.2, .8, 8))]
SIGNALS, GROUPS, VALIDATION = [-.125, 0., .125, .25], [100, 400], [512, 8192]
METHODS = ['signed_variance', 'independent_bernstein', 'independent_betting',
           'pooled_variance', 'pooled_betting']
ALPHA, DELTA, TRAIN = .05, .0001, 8192
FRACTIONS = np.geomspace(1e-4, .99, 64)
KEYS = ['design', 'signal', 'training_rows', 'validation_pairs', 'groups', 'peers', 'total_observations']


def compare(x, y):
    return (x > y) + .5 * (x == y)


def fitted(z, value, count):
    ranks = (rankdata(value, method='average') - 1) / (len(value) - 1)
    return np.array([ranks[z == j].mean() if np.any(z == j) else .5 for j in range(count)])


def learning_budget(f, g, z, v, w):
    count = np.bincount(z[:, 0], minlength=len(f))
    means = [np.array([compare(value[:, 0], value[:, 1])[z[:, 0] == j].mean()
                      if count[j] else 0. for j in range(len(f))]) for value in (v, w)]
    radius = np.full(len(f), np.inf)
    occupied = count > 0
    radius[occupied] = np.sqrt(np.log(8 * len(f) / DELTA) / (2 * count[occupied]))
    lv, lw = [np.maximum(0., mean - radius) for mean in means]
    uv, uw = [np.minimum(1., mean + radius) for mean in means]
    upper = np.array([beta.ppf(1 - DELTA / (2 * len(f)), nc + 1, len(z) - nc)
                      if nc < len(z) else 1. for nc in count])
    corners = np.array([(a - f) * (b - g) for a, b in itertools.product((lv, uv), (lw, uw))])
    objective = corners.max(axis=0)
    fit = linprog(-objective * 1e6, A_eq=np.ones((1, len(f))), b_eq=[1.],
                  bounds=list(zip(np.zeros(len(f)), upper)), method='highs',
                  options={'primal_feasibility_tolerance': 1e-10, 'dual_feasibility_tolerance': 1e-10})
    assert fit.success
    return float(np.dot(objective, fit.x))


def group_u(v, w, f, g):
    # Numeric compression of the four observed (v,w) types; no producer inference.
    states = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    a = compare(states[:, 0, None], states[None, :, 0])
    b = compare(states[:, 1, None], states[None, :, 1])
    codes = (2 * v + w).astype(int)
    counts = np.stack([(codes == k).sum(axis=1) for k in range(4)], axis=1)
    av = np.take_along_axis(counts @ a.T, codes, axis=1) - .5
    bw = np.take_along_axis(counts @ b.T, codes, axis=1) - .5
    joint = np.take_along_axis(counts @ (a*b).T, codes, axis=1) - .25
    n = v.shape[1]
    return ((av*bw-joint)/((n-1)*(n-2)) - (f*bw+g*av)/(n-1) + f*g).mean(axis=1)


def independent_triples(v, w, f, g):
    # Fixed order within each raw group; never treat the six roles as six data.
    end = 3 * (v.shape[1] // 3)
    v, w, f, g = [value[:, :end].reshape(-1, 3) for value in (v, w, f, g)]
    out = np.zeros(len(v))
    for i, j, k in itertools.permutations(range(3)):
        out += (compare(v[:, i], v[:, j]) - f[:, i]) * (compare(w[:, i], w[:, k]) - g[:, i]) / 6
    return out


def variance_inference(mean, h, width, bias, independent=False):
    count, s2 = len(h), float(h.var(ddof=1))
    if independent:
        assert np.isclose(mean, h.mean(), rtol=1e-12, atol=1e-14)
    a = np.sqrt(2*s2/count)
    c = 7*width/(3*(count-1)) if independent else 2*width/np.sqrt(count*(count-1)) + width/(3*count)
    x = np.log(2/(ALPHA-DELTA))
    radius = a*np.sqrt(x)+c*x
    if not independent:
        radius = min(radius, width*np.sqrt(x/(2*count)))
    margin = max(mean-bias, 0.)
    root = 2*margin/(a+np.sqrt(a*a+4*c*margin)) if margin else 0.
    exponent = root*root
    if not independent:
        exponent = max(exponent, 2*count*(margin/width)**2)
    return dict(p=float(min(1., DELTA+2*np.exp(-exponent))), radius=float(radius),
                lower_bound=float(mean-bias-radius))


def one(rep):
    rows = []
    for di, (design, probabilities, success) in enumerate(DESIGNS):
      for si, signal in enumerate(SIGNALS):
        seed = [SEED, rep, di, si]
        streams = [np.random.default_rng(np.random.SeedSequence(seed+[part])) for part in (0, 1, 2)]
        z, v, w = generator.sample(streams[0], (1600, 64), probabilities, success, signal)
        tz, tv, tw = generator.sample(streams[1], (TRAIN,), probabilities, success, signal)
        vz, vv, vw = generator.sample(streams[2], (8192, 2), probabilities, success, signal)
        f, g = fitted(tz, tv, len(probabilities)), fitted(tz, tw, len(probabilities))
        mu = .5 - .5 * probabilities @ success + .5 * success
        bias = float(probabilities @ ((mu-f)*(mu-g)))
        target = float(.25 * signal * probabilities @ (success*(1-success)))
        corners = np.array([(a-f)*(b-g) for a, b in itertools.product((0., 1.), repeat=2)])
        lower, upper = float(corners.min()), float(corners.max())
        gz, gv, gw = z[:400], v[:400], w[:400]
        means = group_u(gv, gw, f[gz], g[gz])
        h_all = independent_triples(gv, gw, f[gz], g[gz])
        budgets = {size: learning_budget(f, g, vz[:size], vv[:size], vw[:size]) for size in VALIDATION}
        for groups in GROUPS:
          h = h_all[:groups*21]
          pv, pw, pz = [value[:groups].reshape(1, -1) for value in (v, w, z)]
          pooled_h = independent_triples(pv, pw, f[pz], g[pz])
          pooled_mean = float(group_u(pv, pw, f[pz], g[pz])[0])
          for validation in VALIDATION:
            allowance = budgets[validation]
            setting = dict(replication=rep, design=design, signal=signal, training_rows=TRAIN,
                           validation_pairs=validation, groups=groups, peers=64,
                           total_observations=TRAIN+2*validation+groups*64,
                           target=target, exact_bias=bias, signed_allowance=allowance,
                           allowance_covers=bias <= allowance, kernel_lower=lower, kernel_upper=upper)
            definitions = [('signed_variance', float(means[:groups].mean()), h),
                           ('independent_bernstein', float(h.mean()), h),
                           ('independent_betting', float(h.mean()), h),
                           ('pooled_variance', pooled_mean, pooled_h),
                           ('pooled_betting', float(pooled_h.mean()), pooled_h)]
            for method, mean, scores in definitions:
              if method.endswith('betting'):
                out = mixture_betting_pvalue(scores, lower, upper, allowance, DELTA, FRACTIONS)
                out.update(radius=np.nan, lower_bound=np.nan)
              else:
                out = variance_inference(mean, scores, upper-lower, allowance,
                                         independent=method == 'independent_bernstein')
                out['log_evalue'] = np.nan
              rows.append(dict(**setting, method=method, mean=mean, effective_triples=len(scores),
                               kernel_sample_variance=float(scores.var(ddof=1)),
                               reject=out['p'] < ALPHA, **out))
    return rows


def exact_interval(events, total):
    return (0. if events == 0 else float(beta.ppf(.025, events, total-events+1)),
            1. if events == total else float(beta.ppf(.975, events+1, total-events)))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    output = parser.parse_args().output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(frozen_utc=datetime.now(timezone.utc).isoformat(), replications=REPS, seed=SEED,
                    original_replications=list(range(REPS)), designs=[entry[0] for entry in DESIGNS],
                    signals=SIGNALS, groups=GROUPS, peers=64, training_rows=TRAIN,
                    validation_pairs=VALIDATION, fits='estimated only', alpha=ALPHA, delta=DELTA,
                    methods=METHODS, fixed_betting_fractions=FRACTIONS.tolist(), uniform_mixture_weights=True,
                    primary='signed_variance versus independent_betting, identical within-group disjoint triple inputs',
                    secondary='independent_bernstein; pooled_variance versus pooled_betting with fixed pooled triples',
                    calibration_levels=[.001, .005, .01, .05], interval='pointwise exact two-sided 95% Clopper-Pearson',
                    generation_shape=[1600,64], analyzed_prefix_groups=GROUPS,
                    scope='Post-exposure comparator extension on original visible synthetic streams; no new domain or independent confirmation. '
                          'Unused generated suffix never enters fitting, validation, evaluation or tuning. Every declared cell is reported; no method selection.',
                    source_sha256={str(p.relative_to(HERE)) if p.is_relative_to(HERE) else p.name:
                                   hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in (Path(__file__), HERE/'classical_reference_comparators.py', GENERATOR)})
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    batches = []
    with ProcessPoolExecutor(max_workers=2) as executor:
        for index, rows in enumerate(executor.map(one, range(REPS))):
            batches.extend(rows)
            if (index+1) % 50 == 0:
                print('Completed replications', index+1, flush=True)
    frame = pd.DataFrame(batches)
    frame.to_csv(output/'replications.csv.gz', index=False, float_format='%.17g',
                 compression={'method':'gzip','mtime':0})
    summary = []
    for key, group in frame.groupby(KEYS+['method'], sort=False):
        for level in protocol['calibration_levels']:
            events = int((group.p < level).sum())
            low, high = exact_interval(events, len(group))
            summary.append(dict(zip(KEYS+['method'], key), level=level, replications=len(group),
                                rejections=events, rejection_rate=events/len(group), ci_lower=low, ci_upper=high,
                                target=float(group.target.iloc[0]),
                                allowance_failures=int((~group.allowance_covers).sum()),
                                p_min=float(group.p.min()), p_median=float(group.p.median()), p_max=float(group.p.max()),
                                mean=float(group['mean'].mean()), signed_allowance=float(group.signed_allowance.mean()),
                                mean_radius=float(group.radius.mean())))
    pd.DataFrame(summary).to_csv(output/'summary.csv', index=False, float_format='%.17g')
    paired = []
    for key, group in frame.groupby(KEYS, sort=False):
        pivot = group.pivot(index='replication', columns='method', values='p')
        for old, new in [('signed_variance','independent_betting'), ('pooled_variance','pooled_betting'),
                         ('independent_bernstein','independent_betting')]:
            a, b = pivot[old] < ALPHA, pivot[new] < ALPHA
            paired.append(dict(zip(KEYS,key), first=old, second=new,
                               first_only=int((a & ~b).sum()), second_only=int((b & ~a).sum()),
                               both=int((a & b).sum()), neither=int((~a & ~b).sum()),
                               difference=float(b.mean()-a.mean())))
    pd.DataFrame(paired).to_csv(output/'paired_decisions.csv', index=False, float_format='%.17g')
    print('Complete',len(frame),'records',len(summary),'level-specific cells',flush=True)


if __name__ == '__main__':
    main()
