"""Formula-based distance summaries and a matched-sample KCI/GCM benchmark.

The KCI baseline calls causal-learn without replacing its kernel statistic.
Partial distance correlation measures a different population quantity and is
reported as a coefficient, never as a conditional-independence p-value here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer

Q_LADDER = np.array([8, 12, 16, 24, 32])
DESIGN = np.column_stack([np.ones(5), 1.0 / Q_LADDER])
WEIGHTS = np.linalg.lstsq(DESIGN.T, np.array([1.0, 0.0]), rcond=None)[0]
METHODS = ("gcm_coarse", "gcm_fine", "gcm_spline", "extrapolated", "kci_gamma")


def _matrix(values):
    a = np.asarray(values, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2 or len(a) < 4 or not np.isfinite(a).all():
        raise ValueError("Expected at least four finite vector observations")
    return a


def u_centered_distances(values):
    """Székely and Rizzo (2014), Eq. (3.1), with Euclidean distances."""
    a = _matrix(values)
    n = len(a)
    d = squareform(pdist(a, metric="euclidean"))
    out = d - d.sum(axis=0)[None, :] / (n - 2)
    out -= d.sum(axis=1)[:, None] / (n - 2)
    out += d.sum() / ((n - 1) * (n - 2))
    np.fill_diagonal(out, 0.0)
    return out


def u_inner(a, b):
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] != a.shape[1] or len(a) < 4:
        raise ValueError("U-inner product requires equal square matrices, n >= 4")
    return float(np.sum(a * b) / (len(a) * (len(a) - 3)))


def partial_distance_summary(x, y, z):
    """U-centered projection definitions (3.6) and (3.7).

    Zero partial distance correlation is not equivalent to conditional
    independence. No permutation conditional-independence claim is made.
    """
    a, b, c = [u_centered_distances(v) for v in (x, y, z)]
    if a.shape != b.shape or a.shape != c.shape:
        raise ValueError("x, y and z must have the same number of observations")
    cc = u_inner(c, c)
    pa = a - (u_inner(a, c) / cc) * c if cc > 0 else a
    pb = b - (u_inner(b, c) / cc) * c if cc > 0 else b
    covariance = u_inner(pa, pb)
    denom = np.sqrt(max(0.0, u_inner(pa, pa) * u_inner(pb, pb)))
    # A projected vector that vanishes mathematically can retain rounding noise.
    scale = np.sqrt(max(0.0, u_inner(a, a) * u_inner(b, b)))
    correlation = covariance / denom if denom > 64 * np.finfo(float).eps * scale else 0.0
    return float(covariance), float(np.clip(correlation, -1, 1))


def binned_predictions(values, z, folds, q):
    """Fit all bin edges and conditional means inside each training fold."""
    values, z, folds = [np.asarray(v) for v in (values, z, folds)]
    if values.ndim == 1:
        values = values[:, None]
    pred = np.empty_like(values, dtype=float)
    for fold in np.unique(folds):
        tr, te = folds != fold, folds == fold
        edges = np.quantile(z[tr], np.arange(1, q) / q)
        train_bin = np.searchsorted(edges, z[tr], side="right")
        test_bin = np.searchsorted(edges, z[te], side="right")
        counts = np.bincount(train_bin, minlength=q)
        means = np.tile(values[tr].mean(axis=0), (q, 1))
        for j in range(values.shape[1]):
            totals = np.bincount(train_bin, weights=values[tr, j], minlength=q)
            np.divide(totals, counts, out=means[:, j], where=counts > 0)
        pred[te] = means[test_bin]
    return pred


def spline_predictions(values, z, folds):
    """Prespecified cubic splines, eight training-quantile knots, ridge 0.001."""
    pred = np.empty_like(values, dtype=float)
    for fold in np.unique(folds):
        tr, te = folds != fold, folds == fold
        spline = SplineTransformer(n_knots=8, degree=3, knots="quantile",
                                   extrapolation="linear", include_bias=False)
        basis = spline.fit_transform(z[tr, None])
        model = Ridge(alpha=0.001, fit_intercept=True).fit(basis, values[tr])
        pred[te] = model.predict(spline.transform(z[te, None]))
    return pred


def covariance_test(products):
    """One-sided studentized GCM score; singleton-cluster population-scale SE."""
    s = np.asarray(products, dtype=float)
    se = np.sqrt(np.sum((s - s.mean()) ** 2)) / len(s)
    if not np.isfinite(s).all() or se <= 0:
        raise ValueError("Non-finite or degenerate residual product")
    statistic = float(s.mean() / se)
    return statistic, float(norm.sf(statistic))


def draw_sample(regime, alternative, seed, n):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    if regime == "heavy_tail":
        ex, ey = rng.standard_t(5, (2, n)) / np.sqrt(5 / 3)
    else:
        ex, ey = rng.normal(size=(2, n))
    if regime == "linear":
        g = 0.8 * z
    elif regime == "nonlinear":
        g = 0.8 * (z * z - 1)
    else:
        g = 0.8 * np.tanh(1.5 * z)
    if alternative == "positive_covariance":
        x, y = g + ex + 0.5 * u, g + ey + 0.5 * u
    elif alternative == "zero_covariance_dependence":
        x, y = g + ex, g + (ex * ex - 1) / np.sqrt(2) + ey
    elif alternative == "null":
        x, y = g + ex, g + ey
    else:
        raise ValueError(alternative)
    return x, y, z


def balanced_folds(n, seed):
    folds = np.arange(n) % 2
    return np.random.default_rng(seed).permutation(folds)


def evaluate_sample(x, y, z, folds):
    from causallearn.utils.KCI.KCI import KCI_CInd
    values = np.column_stack([x, y])
    products = []
    times = {}
    scores = {}
    begin = time.perf_counter()
    for q in Q_LADDER:
        start = time.perf_counter()
        residual = values - binned_predictions(values, z, folds, int(q))
        product = residual[:, 0] * residual[:, 1]
        products.append(product)
        if q in (8, 32):
            name = "gcm_coarse" if q == 8 else "gcm_fine"
            scores[name] = covariance_test(product)
            times[name] = time.perf_counter() - start
    scores["extrapolated"] = covariance_test(np.column_stack(products) @ WEIGHTS)
    times["extrapolated"] = time.perf_counter() - begin
    start = time.perf_counter()
    residual = values - spline_predictions(values, z, folds)
    scores["gcm_spline"] = covariance_test(residual[:, 0] * residual[:, 1])
    times["gcm_spline"] = time.perf_counter() - start
    start = time.perf_counter()
    pvalue, statistic = KCI_CInd(kernelX="Gaussian", kernelY="Gaussian", kernelZ="Gaussian",
                               est_width="empirical", approx=True, use_gp=False).compute_pvalue(
                                   x[:, None], y[:, None], z[:, None])
    scores["kci_gamma"] = (float(statistic), float(pvalue))
    times["kci_gamma"] = time.perf_counter() - start
    for key, (statistic, pvalue) in scores.items():
        if not np.isfinite([statistic, pvalue]).all() or not 0 <= pvalue <= 1:
            raise ValueError(f"Invalid output from {key}")
    distance = partial_distance_summary(x, y, z)
    return scores, times, distance


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol():
    return {
        "schema": 1, "experiment": "matched_iid_canonical_baselines",
        "n": 400, "replications": 300, "alpha": 0.05, "seed_start": 620260905,
        "regimes": ["linear", "smooth", "nonlinear", "heavy_tail"],
        "alternatives": ["null", "positive_covariance"],
        "extra_cell": ["smooth", "zero_covariance_dependence"],
        "observations": "400 independent rows; identical raw x,y,z for every method; no ranking, subsampling, fixed effects, abstention, or selection",
        "nuisance_folds": "One balanced two-fold assignment per replication shared by all GCM variants. Bins, knots and response means are fit only on training rows. KCI uses its native full-sample kernel construction and has no nuisance folds.",
        "gcm_tail": "one-sided positive residual covariance; normal reference; sqrt(sum((s-mean(s))**2))/n",
        "coarse_q": 8, "fine_q": 32, "extrapolation_q": Q_LADDER.tolist(),
        "extrapolation_beta": 1, "extrapolation_weights": WEIGHTS.tolist(),
        "spline": "8 quantile knots, cubic, linear boundary extrapolation, Ridge(alpha=0.001, fit_intercept=True)",
        "kci": {"package": "causal-learn==0.1.4.8", "class": "KCI_CInd", "kernels": "Gaussian", "est_width": "empirical", "approx": True, "use_gp": False, "regularization": 0.001},
        "distance": "Canonical U-centered partial distance covariance and correlation as descriptive coefficients only; its zero null differs from conditional independence.",
        "dgp": "Z,U,ex,ey independent standard normal except heavy_tail ex,ey~t5/sqrt(5/3). g(Z)=0.8Z linear; 0.8(Z^2-1) nonlinear; 0.8tanh(1.5Z) smooth/heavy_tail. Null X=g+ex,Y=g+ey. Positive alternative adds 0.5U to both (conditional covariance 0.25). Extra smooth cell X=g+ex,Y=g+(ex^2-1)/sqrt(2)+ey, conditional covariance zero but dependent.",
        "analysis": "Report every cell, counts, Wilson 95% intervals, all p-values/statistics and paired decisions; no rejection threshold or model tuning after results. Timing excludes imports and coefficient-only distance calculation; extrapolation includes all five resolution fits.",
        "limit": "New i.i.d. diagnostic only; not a rerun or validation of the large dependent panels in the original table; normal/gamma calibration is approximate, never a finite-sample guarantee.",
    }


def freeze(path):
    from causallearn.utils.KCI import KCI
    doc = protocol()
    doc["created_utc"] = datetime.now(timezone.utc).isoformat()
    doc["script_sha256"] = sha256(__file__)
    doc["dependencies"] = {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn", "causal-learn")}
    doc["kci_source_sha256"] = sha256(inspect.getfile(KCI))
    doc["freeze_status"] = "Written locally before the simulation, not an independently timestamped preregistration."
    with path.open("x", encoding="utf-8") as handle:
        json.dump(doc, handle, indent=2)
        handle.write("\n")
    print(f"Protocol written: {path}; sha256={sha256(path)}", flush=True)


def wilson(k, n):
    z = float(norm.ppf(0.975))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [float(centre - half), float(centre + half)]


def run(path, out):
    config = json.loads(path.read_text())
    if config["script_sha256"] != sha256(__file__):
        raise RuntimeError("Script changed after protocol freeze")
    from causallearn.utils.KCI import KCI
    if sha256(inspect.getfile(KCI)) != config["kci_source_sha256"]:
        raise RuntimeError("KCI source changed after protocol freeze")
    for package, expected in config["dependencies"].items():
        if importlib.metadata.version(package) != expected:
            raise RuntimeError(f"Dependency mismatch: {package}")
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "replications.csv"
    cells = [(r, a) for r in config["regimes"] for a in config["alternatives"]]
    cells.append(tuple(config["extra_cell"]))
    summary = []
    begin = time.perf_counter()
    with rows_path.open("x", newline="", encoding="utf-8") as handle:
        fields = ["regime", "alternative", "replication", "sample_sha256", "fold_sha256", "method", "statistic", "pvalue", "reject", "seconds", "partial_distance_covariance", "partial_distance_correlation"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for ci, (regime, alternative) in enumerate(cells):
            counts = {m: 0 for m in METHODS}
            times = {m: 0.0 for m in METHODS}
            distances = []
            for replication in range(config["replications"]):
                # Separate cells have independent samples; all methods within a cell share them.
                seed = config["seed_start"] + ci * 100000 + replication
                x, y, z = draw_sample(regime, alternative, seed, config["n"])
                folds = balanced_folds(config["n"], seed + 1000000000)
                scores, elapsed, distance = evaluate_sample(x, y, z, folds)
                sample_hash = hashlib.sha256(np.column_stack([x, y, z]).astype("<f8").tobytes()).hexdigest()
                fold_hash = hashlib.sha256(folds.astype("<i8").tobytes()).hexdigest()
                distances.append(distance)
                for method in METHODS:
                    statistic, pvalue = scores[method]
                    reject = int(pvalue <= config["alpha"])
                    counts[method] += reject
                    times[method] += elapsed[method]
                    writer.writerow(dict(regime=regime, alternative=alternative, replication=replication, sample_sha256=sample_hash, fold_sha256=fold_hash, method=method, statistic=statistic, pvalue=pvalue, reject=reject, seconds=elapsed[method], partial_distance_covariance=distance[0], partial_distance_correlation=distance[1]))
                if (replication + 1) % 50 == 0:
                    handle.flush()
                    print(f"{regime}/{alternative}: {replication + 1}/{config['replications']}", flush=True)
            for method in METHODS:
                count = counts[method]
                summary.append(dict(regime=regime, alternative=alternative, method=method,
                                    n=config["n"], replications=config["replications"],
                                    rejections=count, rate=count/config["replications"],
                                    wilson95=wilson(count, config["replications"]),
                                    seconds=times[method], mean_partial_distance_correlation=float(np.mean(np.array(distances)[:, 1]))))
            print(json.dumps(summary[-len(METHODS):]), flush=True)
    receipt = {"protocol_sha256": sha256(path), "script_sha256": sha256(__file__),
               "raw_rows_sha256": sha256(rows_path), "python": platform.python_version(),
               "completed_utc": datetime.now(timezone.utc).isoformat(), "seconds": time.perf_counter()-begin,
               "rows": summary, "canonical_null_warning": config["limit"]}
    with (out / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.protocol)
    elif args.output is None:
        parser.error("run requires --output")
    else:
        run(args.protocol, args.output)


if __name__ == "__main__":
    main()
