"""Prespecified smooth-bin illustration, with independent scalar observations.

This is a new validation experiment, not a replay of the fixed panel ladder.
Known uniform bin boundaries isolate approximation bias from quantile estimation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

PROTOCOL = {
    "sample_sizes": [2000, 8000, 32000], "replications": 300,
    "seed_start": 2026090500, "alpha": 0.05,
    "designs": ["linear", "quadratic", "smooth_nonlinear"],
    "noise_correlations": [0.0, 0.2],
    "bin_ladder": "ceil(n**0.2) times (1,2,4)",
    "nuisance": "twofold histogram means; known uniform bin boundaries",
    "methods": ["coarse", "finest", "extrapolation_beta1", "extrapolation_beta2"],
    "sampling": "independent U uniform(0,1) and bivariate standard normal errors",
    "claim": "illustration of a separately proved smooth scalar special case; no panel validity claim",
}


def fit_histogram(y, u, q, folds):
    bins = np.minimum((u * q).astype(int), q - 1)
    predictions = np.empty(len(y))
    for fold in (0, 1):
        train = folds != fold
        counts = np.bincount(bins[train], minlength=q)
        totals = np.bincount(bins[train], weights=y[train], minlength=q)
        means = np.full(q, float(y[train].mean()))
        np.divide(totals, counts, out=means, where=counts > 0)
        predictions[~train] = means[bins[~train]]
    return predictions


def intercept_weights(qs, beta):
    basis = np.column_stack([np.ones(len(qs)), np.asarray(qs, float) ** (-beta)])
    return np.linalg.lstsq(basis.T, np.array([1., 0.]), rcond=None)[0]


def functions(u, design):
    if design == "linear":
        return 12 * u, 12 * u
    if design == "quadratic":
        return 12 * u ** 2, 12 * u ** 2
    if design == "smooth_nonlinear":
        return 6 * np.sin(2 * np.pi * u), 6 * np.sin(2 * np.pi * u)
    raise ValueError(design)


def wilson(hits, total):
    z = float(norm.ppf(.975))
    rate = hits / total
    center = (rate + z*z / (2*total)) / (1 + z*z/total)
    radius = z * np.sqrt(rate*(1-rate)/total + z*z/(4*total*total)) / (1 + z*z/total)
    return center-radius, center+radius


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / "protocol.json"
    payload = {**PROTOCOL, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if args.freeze_only:
        if protocol_path.exists():
            raise FileExistsError(protocol_path)
        protocol_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(protocol_path)
        return
    if not protocol_path.exists() or json.loads(protocol_path.read_text()) != payload:
        raise ValueError("freeze the exact protocol before running")
    if (args.output / "draws.csv").exists():
        raise FileExistsError("refusing to overwrite an existing run")
    rows = []
    started = time.time()
    for n in PROTOCOL["sample_sizes"]:
        q = int(np.ceil(n ** .2))
        qs = q * np.array([1, 2, 4])
        weights = [np.array([1, 0, 0]), np.array([0, 0, 1]), intercept_weights(qs, 1), intercept_weights(qs, 2)]
        for design_index, design in enumerate(PROTOCOL["designs"]):
            for rho in PROTOCOL["noise_correlations"]:
                for seed in range(PROTOCOL["replications"]):
                    rng = np.random.default_rng(PROTOCOL["seed_start"] + seed + 10000*design_index + n)
                    u = rng.uniform(size=n)
                    noise_x, noise_y = rng.normal(size=(2, n))
                    f, g = functions(u, design)
                    x = f + noise_x
                    y = g + rho*noise_x + np.sqrt(1-rho*rho)*noise_y
                    folds = np.arange(n) % 2
                    products = np.column_stack([(x-fit_histogram(x,u,int(b),folds))*(y-fit_histogram(y,u,int(b),folds)) for b in qs])
                    for method, w in zip(PROTOCOL["methods"], weights):
                        score = products @ w
                        statistic = float(score.mean() / (score.std(ddof=1) / np.sqrt(n)))
                        rows.append({"n": n, "q_min": q, "design": design, "noise_correlation": rho,
                                     "seed": seed, "method": method, "statistic": statistic,
                                     "reject": bool(statistic > norm.ppf(.95))})
                print(n, design, rho, "complete", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "draws.csv", index=False)
    summary = []
    for keys, data in frame.groupby(["n", "q_min", "design", "noise_correlation", "method"], sort=False):
        hits = int(data.reject.sum())
        lo, hi = wilson(hits, len(data))
        summary.append(dict(zip(["n", "q_min", "design", "noise_correlation", "method"], keys),
                            hits=hits, draws=len(data), rate=hits/len(data), ci_lower=lo, ci_upper=hi,
                            statistic_mean=float(data.statistic.mean())))
    pd.DataFrame(summary).to_csv(args.output / "summary.csv", index=False)
    (args.output / "receipt.json").write_text(json.dumps({"elapsed_seconds": time.time()-started,
        "draws": len(frame), "generated_datasets": len(frame)//4, "independent_noise_draws": len(frame)//8,
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "numpy": np.__version__, "summary": summary}, indent=2,
        default=lambda value: value.item() if isinstance(value, np.generic) else str(value))+"\n")


if __name__ == "__main__":
    main()
