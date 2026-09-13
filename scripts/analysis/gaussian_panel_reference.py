"""Frozen finite-reference benchmark on explicitly specified Gaussian panels.

All methods share data and controls. This does not establish validity for observed public or
private panels. Incorrect mean/covariance references are retained as negative
controls, never pooled into the cells satisfying the finite theorem.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.linalg import cho_solve
from scipy.stats import beta as beta_distribution, norm, nct

import canonical_baselines as canonical
import conditional_gaussian_reference as reference

METHODS = ["coarse", "fine", "extrapolation_beta1", "extrapolation_beta2", "spline"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def source_hashes():
    return {Path(p).name: sha(p) for p in [__file__, reference.__file__, canonical.__file__]}


def freeze(output):
    output.mkdir(parents=True, exist_ok=False)
    value = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "Local pre-outcome specification; earlier failed scalar/panel benchmarks were known. Not externally registered.",
        "sources": source_hashes(),
        "dependencies": {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "scikit-learn"]},
        "seed": 928202609, "replications": 1000, "periods": [12, 36], "units": 6,
        "serial_correlations": [0., .6], "within_period_correlation": .4,
        "effects": [0., .15, .35], "specifications": ["correct", "omitted_quadratic_mean", "omitted_dependence"],
        "design": "Four fixed covariate designs: one per periods/serial-correlation cell. z is Gaussian with deterministic entity offsets; mean basis is six entity indicators plus z and z squared. Two contiguous whole-period nuisance folds. No ranks, ties or data-dependent covariance fitting.",
        "dgp": "Given fixed Phi and known V=kron(AR1(rho),compound_symmetry(.4)), X=Phi gamma_X+L ex, Y=Phi gamma_Y+delta L ex+L ey. ex,ey and independent pilot ep are iid standard Gaussian vectors. Pilot X0=Phi gamma_X+L ep. Same X, pilot and innovation arrays paired across effects and reference specifications. gamma_X: entity coefficients linspace(-1,1), z=2,z²=3; gamma_Y: entity coefficients linspace(.5,-.5), z=.3,z²=.8.",
        "methods": "Five original residual-product directions (8/32 bins, beta1/beta2 on 8/12/16/24/32, cubic spline); their normal HAC lag2 references and conditional Gaussian t references. Two classical leave-cluster directions, raw X and X minus independent-pilot fitted mean, plus classical GLS score; eight finite-reference methods in total. Every method has access to the same n pilot scores, zero pilot outcome labels and n evaluation outcomes; no claim of a cost advantage.",
        "null_scope": "Y|X,Z has the stated linear Gaussian mean and known covariance up to scalar ONLY in correct reference cells at delta0. Exact t controls the sharp Gaussian no-additional-mean null, not every zero-covariance distribution. Unknown covariance, missing mean directions, ranked outcomes and outcome-chosen score directions are excluded.",
        "negative_controls": "omitted_quadratic_mean removes z² from the declared mean basis but keeps true V; omitted_dependence uses identity covariance but keeps full Phi. These violate the reference theorem by design. They are reported, not treated as alternative valid certificates.",
        "learning": "Fit score-mean coefficients from the independent pilot by ordinary least squares on the declared basis. No evaluation outcomes enter the score direction. This tests whether learning improves the raw leave-cluster direction, not whether this classical construction is novel or better than GLS.",
        "analysis": "Retain all 36 design/effect/reference cells and all 13 methods (468 cells, 468000 rows), every p-value and paired decision. Four fixed designs have 4000 independent primitive innovation draws in total; paired effects create 12000 distinct evaluated panels; three reference specifications reuse each panel. Wilson95 intervals and paired learning-minus-raw differences with descriptive normal MC intervals. Primary learning comparison: delta=.15, correct reference, four fixed designs; one-sided exact discordance binomial tests with Holm correction across all four. Delta=.35 is secondary and all failures remain. All correct-null finite-reference cells receive simultaneous Bonferroni 99% binomial intervals. Empirical consistency criterion: every interval includes .05; this is a diagnostic, not the proof. Record direction cosines, degrees of freedom and GLS conditional noncentral-t power. Do not use ordinary noncentral-t power for off-axis directions.",
        "failure_policy": "Exclusive output creation; stop and preserve partial files on error. No scientific tuning or rerun based on outcomes.",
        "resources": {"cpus": 1, "threads": 1, "gpu": False, "estimated_minutes": 5, "memory_gb": 1},
    }
    write_json(output / "protocol.json", value)
    print(json.dumps({"protocol": str(output / "protocol.json"), "sha256": sha(output / "protocol.json")}))


def design_bank(periods, rho, config, index):
    units = config["units"]
    n = periods * units
    rng = np.random.default_rng(config["seed"] + 100000 * index)
    entity = np.tile(np.arange(units), periods)
    clusters = np.repeat(np.arange(periods), units)
    z = rng.normal(size=n) + .25 * (entity - (units - 1) / 2)
    phi = np.column_stack([np.eye(units)[entity], z, z**2])
    covariance = np.kron(rho ** np.abs(np.subtract.outer(np.arange(periods), np.arange(periods))),
                         .4 * np.ones((units, units)) + .6 * np.eye(units))
    lower = np.linalg.cholesky(covariance)
    gx = np.r_[np.linspace(-1, 1, units), 2., 3.]
    gy = np.r_[np.linspace(.5, -.5, units), .3, .8]
    ex, ey, ep = [lower @ rng.normal(size=(n, config["replications"])) for _ in range(3)]
    x = phi @ gx[:, None] + ex
    pilot = phi @ gx[:, None] + ep
    folds = (clusters >= periods // 2).astype(int)
    residual_operators = [np.eye(n) - canonical.binned_predictions(np.eye(n), z, folds, int(q)) for q in canonical.Q_LADDER]
    spline = np.eye(n) - canonical.spline_predictions(np.eye(n), z, folds)
    b2 = canonical.Q_LADDER.astype(float)**-2
    w2 = np.linalg.lstsq(np.column_stack([np.ones(5), b2]).T, [1., 0.], rcond=None)[0]
    operators = {"coarse": residual_operators[0].T @ residual_operators[0] / n,
                 "fine": residual_operators[-1].T @ residual_operators[-1] / n,
                 "extrapolation_beta1": sum(w * r.T @ r for w, r in zip(canonical.WEIGHTS, residual_operators)) / n,
                 "extrapolation_beta2": sum(w * r.T @ r for w, r in zip(w2, residual_operators)) / n,
                 "spline": spline.T @ spline / n}
    return locals()


def normal_scores(bank, y):
    n, periods, units = bank["n"], bank["periods"], bank["units"]
    products = [(r @ bank["x"]) * (r @ y) for r in bank["residual_operators"]]
    scores = {"coarse": products[0], "fine": products[-1],
              "extrapolation_beta1": sum(w * s for w, s in zip(canonical.WEIGHTS, products)),
              "extrapolation_beta2": sum(w * s for w, s in zip(bank["w2"], products)),
              "spline": (bank["spline"] @ bank["x"]) * (bank["spline"] @ y)}
    result = {}
    for name, score in scores.items():
        mean = score.mean(axis=0)
        sums = (score - mean).reshape(periods, units, -1).sum(axis=1)
        variance = np.sum(sums*sums, axis=0)
        for lag in [1, 2]:
            variance += 2 * (1 - lag / 3) * np.sum(sums[lag:] * sums[:-lag], axis=0)
        statistic = mean / (np.sqrt(np.maximum(variance, 0)) / n)
        np.testing.assert_allclose(mean, np.sum((bank["operators"][name] @ bank["x"]) * y, axis=0), atol=1e-10, rtol=1e-10)
        result[name + "_normal"] = {"statistic": statistic, "pvalue": norm.sf(statistic)}
    return result


def run(output):
    config = json.loads((output / "protocol.json").read_text())
    if source_hashes() != config["sources"]:
        raise ValueError("source drift after freeze")
    if config["dependencies"] != {k: importlib.metadata.version(k) for k in config["dependencies"]}:
        raise ValueError("dependency drift")
    start = time.perf_counter()
    write_json(output / "started.json", {"utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": sha(output / "protocol.json")})
    try:
        _run(output, config, start)
    except BaseException as error:
        write_json(output / "failed.json", {"error": repr(error), "traceback": traceback.format_exc()})
        raise


def _run(output, config, start):
    summaries, paired, fixtures = [], [], {}
    fields = ["design", "periods", "rho", "specification", "effect", "replication", "method", "statistic", "pvalue", "reject", "direction_cosine"]
    null_cell_count = len(config["periods"]) * len(config["serial_correlations"]) * 8
    ci_tail = .01 / (2 * null_cell_count)
    raw_path = output / "replications.csv.gz"
    with raw_path.open("xb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
        import io
        handle = io.TextIOWrapper(compressed, newline="")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        index = 0
        for periods in config["periods"]:
            for rho in config["serial_correlations"]:
                bank = design_bank(periods, rho, config, index)
                for spec in config["specifications"]:
                    phi = bank["phi"][:, :-1] if spec == "omitted_quadratic_mean" else bank["phi"]
                    covariance = np.eye(bank["n"]) if spec == "omitted_dependence" else bank["covariance"]
                    ref = reference.GaussianReference(phi, covariance)
                    leave = reference.leave_cluster_operator(phi, bank["clusters"])
                    learned = phi @ np.linalg.lstsq(phi, bank["pilot"], rcond=None)[0]
                    directions = {name: a @ bank["x"] for name, a in bank["operators"].items()}
                    directions["leave_cluster_raw"] = leave.T @ bank["x"]
                    directions["leave_cluster_learned"] = leave.T @ (bank["x"] - learned)
                    directions["gls"] = cho_solve((ref.cholesky, True), bank["x"])
                    signal = ref.project(ref.whiten(bank["x"]))
                    signal_norm = np.linalg.norm(signal, axis=0)
                    for effect in config["effects"]:
                        y = bank["phi"] @ bank["gy"][:, None] + effect * bank["ex"] + bank["ey"]
                        outputs = normal_scores(bank, y)
                        for name, direction in directions.items():
                            outputs[name + "_gaussian"] = ref.evaluate(y, direction)
                        for name, result in outputs.items():
                            p = result["pvalue"]
                            if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                                raise ArithmeticError("invalid p-values")
                            rejected = p <= .05
                            k, count = int(rejected.sum()), len(p)
                            cosine = np.full(count, np.nan)
                            if "direction" in result:
                                v = result["direction"]
                                cosine = np.sum(v * signal, axis=0) / (np.linalg.norm(v, axis=0) * signal_norm)
                                if np.max(np.abs(cosine)) > 1 + 1e-10:
                                    raise ArithmeticError("Cauchy-Schwarz failed")
                            row = {"design": index, "periods": periods, "rho": rho, "specification": spec, "effect": effect,
                                   "method": name, "replications": count, "rejections": k, "rate": k/count,
                                   "wilson95": canonical.wilson(k, count), "mean_statistic": float(np.mean(result["statistic"])),
                                   "mean_direction_cosine": float(np.mean(cosine)) if "direction" in result else None,
                                   "finite_reference_assumptions_hold": spec == "correct" and name.endswith("_gaussian"),
                                   "df": result.get("df")}
                            if spec == "correct" and effect == 0 and name.endswith("_gaussian"):
                                interval = [float(beta_distribution.ppf(ci_tail, k, count-k+1)) if k else 0.,
                                            float(beta_distribution.ppf(1-ci_tail, k+1, count-k)) if k < count else 1.]
                                row["simultaneous99_binomial_interval"] = interval
                                row["contains_nominal_point05"] = interval[0] <= .05 <= interval[1]
                            if spec == "correct" and name == "gls_gaussian":
                                row["mean_exact_conditional_power"] = float(np.mean(nct.sf(float(reference.t.ppf(.95, ref.dimension-1)), ref.dimension-1, effect*signal_norm)))
                            summaries.append(row)
                            for replication in range(count):
                                writer.writerow({"design": index, "periods": periods, "rho": rho, "specification": spec,
                                                 "effect": effect, "replication": replication, "method": name,
                                                 "statistic": float(result["statistic"][replication]), "pvalue": float(p[replication]),
                                                 "reject": int(rejected[replication]),
                                                 "direction_cosine": "" if np.isnan(cosine[replication]) else float(cosine[replication])})
                        one = outputs["leave_cluster_learned_gaussian"]["pvalue"] <= .05
                        two = outputs["leave_cluster_raw_gaussian"]["pvalue"] <= .05
                        difference = one.astype(float) - two.astype(float)
                        error = float(difference.std(ddof=1) / np.sqrt(len(difference)))
                        paired.append({"design": index, "specification": spec, "effect": effect,
                                       "learned_only": int(np.sum(one & ~two)), "raw_only": int(np.sum(two & ~one)),
                                       "both": int(np.sum(one & two)), "neither": int(np.sum(~one & ~two)),
                                       "difference": float(difference.mean()),
                                       "paired_normal_mc95": [float(difference.mean()-1.96*error), float(difference.mean()+1.96*error)]})
                        key = f"d{index}_{spec}_e{effect}"
                        fixtures[key + "_y"] = y[:, :3]
                    fixtures[f"d{index}_{spec}_phi"] = phi
                    fixtures[f"d{index}_{spec}_covariance"] = covariance
                    fixtures[f"d{index}_{spec}_leave"] = leave
                    for name, direction in directions.items():
                        fixtures[f"d{index}_{spec}_{name}_direction"] = direction[:, :3]
                fixtures[f"d{index}_x"] = bank["x"][:, :3]
                fixtures[f"d{index}_pilot"] = bank["pilot"][:, :3]
                print(f"Completed fixed design {index}, periods={periods}, rho={rho}", flush=True)
                handle.flush()
                index += 1
        handle.flush()
        handle.detach()
    np.savez_compressed(output / "verification_fixtures.npz", **fixtures)
    write_json(output / "summary.json", {"status": "COMPLETE", "protocol_sha256": sha(output / "protocol.json"),
               "source_hashes": source_hashes(), "raw_sha256": sha(raw_path), "fixture_sha256": sha(output / "verification_fixtures.npz"),
               "seconds": time.perf_counter()-start, "method_cells": len(summaries),
               "raw_rows": len(summaries)*config["replications"], "independent_primitive_draws": index*config["replications"],
               "distinct_evaluated_panels": index*config["replications"]*len(config["effects"]), "rows": summaries,
               "paired_learning_comparison": paired,
               "limits": config["null_scope"] + " " + config["negative_controls"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "run"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (freeze if args.command == "freeze" else run)(args.output)
