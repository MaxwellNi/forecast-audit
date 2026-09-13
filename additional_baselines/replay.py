"""Reproduce supplementary synthetic comparisons; see README.md."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run_suite(suite, root_str, source_str):
    root, source = Path(root_str), Path(source_str)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source / "scripts" / "analysis"))
    started = time.monotonic()
    with (root / f"{suite}.stdout.log").open("x") as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                result = _run_suite(suite, root)
                result.update(suite=suite, status="COMPLETE",
                              elapsed_seconds=time.monotonic() - started)
            except BaseException:
                result = dict(suite=suite, status="FAILED", traceback=traceback.format_exc(),
                              elapsed_seconds=time.monotonic() - started)
                (root / f"{suite}.status.json").write_text(json.dumps(result, indent=2) + "\n")
                traceback.print_exc()
                raise
            (root / f"{suite}.status.json").write_text(json.dumps(result, indent=2) + "\n")
            return result

def _run_suite(suite, root):
    import numpy as np
    names = {
        "adaptive_beta": "adaptive_beta",
        "ablation": "ablation_matrix",
        "generalization": "generalization_grid",
        "stress": "null_suite_stress",
    }
    module = importlib.import_module(names[suite])
    native = root / "original_driver_outputs" / suite
    native.mkdir(parents=True, exist_ok=False)
    module.OUT = native
    # Ablation/grid construct OUT locally from ROOT after computation. Imports
    # have already resolved to frozen source; this assignment changes output only.
    if suite in ("ablation", "generalization"):
        module.ROOT = native
    raw_path = root / f"{suite}.raw.csv"
    raw = raw_path.open("x", newline="")
    count = 0
    current = {}
    if suite == "adaptive_beta":
        fields = ["regime", "metric", "seed", "N", "T", "beta_hat",
                  "fixed_statistic", "adaptive_statistic", "fixed_reject",
                  "adaptive_reject", "fixed_mean", "adaptive_mean", "fixed_se",
                  "adaptive_se", "fixed_weight_norm_squared",
                  "adaptive_weight_norm_squared", "spectrum_means"]
    else:
        fields = ["setting", "regime", "metric", "seed", "N", "T", "reject"]
        if suite == "ablation":
            fields += ["statistic", "product_mean", "cluster_se"]
        if suite == "generalization":
            fields += ["overlap"]
    writer = csv.DictWriter(raw, fieldnames=fields)
    writer.writeheader()

    def record(row):
        nonlocal count
        writer.writerow(row)
        count += 1
        if count % 50 == 0:
            raw.flush()

    if suite == "adaptive_beta":
        spectrum_original = module._spectrum
        beta_original = module._beta_hat
        se_original = module._clustered_se
        regimes = ["smooth", "nonlinear", "large_n", "heavy_tail"]

        def spectrum(*args):
            current.clear()
            case, seed_index = divmod(count, 50)
            regime = regimes[case // 2]
            metric = ["typeI", "power"][case % 2]
            P = spectrum_original(*args)
            current.update(regime=regime, metric=metric, seed=505 + seed_index,
                           N=args[5], T=args[6],
                           spectrum_means=json.dumps(P.mean(0).tolist()),
                           fixed_weight_norm_squared=float(module.W1 @ module.W1))
            return P

        def beta(means):
            value = beta_original(means)
            weights = module._w(value)
            current.update(beta_hat=value,
                           adaptive_weight_norm_squared=float(weights @ weights))
            return value

        def se(products, month):
            value = se_original(products, month)
            prefix = "adaptive" if "fixed_se" in current else "fixed"
            statistic = float(products.mean() / (value + 1e-18))
            current.update({prefix + "_mean": float(products.mean()),
                            prefix + "_se": value, prefix + "_statistic": statistic,
                            prefix + "_reject": int(statistic > module.ZA)})
            if prefix == "adaptive":
                record(current)
            return value

        module._spectrum, module._beta_hat, module._clustered_se = spectrum, beta, se
        module.main()  # Original run defaults: 50 seeds beginning at 505.
        expected = 400

    elif suite == "ablation":
        rate_original, cert_original = module._rate, module.cert
        se_original = module._clustered_se

        def rate(cfg, regime, delta, n_seeds, seed0):
            current.clear()
            current.update(setting=cfg[0], regime=regime,
                           metric="typeI" if delta == 0 else "power",
                           seed0=seed0, seed_index=0)
            return rate_original(cfg, regime, delta, n_seeds, seed0)

        def se(products, month):
            value = se_original(products, month)
            current.update(statistic=float(products.mean() / (value + 1e-18)),
                           product_mean=float(products.mean()), cluster_se=value)
            return value

        def cert(*args):
            rejected = cert_original(*args)
            assert rejected == int(current["statistic"] > module.ZA)
            row = {k: current[k] for k in ["setting", "regime", "metric", "statistic",
                                           "product_mean", "cluster_se"]}
            row.update(seed=current["seed0"] + current["seed_index"],
                       N=args[5], T=args[6], reject=int(rejected))
            record(row)
            current["seed_index"] += 1
            return rejected

        module._rate, module.cert, module._clustered_se = rate, cert, se
        module.main()  # Original defaults: 150 seeds, large_n 80, seed0 70707.
        expected = 9540

    elif suite == "generalization":
        rate_original = module._rate
        labels = {id(fn): label for label, fn in module.TESTS}

        def rate(fn, regime, delta, overlap, n_seeds, seed0):
            seed_index = 0

            def logged(*args):
                nonlocal seed_index
                rejected = fn(*args)
                record(dict(setting=labels[id(fn)], regime=regime,
                            metric="typeI" if delta == 0 else "power",
                            overlap=overlap, seed=seed0 + seed_index,
                            N=args[5], T=args[6], reject=int(rejected)))
                seed_index += 1
                return rejected

            return rate_original(logged, regime, delta, overlap, n_seeds, seed0)

        module._rate = rate
        module.main()  # Original defaults: 150 seeds, large_n 80, seed0 80808.
        expected = 42400

    elif suite == "stress":
        ours_original = module.extrapolated_covariance
        seen = {}

        def wrap_dgp(regime, original):
            def dgp(rng, delta, overlap):
                metric = "typeI" if delta == 0 else "power"
                key = (regime, metric)
                index = seen.get(key, 0)
                seen[key] = index + 1
                current.clear()
                current.update(setting="legacy_cluster_extrapolation_beta1",
                               regime=regime, metric=metric, seed=505 + index)
                return original(rng, delta, overlap)
            return dgp

        module.STRESS = {key: wrap_dgp(key, fn) for key, fn in module.STRESS.items()}

        def ours(*args):
            rejected = ours_original(*args)
            record(dict(current, N=args[5], T=args[6], reject=int(rejected)))
            return rejected

        module.extrapolated_covariance = ours
        module.main()  # Original SEEDS: high_dim_fe 100, other regimes 200.
        expected = 1400
    raw.close()
    assert count == expected, (suite, count, expected)
    return dict(raw_rows=count, expected_rows=expected, raw_sha256=sha(raw_path))


PACKAGE = Path(__file__).resolve().parent

def verify_sources(protocol):
    paths=[]
    for item in protocol['sources']:
        path=(PACKAGE/item['path']).resolve()
        if PACKAGE not in path.parents or path.suffix != '.py':
            raise ValueError('Source path leaves the standalone package')
        if sha(path) != item['sha256']:
            raise ValueError('Frozen synthetic source hash mismatch: '+item['path'])
        paths.append(path)
    if len(paths) != 10 or len(set(paths)) != 10:
        raise ValueError('Expected exactly ten distinct frozen synthetic modules')
    return paths

def check_loaders(protocol):
    paths=verify_sources(protocol)
    source=PACKAGE/'implementation/scripts/analysis'
    sys.path.insert(0,str(source))
    loaded=[]
    for path in paths:
        module=importlib.import_module(path.stem)
        if Path(module.__file__).resolve() != path:
            raise RuntimeError('A synthetic module resolved outside this package')
        loaded.append(path.stem)
    grid=importlib.import_module('generalization_grid')
    for label,fn in grid.TESTS:
        if source not in Path(fn.__code__.co_filename).resolve().parents:
            raise RuntimeError('A grid comparator resolved outside this package')
    return dict(status='LOADER_AND_HASH_CHECK_PASS',frozen_modules=len(loaded),
                grid_functions=len(grid.TESTS),simulation_replications_run=0)

def main():
    sys.dont_write_bytecode=True
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[name]='1'
    parser=argparse.ArgumentParser(description='Reproduce supplementary synthetic diagnostic comparisons')
    parser.add_argument('--output',type=Path,help='A NEW output directory; existing destinations are refused')
    parser.add_argument('--check',action='store_true',help='Check hashes and loader graph only; no simulations')
    args=parser.parse_args()
    protocol_path=PACKAGE/'protocol.json'
    protocol=json.loads(protocol_path.read_text())
    verify_sources(protocol)
    if sha(__file__) != protocol['portable_runner_sha256']:
        raise ValueError('Portable runner hash mismatch')
    if args.check:
        print(json.dumps(check_loaders(protocol),indent=2))
        return
    if args.output is None:
        parser.error('--output is required unless --check is used')
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    source=PACKAGE/'implementation'
    (root/'RUN_STARTED.json').write_text(json.dumps(dict(
        protocol_sha256=sha(protocol_path),original_protocol_sha256=protocol['original_protocol_sha256'],
        started_unix=time.time(),suites=protocol['suites'],workers=2))+'\n')
    results=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run_suite,suite,str(root),str(source)) for suite in protocol['suites']]
        for future in as_completed(futures):
            results.append(future.result())
            print(json.dumps({k:results[-1][k] for k in ['suite','status','raw_rows']}),flush=True)
    verify_sources(protocol)
    (root/'RUN_COMPLETED.json').write_text(json.dumps(dict(
        status='ALL_PRESCRIBED_RUNS_COMPLETE',protocol_sha256=sha(protocol_path),
        original_protocol_sha256=protocol['original_protocol_sha256'],
        completed_unix=time.time(),results=results,frozen_input_hashes_unchanged=True),indent=2)+'\n')

if __name__=='__main__':
    main()
