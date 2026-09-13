#!/usr/bin/env python3
"""Reproduce this fixed numerical study in a fresh directory; never overwrite.

The new execution is a reproduction on exposed data, not a new preregistration
or independent statistical application. Supply the public source zip or request
its download explicitly.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

import numpy as np
import pandas as pd
import scipy
import sklearn

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / 'public_export' if (HERE / 'public_export/study.py').exists() else HERE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New, nonexistent directory')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--source', type=Path, help='Existing official source zip')
    source.add_argument('--download-source', action='store_true', help='Download the approximately 249 MiB official zip')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name in ['study.py', 'verify.py']:
        shutil.copyfile(REFERENCE / name, out / name)
    shutil.copytree(REFERENCE / 'helpers', out / 'helpers')
    if args.download_source:
        protocol = json.loads((REFERENCE / 'protocol.json').read_text())
        raw_source = out / 'source.zip'
        with urllib.request.urlopen(protocol['source_url'], timeout=120) as incoming, raw_source.open('wb') as target:
            shutil.copyfileobj(incoming, target)
    else:
        raw_source = args.source.resolve()
    original_protocol = json.loads((REFERENCE / 'protocol.json').read_text())
    digest = hashlib.sha256()
    with raw_source.open('rb') as stream:
        for block in iter(lambda: stream.read(2 ** 20), b''):
            digest.update(block)
    assert digest.hexdigest() == original_protocol['source_sha256'], 'Source differs from frozen study'
    for phase in ['freeze', 'prepare', 'audit']:
        command = [sys.executable, str(out / 'study.py'), phase, '--source', str(raw_source)]
        run = subprocess.run(command, check=True, capture_output=True, text=True)
        (out / f'reproduction_{phase}.log').write_text(run.stdout + run.stderr)
    verification = subprocess.run([sys.executable, str(out / 'verify.py')], check=True,
                                  capture_output=True, text=True)
    (out / 'verification.json').write_text(verification.stdout)
    checks = {}
    table_tolerance = dict(rtol=1e-10, atol=1e-12)
    archive_tolerance = dict(rtol=1e-10, atol=1e-10)
    for name in ['certificate_results.csv', 'census_results.csv', 'nuisance_validation.csv']:
        observed, reference = pd.read_csv(out / name), pd.read_csv(REFERENCE / name)
        assert list(observed) == list(reference) and observed.shape == reference.shape
        columns = {}
        for column in observed:
            if column == 'computational_seconds':
                continue
            new_values, old_values = observed[column].to_numpy(), reference[column].to_numpy()
            numeric = pd.api.types.is_numeric_dtype(observed[column])
            exactly_equal = np.array_equal(new_values, old_values, equal_nan=True) if numeric else np.array_equal(new_values, old_values)
            if numeric:
                tolerance_passed = bool(np.allclose(new_values, old_values, equal_nan=True, **table_tolerance))
                differences = abs(new_values.astype(float) - old_values.astype(float))
                finite = differences[np.isfinite(differences)]
                maximum_error = float(finite.max()) if finite.size else 0.
                assert tolerance_passed, (name, column)
            else:
                tolerance_passed, maximum_error = bool(exactly_equal), None
                assert exactly_equal, (name, column)
            columns[column] = dict(array_equal=bool(exactly_equal), within_tolerance=tolerance_passed,
                                   maximum_absolute_error=maximum_error)
        checks[name] = dict(rows=len(observed), excluded_columns=['computational_seconds'] if 'computational_seconds' in observed else [],
                            all_compared_columns_array_equal=all(item['array_equal'] for item in columns.values()),
                            maximum_numeric_error=max((item['maximum_absolute_error'] or 0.) for item in columns.values()),
                            columns=columns)
    old = np.load(REFERENCE / 'forecast_archive.npz', allow_pickle=False)
    new = np.load(out / 'forecast_archive.npz', allow_pickle=False)
    assert set(old.files) == set(new.files)
    archive_checks = {}
    for key in old.files:
        assert old[key].shape == new[key].shape and old[key].dtype == new[key].dtype, key
        exactly_equal = bool(np.array_equal(old[key], new[key]))
        numeric = np.issubdtype(old[key].dtype, np.number)
        if numeric:
            assert np.isfinite(old[key]).all() and np.isfinite(new[key]).all(), key
            tolerance_passed = bool(np.allclose(old[key], new[key], **archive_tolerance))
            maximum_error = float(np.max(abs(new[key].astype(float) - old[key].astype(float))))
            assert tolerance_passed, key
        else:
            tolerance_passed, maximum_error = exactly_equal, None
            assert exactly_equal, key
        archive_checks[key] = dict(shape=list(old[key].shape), dtype=str(old[key].dtype),
                                   array_equal=exactly_equal, within_tolerance=tolerance_passed,
                                   maximum_absolute_error=maximum_error)
    forecast_checks = [archive_checks[key] for key in original_protocol['candidates']]
    result = dict(passed=True, source_sha256=digest.hexdigest(),
                  environment=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
                                   scipy=scipy.__version__, sklearn=sklearn.__version__),
                  numerical_tables=checks, archive_arrays=archive_checks,
                  comparison_tolerances=dict(numerical_tables=table_tolerance, numeric_archive_arrays=archive_tolerance),
                  all_archive_arrays_equal=all(item['array_equal'] for item in archive_checks.values()),
                  all_forecast_arrays_equal=all(item['array_equal'] for item in forecast_checks),
                  maximum_forecast_absolute_error=max(item['maximum_absolute_error'] for item in forecast_checks),
                  archive_predictions_within_tolerance=all(item['within_tolerance'] for item in forecast_checks),
                  scope='Fresh execution of the exposed fixed numerical study; no new independent statistical confirmation. '
                        'A passed tolerance check does not imply exact equality. Array equality and absolute discrepancies are reported separately; '
                        'runtime columns and execution timestamps are excluded from equality claims.')
    (out / 'reproduction_comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
