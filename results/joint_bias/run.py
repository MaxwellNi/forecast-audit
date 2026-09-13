"""One command: mathematical checks, reconstruction and independent verification.

The default run writes only to a temporary directory and removes it afterward.
Use --output with a new directory to retain the reconstructed result files.
"""
import sys
sys.dont_write_bytecode = True

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pandas as pd

from integrity import check_manifest

HERE = Path(__file__).resolve().parent


def command(script, *arguments):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    result = subprocess.run([sys.executable, str(HERE / script), *map(str, arguments)],
                            check=True, capture_output=True, text=True, env=env)
    return json.loads(result.stdout)


def compare_table(path):
    original = pd.read_csv(HERE / 'recorded' / 'all_candidates.csv', float_precision='round_trip')
    rebuilt = pd.read_csv(path / 'all_candidates.csv', float_precision='round_trip')
    assert list(original.columns) == list(rebuilt.columns)
    assert len(original) == len(rebuilt) == 32
    exact, maximum = True, 0.
    for column in original:
        left, right = original[column].to_numpy(), rebuilt[column].to_numpy()
        if pd.api.types.is_numeric_dtype(original[column]):
            exact = exact and bool(np.array_equal(left, right, equal_nan=True))
            difference = np.abs(left.astype(float) - right.astype(float))
            maximum = max(maximum, float(np.nanmax(difference)))
            np.testing.assert_allclose(left, right, rtol=1e-11, atol=1e-11, equal_nan=True)
        else:
            assert np.array_equal(left, right), column
    return dict(rows=32, all_numeric_fields_array_equal=exact,
                max_absolute_numeric_discrepancy=maximum,
                comparison_rtol=1e-11, comparison_atol=1e-11,
                all_text_fields_equal=True)


def execute(output):
    count = check_manifest()
    checks = command('check_joint_bias.py')
    reconstructed = command('reproduce.py', '--output', output)
    compared = compare_table(output)
    verified = command('verify.py', '--results', output)
    assert check_manifest() == count
    result = dict(status='PASS', classification='Post-exposure exploration; not confirmation',
                  package_files_checked=count, mathematical_checks=checks,
                  reconstruction=reconstructed, numeric_comparison=compared,
                  independent_verification=verified,
                  package_inputs_unchanged=True,
                  statement='All 32 allowances tighten; all reference raw and BY p-values remain one.')
    (output / 'run_report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional new directory to retain replay results')
    args = parser.parse_args()
    if args.output is None:
        with tempfile.TemporaryDirectory(prefix='joint-bias-') as temporary:
            execute(Path(temporary) / 'reconstruction')
    else:
        if args.output.exists():
            raise FileExistsError('Output directory must not already exist')
        execute(args.output.resolve())
