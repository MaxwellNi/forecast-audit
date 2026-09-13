"""Supplementary synthetic comparison; see additional_baselines/README.md
from the package root for assumptions and interpretation limits."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))
from clustered_calibration import panel, ZA
from baseline_implementations import extrapolated_covariance, gcm_fine, gcm_coarse, naive_perm, partial_linear
import importlib.util as _u

def _load(name, fn):
    s = _u.spec_from_file_location(name, ROOT / 'scripts' / 'analysis' / (name + '.py'))
    m = _u.module_from_spec(s)
    s.loader.exec_module(m)
    return getattr(m, fn)
wgcm = _load('wgcm_baseline', 'wgcm_sign')
pcm = _load('pcm', 'pcm')
local_perm = _load('local_permutation_baseline', 'local_permutation')
TESTS = [('ours', extrapolated_covariance), ('GCM coarse', gcm_coarse), ('GCM fine', gcm_fine), ('linear', partial_linear), ('permutation', naive_perm), ('weighted GCM', wgcm), ('projected CM', pcm), ('local perm.', local_perm)]
OVERLAPS = [0.5, 0.65, 0.8, 0.9, 0.95]
REGIMES = ['smooth', 'nonlinear', 'large_n', 'heavy_tail']
VALID_T1, MIN_POWER = (0.18, 0.9)

def _rate(fn, regime, delta, w, n_seeds, seed0):
    rej = 0
    for s in range(n_seeds):
        rng = np.random.default_rng(seed0 + s)
        d = panel(regime, rng, delta, w)
        rej += int(fn(*d, rng))
    return rej / n_seeds

def main(n_seeds=150, seed0=80808):
    t0 = time.time()
    grid = {}
    for tname, fn in TESTS:
        for w in OVERLAPS:
            for reg in REGIMES:
                ns = n_seeds if reg != 'large_n' else max(80, n_seeds // 2)
                t1 = _rate(fn, reg, 0.0, w, ns, seed0)
                pw = _rate(fn, reg, 0.15, w, ns, seed0 + 5000)
                grid[f'{tname}|{w}|{reg}'] = {'typeI': round(t1, 3), 'power': round(pw, 3)}
                print(f'{tname:13s} w={w:.2f} {reg:10s} typeI={t1:.3f} power={pw:.2f}', flush=True)
    summary = {}
    for tname, _ in TESTS:
        cells = [v for k, v in grid.items() if k.startswith(tname + '|')]
        worst_t1 = max((c['typeI'] for c in cells))
        min_pw = min((c['power'] for c in cells))
        generalises = worst_t1 <= VALID_T1 and min_pw >= MIN_POWER
        n_bad = sum((1 for c in cells if c['typeI'] > VALID_T1 or c['power'] < MIN_POWER))
        summary[tname] = {'worst_typeI': worst_t1, 'min_power': min_pw, 'cells_failed': n_bad, 'of_cells': len(cells), 'generalises_everywhere': generalises}
    OUT = ROOT / 'outputs/theory_simulation/audit_generalization'
    OUT.mkdir(parents=True, exist_ok=True)
    out = {'axes': {'overlap': OVERLAPS, 'regime': REGIMES}, 'n_seeds': n_seeds, 'valid_typeI_bar': VALID_T1, 'min_power_bar': MIN_POWER, 'grid': grid, 'per_test': summary, 'only_generaliser': [t for t, s in summary.items() if s['generalises_everywhere']], 'elapsed_sec': round(time.time() - t0, 1)}
    (OUT / 'generalization_grid_summary.json').write_text(json.dumps(out, indent=2) + '\n')
    print('\n=== per-test generalization over the 20-cell grid ===')
    for t, s in summary.items():
        print(f"  {t:13s} worst typeI={s['worst_typeI']:.3f}  min power={s['min_power']:.2f}  failed {s['cells_failed']}/{s['of_cells']}  {('GENERALISES' if s['generalises_everywhere'] else 'FAILS somewhere')}")
    print('only generaliser:', out['only_generaliser'])
if __name__ == '__main__':
    main()
