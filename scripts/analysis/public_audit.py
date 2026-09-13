"""Shared hashing and full-family nominal audit helpers; no forecasting entry point."""
from __future__ import annotations

import argparse

import hashlib

import importlib.metadata

import json

from pathlib import Path

import re

import time

import numpy as np

import pandas as pd

from statsmodels.stats.multitest import multipletests

from threadpoolctl import threadpool_limits

from audit_panel_predictions import audit_extrapolated_panel, audit_panel, by_adjust

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda : handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def audit_family(panels, lag, frequency, include_single_resolution=True):
    extrapolated = []
    single = []
    for (name, frame) in panels:
        extrapolated.extend(({'model': name, **row} for row in audit_extrapolated_panel(frame, lag, frequency, betas=(1, 2))))
        if include_single_resolution:
            for scheme in ('contiguous', 'interleaved'):
                single.extend(({'model': name, **row} for row in audit_panel(frame, [lag], frequency, scheme)))
    results = []
    for (rows, keys) in [(extrapolated, ['beta', 'lag', 'fold_scheme']), (single, ['lag', 'fold_scheme'])]:
        result = pd.DataFrame(rows)
        if rows:
            for (_, group) in result.groupby(keys):
                (rejected, adjusted) = by_adjust(group.p_one_sided)
                oracle = multipletests(group.p_one_sided, alpha=0.05, method='fdr_by')
                np.testing.assert_array_equal(rejected, oracle[0])
                np.testing.assert_allclose(adjusted, oracle[1])
                result.loc[group.index, 'by_reject'] = rejected
                result.loc[group.index, 'by_adjusted_p'] = adjusted
        results.append(result)
    return results
