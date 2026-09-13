#!/usr/bin/env python3
"""Convert official OSAP portfolio CSVs to the audit's parquet input.

Accepts PredictorPortsFull.csv (long) or PredictorLSretWide.csv (wide).
Preserves return units. Does not download files or assert a historical release match.
The official exporter is Portfolios/Code/20_PredictorPorts.R in
https://github.com/OpenSourceAP/CrossSection.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def prepare_frame(path: Path, layout: str = 'long') -> pd.DataFrame:
    if layout not in {'long', 'wide'}:
        raise ValueError('layout must be long or wide')
    frame = pd.read_csv(path, dtype={'signalname': str, 'port': str, 'date': str}, float_precision='round_trip')
    if layout == 'wide':
        if 'date' not in frame or len(frame.columns) < 2:
            raise ValueError('Wide input requires date and at least one signal column')
        frame = frame.melt(id_vars='date', var_name='signalname', value_name='ret')
        frame['port'] = 'LS'
    needed = {'signalname', 'port', 'date', 'ret'}
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    frame = frame.copy()
    if frame[['signalname', 'port', 'date']].isna().any().any():
        raise ValueError('Signal, portfolio and date keys must be present')
    frame['signalname'] = frame['signalname'].astype(str).str.strip()
    frame['port'] = frame['port'].astype(str).str.strip()
    if (frame['signalname'] == '').any() or (frame['port'] == '').any():
        raise ValueError('Signal and portfolio keys cannot be empty')
    dates = frame['date'].astype(str).str.strip()
    if dates.str.fullmatch(r'\d{6}').all():
        frame['date'] = pd.to_datetime(dates, format='%Y%m', errors='raise') + pd.offsets.MonthEnd(0)
    else:
        frame['date'] = pd.to_datetime(dates, errors='raise')
    frame['ret'] = pd.to_numeric(frame['ret'], errors='raise')
    if np.isinf(frame['ret'].to_numpy(dtype=float)).any():
        raise ValueError('Returns may be missing, but cannot be infinite')
    monthly_keys = frame[['signalname', 'port']].assign(month=frame['date'].dt.to_period('M'))
    if monthly_keys.duplicated().any():
        raise ValueError('Duplicate signal/portfolio/month observations')
    if not (frame['port'] == 'LS').any():
        raise ValueError('Input contains no LS portfolios for the alpha audit')
    return frame.sort_values(['signalname', 'port', 'date'], kind='stable').reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_csv', type=Path)
    parser.add_argument('--format', choices=['long', 'wide'], default='long')
    parser.add_argument('--output', type=Path, default=Path('data_public/osap/predictor_ports_full.parquet'))
    parser.add_argument('--source-url', default='https://www.openassetpricing.com/data/')
    args = parser.parse_args()
    output = args.output.resolve()
    sidecar = output.with_suffix(output.suffix + '.provenance.json')
    if output.exists() or sidecar.exists():
        raise SystemExit('Choose a new output path; existing inputs are never overwritten.')
    frame = prepare_frame(args.input_csv, args.format)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    ls = frame[frame['port'] == 'LS']
    receipt = {
        'input_filename': args.input_csv.name,
        'input_sha256': hashlib.sha256(args.input_csv.read_bytes()).hexdigest(),
        'source_url_supplied_by_user': args.source_url,
        'input_layout': args.format,
        'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'rows': len(frame), 'long_short_rows': len(ls),
        'long_short_signals': int(ls['signalname'].nunique()),
        'start': str(frame['date'].min()), 'end': str(frame['date'].max()),
        'return_units': 'unchanged from input; use the same percent units as Ken French factors',
        'historical_release_identity': 'not established by conversion',
    }
    sidecar.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
