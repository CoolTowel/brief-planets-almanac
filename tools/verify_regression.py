"""Compare a 2027/UTC+8/40N/40m new dataset to the preserved historical run."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from astropy.table import Table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from almanac.dataset import load_dataset
from almanac.topo import TopoContext


def verify(run_dir, baseline_root):
    """Verify unchanged calculations with pre-agreed physical tolerances."""
    dataset = load_dataset(run_dir)
    if dataset.topo != TopoContext(2027, 480, 40, 40):
        raise ValueError('Regression requires year=2027, UTC+8, latitude=40, elevation=40')
    baseline_root = Path(baseline_root)
    report = {'run_dir': str(dataset.run_dir), 'baseline': str(baseline_root),
              'tolerances': {'event_seconds': 0.001, 'solar_altitude_degrees': 1e-7},
              'event_max_error_seconds': {}, 'authorized_stellar_changes': {}}
    for key, actual in dataset.tables.items():
        old = Table.read(baseline_root / '40' / f'{key}.fits')
        if old.colnames != actual.colnames:
            raise AssertionError(f'{key}: columns changed')
        np.testing.assert_array_equal(old['Day'], actual['Day'])
        for column in actual.colnames[1:]:
            a, b = np.asarray(actual[column]), np.asarray(old[column])
            valid = (a != -999) & (b != -999)
            name = f'{key}.{column}'
            if column in ('Star4', 'Star5'):
                report['authorized_stellar_changes'][name] = {
                    'median_signed_difference_seconds': float(np.median((a[valid]-b[valid])*3600)),
                    'changed_missing_mask_count': int(np.sum((a == -999) != (b == -999))),
                }
                continue
            np.testing.assert_array_equal(a == -999, b == -999, err_msg=name)
            error = float(np.max(np.abs(a[valid]-b[valid]))*3600) if valid.any() else 0.0
            report['event_max_error_seconds'][name] = error
            if error > 0.001:
                raise AssertionError(f'{name}: {error} seconds exceeds 0.001')
    previous_eot = np.load(baseline_root/'EOT.npy', allow_pickle=False)
    previous_altitudes = np.load(baseline_root/'40'/'sun_alts.npy', allow_pickle=False)
    if previous_eot.shape != dataset.arrays['EOT'].shape:
        raise AssertionError('EOT dimensions changed')
    if previous_altitudes.shape != dataset.arrays['sun_alts'].shape:
        raise AssertionError('Solar altitude dimensions changed')
    eot_error = float(np.max(abs(dataset.arrays['EOT']-previous_eot))*3600)
    altitude_error = float(np.max(abs(dataset.arrays['sun_alts']-previous_altitudes)))
    if eot_error > 0.001 or altitude_error > 1e-7:
        raise AssertionError(f'EOT={eot_error} seconds, altitude={altitude_error} degrees')
    report.update(eot_max_error_seconds=eot_error,
                  solar_altitude_max_error_degrees=altitude_error, passed=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--baseline-root', type=Path,
                        default=ROOT/'output2027/_run/data/2027')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = verify(args.run_dir, args.baseline_root)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text+'\n', encoding='utf-8')
    print(text)
