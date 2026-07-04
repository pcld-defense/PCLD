"""Prove the refactored pipeline reproduces the pre-refactor attack output.

Compares two attack result files (``*_results.parquet`` written by
``attacker()``) row-for-row: the discrete predictions must match exactly and
the softmax probability vectors must match within a tolerance.

    python scripts/verify_equivalence.py baseline.parquet new.parquet

Exit code 0 means equivalent, 1 means a mismatch was found (details printed).
"""

import argparse
import sys

import numpy as np
import pandas as pd

_KEY_COLS = ['image', 't', 'epsilon', 'phase', 'attacked_model']


def _load(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path)
    return df.sort_values(_KEY_COLS).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', help='pre-refactor results file')
    parser.add_argument('new', help='refactored-pipeline results file')
    parser.add_argument('--atol', type=float, default=1e-5,
                        help='absolute tolerance for softmax probs')
    args = parser.parse_args()

    a, b = _load(args.baseline), _load(args.new)

    if len(a) != len(b):
        print(f'MISMATCH: row counts differ ({len(a)} vs {len(b)})')
        return 1

    # 1. Discrete predictions must match exactly.
    pred_mismatch = (a['pred'].values != b['pred'].values).sum()
    if pred_mismatch:
        print(f'MISMATCH: {pred_mismatch}/{len(a)} predictions differ')
        idx = np.where(a['pred'].values != b['pred'].values)[0][:5]
        print(a.loc[idx, _KEY_COLS + ['pred']].to_string(index=False))
        return 1
    print(f'OK: all {len(a)} predictions identical')

    # 2. Softmax vectors must match within tolerance (if present).
    if 'probs' in a.columns and 'probs' in b.columns:
        pa = np.stack(a['probs'].apply(np.asarray).values)
        pb = np.stack(b['probs'].apply(np.asarray).values)
        max_abs = float(np.max(np.abs(pa - pb)))
        print(f'Max abs softmax diff: {max_abs:.2e} (tol {args.atol:.0e})')
        if max_abs > args.atol:
            return 1
        print('OK: softmax vectors match within tolerance')

    print('\nEQUIVALENT: refactored pipeline reproduces baseline output.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
