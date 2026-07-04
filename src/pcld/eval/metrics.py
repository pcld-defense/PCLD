"""Metrics aggregation and paper-table emitters.

Turns the per-(image x paint-step) result rows written by ``attacker()`` into
robust-accuracy summaries, a structured ``metrics.json`` per run, and a
multi-model comparison table exportable to CSV and LaTeX.
"""

import json
import os
from typing import Optional

import pandas as pd

# The paint-step sentinel used for the original image / final decision row.
FINAL_STEP = 999999


def robust_accuracy(df: pd.DataFrame,
                    attacked_model: str = 'adaptive') -> pd.DataFrame:
    """Computes accuracy per epsilon from an attack results DataFrame.

    Uses only the final-decision rows (``t == FINAL_STEP``) so each image
    contributes exactly one prediction per epsilon, then compares ``pred`` to
    ``actual``.

    Args:
        df: Results frame with at least ``t``, ``epsilon``, ``pred``,
            ``actual``, and ``attacked_model`` columns (as written by
            ``attacker()``).
        attacked_model: Which attack rows to score (``'adaptive'`` or
            ``'naive'``).

    Returns:
        DataFrame with columns ``epsilon``, ``n``, ``correct``, ``accuracy``,
        sorted by epsilon.
    """
    final = df[(df['t'] == FINAL_STEP) &
               (df['attacked_model'] == attacked_model)].copy()
    final['is_correct'] = (final['pred'] == final['actual']).astype(int)

    grouped = final.groupby('epsilon')['is_correct'].agg(['size', 'sum'])
    grouped = grouped.rename(columns={'size': 'n', 'sum': 'correct'})
    grouped['accuracy'] = grouped['correct'] / grouped['n']
    return grouped.reset_index().sort_values('epsilon').reset_index(drop=True)


def write_metrics_json(metrics: dict, output_dir: str,
                       filename: str = 'metrics.json') -> str:
    """Writes a metrics dict to JSON in the run directory.

    Args:
        metrics: JSON-serialisable metrics (e.g. clean/robust accuracy).
        output_dir: Run directory to write into.
        filename: Output filename.

    Returns:
        The path written.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    return path


def summarize_run(df: pd.DataFrame,
                  attacked_model: str = 'adaptive') -> dict:
    """Builds a compact metrics dict for one attack run.

    Args:
        df: Attack results frame.
        attacked_model: Which attack rows to score.

    Returns:
        Dict with ``clean_accuracy`` (epsilon 0, if present) and
        ``robust_accuracy`` (a ``{epsilon: accuracy}`` map).
    """
    acc = robust_accuracy(df, attacked_model=attacked_model)
    per_eps = {int(r.epsilon): float(r.accuracy) for r in acc.itertuples()}
    return {
        'attacked_model': attacked_model,
        'clean_accuracy': per_eps.get(0),
        'robust_accuracy': per_eps,
        'num_images': int(acc['n'].max()) if len(acc) else 0,
    }


def comparison_table(rows: list[dict]) -> pd.DataFrame:
    """Assembles a multi-model comparison table from per-model summaries.

    Args:
        rows: One dict per model, e.g.
            ``{'model': 'wrn-28-10-standard', 'clean': 0.94, 'eps_8': 0.0}``.

    Returns:
        A DataFrame with models as rows, one column per reported metric.
    """
    return pd.DataFrame(rows)


def emit_table(df: pd.DataFrame, output_dir: str, stem: str = 'comparison',
               float_format: str = '%.4f',
               caption: Optional[str] = None) -> tuple[str, str]:
    """Writes a comparison DataFrame to both CSV and LaTeX.

    Args:
        df: The comparison table.
        output_dir: Directory to write into.
        stem: Filename stem; produces ``<stem>.csv`` and ``<stem>.tex``.
        float_format: Float format string for both outputs.
        caption: Optional LaTeX caption.

    Returns:
        Tuple of (csv_path, tex_path).
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f'{stem}.csv')
    tex_path = os.path.join(output_dir, f'{stem}.tex')

    df.to_csv(csv_path, index=False, float_format=float_format)
    df.to_latex(tex_path, index=False, float_format=float_format,
                caption=caption, escape=True)
    return csv_path, tex_path
