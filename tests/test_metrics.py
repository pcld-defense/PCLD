"""Tests for robust-accuracy aggregation and table emitters (no torch)."""

import pandas as pd

from pcld.eval.metrics import (FINAL_STEP, robust_accuracy, summarize_run,
                               comparison_table, emit_table, sweep_row)


def _toy_results():
    # Two images, two epsilons. eps 0: both correct. eps 8: one correct.
    # Includes an intermediate paint-step row that must be ignored.
    return pd.DataFrame([
        # intermediate step (t != FINAL_STEP) — must be excluded
        {'t': 50, 'epsilon': 8, 'pred': 1, 'actual': 1, 'attacked_model': 'adaptive'},
        # eps 0 finals
        {'t': FINAL_STEP, 'epsilon': 0, 'pred': 3, 'actual': 3, 'attacked_model': 'adaptive'},
        {'t': FINAL_STEP, 'epsilon': 0, 'pred': 5, 'actual': 5, 'attacked_model': 'adaptive'},
        # eps 8 finals: one right, one wrong
        {'t': FINAL_STEP, 'epsilon': 8, 'pred': 3, 'actual': 3, 'attacked_model': 'adaptive'},
        {'t': FINAL_STEP, 'epsilon': 8, 'pred': 9, 'actual': 5, 'attacked_model': 'adaptive'},
    ])


def test_robust_accuracy_uses_only_final_rows():
    acc = robust_accuracy(_toy_results())
    by_eps = {int(r.epsilon): r.accuracy for r in acc.itertuples()}
    assert by_eps[0] == 1.0
    assert by_eps[8] == 0.5
    # eps 8 had 2 finals (intermediate step excluded)
    assert int(acc[acc['epsilon'] == 8]['n'].iloc[0]) == 2


def test_summarize_run():
    s = summarize_run(_toy_results())
    assert s['clean_accuracy'] == 1.0
    assert s['robust_accuracy'][8] == 0.5
    assert s['num_images'] == 2


def test_emit_table_writes_csv_and_tex(tmp_path):
    table = comparison_table([
        {'model': 'a', 'clean': 0.94, 'eps_8': 0.0},
        {'model': 'b', 'clean': 0.90, 'eps_8': 0.6},
    ])
    csv_path, tex_path = emit_table(table, str(tmp_path))
    assert (tmp_path / 'comparison.csv').exists()
    assert (tmp_path / 'comparison.tex').exists()
    text = (tmp_path / 'comparison.csv').read_text()
    assert 'model' in text and 'eps_8' in text


def test_sweep_row_from_robust_accuracy():
    acc = robust_accuracy(_toy_results())
    row = sweep_row('wrn-28-10-standard', 'Linf', acc)
    assert row['model'] == 'wrn-28-10-standard'
    assert row['threat_model'] == 'Linf'
    assert row['clean'] == 1.0
    assert row['eps_8'] == 0.5


def test_sweep_row_preserves_float_epsilon_columns():
    # l2 runs use absolute float budgets (e.g. 0.5); the eps column must not
    # be truncated to 'eps_0'.
    acc = pd.DataFrame([
        {'epsilon': 0.0, 'n': 2, 'correct': 2, 'accuracy': 1.0},
        {'epsilon': 0.5, 'n': 2, 'correct': 1, 'accuracy': 0.5},
        {'epsilon': 8.0, 'n': 2, 'correct': 0, 'accuracy': 0.0},
    ])
    row = sweep_row('wrn-28-10-rebuffi2021-l2', 'L2', acc)
    assert row['clean'] == 1.0
    assert row['eps_0.5'] == 0.5
    assert row['eps_8'] == 0.0     # integral floats collapse to int-style names


def test_summarize_run_preserves_float_epsilons():
    df = pd.DataFrame([
        {'t': FINAL_STEP, 'epsilon': 0, 'pred': 1, 'actual': 1,
         'attacked_model': 'adaptive'},
        {'t': FINAL_STEP, 'epsilon': 0.5, 'pred': 2, 'actual': 1,
         'attacked_model': 'adaptive'},
    ])
    s = summarize_run(df)
    assert s['clean_accuracy'] == 1.0
    assert s['robust_accuracy'][0.5] == 0.0