"""Multi-model RobustBench comparison sweep.

Runs the configured attack against every model listed in
``comparison.models`` and emits a single comparison table (CSV + LaTeX) of
robust accuracy per epsilon.

    python scripts/sweep.py experiment=rb_sweep

Add a model to compare against by adding one line to ``comparison.models`` in
the experiment config — no code changes required.
"""

import copy
import os

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from pcld.utils.config import config_to_namespace
from pcld.utils.consts import RESOURCES_RESULTS_DIR
from pcld.utils.seeding import seed_everything
from pcld.experiments.experiment_navigator import apply_experiment
from pcld.eval.metrics import robust_accuracy, emit_table, sweep_row


def _load_results(run_dir: str) -> pd.DataFrame:
    """Concatenates all attack result parquet/CSV files under a run dir."""
    frames = []
    for f in sorted(os.listdir(run_dir)):
        p = os.path.join(run_dir, f)
        if f.endswith('_results.parquet'):
            frames.append(pd.read_parquet(p))
        elif f.endswith('_results.csv'):
            frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError(f'No *_results.* files found in {run_dir}')
    return pd.concat(frames, ignore_index=True)


@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(cfg: DictConfig) -> None:
    """Runs the attack per comparison model and writes one comparison table."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base = config_to_namespace(cfg)
    seed_everything(base.seed, deterministic=getattr(base, 'deterministic', False))

    models = OmegaConf.to_container(cfg.comparison.models, resolve=True)
    sweep_root = os.path.join(RESOURCES_RESULTS_DIR, base.experiment_name)
    os.makedirs(sweep_root, exist_ok=True)

    table_rows = []
    for entry in models:
        name = entry['name']
        print('=' * 60)
        print(f'Sweep: attacking {name} ({entry.get("threat_model")})')

        args = copy.deepcopy(base)
        args.model_type = name
        args.dataset_type = entry.get('dataset', args.dataset_type)
        args.experiment_name = f'{base.experiment_name}/{name}'
        args.experiment_type = 'attack_classifier'
        # Attack each model under its own RobustBench threat model.
        args.attack_norm = ('linf' if entry.get('threat_model', 'Linf') == 'Linf'
                            else 'l2')
        args.data_source = getattr(base, 'data_source', 'robustbench')

        # Re-seed per model so each run starts from the same RNG state.
        seed_everything(base.seed, deterministic=getattr(base, 'deterministic', False))
        apply_experiment(args=args, device=device)

        run_dir = os.path.join(RESOURCES_RESULTS_DIR, args.experiment_name)
        try:
            df = _load_results(run_dir)
            acc = robust_accuracy(df)
            table_rows.append(sweep_row(name, entry.get('threat_model'), acc))
        except FileNotFoundError as e:
            print(f'  [warn] no attack results for {name}: {e}')

    if table_rows:
        table = pd.DataFrame(table_rows)
        csv_path, tex_path = emit_table(
            table, sweep_root, stem='comparison',
            caption='Robust accuracy across RobustBench models.')
        print(f'\nComparison table written:\n  {csv_path}\n  {tex_path}')
        print(table.to_string(index=False))


if __name__ == '__main__':
    main()