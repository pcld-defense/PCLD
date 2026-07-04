"""Config-driven experiment runner (Hydra entry point).

Usage:
    python scripts/run.py experiment=smoke_test
    python scripts/run.py experiment=paper_pcld_pgd10 batch_size=4
    python scripts/run.py experiment_type=attack_pcl dataset=cifar10 attack=fgsm

Resolves the composed config, seeds every RNG, snapshots the exact config into
the run directory, then dispatches to the matching experiment entry point via
the same Namespace the legacy CLI produced.
"""

import os
import warnings

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from pcld.utils.config import config_to_namespace
from pcld.utils.consts import NUM_OF_HYPHENS
from pcld.utils.seeding import seed_everything
from pcld.experiments.experiment_navigator import apply_experiment


@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(cfg: DictConfig) -> None:
    """Runs one experiment from a composed Hydra config.

    Args:
        cfg: The composed config tree (root + dataset/model/attack groups,
            optionally an ``experiment=`` preset), with CLI overrides applied.
    """
    warnings.filterwarnings('ignore')

    args = config_to_namespace(cfg)
    seed_everything(args.seed, deterministic=getattr(args, 'deterministic', False))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    # Snapshot the exact resolved config next to the results.
    run_dir = os.getcwd()
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config_snapshot.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    print('-' * NUM_OF_HYPHENS)
    print(f'Run {args.experiment_name} ({args.experiment_type})...')
    apply_experiment(args=args, device=device)
    print(f'Finished experiment {args.experiment_name}')


if __name__ == '__main__':
    main()