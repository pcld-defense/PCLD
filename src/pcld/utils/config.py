"""Bridge between the Hydra/OmegaConf config tree and the legacy argument API.

The experiment entry-points in :mod:`pcld.experiments` consume a flat
``argparse.Namespace``.  Rather than rewrite every experiment, the config layer
resolves a structured YAML config and flattens it into the exact same Namespace
those functions already expect.  This keeps behaviour identical while making
every knob configurable from one YAML file (with CLI overrides via Hydra).
"""

import argparse
from typing import Union

from omegaconf import DictConfig, OmegaConf

# Canonical list of every field the experiments read off the Namespace, with
# the default that the original ``parse_args()`` assigned.  Anything the config
# omits falls back to these, so a minimal experiment YAML stays minimal.
_DEFAULTS: dict = {
    'experiment_type': None,
    'experiment_name': 'test',
    # dataset
    'dataset': None,
    'dataset_type': None,
    'splits': ['test'],
    'batch_size': 16,
    # painter
    'output_every': [50, 100, 200, 300, 400, 500, 600, 700,
                     950, 1200, 1700, 2200, 3200, 4200, 5200],
    # classifier / decisioner
    'preprocessing': None,
    'train_preprocessing': None,
    'model_type': 'wrn-70-16',
    'pretrained_weights': None,
    'max_epochs': 51,
    'lr': 0.01,
    'patience': 5,
    'classifier_experiment': None,
    'decisioner_experiment': None,
    'decisioner_architechture': None,
    'find_best_epoch': 1,
    'max_train_epsilon': 20,
    # attack
    'epsilons': [8],
    'attack': 'pgd',
    'attack_direction': 'untargeted',
    'attack_nb_iter': 10,
    'run_naive_attack': 0,
    'save_parquet': 1,
    'attack_norm': 'linf',
    'targeted_jumps_allowed': 6,
    'attack_nb_restarts': 1,
    'multi_step_loss_weight': 0.0,
    'eot_samples': 1,
    'use_apgd': 0,
    'resume': 0,
    'joint_surrogate': None,
    # reproducibility (new; not present in the legacy CLI)
    'seed': 42,
    'deterministic': False,
}

# Config-tree leaf names that map onto a differently-named Namespace field.
# The dataset group stores the dataset folder as ``dataset.name`` for
# readability, but the experiments read it off ``args.dataset``.
_ALIASES = {'name': 'dataset'}

# Fields that must end up as a list[int] on the Namespace even if the YAML/CLI
# provides them as a scalar or a delimited string (legacy CLI accepted
# "50,100" and "3|9").
_INT_LIST_FIELDS = {'output_every': ',', 'epsilons': '|'}


def _coerce_int_list(value: Union[str, int, list], sep: str) -> list:
    """Normalises a scalar / delimited-string / list value to ``list[int]``."""
    if isinstance(value, str):
        parts = value.split(sep) if sep in value else [value]
        return [int(v) for v in parts]
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)]


def config_to_namespace(cfg: DictConfig) -> argparse.Namespace:
    """Flattens a resolved config tree into a legacy ``argparse.Namespace``.

    Walks every group in ``cfg`` and copies leaf values into a flat namespace,
    filling any missing field from ``_DEFAULTS`` and coercing list-valued
    fields (``output_every``, ``epsilons``) to ``list[int]``.  Unknown leaves
    are still copied through so custom experiments can read extra keys.

    Args:
        cfg: The composed Hydra/OmegaConf config. May be grouped (e.g.
            ``cfg.dataset.name``) or flat; both are flattened by leaf key.

    Returns:
        A Namespace exposing every attribute in ``_DEFAULTS`` (plus any extra
        leaves), matching what ``parse_args()`` used to produce.
    """
    container = OmegaConf.to_container(cfg, resolve=True)
    flat: dict = {}

    def _walk(node: dict) -> None:
        for key, val in node.items():
            if isinstance(val, dict):
                _walk(val)
            else:
                flat[key] = val

    _walk(container)

    for src, dst in _ALIASES.items():
        if src in flat and dst not in flat:
            flat[dst] = flat.pop(src)

    ns = dict(_DEFAULTS)
    ns.update({k: v for k, v in flat.items() if k in _DEFAULTS})
    # keep any extra keys the experiment might read
    ns.update({k: v for k, v in flat.items() if k not in _DEFAULTS})

    for field, sep in _INT_LIST_FIELDS.items():
        if ns.get(field) is not None:
            ns[field] = _coerce_int_list(ns[field], sep)

    if isinstance(ns.get('splits'), str):
        ns['splits'] = [s.strip() for s in ns['splits'].split(',')]

    return argparse.Namespace(**ns)