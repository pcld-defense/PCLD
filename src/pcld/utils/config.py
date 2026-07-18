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
    # eval subset cap; comparison.num_samples flattens here
    'num_samples': None,
    'data_source': 'folder',    # 'folder' | 'robustbench'
    # painter
    'output_every': [50, 100, 200, 300, 400, 500, 600, 700,
                     950, 1200, 1700, 2200, 3200, 4200, 5200],
    'painter_max_step': 80,
    'painter_divide': 5,
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
    'aa_version': 'standard',
    'resume': 0,
    'joint_surrogate': None,
    'surrogate_type': 'learned',
    # gradient-validity battery (R01 gate)
    'battery_epsilons': [8, 16, 32, 64],
    'battery_fd_dirs': 8,
    'battery_fd_delta': 1e-3,
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
_INT_LIST_FIELDS = {'output_every': ',', 'epsilons': '|',
                    'battery_epsilons': ','}

# Fields in _INT_LIST_FIELDS that may carry non-integral floats: l2 epsilons
# are absolute budgets (e.g. 0.5), unlike the integral /255 linf budgets.
_FLOAT_OK_FIELDS = {'epsilons'}


def _coerce_int_list(value: Union[str, int, list], sep: str,
                     allow_float: bool = False) -> list:
    """Normalises a scalar / delimited-string / list value to a number list.

    Args:
        value: Scalar, ``sep``-delimited string, or list/tuple of values.
        sep: Delimiter for string inputs.
        allow_float: If True, non-integral values are preserved as floats
            (integral values still become ints); if False every value is
            coerced with ``int()`` exactly as the legacy CLI did.

    Returns:
        List of ints (plus floats for non-integral values when
        ``allow_float`` is True).
    """
    def _num(v: Union[str, int, float]) -> Union[int, float]:
        if allow_float:
            f = float(v)
            return int(f) if f.is_integer() else f
        return int(v)

    if isinstance(value, str):
        parts = value.split(sep) if sep in value else [value]
        return [_num(v) for v in parts]
    if isinstance(value, (list, tuple)):
        return [_num(v) for v in value]
    return [_num(value)]


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
            ns[field] = _coerce_int_list(ns[field], sep,
                                         allow_float=field in _FLOAT_OK_FIELDS)

    if isinstance(ns.get('splits'), str):
        ns['splits'] = [s.strip() for s in ns['splits'].split(',')]

    return argparse.Namespace(**ns)