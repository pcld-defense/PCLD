"""Tests that the Hydra config tree flattens to the legacy Namespace correctly.

These run without torch: they exercise config composition and the
``config_to_namespace`` adapter only.
"""

import os

from hydra import compose, initialize_config_dir

from pcld.utils.config import config_to_namespace

_CFG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'configs'))


def _load(overrides):
    with initialize_config_dir(version_base=None, config_dir=_CFG_DIR):
        cfg = compose(config_name='config', overrides=overrides)
    return config_to_namespace(cfg)


def test_base_defaults_match_legacy():
    a = _load([])
    assert a.dataset == 'subset_of_imagenet'
    assert a.dataset_type == 'imagenet'
    assert a.attack == 'pgd'
    assert a.epsilons == [8]
    assert a.output_every[0] == 50 and len(a.output_every) == 15
    assert a.model_type == 'wrn-70-16'


def test_smoke_preset():
    s = _load(['experiment=smoke_test'])
    assert s.experiment_type == 'attack_pcld'
    assert s.experiment_name == 'smoke_test'
    assert s.batch_size == 4
    assert s.attack == 'fgsm'
    assert s.epsilons == [0, 3]
    assert s.deterministic is True
    assert s.splits == ['test']


def test_paper_preset():
    p = _load(['experiment=paper_pcld_pgd10'])
    assert p.attack_direction == 'targeted'
    assert p.epsilons == [3, 9]
    assert p.attack_nb_iter == 10
    assert p.batch_size == 10


def test_cli_overrides_and_int_list_coercion():
    o = _load(['experiment=smoke_test', 'batch_size=16',
               'attack.epsilons=[4,8,12]'])
    assert o.batch_size == 16
    assert o.epsilons == [4, 8, 12]


def test_rb_sweep_exposes_comparison_list():
    r = _load(['experiment=rb_sweep'])
    assert r.dataset == 'cifar10'
    assert r.dataset_type == 'cifar10'
    assert r.model_type == 'wrn-28-10-standard'
    assert r.targeted_jumps_allowed == 1
    assert hasattr(r, 'models') and len(r.models) == 3