"""Registry entries for the leaderboard backbones, and threat_model plumbing."""

import sys
import types
from unittest import mock

import pytest

from pcld.models.registry import CLASSIFIER_REGISTRY, ROBUSTBENCH_STANDARD_MODELS

EXPECTED_NEW_KEYS = [
    'wrn-70-16-wang2023',
    'wrn-70-16-wang2023-l2',
    'wrn-28-10-rebuffi2021-l2',
    'wrn-94-16-bartoldson2024',
    'swin-l-liu2023',
]


@pytest.mark.parametrize('key', EXPECTED_NEW_KEYS)
def test_new_backbone_registered(key):
    assert key in CLASSIFIER_REGISTRY


def test_robustbench_entries_are_allowlisted():
    for key, cfg in CLASSIFIER_REGISTRY.items():
        if cfg.family != 'robustbench':
            continue
        assert cfg.robustbench_name in ROBUSTBENCH_STANDARD_MODELS, key
        assert cfg.threat_model in {'Linf', 'L2'}, key


def test_threat_model_follows_key_suffix():
    for key, cfg in CLASSIFIER_REGISTRY.items():
        expected = 'L2' if key.endswith('-l2') else 'Linf'
        assert cfg.threat_model == expected, key


def test_threat_model_passthrough():
    pytest.importorskip('torch')
    pytest.importorskip('robustbench')  # classifier.py imports it at module load
    import torch.nn as nn
    from pcld.models.classifier import _build_model

    recorded = {}

    def fake_load_model(**kwargs):
        recorded.update(kwargs)
        return nn.Identity()

    fake_utils = types.ModuleType('robustbench.utils')
    fake_utils.load_model = fake_load_model
    # _build_model lazily does `from robustbench.utils import load_model`;
    # injecting the fake module into sys.modules intercepts that import.
    with mock.patch.dict(sys.modules, {'robustbench.utils': fake_utils}):
        model = _build_model(CLASSIFIER_REGISTRY['wrn-28-10-rebuffi2021-l2'],
                             n_classes=10, dataset_type='cifar10')

    assert isinstance(model, nn.Identity)
    assert recorded['threat_model'] == 'L2'
    assert recorded['dataset'] == 'cifar10'
    assert recorded['model_name'] == 'Rebuffi2021Fixing_28_10_cutmix_ddpm'


def test_cifar100_maps():
    pytest.importorskip('torch')
    pytest.importorskip('robustbench')  # classifier.py imports it at module load
    from pcld.models.classifier import _DATASET_NUM_CLASSES
    assert _DATASET_NUM_CLASSES['cifar100'] == 100

    # _RB_DATASET_MAP is local to _build_model; prove the mapping by driving
    # _build_model with a fake robustbench loader and dataset_type='cifar100'.
    import torch.nn as nn
    from pcld.models.classifier import _build_model

    recorded = {}

    def fake_load_model(**kwargs):
        recorded.update(kwargs)
        return nn.Identity()

    fake_utils = types.ModuleType('robustbench.utils')
    fake_utils.load_model = fake_load_model
    with mock.patch.dict(sys.modules, {'robustbench.utils': fake_utils}):
        _build_model(CLASSIFIER_REGISTRY['wrn-70-16-wang2023'],
                     n_classes=100, dataset_type='cifar100')

    assert recorded['dataset'] == 'cifar100'
