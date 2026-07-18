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
    assert hasattr(r, 'models') and len(r.models) == 5


def test_surrogate_type_default_learned():
    a = _load([])
    assert a.surrogate_type == 'learned'


def test_aa_version_default_standard():
    a = _load([])
    assert a.aa_version == 'standard'
    r = _load(['attack=aa_rand'])
    assert r.attack == 'aa'
    assert r.aa_version == 'rand'


def test_painter_step_defaults():
    a = _load([])
    assert a.painter_max_step == 80
    assert a.painter_divide == 5


def test_num_samples_and_data_source_defaults():
    a = _load([])
    assert a.num_samples is None
    assert a.data_source == 'folder'

    r = _load(['experiment=rb_sweep'])
    assert r.num_samples == 512


def test_rb_sweep_runs_real_attack_path():
    r = _load(['experiment=rb_sweep'])
    assert r.experiment_type == 'attack_classifier'
    assert r.data_source == 'robustbench'
    assert 0 in r.epsilons


def test_l2_float_epsilons_preserved():
    o = _load(['attack.epsilons=[0.5]', 'attack.attack_norm=l2'])
    assert o.epsilons == [0.5]
    assert isinstance(o.epsilons[0], float)
    # Integral values still flatten to ints.
    i = _load(['attack.epsilons=[0,8]'])
    assert i.epsilons == [0, 8]
    assert all(isinstance(e, int) for e in i.epsilons)

def test_cluster_sweep_presets():
    """The three cluster comparison presets compose with the right norms,
    epsilons, and model lists."""
    l2 = _load(['experiment=rb_sweep_l2'])
    assert l2.experiment_type == 'attack_classifier'
    assert l2.dataset_type == 'cifar10'
    assert l2.attack_norm == 'l2'
    assert l2.epsilons == [0, 0.5]
    assert isinstance(l2.epsilons[1], float)
    assert [m['name'] for m in l2.models] == [
        'wrn-28-10-rebuffi2021-l2', 'wrn-70-16-wang2023-l2']
    assert all(m['threat_model'] == 'L2' for m in l2.models)

    c100 = _load(['experiment=rb_sweep_cifar100'])
    assert c100.experiment_type == 'attack_classifier'
    assert c100.dataset_type == 'cifar100'
    assert c100.attack_norm == 'linf'
    assert c100.epsilons == [0, 8]
    assert [m['name'] for m in c100.models] == ['wrn-70-16-wang2023']

    inet = _load(['experiment=rb_sweep_imagenet'])
    assert inet.experiment_type == 'attack_classifier'
    assert inet.dataset_type == 'imagenet'
    assert inet.attack_norm == 'linf'
    assert inet.epsilons == [0, 4]
    assert inet.data_source == 'robustbench'
    assert [m['name'] for m in inet.models] == [
        'swin-l-liu2023', 'resnet50-standard']


def test_gradient_battery_preset():
    g = _load(['experiment=gradient_battery'])
    assert g.experiment_type == 'gradient_battery'
    assert g.battery_epsilons == [8, 16, 32, 64]
    assert all(isinstance(e, int) for e in g.battery_epsilons)
    assert g.battery_fd_dirs == 8
    assert g.battery_fd_delta == 1e-3
    assert g.num_samples == 32
    assert g.attack == 'pgd'
