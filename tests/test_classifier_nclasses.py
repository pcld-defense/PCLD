"""The classifier head must follow the requested n_classes, not the dataset type."""

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('robustbench')  # classifier.py imports it at module load


def _tiny_wrn_cfg():
    from pcld.models.registry import ClassifierConfig
    return ClassifierConfig(family='wrn', optimizer='sgd',
                            weight_decay_imagenet=1e-4, weight_decay_cifar10=5e-4,
                            wrn_depth=16, wrn_width=1)


@pytest.mark.parametrize('n_classes', [7, 10, 100])
def test_head_follows_n_classes(n_classes):
    from pcld.models.classifier import _build_model
    model = _build_model(_tiny_wrn_cfg(), n_classes, dataset_type='cifar10').eval()
    out = model(torch.randn(2, 3, 32, 32))
    assert out.shape[1] == n_classes


def test_none_falls_back_to_dataset_size():
    from pcld.models.classifier import _DATASET_NUM_CLASSES
    # The fallback table is only consulted when n_classes is None in get_net.
    assert _DATASET_NUM_CLASSES['imagenet'] == 1000
    assert _DATASET_NUM_CLASSES['cifar10'] == 10
