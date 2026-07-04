"""Classifier factory for the PCLD pipeline.

To add a new architecture, add a ``ClassifierConfig`` entry to
``CLASSIFIER_REGISTRY`` in ``model/consts.py``.  No changes to this file
are needed unless you are introducing a brand-new architecture *family*.
"""

from typing import Optional

import timm
import torch.nn as nn
import torch.optim as optim
from robustbench.model_zoo.architectures.wide_resnet import WideResNet

from pcld.models.registry import CLASSIFIER_REGISTRY, ClassifierConfig, ROBUSTBENCH_STANDARD_MODELS
from pcld.utils.consts import CIFAR10Consts, IMAGENETConsts
from pcld.models.train_utils import load_model

# Derived once from dataset consts so they stay the single source of truth.
_DATASET_NUM_CLASSES: dict = {
    'imagenet': IMAGENETConsts.NUM_CLASSES,
    'cifar10': CIFAR10Consts.NUM_CLASSES,
}

SUPPORTED_MODELS: list = sorted(CLASSIFIER_REGISTRY)


# ---------------------------------------------------------------------------
# Internal helpers — extend these when adding a new family
# ---------------------------------------------------------------------------

def _build_model(cfg: ClassifierConfig, n_classes: int,
                 use_pretrained: bool = True,
                 dataset_type: str = 'cifar10') -> nn.Module:
    """Instantiates a bare model from its registry config.

    Args:
        cfg: Architecture config from ``CLASSIFIER_REGISTRY``.
        n_classes: Number of output classes.
        use_pretrained: When ``False``, suppresses timm hub weight loading
            even if ``cfg.timm_pretrained`` is ``True`` (used when a local
            checkpoint will be loaded afterwards). Ignored for
            ``family='robustbench'`` since robustbench always bundles weights.
        dataset_type: Dataset family; used by the ``'robustbench'`` branch to
            select the correct model zoo (currently only ``'cifar10'``
            is supported).

    Returns:
        Uninitialised (or hub-/robustbench-pretrained) ``nn.Module``.

    Raises:
        ValueError: If ``cfg.family`` is not a recognised build family, or if
            a ``'robustbench'`` model name is not in
            ``ROBUSTBENCH_STANDARD_MODELS``.
    """
    if cfg.family == 'wrn':
        return WideResNet(depth=cfg.wrn_depth, num_classes=n_classes,
                          widen_factor=cfg.wrn_width)

    if cfg.family == 'timm':
        pretrained = cfg.timm_pretrained and use_pretrained
        return timm.create_model(cfg.timm_name, pretrained=pretrained,
                                 num_classes=n_classes)

    if cfg.family == 'robustbench':
        if cfg.robustbench_name not in ROBUSTBENCH_STANDARD_MODELS:
            raise ValueError(
                f"RobustBench model {cfg.robustbench_name!r} is not in the "
                f"non-adversarially-trained allowlist "
                f"{ROBUSTBENCH_STANDARD_MODELS}. "
                f"Using adversarially pre-trained weights would compromise "
                f"PCLD's research validity."
            )
        _RB_DATASET_MAP = {'cifar10': 'cifar10', 'imagenet': 'imagenet'}
        rb_dataset = _RB_DATASET_MAP.get(dataset_type)
        if rb_dataset is None:
            raise ValueError(
                f"RobustBench standard models are only available for "
                f"{list(_RB_DATASET_MAP)}, got {dataset_type!r}."
            )
        from robustbench.utils import load_model as rb_load_model  # lazy import
        return rb_load_model(model_name=cfg.robustbench_name,
                             dataset=rb_dataset, threat_model='Linf')

    raise ValueError(
        f"Unknown architecture family {cfg.family!r}. "
        f"Add an 'elif cfg.family == ...' branch in _build_model() "
        f"(model/classifier.py)."
    )


def _build_optimizer(cfg: ClassifierConfig, net: nn.Module,
                     lr: float, dataset_type: str) -> optim.Optimizer:
    """Creates the optimizer matching the architecture's training convention.

    Args:
        cfg: Architecture config from ``CLASSIFIER_REGISTRY``.
        net: The model whose parameters will be optimised.
        lr: Initial learning rate.
        dataset_type: ``'imagenet'`` or ``'cifar10'``; selects weight decay.

    Returns:
        Configured ``Optimizer`` instance.
    """
    wd = (cfg.weight_decay_imagenet if dataset_type == 'imagenet'
          else cfg.weight_decay_cifar10)

    if cfg.optimizer == 'adamw':
        return optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)

    # default: sgd
    return optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_net(dataset_type: str, device: str, model_type: str = 'wrn-70-16',
            weights: Optional[str] = None,
            n_classes: Optional[int] = None) -> nn.Module:
    """Builds and returns a classifier for the specified dataset and architecture.

    Args:
        dataset_type: Dataset family; ``'imagenet'`` or ``'cifar10'``.
        device: Target device string (e.g. ``'cuda'`` or ``'cpu'``).
        model_type: Key in ``CLASSIFIER_REGISTRY`` (e.g. ``'wrn-70-16'``,
            ``'wrn-34-10'``, ``'xcit-m12'``).
        weights: Path to a ``.pth`` checkpoint to load after construction.
            For timm models with ``timm_pretrained=True``, omitting this
            loads hub pretrained weights automatically.
        n_classes: Number of output classes. When given, this sizes the model
            head — pass the number of classes in the actual dataset (e.g. 7 for
            a 7-class ImageNet subset). When ``None``, falls back to the full
            dataset size for ``dataset_type`` (1000 for ImageNet, 10 for
            CIFAR-10). Ignored by ``family='robustbench'`` models, whose head
            is fixed by their pretrained weights.

    Returns:
        The classifier network moved to ``device``.

    Raises:
        ValueError: If ``model_type`` is not in ``CLASSIFIER_REGISTRY``.
    """
    if model_type not in CLASSIFIER_REGISTRY:
        raise ValueError(
            f"Unknown model_type {model_type!r}. "
            f"Registered models: {SUPPORTED_MODELS}"
        )

    cfg = CLASSIFIER_REGISTRY[model_type]
    if n_classes is None:
        n_classes = _DATASET_NUM_CLASSES[dataset_type]

    # Suppress hub weights when a local checkpoint will be loaded next.
    model = _build_model(cfg, n_classes, use_pretrained=(weights is None),
                         dataset_type=dataset_type)

    if weights is not None:
        model = load_model(model, weights, device)

    return model.to(device)


def get_net_and_optim(dataset_type: str, device: str, lr: float,
                      model_type: str = 'wrn-70-16',
                      weights: Optional[str] = None,
                      n_classes: Optional[int] = None) -> tuple:
    """Builds the network together with its loss function, optimiser, and scheduler.

    Optimizer and weight-decay are chosen per architecture from the registry;
    both WRN and timm families use ``CosineAnnealingLR``.

    Args:
        dataset_type: Dataset family; ``'imagenet'`` or ``'cifar10'``.
        device: Target device string.
        lr: Initial learning rate.
        model_type: Key in ``CLASSIFIER_REGISTRY``, forwarded to ``get_net``.
        weights: Optional checkpoint path, forwarded to ``get_net``.
        n_classes: Number of output classes, forwarded to ``get_net``. When
            ``None``, falls back to the full dataset size for ``dataset_type``.

    Returns:
        Tuple of ``(net, criterion, optimizer, scheduler)`` where:
            net: The initialised classifier.
            criterion: ``CrossEntropyLoss`` with label smoothing 0.1.
            optimizer: SGD (WRN) or AdamW (timm), per registry config.
            scheduler: ``CosineAnnealingLR`` over 200 epochs.
    """
    cfg = CLASSIFIER_REGISTRY[model_type]
    net = get_net(dataset_type, device, model_type=model_type, weights=weights,
                  n_classes=n_classes)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = _build_optimizer(cfg, net, lr, dataset_type)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    return net, criterion, optimizer, scheduler
