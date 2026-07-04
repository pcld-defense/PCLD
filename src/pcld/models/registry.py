"""Architecture registry for classifier models in the PCLD pipeline.

To add a new classifier, add a single ``ClassifierConfig`` entry to
``CLASSIFIER_REGISTRY`` below.  See the README for full instructions.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ClassifierConfig:
    """Immutable configuration record for a single classifier architecture.

    Args:
        family: Architecture source family; determines the build branch in
            ``model/classifier.py``.  Built-in families:

            - ``'wrn'``         — Wide ResNet architecture (random init; load
              local weights via ``get_net(weights=...)``.
            - ``'timm'``        — Any model available in the ``timm`` hub.
            - ``'robustbench'`` — Model fetched from the RobustBench model zoo.
              Only **non-adversarially-trained** entries are allowed (see
              ``ROBUSTBENCH_STANDARD_MODELS``).

        optimizer: Optimizer type; ``'sgd'`` or ``'adamw'``.
        weight_decay_imagenet: L2 regularisation weight for ImageNet training.
        weight_decay_cifar10: L2 regularisation weight for CIFAR-10 training.
        wrn_depth: WideResNet total depth (``family='wrn'`` only).
        wrn_width: WideResNet widen factor (``family='wrn'`` only).
        timm_name: Model identifier passed to ``timm.create_model``
            (``family='timm'`` only).
        timm_pretrained: If ``True``, load timm hub pretrained weights when
            no local checkpoint path is provided (``family='timm'`` only).
        robustbench_name: Model name passed to ``robustbench.utils.load_model``
            (``family='robustbench'`` only).  Must be in
            ``ROBUSTBENCH_STANDARD_MODELS``.
    """

    family: str
    optimizer: str
    weight_decay_imagenet: float
    weight_decay_cifar10: float
    wrn_depth: Optional[int] = None
    wrn_width: Optional[int] = None
    timm_name: Optional[str] = None
    timm_pretrained: bool = False
    robustbench_name: Optional[str] = None


# ---------------------------------------------------------------------------
# Allowlist of RobustBench CIFAR-10 models permitted for use in the registry.
# Includes both standard (non-AT) and adversarially-trained baseline models.
# AT models are allowed for baseline evaluation only, not as the PCLD
# classifier used during adversarial training.
# ---------------------------------------------------------------------------
ROBUSTBENCH_STANDARD_MODELS: list = [
    'Standard',                           # WRN-28-10, non-AT — Rice et al. 2020
    'Carmon2019Unlabeled',               # WRN-28-10, AT + unlabeled data
    'Gowal2021Improving_28_10_ddpm_100m',  # WRN-28-10, AT + DDPM data
    'Rebuffi2021Fixing_28_10_cutmix_ddpm', # WRN-28-10, AT + cutmix + DDPM
    'Wang2023Better_WRN-28-10',           # WRN-28-10, AT — Wang et al. 2023
    # WRN-34-10 adversarially-trained baselines
    'Cui2020Learnable_34_10',             # WRN-34-10, AT — TRADES variant
    'Chen2021LTD_WRN34_10',              # WRN-34-10, AT — LTD
    'Jia2022LAS-AT_34_10',               # WRN-34-10, AT — LAS-AT
    'Addepalli2022Efficient_WRN_34_10',   # WRN-34-10, AT — Addepalli et al.
    # ImageNet standard (non-AT) baselines
    'Standard_R50',                       # ResNet-50, non-AT ImageNet — He et al. 2016
]


# ---------------------------------------------------------------------------
# Add new architectures here — no other file needs to change for 'wrn'/'timm'
# ---------------------------------------------------------------------------
CLASSIFIER_REGISTRY: dict = {

    # --- Wide ResNet (robustbench) -------------------------------------------
    'wrn-70-16': ClassifierConfig(
        family='wrn',
        optimizer='sgd',
        weight_decay_imagenet=1e-4,
        weight_decay_cifar10=5e-4,
        wrn_depth=70,
        wrn_width=16,
    ),
    'wrn-34-10': ClassifierConfig(
        family='wrn',
        optimizer='sgd',
        weight_decay_imagenet=1e-4,
        weight_decay_cifar10=5e-4,
        wrn_depth=34,
        wrn_width=10,
    ),

    # --- RobustBench CIFAR-10 models (standard + adversarially-trained baselines) ---
    'wrn-28-10-standard': ClassifierConfig(
        family='robustbench',
        optimizer='sgd',
        weight_decay_imagenet=1e-4,
        weight_decay_cifar10=5e-4,
        wrn_depth=28,
        wrn_width=10,
        robustbench_name='Standard',
    ),
    'wrn-28-10-carmon2019': ClassifierConfig(
        family='robustbench',
        optimizer='sgd',
        weight_decay_imagenet=1e-4,
        weight_decay_cifar10=5e-4,
        wrn_depth=28,
        wrn_width=10,
        robustbench_name='Carmon2019Unlabeled',
    ),
    'wrn-28-10-gowal2021': ClassifierConfig(
        family='robustbench',
        optimizer='sgd',
        weight_decay_imagenet=1e-4,
        weight_decay_cifar10=5e-4,
        wrn_depth=28,
        wrn_width=10,
        robustbench_name='Gowal2021Improving_28_10_ddpm_100m',
    ),
    'wrn-28-10-rebuffi2021': ClassifierConfig(
        family='robustbench',
        optimizer='sgd',
        weight_decay_imagenet=1e-4,
        weight_decay_cifar10=5e-4,
        wrn_depth=28,
        wrn_width=10,
        robustbench_name='Rebuffi2021Fixing_28_10_cutmix_ddpm',
    ),
    'wrn-28-10-wang2023': ClassifierConfig(
        family='robustbench',
        optimizer='sgd',
        weight_decay_imagenet=1e-4,
        weight_decay_cifar10=5e-4,
        wrn_depth=28,
        wrn_width=10,
        robustbench_name='Wang2023Better_WRN-28-10',
    ),

    # --- RobustBench WRN-34-10 adversarially-trained baselines ----------------
    'wrn-34-10-cui2020': ClassifierConfig(
        family='robustbench',
        optimizer='sgd',
        weight_decay_imagenet=1e-4,
        weight_decay_cifar10=5e-4,
        wrn_depth=34,
        wrn_width=10,
        robustbench_name='Cui2020Learnable_34_10',
    ),
    'wrn-34-10-chen2021': ClassifierConfig(
        family='robustbench',
        optimizer='sgd',
        weight_decay_imagenet=1e-4,
        weight_decay_cifar10=5e-4,
        wrn_depth=34,
        wrn_width=10,
        robustbench_name='Chen2021LTD_WRN34_10',
    ),
    'wrn-34-10-jia2022': ClassifierConfig(
        family='robustbench',
        optimizer='sgd',
        weight_decay_imagenet=1e-4,
        weight_decay_cifar10=5e-4,
        wrn_depth=34,
        wrn_width=10,
        robustbench_name='Jia2022LAS-AT_34_10',
    ),
    'wrn-34-10-addepalli2022': ClassifierConfig(
        family='robustbench',
        optimizer='sgd',
        weight_decay_imagenet=1e-4,
        weight_decay_cifar10=5e-4,
        wrn_depth=34,
        wrn_width=10,
        robustbench_name='Addepalli2022Efficient_WRN_34_10',
    ),

    # --- RobustBench ImageNet standard (non-AT) baseline ----------------------
    'resnet50-standard': ClassifierConfig(
        family='robustbench',
        optimizer='sgd',
        weight_decay_imagenet=1e-4,
        weight_decay_cifar10=5e-4,
        robustbench_name='Standard_R50',
    ),

    # --- ResNet (timm) -------------------------------------------------------
    'resnet50': ClassifierConfig(
        family='timm',
        optimizer='sgd',
        weight_decay_imagenet=1e-4,
        weight_decay_cifar10=5e-4,
        timm_name='resnet50',
        timm_pretrained=True,
    ),

    # --- Vision Transformers (timm) ------------------------------------------
    'xcit-m12': ClassifierConfig(
        family='timm',
        optimizer='adamw',
        weight_decay_imagenet=0.05,
        weight_decay_cifar10=0.05,
        timm_name='xcit_medium_24_p16_224',
        timm_pretrained=True,
    ),
}
