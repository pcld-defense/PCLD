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

            - ``'wrn'``  — Wide ResNet via ``robustbench``.
            - ``'timm'`` — Any model available in the ``timm`` hub.

        optimizer: Optimizer type; ``'sgd'`` or ``'adamw'``.
        weight_decay_imagenet: L2 regularisation weight for ImageNet training.
        weight_decay_cifar10: L2 regularisation weight for CIFAR-10 training.
        wrn_depth: WideResNet total depth (``family='wrn'`` only).
        wrn_width: WideResNet widen factor (``family='wrn'`` only).
        timm_name: Model identifier passed to ``timm.create_model``
            (``family='timm'`` only).
        timm_pretrained: If ``True``, load timm hub pretrained weights when
            no local checkpoint path is provided (``family='timm'`` only).
    """

    family: str
    optimizer: str
    weight_decay_imagenet: float
    weight_decay_cifar10: float
    wrn_depth: Optional[int] = None
    wrn_width: Optional[int] = None
    timm_name: Optional[str] = None
    timm_pretrained: bool = False


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

    # --- Vision Transformers (timm) ------------------------------------------
    'xcit-m12': ClassifierConfig(
        family='timm',
        optimizer='adamw',
        weight_decay_imagenet=0.05,
        weight_decay_cifar10=0.05,
        timm_name='xcit_medium_12_p16_224',
        timm_pretrained=True,
    ),
}
