"""Dataset-specific painter configurations.

Each dataset gets a PainterConfig that specifies the native canvas resolution,
the coarse-to-fine patch grid, and the stroke budget.  The actor and renderer
are instantiated with ``config.width`` so they operate at native resolution
— no image resize is needed before or after painting.

DIVIDE only affects inference (Phase 2 patching).  Training always uses a
single WIDTH×WIDTH canvas regardless of DIVIDE.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PainterConfig:
    """Immutable painting configuration for a single dataset.

    Args:
        width: Canvas side length in pixels.  The actor and renderer are
            built for this resolution.  Must be divisible by 8 (three
            PixelShuffle(2) stages in the renderer = 8× spatial upscale).
        divide: Number of patches per spatial dimension in Phase 2.
            ``1`` disables Phase 2 entirely.
        max_step: Total actor forward-pass budget.  When ``divide > 1``
            the budget is split equally between Phase 1 and Phase 2.
            Each forward pass produces 5 strokes.
    """

    width: int
    divide: int
    max_step: int

    def __post_init__(self) -> None:
        if self.width % 8 != 0:
            raise ValueError(
                f"width must be divisible by 8 (3 PixelShuffle stages), "
                f"got {self.width}"
            )


# ── Dataset-specific configs ─────────────────────────────────────────────────

CIFAR10_PAINTER = PainterConfig(width=32, divide=1, max_step=40)
CIFAR100_PAINTER = PainterConfig(width=32, divide=1, max_step=40)
IMAGENET_PAINTER = PainterConfig(width=224, divide=5, max_step=80)

# Backward-compatible config matching the original PCLD paper (128×128 patches).
LEGACY_PAINTER = PainterConfig(width=128, divide=5, max_step=80)

PAINTER_CONFIGS: dict[str, PainterConfig] = {
    'cifar10': CIFAR10_PAINTER,
    'cifar100': CIFAR100_PAINTER,
    'imagenet': IMAGENET_PAINTER,
}


def get_painter_config(dataset_type: str) -> PainterConfig:
    """Returns the PainterConfig for a dataset, falling back to LEGACY.

    Args:
        dataset_type: Dataset family key (e.g. ``'cifar10'``, ``'imagenet'``).

    Returns:
        The matching PainterConfig.
    """
    return PAINTER_CONFIGS.get(dataset_type, LEGACY_PAINTER)
