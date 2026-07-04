"""Normalization wrapper following the RobustBench convention.

RobustBench models normalize internally so that attacks can operate in [0, 1]
pixel space with correct clip bounds.  This wrapper adds the same behaviour
to any classifier that was trained with ``transforms.Normalize(mean, std)``
in the DataLoader — the weights stay the same, only the normalization point
moves from the DataLoader into the model.
"""

import torch
import torch.nn as nn


class NormalizedModel(nn.Module):
    """Wraps a classifier to normalize [0, 1] input internally.

    Args:
        model: Classifier expecting normalised input.
        mean: Per-channel mean, e.g. ``(0.4914, 0.4822, 0.4465)``.
        std: Per-channel std, e.g. ``(0.2471, 0.2435, 0.2616)``.
    """

    def __init__(self, model: nn.Module, mean, std) -> None:
        super().__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes ``x`` from [0, 1] then forwards to the wrapped model."""
        return self.model((x - self.mean) / self.std)
