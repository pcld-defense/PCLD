"""Global seeding for reproducible experiments."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seeds all RNGs used by the pipeline.

    Seeds Python's ``random``, NumPy, and PyTorch (CPU and CUDA), and also sets
    ``PYTHONHASHSEED`` so hash-based ordering is stable. When ``deterministic``
    is True, additionally forces cuDNN into deterministic mode, which makes
    convolutions reproducible at the cost of some throughput.

    Args:
        seed: The integer seed applied to every RNG.
        deterministic: If True, force cuDNN deterministic algorithms and
            disable its autotuner. Leave False for maximum speed when exact
            bit-for-bit reproducibility of GPU kernels is not required.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False