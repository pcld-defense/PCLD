import os
import re
from typing import Union

import torch
import torch.nn as nn
from torchvision import models

from util.models import load_model
from painter.painter_surrogate import PainterSurrogate_

torch.manual_seed(42)


def load_painter_surrogate(models_folder: str, device: str,
                           output_every: Union[list[int], None] = None) -> list[nn.Module]:
    """Loads per-step painter surrogate models from a directory.

    Scans `models_folder` for files matching `model_t<step>.pth`, sorts them
    by stroke count ascending, and optionally filters to only the steps listed
    in `output_every`. Returns an ordered list of loaded PainterSurrogate_
    models ready for inference.

    Args:
        models_folder: Path to the directory containing surrogate checkpoints
            named `model_t<step>.pth`.
        device: Target device string (e.g. 'cuda' or 'cpu').
        output_every: If provided, only load surrogates whose stroke count
            matches an entry in this list. Loads all files when None.

    Returns:
        List of PainterSurrogate_ models sorted by ascending stroke count,
        each moved to `device` with weights loaded.
    """
    output_every_names = [f'model_t{oe}.pth' for oe in output_every] if output_every else []
    models_names = os.listdir(models_folder)
    pattern = re.compile(r'model_t(\d+)\.pth')

    def sort_key(filename: str) -> int:
        match = pattern.match(filename)
        if match:
            return int(match.group(1))
        else:
            return float('inf')

    sorted_list = sorted(models_names, key=sort_key)
    surrogate_list = []
    for name in sorted_list:
        if output_every:
            if name in output_every_names:
                model_path = os.path.join(models_folder, name)
            else:
                continue
        else:
            model_path = os.path.join(models_folder, name)

        # Create the model
        encoder = models.resnet18(weights='IMAGENET1K_V1')
        encoder = nn.Sequential(*list(encoder.children())[:-3])
        painter_surr = PainterSurrogate_(encoder)
        painter_surr = painter_surr.to(device)
        model = load_model(painter_surr, model_path, device)
        surrogate_list.append(model)

    return surrogate_list
