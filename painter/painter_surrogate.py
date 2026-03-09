import os
import re
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from util.models import load_model

torch.manual_seed(42)


class PainterSurrogate_(nn.Module):
    """Differentiable surrogate for a single paint-step of the neural painter.

    Trained to approximate the output of the non-differentiable painter at a
    specific stroke count t, enabling gradient flow through BPDA. The encoder
    is a truncated ResNet18 (up to layer3), and the decoder is a series of
    transposed convolutions that upsample back to the original image resolution.
    """

    def __init__(self, encoder: nn.Module) -> None:
        """Initialises the surrogate with a shared encoder and a fixed decoder.

        Args:
            encoder: A pretrained feature extractor (e.g. truncated ResNet18)
                that maps (B, 3, H, W) → (B, 256, h, w).
        """
        super(PainterSurrogate_, self).__init__()
        self.encoder = encoder
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1,
                                          output_padding=1)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1,
                                          output_padding=1)
        self.upconv4 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.conv_final = nn.Conv2d(16, 3, kernel_size=27, stride=1, padding=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes the input and decodes it into an approximated painted canvas.

        Args:
            x: Input image batch of shape (B, 3, H, W) in [0, 1].

        Returns:
            Approximated painted canvas of shape (B, 3, 300, 300) in [0, 1].
        """
        x = self.encoder(x)
        x = F.relu(self.upconv1(x))
        x = F.relu(self.upconv2(x))
        x = F.relu(self.upconv3(x))
        x = F.relu(self.upconv4(x))
        x = self.conv_final(x)
        x = torch.sigmoid(x)
        return x


class IdentitySurrogate_(nn.Module):
    """Pass-through surrogate representing the original image at t=∞.

    Used as the last element in the surrogate list to model the unmodified
    input image as if it were the output of an infinitely-stroked painter.
    """

    def __init__(self) -> None:
        super(IdentitySurrogate_, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the input unchanged.

        Args:
            x: Any tensor.

        Returns:
            The same tensor, unmodified.
        """
        return x


class PainterSurrogate(torch.nn.Module):
    """Aggregates multiple per-step surrogates into a single differentiable painter.

    Runs each surrogate in the list over the input and stacks the results along
    a new step dimension, producing a tensor compatible with the PCLD pipeline.
    """

    def __init__(self, surrogates: list[nn.Module]) -> None:
        """Wraps a list of step-specific surrogate models.

        Args:
            surrogates: Ordered list of surrogate models, one per paint step
                (including the identity surrogate for t=∞ as the last entry).
        """
        super(PainterSurrogate, self).__init__()
        self.surrogates = nn.ModuleList(surrogates)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies every surrogate to the input and stacks the outputs.

        Args:
            x: Input image batch of shape (B, 3, H, W).

        Returns:
            Stacked canvases of shape (B, Steps, 3, H, W), where Steps equals
            the number of surrogates.
        """
        h, w = x.shape[-2], x.shape[-1]
        canvases = []
        for surrogate in self.surrogates:
            out = surrogate(x)
            if out.shape[-2:] != (h, w):
                out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
            canvases.append(out)
        canvases = torch.stack(canvases, dim=1)
        return canvases


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
