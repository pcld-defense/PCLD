import torch
import torch.nn.functional as F
import torch.nn as nn

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
        canvases = [surrogate(x) for surrogate in self.surrogates]
        canvases = torch.stack(canvases, dim=1)
        return canvases
