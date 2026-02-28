import torch.nn as nn
import torch

torch.manual_seed(42)

import torch
import torch.nn as nn
from torchvision import models

model = models.resnet18(weights='IMAGENET1K_V1')


class Encoder(nn.Module):
    """ResNet18-based image encoder that removes the classification head.

    Keeps all layers up to (but not including) the average-pooling and the
    fully-connected head, producing a spatial feature map instead of a class
    vector. Useful as a pretrained backbone for downstream tasks such as
    autoencoder-based surrogate training.
    """

    def __init__(self, resnet18: nn.Module) -> None:
        """Strips the last two layers from the provided ResNet18.

        Args:
            resnet18: A torchvision ResNet18 instance (pretrained or not).
        """
        super(Encoder, self).__init__()
        self.features = nn.Sequential(*list(resnet18.children())[:-2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts a spatial feature map from the input image.

        Args:
            x: Input image batch of shape (B, 3, H, W).

        Returns:
            Feature map of shape (B, 512, H/32, W/32).
        """
        x = self.features(x)
        return x


class Decoder(nn.Module):
    """Upsampling decoder that reconstructs a 300×300 RGB image from features.

    Takes the 512-channel spatial feature map produced by the Encoder and
    progressively upsamples it through five transposed convolutions, ending
    with a 3-channel output scaled to [0, 1] via sigmoid.
    """

    def __init__(self) -> None:
        """Builds the upsampling decoder as a sequential container."""
        super(Decoder, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=21, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsamples the feature map to a full-resolution RGB image.

        Args:
            x: Feature map of shape (B, 512, H/32, W/32).

        Returns:
            Reconstructed image batch of shape (B, 3, 300, 300) in [0, 1].
        """
        x = self.up(x)
        return x


class Autoencoder(nn.Module):
    """Full autoencoder combining an Encoder and a Decoder.

    Encodes an input image into a compact spatial feature map and then decodes
    it back to the original spatial resolution. Designed for 300×300 RGB inputs
    when paired with the ResNet18-based Encoder and matching Decoder.
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        """Composes the encoder and decoder.

        Args:
            encoder: Feature extractor (e.g. Encoder wrapping ResNet18).
            decoder: Upsampling network (e.g. Decoder).
        """
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes the input and reconstructs it with the decoder.

        Args:
            x: Input image batch of shape (B, 3, H, W).

        Returns:
            Reconstructed image batch of shape (B, 3, 300, 300) in [0, 1].
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x
