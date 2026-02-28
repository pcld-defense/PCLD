import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models

torch.manual_seed(42)


class DenoisingAutoencoder(nn.Module):
    """Convolutional denoising autoencoder for 300×300 RGB images.

    Encodes the input through three strided convolutions (3→32→64→128 channels)
    and decodes it back with transposed convolutions, ending with a sigmoid
    activation to keep outputs in [0, 1].
    """

    def __init__(self) -> None:
        """Builds the encoder and decoder as sequential containers."""
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes and then decodes the input image.

        Args:
            x: Noisy input image batch of shape (B, 3, 300, 300) in [0, 1].

        Returns:
            Reconstructed image batch of shape (B, 3, 300, 300) in [0, 1].
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class AutoEncoder(nn.Module):
    """Autoencoder that uses a pretrained ResNet encoder with a learned decoder.

    The encoder is a truncated feature extractor (e.g. ResNet18 up to layer3),
    and the decoder upsamples back to 300×300 via four transposed convolutions
    and a final 27×27 conv to fine-tune spatial resolution.
    """

    def __init__(self, encoder: nn.Module) -> None:
        """Attaches the provided encoder and builds the fixed decoder.

        Args:
            encoder: Pretrained feature extractor that maps
                (B, 3, H, W) → (B, 256, h, w).
        """
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2,
                                          padding=1, output_padding=1)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2,
                                          padding=1, output_padding=1)
        self.upconv4 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.conv_final = nn.Conv2d(16, 3, kernel_size=27, stride=1, padding=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes the input and decodes it back to 300×300 RGB.

        Args:
            x: Input image batch of shape (B, 3, H, W) in [0, 1].

        Returns:
            Reconstructed image batch of shape (B, 3, 300, 300) in [0, 1].
        """
        x = self.encoder(x)
        x = F.relu(self.upconv1(x))
        x = F.relu(self.upconv2(x))
        x = F.relu(self.upconv3(x))
        x = F.relu(self.upconv4(x))
        x = self.conv_final(x)
        x = torch.sigmoid(x)
        return x
