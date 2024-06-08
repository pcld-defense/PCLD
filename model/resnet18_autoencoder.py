import torch.nn as nn
import torch


torch.manual_seed(42)

import torch
import torch.nn as nn
from torchvision import models

# Load pre-trained ResNet18
# resnet18 = models.resnet18(pretrained=True)
model = models.resnet18(weights='IMAGENET1K_V1')

# Modify ResNet18 for encoding
class Encoder(nn.Module):
    def __init__(self, resnet18):
        super(Encoder, self).__init__()
        self.features = nn.Sequential(*list(resnet18.children())[:-2]) # Remove last two layers

    def forward(self, x):
        x = self.features(x)
        return x

# Decoder architecture (example)
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Assuming input features are 7x7x512 (from ResNet18)
        self.up = nn.Sequential(
            # Upsample to 14x14
            # nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            # nn.ReLU(),
            # # Upsample to 28x28
            # nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            # nn.ReLU(),
            # # Upsample to 112x112 (bigger stride and kernel)
            # nn.ConvTranspose2d(128, 64, kernel_size=4, stride=4),
            # nn.ReLU(),
            # # Upsample to 300x300 - need a different kernel size and stride
            # nn.ConvTranspose2d(64, 32, kernel_size=6, stride=3, padding=1),
            # nn.ReLU(),

            # Upsample to 512x20x20
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # Upsample to 256x40x40
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # Upsample to 128x80x80
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # Upsample to 64x160x160
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # Upsample to 32x320x320
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # Adjust to 16x300x300 - using a convolution with kernel_size=21, stride=1, padding=0
            nn.Conv2d(16, 3, kernel_size=21, stride=1, padding=0),
            nn.Sigmoid()

            # Final convolution to get 3 channels (RGB)
            # nn.Conv2d(32, 3, kernel_size=3, padding=1),
            # nn.Sigmoid()  # Sigmoid activation to scale the output to [0,1] range
        )

    def forward(self, x):
        x = self.up(x)
        return x

# Combine encoder and decoder
class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

