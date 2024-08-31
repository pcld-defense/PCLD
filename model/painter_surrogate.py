import torch
import torch.nn.functional as F
import torch.nn as nn
torch.manual_seed(42)


class PainterSurrogate_(nn.Module):
    def __init__(self, encoder):
        super(PainterSurrogate_, self).__init__()
        self.encoder = encoder
        # Decoder - adjusting layers to achieve 3x300x300 output
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1) # Output: 128x40x40
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # Output: 64x80x80
        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=1)   # Output: 32x160x160
        self.upconv4 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)   # Output: 16x320x320
        self.conv_final = nn.Conv2d(16, 3, kernel_size=27, stride=1, padding=10)        # Output: 3x300x300

    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(self.upconv1(x))
        x = F.relu(self.upconv2(x))
        x = F.relu(self.upconv3(x))
        x = F.relu(self.upconv4(x))
        # Final convolution to get to 300x300 and reduce to 1 channel
        x = self.conv_final(x)
        x = torch.sigmoid(x)
        return x


# a surrogate for the original image (t=∞)
class IdentitySurrogate_(nn.Module):
    def __init__(self):
        super(IdentitySurrogate_, self).__init__()

    def forward(self, x):
        return x


class PainterSurrogate(torch.nn.Module):
    def __init__(self, surrogates):
        super(PainterSurrogate, self).__init__()
        # self.surrogates = surrogates
        self.surrogates = nn.ModuleList(surrogates)

    def forward(self, x):
        canvases = [surrogate(x) for surrogate in self.surrogates]
        canvases = torch.stack(canvases, dim=1)
        return canvases





