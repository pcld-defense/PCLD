import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(42)


class DecisionerStepAttention(nn.Module):
    """Attention-based decisioner that learns which paint steps to trust.

    Encodes the confidence trajectory with a 1D conv, computes a soft
    attention weight per paint step, then forms a weighted combination of
    the per-step softmax vectors and maps it to class logits via a linear
    head. This lets the model explicitly learn to prefer paint steps whose
    classifier output is more reliable.
    """

    def __init__(self, n_classes: int, paint_steps: int,
                 hidden_dim: int = 32) -> None:
        """Builds the step-attention decisioner.

        Args:
            n_classes: Number of output classes (feature size per step).
            paint_steps: Number of paint steps (temporal sequence length).
            hidden_dim: Number of filters in the 1D conv encoder.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_classes, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.attn_head = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(n_classes, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classifies via a learned weighted average of per-step softmax outputs.

        Args:
            x: Softmax probability sequence of shape
                (batch_size, paint_steps, n_classes).

        Returns:
            Class logits of shape (batch_size, n_classes).
        """
        h = self.encoder(x.permute(0, 2, 1))        # (B, hidden, T)
        attn = torch.softmax(
            self.attn_head(h.permute(0, 2, 1)), dim=1
        )                                            # (B, T, 1)
        weighted = (x * attn).sum(dim=1)             # (B, n_classes)
        return self.classifier(weighted)             # (B, n_classes) logits


class Decisioner1DConv(nn.Module):
    """1D convolutional decisioner that reads confidence trajectories over paint steps.

    Treats each paint step as a temporal position and applies two layers of 1D
    convolutions over the softmax probability sequence, then compresses the
    temporal dimension via adaptive max pooling before the final linear layer.
    """

    def __init__(self, num_classes: int, num_steps: int,
                 num_filters: int = 64) -> None:
        """Builds the 1D conv decisioner.

        Args:
            num_classes: Number of output classes (also the feature size per step).
            num_steps: Number of paint steps (temporal length of the sequence).
            num_filters: Number of convolutional filters in both conv layers.
        """
        super(Decisioner1DConv, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_classes, out_channels=num_filters,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.dropout1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_filters)
        self.dropout2 = nn.Dropout(0.5)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classifies the confidence trajectory via 1D convolutions.

        Args:
            x: Softmax probability sequence of shape
                (batch_size, num_steps, num_classes).

        Returns:
            Class logits of shape (batch_size, num_classes).
        """
        # Transpose to (B, num_classes, num_steps) so the kernel slides over
        # the temporal axis (paint steps) and treats class probabilities as features.
        x = x.transpose(1, 2)
        x = F.relu(self.dropout1(self.bn1(self.conv1(x))))
        x = F.relu(self.dropout2(self.bn2(self.conv2(x))))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class DecisionerFC(nn.Module):
    """Fully-connected decisioner that classifies a flattened confidence trajectory.

    Concatenates all per-step softmax vectors into a single flat vector and
    passes it through a three-layer MLP with dropout regularisation.
    """

    def __init__(self, num_classes: int, num_steps: int) -> None:
        """Builds the fully-connected decisioner.

        Args:
            num_classes: Number of output classes (also the feature size per step).
            num_steps: Number of paint steps; input size is num_steps * num_classes.
        """
        super(DecisionerFC, self).__init__()
        input_dim = num_steps * num_classes
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classifies the flattened confidence trajectory.

        Args:
            x: Flattened softmax sequence of shape
                (batch_size, num_steps * num_classes).

        Returns:
            Class logits of shape (batch_size, num_classes).
        """
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
