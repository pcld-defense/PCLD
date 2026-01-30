import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(42)


# 1DConv decisioner
class Decisioner1DConv(nn.Module):
    """
    A 1D Convolutional Neural Network for a Decisioner Neural Network.

    This module applies two layers of 1D convolutions followed by adaptive
    max pooling to extract local temporal patterns. It is designed to process
    input tensors where the sequence length is treated as the channel dimension.
    """

    def __init__(self, num_classes: int, num_steps: int, num_filters: int = 64):
        super(Decisioner1DConv, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_steps, out_channels=num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.dropout1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_filters)
        self.dropout2 = nn.Dropout(0.5)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass on the input tensor.

        Args:
            x: Input tensor of shape (batch_size, num_steps, features).

        Returns:
            Logits of shape (batch_size, num_classes).

        """
        x = F.relu(self.dropout1(self.bn1(self.conv1(x))))
        x = F.relu(self.dropout2(self.bn2(self.conv2(x))))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# Fully connected decisioner
class DecisionerFC(nn.Module):
    """
    A Fully Connected (MLP) architecture for classification decisions.

    This module implements a three-layer deep neural network designed to process
    flattened temporal or feature-based data. It utilizes ReLU activations and
    Dropout for regularization to prevent overfitting during training.
    """

    def __init__(self, num_classes: int, num_steps: int):
        super(DecisionerFC, self).__init__()
        input_dim = num_steps * num_classes
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the computation performed at every call.

        Args:
            x: A tensor of shape (batch_size, input_dim).
                Note: Input must be flattened prior to passing through this layer.

        Returns:
            The output logits of shape (batch_size, num_classes).

        """
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x
