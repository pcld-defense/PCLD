import torch
from torch import nn
import torch.nn.functional as F
torch.manual_seed(42)

# 1DConv decisioner
class Decisioner1DConv(nn.Module):
    def __init__(self, num_classes, num_steps, num_filters=64):
        super(Decisioner1DConv, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_steps, out_channels=num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.dropout1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_filters)
        self.dropout2 = nn.Dropout(0.5)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        # x = x.permute(0, 2, 1)
        x = F.relu(self.dropout1(self.bn1(self.conv1(x))))
        x = F.relu(self.dropout2(self.bn2(self.conv2(x))))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Fully connected decisioner
class DecisionerFC(nn.Module):
    def __init__(self, num_classes, num_steps):
        super(DecisionerFC, self).__init__()
        input_dim = num_steps * num_classes
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
