import torch
import torch.nn as nn
import torch.nn.functional as F


class Decisioner(nn.Module):
    def __init__(self):
        super(Decisioner, self).__init__()
        self.fc1 = nn.Linear(12 * 3, 64)
        self.bn1 = nn.BatchNorm1d(64)  # Batch Normalization after first linear layer
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)  # Batch Normalization after second linear layer
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        # Return logits for CrossEntropyLoss, which applies softmax
        return x  # remove softmax here if using CrossEntropyLoss


class AECLD(nn.Module):
    def __init__(self, ae_painters: nn.ModuleList,
                 classifier: nn.Module, decisioner: nn.Module) -> None:
        super(AECLD, self).__init__()
        self.ae_painters = ae_painters
        self.classifier = classifier
        self.decisioner = decisioner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        aecl_output = []
        for ae in self.ae_painters:
            x_aecl = ae(x)
            x_aecl = self.classifier(x_aecl)
            x_aecl = F.softmax(x_aecl, dim=1)
            aecl_output.append(x_aecl)
        x_aecl = self.classifier(x)
        x_aecl = F.softmax(x_aecl, dim=1)
        aecl_output.append(x_aecl)
        x_aecl = torch.stack(aecl_output, dim=0)
        x_aecl = torch.permute(x_aecl, (1, 0, 2))
        x_aecl_reshaped = x_aecl.reshape(x_aecl.shape[0] * x_aecl.shape[1], x_aecl.shape[2])

        # x = self.classifier(x_aecl)  # 256x3
        # x = F.softmax(x, dim=1)  # Shape remains 256x3
        x_aecl_reshaped = x_aecl_reshaped.reshape(int(x_aecl_reshaped.shape[0] / x_aecl.shape[1]),
                                                  x_aecl.shape[1] * x_aecl.shape[2])  # 64x12
        x_aecld = self.decisioner(x_aecl_reshaped)  # 64x3
        return x_aecld
