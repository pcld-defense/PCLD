import torch.nn as nn
import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F

# class Decisioner(nn.Module):
#     def __init__(self):
#         super(Decisioner, self).__init__()
#         self.fc1 = nn.Linear(12 * 3, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.dropout = nn.Dropout(0.5)
#         self.fc3 = nn.Linear(32, 3)
#
#     def forward(self, x):
#         # Flatten input from [bsx12x3] to [bsx36]
#         # x = x.view(-1, 12 * 3)
#         x = torch.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = torch.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return torch.softmax(x, dim=1)

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
        # ae_num = 0
        # ae_output = []
        # print(f'len self.ae_painters: {len(self.ae_painters)}')
        # for ae in self.ae_painters:
        #     ae_output.append(ae(x))
        #     ae_num += 1
        #     print(f'AE {ae_num} loaded')
        # ae_output.append(x)  # original image (∞) as well
        # x = torch.stack(ae_output, dim=0)  # 4x64x3x300x300
        # x = torch.permute(x, (1, 0, 2, 3, 4))  # 64x4x3x300x300
        # x = x.reshape(x.shape[0]*x.shape[1], 3, 300, 300)  # 256x3x300x300

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
        x_aecl_reshaped = x_aecl.reshape(x_aecl.shape[0]*x_aecl.shape[1], x_aecl.shape[2])

        # x = self.classifier(x_aecl)  # 256x3
        # x = F.softmax(x, dim=1)  # Shape remains 256x3
        x_aecl_reshaped = x_aecl_reshaped.reshape(int(x_aecl_reshaped.shape[0]/x_aecl.shape[1]), x_aecl.shape[1]*x_aecl.shape[2])  # 64x12
        x_aecld = self.decisioner(x_aecl_reshaped)  # 64x3
        return x_aecld




# from denoising_autoencoder import DenoisingAutoencoder
# from alex_net import AlexNet
#
# ae = DenoisingAutoencoder()
# AEs = nn.ModuleList([DenoisingAutoencoder() for i in range(11)])
# clf = AlexNet(num_classes=3)
# decisioner = Decisioner()
#
# pcl = PCL(ae, clf)
# pcld = PCLD(AEs, clf, decisioner)
#
# input = torch.randn(128, 3, 300, 300)
# output_pcl = pcl(input)
# output_pcld = pcld(input)



# x = torch.randn(1408, 3)
# decisioner = Decisioner()
# y = decisioner(x)
# print(y)


# import torch
# x1 = torch.randn(size=(128, 3, 300, 300), requires_grad=True)
# x2 = torch.randn(size=(128, 3, 300, 300), requires_grad=True)
# x3 = torch.randn(size=(128, 3, 300, 300), requires_grad=True)
# Create tensors with requires_grad=True to enable gradient tracking
# x1 = torch.tensor([1.0, 2.0], requires_grad=True)
# x2 = torch.tensor([4.0, 5.0], requires_grad=True)
# x3 = torch.tensor([7.0, 8.0], requires_grad=True)

# Stack tensors along a new dimension
# stacked_tensor = torch.stack([x1, x2, x3], dim=0)

# Perform some operation on the stacked tensor
# output = stacked_tensor.mean()

# Compute gradients
# output.backward()

# Check gradients of the input tensors
# print(x1.grad)  # Gradient of x1
# print(x2.grad)  # Gradient of x2
# print(x3.grad)  # Gradient of x3
# print(stacked_tensor.shape)
