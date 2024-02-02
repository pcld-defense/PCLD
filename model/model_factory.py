import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import models
torch.manual_seed(42)


torch.manual_seed(42)


def get_net(device: str, n_classes: int):
    net = models.resnet18(weights='IMAGENET1K_V1')
    net.fc = nn.Linear(net.fc.in_features, n_classes)
    net = net.to(device)

    return net


def get_net_trainers(net: models.resnet.ResNet, lr: float):
    optimizer = optim.SGD(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    return optimizer, scheduler
