import torch.nn as nn
import torch
from torchvision import datasets, models, transforms
import torch.optim as optim
import os

torch.manual_seed(42)


def get_net(net_name, pretrained=True):
    if net_name == 'res_net_18_v1':
        if pretrained:
            model = models.resnet18(weights='IMAGENET1K_V1')
        else:
            model = models.resnet18()
        return model


def get_net_and_optim(n_classes, device, lr):
    net = models.resnet18(weights='IMAGENET1K_V1')
    net.fc = nn.Linear(net.fc.in_features, n_classes)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    return net, criterion, optimizer, scheduler
