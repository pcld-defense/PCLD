import torch.nn as nn
import torch
from torchvision import models
import torch.optim as optim

torch.manual_seed(42)


def get_net(pretrained=True, weights='IMAGENET1K_V2'):
    if pretrained:
        model = models.resnet18(weights=weights)
    else:
        model = models.resnet18()
    return model


def get_net_and_optim(n_classes, device, lr, weights='IMAGENET1K_V2', pretrained=False):
    net = get_net( weights=weights, pretrained=pretrained)
    net.fc = nn.Linear(net.fc.in_features, n_classes)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    return net, criterion, optimizer, scheduler
