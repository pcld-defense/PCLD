import torch.nn as nn
import torch
from torchvision import datasets, models, transforms
import os

torch.manual_seed(42)


def get_net(net_name, pretrained=True):
    if net_name == 'res_net_18_v1':
        if pretrained:
            model = models.resnet18(weights='IMAGENET1K_V1')
        else:
            model = models.resnet18()
        return model
