from typing import Union

from torchvision import models
import torch.optim as optim

from model.resnet import *
from util.models import load_model

torch.manual_seed(42)


def get_net(dataset_type: str, device: str, model_type: str = "resnet18", weights: Union[str, None] = None) -> nn.Module:

    model_map = {
        'imagenet': {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152
        },
        'cifar10': {
            'resnet50': ResNet50(),
        }

    }
    if dataset_type == 'imagenet':
        model = model_map['imagenet'].get(model_type)(pretrained=True)
    else:
        model = model_map['cifar10']['resnet50']

        if weights is not None:
            try:
                state_dict = torch.load(weights, map_location=device)['net']
                model.load_state_dict(state_dict)
            except Exception as e:
                model = load_model(model, weights, device)
    model.to(device)
    return model


def get_net_and_optim(dataset_type, device, lr, model_type="resnet18", weights=None):
    net = get_net(dataset_type, device, model_type=model_type, weights=weights)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    if dataset_type == 'imagenet':
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    else :
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    return net, criterion, optimizer, scheduler
