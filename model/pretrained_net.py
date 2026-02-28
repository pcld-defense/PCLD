from typing import Union

from torchvision import models
import torch.optim as optim

from model.resnet import *
from util.models import load_model

torch.manual_seed(42)


def get_net(dataset_type: str, device: str, model_type: str = "resnet18",
            weights: Union[str, None] = None) -> nn.Module:
    """Builds and returns a classifier network for the specified dataset type.

    For ImageNet the network is loaded with pretrained ImageNet weights. For
    CIFAR-10 a custom ResNet50 is created; if a weights path is supplied, the
    checkpoint is loaded either from a standard state-dict or from the 'net'
    key produced by some training frameworks.

    Args:
        dataset_type: Dataset family; either 'imagenet' or 'cifar10'.
        device: Target device string (e.g. 'cuda' or 'cpu').
        model_type: Model architecture name for ImageNet; one of 'resnet18',
            'resnet34', 'resnet50', 'resnet101', 'resnet152'.
        weights: Optional path to a checkpoint file. Only used for CIFAR-10.

    Returns:
        The classifier network moved to `device`.
    """
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


def get_net_and_optim(dataset_type: str, device: str, lr: float,
                      model_type: str = "resnet18",
                      weights: Union[str, None] = None) -> tuple:
    """Builds the network together with its loss function, optimiser, and scheduler.

    Uses SGD with momentum and a StepLR learning-rate scheduler. Hyperparameters
    are chosen per dataset: ImageNet uses weight-decay 1e-4 and step-size 15;
    CIFAR-10 uses weight-decay 5e-4 and step-size 7.

    Args:
        dataset_type: Dataset family; either 'imagenet' or 'cifar10'.
        device: Target device string.
        lr: Initial learning rate for the SGD optimiser.
        model_type: Architecture name forwarded to `get_net`.
        weights: Optional path to pretrained weights forwarded to `get_net`.

    Returns:
        A tuple (net, criterion, optimizer, scheduler) where:
            net: The initialised classifier.
            criterion: CrossEntropyLoss with label smoothing 0.1.
            optimizer: SGD optimiser.
            scheduler: StepLR learning-rate scheduler.
    """
    net = get_net(dataset_type, device, model_type=model_type, weights=weights)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    if dataset_type == 'imagenet':
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    else:
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    return net, criterion, optimizer, scheduler
