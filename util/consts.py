import os
from pathlib import Path

from dotenv import load_dotenv
from imagenet_stubs.imagenet_2012_labels import IMAGENET_2012_LABELS
from torchvision import transforms

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
NUM_OF_HYPHENS = 50
RESOURCES_DIR = os.getenv("RESOURCES_DIR")
RESOURCES_DATASETS_DIR = os.getenv("RESOURCES_DATASETS_DIR")
RESOURCES_RESULTS_DIR = os.getenv("RESOURCES_RESULTS_DIR")
RESOURCES_MODELS_DIR = os.getenv("RESOURCES_MODELS_DIR")
ACTOR_WEIGHTS_PATH = os.getenv("ACTOR_WEIGHTS_PATH")
RENDERER_WEIGHTS_PATH = os.getenv("RENDERER_WEIGHTS_PATH")


class IMAGENETConsts:
    SHAPE = 224
    NUM_CLASSES = 1000
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    IMAGENET_MAPPING = IMAGENET_2012_LABELS
    PREPROCESSINGS = {
        'Res256Crop224': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ]),
        'Crop288': transforms.Compose([transforms.CenterCrop(288),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=MEAN, std=STD)]),
        None: transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(mean=MEAN, std=STD)]),
        'Res224': transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ]),
        'BicubicRes256Crop224': transforms.Compose([
            transforms.Resize(
                256,
                interpolation=transforms.InterpolationMode("bicubic")),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
    }


class CIFAR10Consts:
    CIFAR10_MAPPING = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }
    SHAPE = 32
    NUM_CLASSES = 10
    MEAN = (0.4914, 0.4822, 0.4465)
    STD = (0.2471, 0.2435, 0.2616)
    PREPROCESSINGS = {
        None: transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(mean=MEAN, std=STD)]),
        'augmented': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
            transforms.RandomErasing(p=0.5),
        ]),
    }


class PainterConsts:
    MAX_STEP = 80
    WIDTH = 128
    DIVIDE = 5
