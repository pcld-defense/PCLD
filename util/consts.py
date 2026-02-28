import os
from pathlib import Path

from dotenv import load_dotenv
from imagenet_stubs.imagenet_2012_labels import IMAGENET_2012_LABELS

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


class PainterConsts:
    MAX_STEP = 80
    WIDTH = 128
    DIVIDE = 5
