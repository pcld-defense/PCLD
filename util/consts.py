import os

from imagenet_stubs.imagenet_2012_labels import IMAGENET_2012_LABELS

NUM_OF_HYPHENS = 50
RESOURCES_DIR = r"/home/idanbib/PCLD/code/resources/"
RESOURCES_DATASETS_DIR = r"/home/idanbib/PCLD/data/"
RESOURCES_RESULTS_DIR = r"/home/idanbib/PCLD/results"
RESOURCES_MODELS_DIR = r"/home/idanbib/PCLD/models"
IMAGENET_SHAPE = 224

ACTOR_PATH = r'/home/idanbib/PCLD/code/resources/models/painter_actor/actor.pkl'
RENDERER_PATH = r'/home/idanbib/PCLD/code/resources/models/painter_renderer/renderer.pkl'


class CIFAR10:
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
    CIFAR_SHAPE = 32


class PainterConsts:
    MAX_STEP = 80
    WIDTH = 128
    DIVIDE = 5
