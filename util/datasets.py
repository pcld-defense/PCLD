import os
from typing import Union

import torchvision
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms

from util.consts import RESOURCES_DATASETS_DIR, IMAGENETConsts


def load_image(path: str) -> np.ndarray:
    """
    Loads, resizes, and normalizes an image from a given path.

    The function opens an image, scales it to a fixed 300x300 resolution,
    and converts the pixel values from an integer range [0, 255] to a
    floating-point range [0.0, 1.0].

    Args:
        path: The file path to the image to be loaded.

    Returns:
        A 3D float32 array representing the processed image with shape (300, 300, channels).

    """
    img = Image.open(path)
    img = img.resize((300, 300))
    img = np.asarray(img)
    img = img / 255.0
    img = img.astype(np.float32)

    return img


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index: int):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def transform_dataset(augmentations: bool, dataset_type: str):
    composition = []
    if augmentations:
        if dataset_type == "imagenet":
            composition.extend([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(45)
            ])
        elif dataset_type == "cifar10":
            composition.extend([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ])

    composition.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENETConsts.MEAN, std=IMAGENETConsts.STD)
    ])

    return transforms.Compose(composition)


def create_ds_loader(path: str, transform: transforms.Compose,
                     batch_size: int, shuffle: bool = True, num_workers: int = os.cpu_count() - 1):
    ds = ImageFolderWithPaths(path, transform=transform)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                         num_workers=num_workers, pin_memory=True)
    return ds, loader


def get_loaders(dataset: str, splits: Union[list, str], transform_dict: dict[str, transforms.Compose], batch_size: int):
    """
    Args:
        dataset: Name of the dataset directory.
        splits: A tuple/list of strings like ('train', 'val', 'test').
        transform_dict: Transform for training/augmentation.
        batch_size: Number of samples per batch.

    """
    ds_local_dir = os.path.join(RESOURCES_DATASETS_DIR, dataset)
    loaders = {}

    for split in splits:
        path = os.path.join(ds_local_dir, split)

        ds, loader = create_ds_loader(path=path, transform=transform_dict[split], batch_size=batch_size,
                                      num_workers=4)

        loaders[split] = [ds, loader]
        print(f'{split} batches {len(loader)} size {len(ds)}')

    return loaders


def concat_to_one_decisioner_dataset(ds_local_dir: str) -> pd.DataFrame:
    df_dataset = pd.DataFrame()
    for file_name in os.listdir(ds_local_dir):
        file_path = os.path.join(ds_local_dir, file_name)
        df = pd.read_csv(file_path)
        df = df[df['attacked_model'] == 'pcl']  # only those records are relevant
        df_dataset = pd.concat([df_dataset, df], axis=0, ignore_index=True)
    return df_dataset
