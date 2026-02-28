import os
from typing import Union

from PIL import Image
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms

from util.consts import RESOURCES_DATASETS_DIR, IMAGENETConsts


def load_image(path: str, width: int = 300, height: int = 300) -> np.ndarray:
    """Loads, resizes, and normalises an image from disk.

    Opens the image at `path`, resizes it to the specified dimensions, and
    converts pixel values from uint8 [0, 255] to float32 [0.0, 1.0].

    Args:
        path: Filesystem path to the image file.
        width: Target width in pixels after resizing.
        height: Target height in pixels after resizing.

    Returns:
        Float32 array of shape (height, width, channels) with values in [0, 1].
    """
    img = Image.open(path)
    img = img.resize((width, height))
    img = np.asarray(img)
    img = img / 255.0
    img = img.astype(np.float32)
    return img


class ImageFolderWithPaths(datasets.ImageFolder):
    """ImageFolder variant that also returns the file path for each sample.

    Extends the standard torchvision ImageFolder so that each item returned
    by __getitem__ is a tuple (image_tensor, label, path_string) instead of
    the usual (image_tensor, label). This is used throughout the codebase to
    track which files were attacked or painted.
    """

    def __getitem__(self, index: int) -> tuple:
        """Returns the image, label, and file path for the given index.

        Args:
            index: Sample index within the dataset.

        Returns:
            Tuple of (image_tensor, label_int, file_path_str).
        """
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def transform_dataset(augmentations: bool, dataset_type: str = 'imagenet') -> transforms.Compose:
    """Builds a torchvision transform pipeline for the given dataset type.

    For training splits, random crop, flip, and (for ImageNet) rotation
    augmentations are prepended. All splits end with ToTensor and ImageNet
    normalisation (mean/std from IMAGENETConsts).

    Args:
        augmentations: If True, include data-augmentation transforms before
            normalisation.
        dataset_type: Dataset family; 'imagenet' or 'cifar10'. Determines
            which augmentation operations are applied.

    Returns:
        A composed transform ready to pass to ImageFolderWithPaths.
    """
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
                     batch_size: int, shuffle: bool = True,
                     num_workers: int = os.cpu_count() - 1) -> tuple:
    """Creates an ImageFolderWithPaths dataset and a DataLoader for it.

    Args:
        path: Root directory of the dataset split (expects class subdirectories).
        transform: Composed transform applied to each image.
        batch_size: Number of samples per mini-batch.
        shuffle: Whether to shuffle the dataset each epoch.
        num_workers: Number of worker processes for data loading.

    Returns:
        Tuple of (dataset, dataloader) where dataset is an
        ImageFolderWithPaths and dataloader is a torch DataLoader.
    """
    ds = ImageFolderWithPaths(path, transform=transform)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                         num_workers=num_workers, pin_memory=True)
    return ds, loader


def get_loaders(dataset: str, splits: Union[list, str],
                transform_dict: dict, batch_size: int) -> dict:
    """Builds a DataLoader for each requested dataset split.

    Resolves the dataset directory from RESOURCES_DATASETS_DIR and creates one
    loader per split using the transform specified in `transform_dict`.

    Args:
        dataset: Name of the dataset directory inside RESOURCES_DATASETS_DIR.
        splits: List of split names to load (e.g. ['train', 'val', 'test']).
        transform_dict: Mapping from split name to its torchvision transform.
        batch_size: Number of samples per mini-batch.

    Returns:
        Dictionary mapping each split name to [dataset, dataloader].
    """
    ds_local_dir = os.path.join(RESOURCES_DATASETS_DIR, dataset)
    loaders = {}

    for split in splits:
        path = os.path.join(ds_local_dir, split)
        ds, loader = create_ds_loader(path=path, transform=transform_dict[split],
                                      batch_size=batch_size, num_workers=4)
        loaders[split] = [ds, loader]
        print(f'{split} batches {len(loader)} size {len(ds)}')

    return loaders


def concat_to_one_decisioner_dataset(ds_local_dir: str) -> pd.DataFrame:
    """Concatenates all PCL attack CSV files in a directory into one DataFrame.

    Reads every CSV in `ds_local_dir`, filters rows where attacked_model is
    'pcl' (the only rows relevant for decisioner training), and returns the
    combined result.

    Args:
        ds_local_dir: Path to the directory containing per-epsilon attack CSV
            files produced by the attack_pcl experiment.

    Returns:
        Concatenated DataFrame of all PCL attack records.
    """
    df_dataset = pd.DataFrame()
    for file_name in os.listdir(ds_local_dir):
        file_path = os.path.join(ds_local_dir, file_name)
        df = pd.read_csv(file_path)
        df = df[df['attacked_model'] == 'pcl']
        df_dataset = pd.concat([df_dataset, df], axis=0, ignore_index=True)
    return df_dataset
