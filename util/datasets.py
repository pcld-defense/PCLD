import os
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import datasets, transforms

from util.consts import RESOURCES_DATASETS_DIR, IMAGENETConsts, CIFAR10Consts


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


def transform_dataset(dataset_type: str = 'imagenet',
                      preprocessing: Optional[str] = None) -> transforms.Compose:
    """Builds a torchvision transform pipeline for the given dataset type.

    Returns the ready-made pipeline stored in the dataset's ``PREPROCESSINGS`` dict under the
    ``preprocessing`` key. If ``preprocessing`` is None, returns a simple ToTensor + Normalize pipeline.

    Args:
        dataset_type: Dataset family; ``'imagenet'`` or ``'cifar10'``. Selects
            the ``PREPROCESSINGS`` dict.
        preprocessing: Key into the dataset's ``PREPROCESSINGS`` dict (e.g.
            ``'Res256Crop224'``, ``'Res224'``, ``'BicubicRes256Crop224'``).
            ``None`` selects plain ToTensor + Normalize.

    Returns:
        A composed transform ready to pass to ``ImageFolderWithPaths``.

    Raises:
        ValueError: If ``preprocessing`` is not a recognised key for the
            chosen dataset type.
    """
    consts = IMAGENETConsts if dataset_type == 'imagenet' else CIFAR10Consts

    if preprocessing not in consts.PREPROCESSINGS:
        raise ValueError(
            f"Preprocessing {preprocessing!r} is not valid for "
            f"{dataset_type!r}. "
            f"Available options: {list(consts.PREPROCESSINGS.keys())}"
        )
    return consts.PREPROCESSINGS[preprocessing]


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
                                         num_workers=num_workers, pin_memory=False)
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


def load_decisioner_dataset(results_dir: str,
                            attacked_model: Union[str, None] = 'adaptive') -> pd.DataFrame:
    """Loads and concatenates all attacker Parquet files from a results directory.

    Reads every `*_results.parquet` file written by `attacker()`, concatenates
    them into a single DataFrame, and optionally filters to a specific
    attacked_model type. The resulting DataFrame contains both metadata columns
    and inline `prob_<classname>` probability columns, making it directly
    usable for decisioner training without any additional joins.

    Args:
        results_dir: Path to the directory containing `*_results.parquet` files
            produced by the attack_pcl or attack_pcld experiments.
        attacked_model: If provided, only rows where the 'attacked_model' column
            matches this value are returned. Pass 'adaptive' (default) to get
            the BPDA-attacked outputs used for decisioner training, 'naive' for
            the CLD baseline, or None to return all rows.

    Returns:
        Concatenated DataFrame with all metadata and probability columns.

    Raises:
        FileNotFoundError: If no `*_results.parquet` files are found in
            `results_dir`.
    """
    parquet_files = [
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.endswith('_results.parquet')
    ]

    if not parquet_files:
        raise FileNotFoundError(
            f'No *_results.parquet files found in {results_dir}. '
            f'Run the attack_pcl experiment first.'
        )

    frames = [pd.read_parquet(p) for p in sorted(parquet_files)]
    df = pd.concat(frames, axis=0, ignore_index=True)

    if attacked_model is not None:
        df = df[df['attacked_model'] == attacked_model].reset_index(drop=True)

    print(f'Loaded {len(df)} rows from {len(parquet_files)} files in {results_dir}')
    return df


def concat_to_one_decisioner_dataset(ds_local_dir: str) -> pd.DataFrame:
    """Loads the decisioner training dataset from a results directory.

    Delegates to `load_decisioner_dataset`, returning only the 'adaptive'
    attacked_model rows (the BPDA-attacked outputs). This function exists for
    backwards compatibility; prefer calling `load_decisioner_dataset` directly
    for new code.

    Args:
        ds_local_dir: Path to the directory containing `*_results.parquet` files
            produced by the attack_pcl experiment.

    Returns:
        Concatenated DataFrame of all adaptive PCL attack records.
    """
    return load_decisioner_dataset(ds_local_dir, attacked_model='adaptive')
