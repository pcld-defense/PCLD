import os
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import datasets, transforms

from pcld.utils.consts import (RESOURCES_DATASETS_DIR, IMAGENETConsts,
                               CIFAR10Consts, CIFAR100Consts)

# Dataset-type string -> consts class holding MEAN/STD/PREPROCESSINGS.
_DATASET_CONSTS = {
    'imagenet': IMAGENETConsts,
    'cifar10': CIFAR10Consts,
    'cifar100': CIFAR100Consts,
}


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
        dataset_type: Dataset family; ``'imagenet'``, ``'cifar10'`` or
            ``'cifar100'``. Selects the ``PREPROCESSINGS`` dict.
        preprocessing: Key into the dataset's ``PREPROCESSINGS`` dict (e.g.
            ``'Res256Crop224'``, ``'Res224'``, ``'BicubicRes256Crop224'``).
            ``None`` selects plain ToTensor + Normalize.

    Returns:
        A composed transform ready to pass to ``ImageFolderWithPaths``.

    Raises:
        ValueError: If ``dataset_type`` is unknown, or if ``preprocessing``
            is not a recognised key for the chosen dataset type.
    """
    if dataset_type not in _DATASET_CONSTS:
        raise ValueError(
            f'Unknown dataset_type {dataset_type!r}. '
            f'Available options: {list(_DATASET_CONSTS.keys())}')
    consts = _DATASET_CONSTS[dataset_type]

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
    # When folder names are plain integers (e.g. painted datasets that store
    # classes as "0", "1", ..., "999"), ImageFolder's alphabetical sort
    # produces a wrong class_to_idx ("100" → 3 instead of 100).  Remap so
    # that folder name == label index, matching what the model expects.
    if all(c.isdigit() for c in ds.classes):
        ds.class_to_idx = {c: int(c) for c in ds.classes}
        ds.samples = [(p, int(ds.classes[t])) for p, t in ds.samples]
        ds.imgs = ds.samples
        ds.targets = [t for _, t in ds.samples]
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
    from pcld.data.registry import ensure_dataset

    loaders = {}

    for split in splits:
        # No-op when the folder already exists; downloads on first use for
        # registered auto-downloadable datasets (e.g. cifar10).
        path = ensure_dataset(dataset, split, root=RESOURCES_DATASETS_DIR)
        ds, loader = create_ds_loader(path=path, transform=transform_dict[split],
                                      batch_size=batch_size, num_workers=0)
        loaders[split] = [ds, loader]
        print(f'{split} batches {len(loader)} size {len(ds)}')

    return loaders


def build_eval_loaders(args: object, batch_size: int,
                       run_dir: Optional[str] = None) -> dict:
    """Builds evaluation loaders from either the folder pipeline or RobustBench.

    Central data entry point for evaluation experiments. Two data sources:

    * ``args.data_source == 'robustbench'``: loads a fixed test-set prefix of
      ``args.num_samples`` (default 1000) examples via ``robustbench.data``,
      guaranteeing the exact same images in the same order on every machine.
      Only the ``'test'`` split is produced. If ``run_dir`` is given, a
      SHA-256 fingerprint of the loaded tensors is written there as
      ``rb_prefix_fingerprint.json`` for cross-machine verification.
    * ``args.data_source == 'folder'`` (default): delegates to ``get_loaders``
      unchanged. If ``args.num_samples`` is set, each split's dataset is
      wrapped in a ``torch.utils.data.Subset`` over its first
      ``num_samples`` indices and the loader is rebuilt unshuffled (with the
      same batch size / worker settings ``get_loaders`` uses); when
      ``num_samples`` is None the ``get_loaders`` result is returned as-is.

    Both sources apply the RobustBench convention: images in [0, 1] with no
    normalization (the classifier normalizes internally via
    ``NormalizedModel``), matching the ``'ToTensorOnly'`` preprocessing.

    Args:
        args: Namespace read for ``dataset``, ``dataset_type``, ``splits``,
            and optionally ``data_source`` and ``num_samples``.
        batch_size: Number of samples per mini-batch.
        run_dir: If given (robustbench source only), directory to write the
            data fingerprint JSON into.

    Returns:
        Dictionary mapping each split name to [dataset, dataloader], the same
        structure ``get_loaders`` returns.
    """
    if getattr(args, 'data_source', 'folder') == 'robustbench':
        from pcld.data.robustbench_data import (get_rb_prefix_loader,
                                                prefix_fingerprint,
                                                write_fingerprint)

        n = getattr(args, 'num_samples', None) or 1000
        ds, loader = get_rb_prefix_loader(args.dataset_type, n, batch_size)
        if run_dir is not None:
            fp_path = write_fingerprint(prefix_fingerprint(ds.x, ds.y), run_dir)
            print(f'[data] robustbench prefix fingerprint -> {fp_path}')
        print(f'test batches {len(loader)} size {len(ds)}')
        return {'test': [ds, loader]}

    split_transform = transform_dataset(dataset_type=args.dataset_type,
                                        preprocessing='ToTensorOnly')
    transform_dict = {split: split_transform for split in args.splits}
    loaders = get_loaders(args.dataset, args.splits, transform_dict, batch_size)

    n = getattr(args, 'num_samples', None)
    if n:
        for split, (ds, _) in loaders.items():
            sub = torch.utils.data.Subset(ds, range(min(n, len(ds))))
            # Preserve the class metadata the experiments read off the dataset.
            sub.classes = ds.classes
            sub.class_to_idx = ds.class_to_idx
            # Same loader params as create_ds_loader/get_loaders (num_workers=0,
            # pin_memory=False) but unshuffled: eval order is deterministic.
            sub_loader = torch.utils.data.DataLoader(
                sub, batch_size=batch_size, shuffle=False, num_workers=0,
                pin_memory=False)
            loaders[split] = [sub, sub_loader]
            print(f'{split} capped to {len(sub)} samples '
                  f'({len(sub_loader)} batches)')

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
