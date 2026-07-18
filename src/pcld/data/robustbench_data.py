"""RobustBench fixed-prefix evaluation data.

Loads the first ``n`` examples of a benchmark test set through
``robustbench.data`` so every machine evaluates on the exact same images in
the exact same order (the RobustBench convention: deterministic, unshuffled
test-set prefix).  A SHA-256 fingerprint of the loaded tensors can be written
next to the run results to prove two machines saw identical data.
"""

import hashlib
import json
import os
from typing import Optional

import torch

from pcld.utils.consts import RESOURCES_DATASETS_DIR, CIFAR10Consts

# Canonical CIFAR-100 fine-label names (alphabetical, index order used by the
# dataset and by robustbench's loaders).
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee',
    'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus',
    'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair',
    'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab',
    'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish',
    'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man',
    'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree',
    'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck',
    'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit',
    'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark',
    'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel',
    'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone',
    'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle',
    'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm',
]


class RBPrefixDataset(torch.utils.data.Dataset):
    """In-memory dataset over a fixed RobustBench test-set prefix.

    Holds the image and label tensors returned by ``robustbench.data`` and
    serves them in their original order.  Each item is a
    ``(image, label, pseudo_path)`` triple: the third element mimics an image
    file path so the dataset satisfies the same loader contract as
    ``ImageFolderWithPaths`` (``attacker()`` unpacks ``x, y, img_paths``).
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor,
                 classes: list) -> None:
        """Initialises the dataset from pre-loaded tensors.

        Args:
            x: Image tensor of shape (N, 3, H, W) with values in [0, 1].
            y: Label tensor of shape (N,).
            classes: Class names, indexed by label value.
        """
        self.x = x
        self.y = y
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self) -> int:
        """Returns the number of examples in the prefix."""
        return self.x.shape[0]

    def __getitem__(self, i: int) -> tuple:
        """Returns the i-th example.

        Args:
            i: Sample index.

        Returns:
            Tuple of (image_tensor, label_int, pseudo_path_str) where the
            pseudo path is ``'rb_<index:05d>.png'``.
        """
        return self.x[i], int(self.y[i]), f'rb_{i:05d}.png'


def get_rb_prefix_loader(dataset_type: str, n_examples: int, batch_size: int,
                         data_dir: Optional[str] = None) -> tuple:
    """Loads the first ``n_examples`` of a benchmark test set via RobustBench.

    Uses ``robustbench.data`` so the prefix is byte-identical on every
    machine (same download, same deterministic order, no shuffling).

    Args:
        dataset_type: One of ``'cifar10'``, ``'cifar100'``, ``'imagenet'``.
        n_examples: Number of leading test-set examples to load.
        batch_size: Mini-batch size for the returned DataLoader.
        data_dir: Directory RobustBench caches the raw dataset in. Defaults
            to ``RESOURCES_DATASETS_DIR/robustbench`` (created if missing).

    Returns:
        Tuple of (RBPrefixDataset, DataLoader). The loader is unshuffled with
        ``num_workers=0`` so iteration order is deterministic.

    Raises:
        ImportError: If robustbench is not installed.
        ValueError: If ``dataset_type`` is not supported.
    """
    try:
        from robustbench.data import (load_cifar10, load_cifar100,
                                      load_imagenet)
    except ImportError as e:
        raise ImportError(
            'robustbench is required for data_source=robustbench. '
            'Install it with: pip install '
            'git+https://github.com/RobustBench/robustbench.git'
        ) from e

    if data_dir is None:
        data_dir = os.path.join(RESOURCES_DATASETS_DIR, 'robustbench')
    os.makedirs(data_dir, exist_ok=True)

    if dataset_type == 'cifar10':
        x, y = load_cifar10(n_examples=n_examples, data_dir=data_dir)
        mapping = CIFAR10Consts.CIFAR10_MAPPING
        classes = [mapping[i] for i in sorted(mapping)]
    elif dataset_type == 'cifar100':
        x, y = load_cifar100(n_examples=n_examples, data_dir=data_dir)
        classes = list(CIFAR100_CLASSES)
    elif dataset_type == 'imagenet':
        x, y = load_imagenet(n_examples=n_examples, data_dir=data_dir)
        classes = [str(i) for i in range(1000)]
    else:
        raise ValueError(
            f'Unsupported dataset_type {dataset_type!r} for the RobustBench '
            f"prefix loader. Expected one of: 'cifar10', 'cifar100', "
            f"'imagenet'.")

    ds = RBPrefixDataset(x, y, classes)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                         shuffle=False, num_workers=0,
                                         pin_memory=False)
    return ds, loader


def prefix_fingerprint(x: torch.Tensor, y: torch.Tensor) -> dict:
    """Computes a SHA-256 fingerprint of a loaded data prefix.

    Two machines that loaded the same images in the same order produce
    identical fingerprints, so comparing the JSON proves data equivalence.

    Args:
        x: Image tensor of shape (N, 3, H, W).
        y: Label tensor of shape (N,).

    Returns:
        Dict with keys ``n`` (int), ``x_sha256``, ``y_sha256`` (hex digests)
        and ``shape`` (list of ints, the shape of ``x``).
    """
    return {
        'n': int(x.shape[0]),
        'x_sha256': hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest(),
        'y_sha256': hashlib.sha256(y.cpu().numpy().tobytes()).hexdigest(),
        'shape': list(x.shape),
    }


def write_fingerprint(fp: dict, out_dir: str) -> str:
    """Writes a prefix fingerprint to ``<out_dir>/rb_prefix_fingerprint.json``.

    Args:
        fp: Fingerprint dict as returned by ``prefix_fingerprint``.
        out_dir: Directory to write the JSON file into (created if missing).

    Returns:
        Absolute path of the written JSON file.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'rb_prefix_fingerprint.json')
    with open(path, 'w') as f:
        json.dump(fp, f, indent=2)
    return path
