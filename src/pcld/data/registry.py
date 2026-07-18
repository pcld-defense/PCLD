"""Dataset registry with download-on-first-use.

Each dataset registers a *builder*: a callable ``builder(root, split) -> None``
that ensures the split's image folder exists under ``root``, downloading and
caching it if necessary.  ``ensure_dataset`` is called by ``get_loaders`` before
a split is read, so loading a registered dataset is one config line and the
data appears automatically on first use.

Datasets that cannot be auto-downloaded (ImageNet, custom subsets) register a
builder that verifies the folder exists and otherwise raises with setup
instructions.
"""

import os
from typing import Callable

from pcld.utils.consts import RESOURCES_DATASETS_DIR
from pcld.utils.registry import Registry

# builder signature: (root: str, split: str) -> None
DATASETS: Registry = Registry('dataset')


def ensure_dataset(name: str, split: str, root: str = None) -> str:
    """Ensures a dataset split is present on disk, downloading if registered.

    Args:
        name: Registered dataset name (e.g. ``'cifar10'``). If the name is not
            registered, no download is attempted and the on-disk path is
            returned as-is (backwards compatible with folder-only datasets).
        split: Split subfolder to ensure (e.g. ``'train'``, ``'test'``).
        root: Datasets root; defaults to ``RESOURCES_DATASETS_DIR``.

    Returns:
        Absolute path to the dataset's split directory.
    """
    root = root or RESOURCES_DATASETS_DIR
    split_dir = os.path.join(root, name, split)

    if name in DATASETS and not os.path.isdir(split_dir):
        print(f'[data] {name}/{split} not found — running its builder...')
        DATASETS.get(name)(root, split)

    return split_dir


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

@DATASETS.register('cifar10')
def _build_cifar10(root: str, split: str) -> None:
    """Downloads CIFAR-10 from HuggingFace into ImageFolder layout.

    Saves one PNG per image under ``root/cifar10/<split>/<class_name>/``. The
    HuggingFace ``test`` split is mapped from its ``test`` set; ``train`` and
    ``val`` both derive from the HF ``train`` set (val is not a separate HF
    split, so callers that need a held-out val should carve it from train).

    Args:
        root: Datasets root directory.
        split: ``'train'``, ``'val'``, or ``'test'``.
    """
    from datasets import load_dataset
    from tqdm import tqdm

    hf_split = 'test' if split == 'test' else 'train'
    ds = load_dataset('uoft-cs/cifar10', split=hf_split, trust_remote_code=True)
    class_names = ds.features['label'].names
    out_dir = os.path.join(root, 'cifar10', split)

    for i, example in enumerate(tqdm(ds, desc=f'cifar10/{split}')):
        class_name = class_names[example['label']]
        class_path = os.path.join(out_dir, class_name)
        os.makedirs(class_path, exist_ok=True)
        example['img'].save(os.path.join(class_path, f'{i}.png'))
    print(f'[data] cifar10/{split} saved to {out_dir}')


@DATASETS.register('cifar100')
def _build_cifar100(root: str, split: str) -> None:
    """Downloads CIFAR-100 from HuggingFace into ImageFolder layout.

    Saves one PNG per image under ``root/cifar100/<split>/<class_name>/``
    using the fine labels (100 classes). The HuggingFace ``test`` split is
    mapped from its ``test`` set; ``train`` and ``val`` both derive from the
    HF ``train`` set (val is not a separate HF split, so callers that need a
    held-out val should carve it from train).

    Args:
        root: Datasets root directory.
        split: ``'train'``, ``'val'``, or ``'test'``.
    """
    from datasets import load_dataset
    from tqdm import tqdm

    hf_split = 'test' if split == 'test' else 'train'
    ds = load_dataset('uoft-cs/cifar100', split=hf_split, trust_remote_code=True)
    class_names = ds.features['fine_label'].names
    out_dir = os.path.join(root, 'cifar100', split)

    for i, example in enumerate(tqdm(ds, desc=f'cifar100/{split}')):
        class_name = class_names[example['fine_label']]
        class_path = os.path.join(out_dir, class_name)
        os.makedirs(class_path, exist_ok=True)
        example['img'].save(os.path.join(class_path, f'{i}.png'))
    print(f'[data] cifar100/{split} saved to {out_dir}')


def _require_folder(name: str) -> Callable[[str, str], None]:
    """Builds a no-download builder that just checks the folder exists."""
    def _builder(root: str, split: str) -> None:
        split_dir = os.path.join(root, name, split)
        raise FileNotFoundError(
            f'Dataset {name!r} split {split!r} not found at {split_dir}. '
            f'{name} is not auto-downloadable; place it there manually '
            f'(see scripts/data/download_imagenet.py for ImageNet setup).')
    return _builder


DATASETS.register('imagenet')(_require_folder('imagenet'))
DATASETS.register('subset_of_imagenet')(_require_folder('subset_of_imagenet'))
