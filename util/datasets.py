import os

from PIL import Image
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms

from util.consts import RESOURCES_DATASETS_DIR


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


class IntegersScaler(object):
    def __call__(self, img):
        return img * 255


def transform_dataset(augmentations: bool, to_integers: bool = True):
    composition = []
    if augmentations:
        composition.extend([transforms.RandomRotation(45), transforms.RandomHorizontalFlip(p=0.5)])

    composition.extend([transforms.ToTensor()])

    if to_integers:
        composition.extend([IntegersScaler()])

    return transforms.Compose(composition)


def create_ds_loader(path: str, transform: transforms.Compose,
                     batch_size: int, shuffle: bool = True, num_workers: int = os.cpu_count() - 1):
    ds = ImageFolderWithPaths(path, transform=transform)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                         num_workers=num_workers, pin_memory=True)
    return ds, loader


def generator_loader_train_full(*itrs):
    for itr in itrs:
        for v in itr:
            yield v


# def get_loaders(dataset, train_transform, test_transform, batch_size):
#     ds_local_dir = os.path.join(RESOURCES_DATASETS_DIR, dataset)
#     ds_train_path = ds_local_dir #os.path.join(ds_local_dir, 'train')
#     ds_val_path = os.path.join(ds_local_dir, 'val')
#     ds_test_path = os.path.join(ds_local_dir, 'test')
#     ds_train, loader_train = create_ds_loader(path=ds_train_path, transform=train_transform, batch_size=batch_size,
#                                                num_workers=4)
#     ds_val, loader_val = create_ds_loader(path=ds_val_path, transform=test_transform, batch_size=batch_size,
#                                           num_workers=4)
#     ds_test, loader_test = create_ds_loader(path=ds_test_path, transform=test_transform, batch_size=batch_size,
#                                             num_workers=-1)
#     # we will use this validation set for concat to the training set
#     ds_val_to_concat, loader_val_to_concat = create_ds_loader(path=ds_val_path, transform=train_transform,
#                                                               batch_size=batch_size)
#     print(f'train batches {len(loader_train)} size {len(ds_train)}')
#     print(f'validation batches {len(loader_val)} size {len(ds_val)}')
#     print(f'test batches {len(loader_test)} size {len(ds_test)}')
#     loaders = {
#         'train': [ds_train, loader_train],
#         'val': [ds_val, loader_val],
#         'test': [ds_test, loader_test],
#         'val_to_concat': [ds_val_to_concat, loader_val_to_concat]
#     }
#     return loaders


def get_loaders(dataset, splits, transform, batch_size):
    """
    Args:
        dataset: Name of the dataset directory.
        splits: A tuple/list of strings like ('train', 'val', 'test').
        transform: Transform for training/augmentation.
        batch_size: Number of samples per batch.

    """
    ds_local_dir = os.path.join(RESOURCES_DATASETS_DIR, dataset)
    loaders = {}

    for split in splits:
        path = os.path.join(ds_local_dir, split)

        ds, loader = create_ds_loader(path=path, transform=transform, batch_size=batch_size,
                                  num_workers=os.cpu_count() - 1)

        loaders[split] = [ds, loader]
        print(f'{split} batches {len(loader)} size {len(ds)}')

    # Handle the specific "concat" logic if requested specifically or as an option
    if 'val_to_concat' in splits and 'val' in splits:
        ds_v_c, loader_v_c = create_ds_loader(
            path=os.path.join(ds_local_dir, 'val'),
            transform=transform,
            batch_size=batch_size
        )
        loaders['val_to_concat'] = [ds_v_c, loader_v_c]

    return loaders


def concat_to_one_decisioner_dataset(ds_local_dir: str) -> pd.DataFrame:
    df_dataset = pd.DataFrame()
    for file_name in os.listdir(ds_local_dir):
        file_path = os.path.join(ds_local_dir, file_name)
        df = pd.read_csv(file_path)
        df = df[df['attacked_model'] == 'pcl']  # only those records are relevant
        df_dataset = pd.concat([df_dataset, df], axis=0, ignore_index=True)
    return df_dataset
