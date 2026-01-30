import os
import shutil
import re

from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
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

    def get_items_by_match_names(self, names):
        items = []
        found_mask = []
        for filename in names:
            item_found = False
            for idx, p in enumerate(self.imgs):
                p_list = p[0].split('/')[-1].split('_')
                p_name = p_list[0] + '_' + p_list[1]
                if filename == p_name:
                    item = self.__getitem__(idx)
                    items.append(item)
                    item_found = True
                    break
            found_mask.append(item_found)
        return items, found_mask


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


def get_loaders(dataset, train_transform, test_transform, batch_size):
    ds_local_dir = os.path.join(RESOURCES_DATASETS_DIR, dataset)
    ds_train_path = os.path.join(ds_local_dir, 'train')
    ds_val_path = os.path.join(ds_local_dir, 'val')
    ds_test_path = os.path.join(ds_local_dir, 'test')
    ds_train, loader_train = create_ds_loader(path=ds_train_path, transform=train_transform, batch_size=batch_size,
                                              num_workers=1)
    ds_val, loader_val = create_ds_loader(path=ds_val_path, transform=test_transform, batch_size=batch_size,
                                          num_workers=1)
    ds_test, loader_test = create_ds_loader(path=ds_test_path, transform=test_transform, batch_size=batch_size,
                                            num_workers=1)
    # we will use this validation set for concat to the training set
    ds_val_to_concat, loader_val_to_concat = create_ds_loader(path=ds_val_path, transform=train_transform,
                                                              batch_size=batch_size)
    print(f'train batches {len(loader_train)} size {len(ds_train)}')
    print(f'validation batches {len(loader_val)} size {len(ds_val)}')
    print(f'test batches {len(loader_test)} size {len(ds_test)}')
    loaders = {
        'train': [ds_train, loader_train],
        'val': [ds_val, loader_val],
        'test': [ds_test, loader_test],
        'val_to_concat': [ds_val_to_concat, loader_val_to_concat]
    }
    return loaders


def concat_to_one_decisioner_dataset(ds_local_dir: str) -> pd.DataFrame:
    df_dataset = pd.DataFrame()
    for file_name in os.listdir(ds_local_dir):
        file_path = os.path.join(ds_local_dir, file_name)
        df = pd.read_csv(file_path)
        df = df[df['attacked_model'] == 'pcl']  # only those records are relevant
        df_dataset = pd.concat([df_dataset, df], axis=0, ignore_index=True)
    return df_dataset


def prepare_imagent_val(sh_path: str, output_path: str) -> None:
    """
    Parses valprep.sh (imagenet script to move images into the appropriate folders)
     and executes file movements with a progress bar.

    Args:
        sh_path: Path to the original valprep.sh file.
        output_path: Path to the directory output.
    """
    if not os.path.exists(sh_path):
        print(f"Error: {sh_path} not found.")
        return

    # Read the script into memory
    with open(sh_path, 'r') as f:
        lines = f.readlines()

    # Regex patterns to extract folder creation and move commands
    mkdir_re = re.compile(r'mkdir -p\s+([\w\d]+)')
    mv_re = re.compile(r'mv\s+([\w\d\.]+)\s+([\w\d]+)/')

    # 1. Handle Directory Creation
    directories = [mkdir_re.search(line).group(1) for line in lines if mkdir_re.search(line)]
    unique_dirs = sorted(list(set(directories)))

    print(f"Checking/Creating {len(unique_dirs)} directories...")
    for d in tqdm(unique_dirs, desc="Creating Folders", unit="folder"):
        os.makedirs(os.path.join(output_path, d), exist_ok=True)

    # 2. Handle File Movements
    # Pre-parse the move commands to know how many there are
    move_commands = []
    for line in lines:
        match = mv_re.search(line)
        if match:
            move_commands.append(match.groups())

    print(f"Moving {len(move_commands)} files...")
    moved_count = 0

    # tqdm progress bar for the file moves
    for src_name, dest_folder in tqdm(move_commands, desc="Organizing Images", unit="file"):
        current_file_path = os.path.join(output_path, src_name)
        target_file_path = os.path.join(output_path, dest_folder, src_name)

        if os.path.exists(current_file_path):
            try:
                # shutil.move is efficient for same-drive renames
                shutil.move(current_file_path, target_file_path)
                moved_count += 1
            except Exception as e:
                # Quietly handle errors like permission issues
                print(e)
                pass

    print(f"\nTask Complete!")
    print(f"Total files moved: {moved_count}")
