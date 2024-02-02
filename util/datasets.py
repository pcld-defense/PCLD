import torch
from torchvision import datasets, transforms
import os
import errno


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
                     batch_size: int, shuffle: bool = True, num_workers: int = os.cpu_count()-1):
    ds = ImageFolderWithPaths(path, transform=transform)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                         num_workers=num_workers, pin_memory=True)
    return ds, loader


def generator_loader_train_full(*itrs):
    for itr in itrs:
        for v in itr:
            yield v


def assert_dir_exists(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise



