import os
import time

from tqdm import tqdm
from torchvision.utils import save_image

from model.painter_utils import load_painter
from util.consts import RESOURCES_DATASETS_DIR, NUM_OF_HYPHENS, IMAGENET_2012_LABELS, ACTOR_PATH, RENDERER_PATH
from util.datasets import transform_dataset, generator_loader_train_full, get_loaders


def paint_dataset(actor, renderer, loaders, loader_name, device, output_every, ds_local_dir_new):
    print('-' * NUM_OF_HYPHENS)
    print(f'Paint {loader_name}...')

    ds, loader = loaders
    idx_to_class = {idx: cls for cls, idx in ds.class_to_idx.items()}

    for label in ds.classes:
        os.makedirs(os.path.join(ds_local_dir_new, loader_name, label), exist_ok=True)

    painting_avg_time = 0
    i = 0
    for i, data in enumerate(tqdm(loader)):
        print(f'painting batch {i}...')

        x, y, paths = data[0].to(device), data[1].to(device), data[2]
        img_names = [p.split('/')[-1].split('.')[0] for p in paths]
        start_time = time.time()
        canvases = paint_images(x=x,
                                output_every=output_every,
                                device=device,
                                actor=actor,
                                renderer=renderer,
                                add_original=True)
        end_time = time.time()
        painting_avg_time += (end_time - start_time) / len(img_names)
        for img_i in range(canvases.shape[0]):
            img_name = img_names[img_i]
            img_label = idx_to_class[y[img_i]]
            img_dir = os.path.join(ds_local_dir_new, loader_name, img_label)
            for c_i in range(canvases.shape[1]):
                img = canvases[img_i, c_i]
                img_save_path = os.path.join(img_dir, img_name + '_generated999999.png')
                if c_i < len(output_every):
                    img_save_path = os.path.join(img_dir, img_name + f'_generated{output_every[c_i]}.png')
                save_image(img, img_save_path)

    painting_avg_time /= (i + 1)
    print(f'Finished painting {loader_name} (avg sec per image {painting_avg_time})')


def main_paint_dataset(args, device):
    # Fail fast
    dataset, splits, experiment_name, batch_size, output_every = \
        args.dataset, args.splits, args.experiment_name, args.batch_size, args.output_every

    if isinstance(splits, str):
        split_list = [s.strip() for s in splits.split(',')]
    else:
        split_list = list(splits)

    actor, renderer = load_painter(ACTOR_PATH, RENDERER_PATH, device)

    # =================== Load the dataset =================== #
    transform = transform_dataset(augmentations=False, to_integers=False)
    loaders = get_loaders(dataset, split_list, transform, batch_size)

    ds_local_dir_new = os.path.join(RESOURCES_DATASETS_DIR, f'{experiment_name}', dataset)

    for split in split_list:
        paint_dataset(actor, renderer, loaders[split], split, device, output_every, ds_local_dir_new)
    # paint_dataset(actor, renderer, loaders['val'][1], 'val', device, output_every, ds_local_dir_new)
    # paint_dataset(actor, renderer, loaders['test'][1], 'test', device, output_every, ds_local_dir_new)
    # paint_dataset(actor, renderer, generator_loader_train_full(loaders['train'][1], loaders['val_to_concat'][1]),
    #               'train_full', device, output_every, ds_local_dir_new)
