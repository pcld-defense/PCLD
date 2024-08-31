import os
import time
from torchvision.utils import save_image

from util.consts import RESOURCES_DATASETS_DIR, NUM_OF_HYPHENS, IMAGENET_7_LABELS
from model.painter_utils import load_painter, paint_images
from util.datasets import create_ds_loader, transform_dataset, generator_loader_train_full, get_loaders


def paint_dataset(actor, renderer, loader, loader_name, device, output_every, ds_local_dir_new):
    print('-' * NUM_OF_HYPHENS)
    print(f'Paint {loader_name}...')

    for animal in IMAGENET_7_LABELS.values():
        animal_dir_new = os.path.join(ds_local_dir_new, loader_name, animal)
        os.makedirs(os.path.join(ds_local_dir_new, loader_name, animal), exist_ok=True)

    painting_avg_time = 0
    i = 0
    for i, data in enumerate(loader, 0):
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
        painting_avg_time += (end_time - start_time)/len(img_names)
        for img_i in range(canvases.shape[0]):
            img_name = img_names[img_i]
            img_animal = IMAGENET_7_LABELS[int(y[img_i])]
            animal_dir_new = os.path.join(ds_local_dir_new, loader_name, img_animal)
            for c_i in range(canvases.shape[1]):
                img = canvases[img_i, c_i]
                img_save_path = os.path.join(animal_dir_new, img_name+'_generated999999.png')
                if c_i < len(output_every):
                    img_save_path = os.path.join(animal_dir_new, img_name+f'_generated{output_every[c_i]}.png')
                save_image(img, img_save_path)
    painting_avg_time /= (i+1)
    print(f'Finished painting {loader_name} (avg sec per image {painting_avg_time})')


def main_paint_dataset(args, device):
    # Fail fast
    dataset, experiment_name, batch_size, output_every = \
        args.dataset, args.experiment_name, args.batch_size, args.output_every

    actor, renderer = load_painter(device)

    # =================== Load the dataset =================== #
    transform = transform_dataset(augmentations=False, to_integers=False)
    loaders = get_loaders(dataset, transform, transform, batch_size)

    ds_local_dir_new = os.path.join(RESOURCES_DATASETS_DIR, f'{experiment_name}_paints' + dataset)

    paint_dataset(actor, renderer, loaders['train'][1], 'train', device, output_every, ds_local_dir_new)
    paint_dataset(actor, renderer, loaders['val'][1], 'val', device, output_every, ds_local_dir_new)
    paint_dataset(actor, renderer, loaders['test'][1], 'test', device, output_every, ds_local_dir_new)
    paint_dataset(actor, renderer, generator_loader_train_full(loaders['train'][1], loaders['val_to_concat'][1]),
                  'train_full', device, output_every, ds_local_dir_new)









