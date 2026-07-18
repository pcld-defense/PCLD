import argparse
import os
import time
import glob

from tqdm import tqdm
from torchvision.utils import save_image

from pcld.painter.painter_utils import load_painter, paint_images
from pcld.utils.consts import RESOURCES_DATASETS_DIR, NUM_OF_HYPHENS, ACTOR_WEIGHTS_PATH, RENDERER_WEIGHTS_PATH
from pcld.data.datasets import transform_dataset, get_loaders
from pcld.utils.integrative import save_args_json


def paint_dataset(actor, renderer, loaders: tuple, loader_name: str,
                  device: str, output_every: list[int],
                  ds_local_dir_new: str,
                  painter_max_step: int = 80,
                  painter_divide: int = 5) -> None:
    """Paints all images in a single dataset split and saves canvases to disk.

    Iterates over each batch, calls paint_images to obtain canvases at the
    requested stroke counts (plus the original at t=∞), and saves one PNG per
    image per step. Batches whose output files already exist on disk are
    skipped so the job can be resumed after interruption.

    Saved filenames follow the pattern:
        <img_name>_generated<stroke_count>.png
    The original image uses stroke count 999999 as a placeholder for t=∞.

    Args:
        actor: Pretrained ActorResNet stroke-parameter predictor.
        renderer: Pretrained RendererFCN stroke renderer.
        loaders: Tuple of (dataset, dataloader) for this split.
        loader_name: Name of the split (e.g. 'train', 'val', 'test'), used as
            a subdirectory name and in log messages.
        device: Target device string (e.g. 'cuda').
        output_every: Ordered list of stroke-count checkpoints at which to save
            canvases.
        ds_local_dir_new: Root directory where painted images will be written.
            Each class gets its own subdirectory.
        painter_max_step: Total painter step budget forwarded to paint_images.
        painter_divide: Phase-2 patch-grid side length forwarded to
            paint_images.
    """
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

        if all(len(glob.glob(
                f'{os.path.join(ds_local_dir_new, loader_name, idx_to_class[y[img_i].detach().cpu().item()])}/{img_names[img_i]}*')) > 0
               for img_i in range(len(img_names))):
            print(f"Batch {i} exists, skipping...")
            continue

        start_time = time.time()
        canvases = paint_images(x=x,
                                output_every=output_every,
                                device=device,
                                actor=actor,
                                renderer=renderer,
                                add_original=True,
                                max_step=painter_max_step,
                                divide=painter_divide)
        end_time = time.time()
        painting_avg_time += (end_time - start_time) / len(img_names)

        for img_i in range(canvases.shape[0]):
            img_name = img_names[img_i]
            img_label = idx_to_class[y[img_i].detach().cpu().item()]
            img_dir = os.path.join(ds_local_dir_new, loader_name, img_label)
            for c_i in range(canvases.shape[1]):
                img = canvases[img_i, c_i]
                img_save_path = os.path.join(img_dir, img_name + '_generated999999.png')
                if c_i < len(output_every):
                    img_save_path = os.path.join(img_dir, img_name + f'_generated{output_every[c_i]}.png')
                save_image(img, img_save_path)

    painting_avg_time /= (i + 1)
    print(f'Finished painting {loader_name} (avg sec per image {painting_avg_time})')


def main_paint_dataset(args: argparse.Namespace, device: str) -> None:
    """Entry point for the paint_dataset experiment.

    Loads the pretrained painter, builds data loaders for the requested splits,
    and paints each split, writing canvases to the datasets output directory.

    Args:
        args: Parsed argument namespace. Reads: dataset, splits, experiment_name,
            batch_size, output_every.
        device: Target device string resolved by main.py.
    """
    dataset, splits, experiment_name, batch_size, output_every = \
        args.dataset, args.splits, args.experiment_name, args.batch_size, args.output_every

    actor, renderer = load_painter(ACTOR_WEIGHTS_PATH, RENDERER_WEIGHTS_PATH, device)

    transform = transform_dataset(dataset_type=args.dataset_type,
                                  preprocessing=args.preprocessing)
    transform_dict = {split: transform for split in splits}
    loaders = get_loaders(dataset, splits, transform_dict, batch_size)

    ds_local_dir_new = os.path.join(RESOURCES_DATASETS_DIR, f'{experiment_name}', dataset)
    os.makedirs(os.path.join(RESOURCES_DATASETS_DIR, experiment_name), exist_ok=True)
    save_args_json(args, os.path.join(RESOURCES_DATASETS_DIR, experiment_name))

    for split in splits:
        paint_dataset(actor, renderer, loaders[split], split, device, output_every, ds_local_dir_new,
                      painter_max_step=args.painter_max_step,
                      painter_divide=args.painter_divide)
