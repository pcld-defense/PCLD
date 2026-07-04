import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import models
from tqdm import tqdm

from pcld.painter.painter_surrogate import PainterSurrogate_, JointPainterSurrogate
from pcld.painter.painter_utils import load_painter, paint_images
from pcld.utils.consts import (NUM_OF_HYPHENS, RESOURCES_MODELS_DIR,
                         ACTOR_WEIGHTS_PATH, RENDERER_WEIGHTS_PATH,
                         IMAGENETConsts, CIFAR10Consts)
from pcld.data.datasets import transform_dataset, get_loaders
from pcld.utils.integrative import save_args_json
from pcld.models.train_utils import load_model


def _make_surrogate(device: str) -> PainterSurrogate_:
    """Builds a fresh PainterSurrogate_ with an ImageNet-pretrained encoder.

    Architecture: truncated ResNet18 (up to layer3) as encoder,
    four transposed-conv layers + final conv as decoder.

    Args:
        device: Target device string.

    Returns:
        Uninitialised PainterSurrogate_ on ``device``.
    """
    encoder = models.resnet18(weights='IMAGENET1K_V1')
    encoder = nn.Sequential(*list(encoder.children())[:-3])
    surrogate = PainterSurrogate_(encoder)
    return surrogate.to(device)


def _train_one_surrogate(
        step: int,
        actor: nn.Module,
        renderer: nn.Module,
        loaders: dict,
        splits: list[str],
        device: str,
        max_epochs: int,
        lr: float,
        output_dir: str,
        resume: bool,
        mean: torch.Tensor,
        std: torch.Tensor,
) -> None:
    """Trains a single PainterSurrogate_ for a given paint step and saves it.

    For each batch the real painter is run (no grad) to produce the ground-truth
    canvas at ``step`` strokes. The surrogate then predicts that canvas from the
    original image. MSE loss is minimised with Adam.

    Both the real painter and the surrogate operate on images in **[0, 1]**.
    DataLoader images are normalised, so they are unnormalised before use and
    the model saves in [0, 1] space so it is compatible with BPDAPainterLayer.

    The best checkpoint (lowest validation MSE) is saved to
    ``<output_dir>/model_t<step>.pth``.

    Args:
        step: Stroke-count checkpoint to train this surrogate for.
        actor: Pretrained ActorResNet (eval mode, no grad needed).
        renderer: Pretrained RendererFCN (eval mode, no grad needed).
        loaders: Dict mapping split name → [dataset, dataloader].
        splits: Split names to iterate; first split used for training,
            second (if present) used for validation.
        device: Target device string.
        max_epochs: Maximum number of training epochs.
        lr: Adam learning rate.
        output_dir: Directory where model checkpoints are written.
        resume: If True and a checkpoint already exists, load it and continue.
        mean: Per-channel normalisation mean of shape (3,), used to
            unnormalise DataLoader images back to [0, 1] before painting.
        std: Per-channel normalisation std of shape (3,).
    """
    save_path = os.path.join(output_dir, f'model_t{step}.pth')

    surrogate = _make_surrogate(device)
    optimizer = Adam(surrogate.parameters(), lr=lr)

    start_epoch = 0
    best_val_mse = float('inf')

    if resume and os.path.exists(save_path):
        print(f'  Resuming from {save_path}')
        surrogate = load_model(surrogate, save_path, device)
        best_val_mse = 0.0  # will be overwritten on first val pass

    mean4d = mean.to(device).view(1, 3, 1, 1)
    std4d  = std.to(device).view(1, 3, 1, 1)

    train_split = splits[0]
    val_split   = splits[1] if len(splits) > 1 else None

    for epoch in range(start_epoch, max_epochs):
        # ---- Training ----
        surrogate.train()
        train_loss = 0.0
        n_batches = 0
        for x_norm, _y, _paths in tqdm(loaders[train_split][1],
                                        desc=f'  t={step} epoch {epoch} train'):
            x_norm = x_norm.to(device)
            # Unnormalise to [0, 1] — painter and surrogate both expect [0, 1].
            x_01 = (x_norm * std4d + mean4d).clamp(0.0, 1.0)

            # Real painter output at this step (no gradient needed).
            with torch.no_grad():
                canvases = paint_images(x_01, [step], device, actor, renderer,
                                        add_original=False)
                target = canvases[:, 0]     # (B, 3, H, W) in [0, 1]

            pred = surrogate(x_01)          # (B, 3, H, W) in [0, 1]
            loss = F.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        avg_train_mse = train_loss / max(n_batches, 1)

        # ---- Validation ----
        val_mse_str = ''
        if val_split is not None:
            surrogate.eval()
            val_loss = 0.0
            v_batches = 0
            with torch.no_grad():
                for x_norm, _y, _paths in loaders[val_split][1]:
                    x_norm = x_norm.to(device)
                    x_01 = (x_norm * std4d + mean4d).clamp(0.0, 1.0)
                    canvases = paint_images(x_01, [step], device, actor, renderer,
                                            add_original=False)
                    target = canvases[:, 0]
                    pred = surrogate(x_01)
                    val_loss += F.mse_loss(pred, target).item()
                    v_batches += 1
            avg_val_mse = val_loss / max(v_batches, 1)
            val_mse_str = f'  val_mse={avg_val_mse:.6f}'

            if avg_val_mse < best_val_mse:
                best_val_mse = avg_val_mse
                torch.save(surrogate.state_dict(), save_path)
                print(f'  → Saved best checkpoint (val_mse={best_val_mse:.6f})')
        else:
            # No validation split: save after every epoch.
            torch.save(surrogate.state_dict(), save_path)

        print(f'  t={step} | epoch {epoch}/{max_epochs - 1} | '
              f'train_mse={avg_train_mse:.6f}{val_mse_str}')

    if val_split is None:
        print(f'  Saved final checkpoint to {save_path}')
    else:
        print(f'  Best val_mse={best_val_mse:.6f} → {save_path}')


def _train_joint_surrogate(
        output_every: list[int],
        actor: nn.Module,
        renderer: nn.Module,
        loaders: dict,
        splits: list[str],
        device: str,
        max_epochs: int,
        lr: float,
        save_path: str,
        resume: bool,
        mean: torch.Tensor,
        std: torch.Tensor,
) -> None:
    """Trains a single JointPainterSurrogate covering all steps in output_every.

    One shared encoder is run once per forward pass; its features are fed to a
    per-step decoder bank. All step MSE losses are summed into a single backward
    pass, so the encoder receives gradient signal from every step simultaneously.

    The checkpoint is saved to ``save_path``.

    Args:
        output_every: All stroke-count checkpoints to model.
        actor: Pretrained ActorResNet (no grad).
        renderer: Pretrained RendererFCN (no grad).
        loaders: Dict mapping split name → [dataset, dataloader].
        splits: [train_split] or [train_split, val_split].
        device: Target device string.
        max_epochs: Maximum training epochs.
        lr: Adam learning rate.
        save_path: Full file path where the checkpoint is saved.
        resume: If True and checkpoint exists, load and continue.
        mean: Per-channel normalisation mean for unnormalisation.
        std: Per-channel normalisation std for unnormalisation.
    """
    num_steps = len(output_every)

    model = JointPainterSurrogate(num_steps=num_steps).to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    if resume and os.path.exists(save_path):
        print(f'  Resuming joint surrogate from {save_path}')
        model = load_model(model, save_path, device)

    mean4d = mean.to(device).view(1, 3, 1, 1)
    std4d  = std.to(device).view(1, 3, 1, 1)

    train_split = splits[0]
    val_split   = splits[1] if len(splits) > 1 else None
    best_val_mse = float('inf')

    plot_path = os.path.join(os.path.dirname(os.path.abspath(save_path)),
                             'joint_surrogate_loss.png')
    train_mses: list[float] = []
    val_mses: list[float] = []

    for epoch in range(max_epochs):
        # ---- Training ----
        model.train()
        train_loss = 0.0
        n_batches = 0
        for x_norm, _y, _paths in tqdm(loaders[train_split][1],
                                        desc=f'  joint epoch {epoch} train'):
            x_norm = x_norm.to(device)
            x_01 = (x_norm * std4d + mean4d).clamp(0.0, 1.0)

            # Paint all steps in one call.
            with torch.no_grad():
                targets = paint_images(x_01, output_every, device, actor, renderer,
                                       add_original=False)  # (B, Steps, 3, H, W)

            preds = model(x_01)   # (B, Steps, 3, H, W)
            loss = F.mse_loss(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        avg_train_mse = train_loss / max(n_batches, 1)
        train_mses.append(avg_train_mse)

        # ---- Validation ----
        val_mse_str = ''
        avg_val_mse = None
        if val_split is not None:
            model.eval()
            val_loss = 0.0
            v_batches = 0
            with torch.no_grad():
                for x_norm, _y, _paths in loaders[val_split][1]:
                    x_norm = x_norm.to(device)
                    x_01 = (x_norm * std4d + mean4d).clamp(0.0, 1.0)
                    targets = paint_images(x_01, output_every, device, actor, renderer,
                                           add_original=False)
                    preds = model(x_01)
                    val_loss += F.mse_loss(preds, targets).item()
                    v_batches += 1
            avg_val_mse = val_loss / max(v_batches, 1)
            val_mses.append(avg_val_mse)
            val_mse_str = f'  val_mse={avg_val_mse:.6f}'

            if avg_val_mse < best_val_mse:
                best_val_mse = avg_val_mse
                torch.save(model.state_dict(), save_path)
                print(f'  → Saved best joint checkpoint (val_mse={best_val_mse:.6f})')
        else:
            torch.save(model.state_dict(), save_path)

        print(f'  joint | epoch {epoch}/{max_epochs - 1} | '
              f'train_mse={avg_train_mse:.6f}{val_mse_str}')

        # ---- Plot ----
        epochs_axis = list(range(len(train_mses)))
        fig, ax = plt.subplots()
        ax.plot(epochs_axis, train_mses, label='train')
        if val_mses:
            ax.plot(epochs_axis, val_mses, label='val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE')
        ax.set_title('Joint surrogate loss')
        ax.legend()
        fig.savefig(plot_path)
        plt.close(fig)

    print(f'  Joint surrogate saved to {save_path}')
    print(f'  Loss plot saved to {plot_path}')


def main_train_surrogate_painter(args: argparse.Namespace, device: str) -> None:
    """Entry point for the train_surrogate_painter experiment.

    Supports two modes controlled by ``--joint_surrogate``:

    **Separate mode** (``--joint_surrogate 0``, default):
        Trains one independent ``PainterSurrogate_`` per step in
        ``output_every``. Steps are trained sequentially. Each model is saved
        as ``model_t<step>.pth`` and loaded by ``load_painter_surrogate()``.

    **Joint mode** (``--joint_surrogate 1``):
        Trains a single ``JointPainterSurrogate`` with one shared encoder and
        one decoder per step. All steps are trained in one backward pass per
        batch — the encoder gets gradient signal from every step simultaneously.
        Saved as ``joint_surrogate.pth`` and loaded by ``load_joint_surrogate()``.
        Use ``--joint_surrogate 1`` in attack experiments to activate this model.

    Both modes save to ``RESOURCES_MODELS_DIR/train_surrogate_painter/``.

    Args:
        args: Parsed argument namespace. Reads: dataset, dataset_type, splits,
            batch_size, output_every, max_epochs, lr, preprocessing,
            joint_surrogate (path string or None), resume (default 0).
        device: Target device string resolved by main.py.
    """
    joint_path = getattr(args, 'joint_surrogate', None) or None
    joint: bool = joint_path is not None
    resume: bool = bool(getattr(args, 'resume', 0))

    print('-' * NUM_OF_HYPHENS)
    print(f'train_surrogate_painter  [mode: {"joint" if joint else "separate"}]')
    print(f'  steps     : {args.output_every}')
    print(f'  epochs    : {args.max_epochs}')
    print(f'  lr        : {args.lr}')
    print(f'  dataset   : {args.dataset}')

    output_dir = os.path.join(RESOURCES_MODELS_DIR, 'train_surrogate_painter')
    os.makedirs(output_dir, exist_ok=True)
    save_args_json(args, output_dir)

    consts = CIFAR10Consts if args.dataset_type == 'cifar10' else IMAGENETConsts
    mean = torch.tensor(consts.MEAN)
    std  = torch.tensor(consts.STD)

    split_transform = transform_dataset(dataset_type=args.dataset_type,
                                        preprocessing=args.preprocessing)
    transform_dict = {split: split_transform for split in args.splits}
    loaders = get_loaders(args.dataset, args.splits, transform_dict, args.batch_size)

    actor, renderer = load_painter(ACTOR_WEIGHTS_PATH, RENDERER_WEIGHTS_PATH, device)

    if joint:
        os.makedirs(os.path.dirname(os.path.abspath(joint_path)), exist_ok=True)
        print('-' * NUM_OF_HYPHENS)
        print(f'Training joint surrogate for {len(args.output_every)} steps')
        print(f'  save path : {joint_path}')
        _train_joint_surrogate(
            output_every=args.output_every,
            actor=actor, renderer=renderer,
            loaders=loaders, splits=args.splits,
            device=device, max_epochs=args.max_epochs, lr=args.lr,
            save_path=joint_path, resume=resume,
            mean=mean, std=std,
        )
    else:
        for step in args.output_every:
            print('-' * NUM_OF_HYPHENS)
            print(f'Training surrogate for step t={step}')
            _train_one_surrogate(
                step=step,
                actor=actor, renderer=renderer,
                loaders=loaders, splits=args.splits,
                device=device, max_epochs=args.max_epochs, lr=args.lr,
                output_dir=output_dir, resume=resume,
                mean=mean, std=std,
            )

    print('-' * NUM_OF_HYPHENS)
    print('train_surrogate_painter finished!')
    print(f'Saved surrogates to {output_dir}')
