"""Single-image FGSM attack example with painting defense visualization.

Loads one CIFAR-10 image, runs an untargeted FGSM L-inf attack, paints both
the clean and adversarial images across all paint steps, classifies each step,
and saves a figure showing the images and confidence trajectories.

Usage:
    python plots/fgsm_example.py
    python plots/fgsm_example.py --img_path /path/to/image.png --true_class 3 --epsilon 8
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image as PILImg
from torchvision import transforms

from model.classifier import get_net
from painter.painter_utils import load_painter, paint_images
from util.attacks import attack_batch
from util.consts import CIFAR10Consts
from util.models import load_model

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
OUTPUT_EVERY = [50, 100, 200, 300, 400, 500, 600, 700, 950, 1200, 1700, 2200, 3200, 4200, 5200]
SELECTED_STEPS_IDX = [0, 3, 6, 10, 14, 15]  # steps 50, 300, 600, 1700, 5200, original


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description='FGSM single-image example')
    parser.add_argument('--img_path', default='/home/idanbib/PCLD/data/cifar10/val/cat/0.png')
    parser.add_argument('--true_class', type=int, default=3, help='Ground-truth class index (0-9)')
    parser.add_argument('--epsilon', type=int, default=8, help='FGSM epsilon in [0, 255]')
    parser.add_argument('--clf_weights', default='/home/idanbib/PCLD/models/train_classifier_wrn34_cifar10_augmented/best_model.pth')
    parser.add_argument('--actor_weights', default='/home/idanbib/PCLD/models/painter_actor/actor.pkl')
    parser.add_argument('--renderer_weights', default='/home/idanbib/PCLD/models/painter_renderer/renderer.pkl')
    parser.add_argument('--save_path', default='/home/idanbib/PCLD/results/fgsm_example_cat.png')
    return parser.parse_args()


def classify_paints(paints: torch.Tensor, clf: torch.nn.Module,
                    normalize: transforms.Normalize, steps: int) -> np.ndarray:
    """Classifies each paint step of a batch-1 painted tensor.

    Args:
        paints: Painted tensor of shape (1, Steps, 3, H, W) in [0, 1].
        clf: Classifier model in eval mode.
        normalize: Normalization transform matching the classifier's training.
        steps: Number of paint steps (including optional original).

    Returns:
        Float array of shape (Steps, n_classes) with softmax probabilities.
    """
    probs_list = []
    with torch.no_grad():
        for s in range(steps):
            inp = normalize(paints[0, s])
            logits = clf(inp.unsqueeze(0))
            probs_list.append(torch.softmax(logits, dim=1)[0].detach().cpu().numpy())
    return np.array(probs_list)


def to_display(tensor_chw: torch.Tensor, size: int = 128) -> np.ndarray:
    """Converts a (3, H, W) tensor to a uint8 numpy array upsampled to size×size.

    Args:
        tensor_chw: Image tensor of shape (3, H, W) in [0, 1].
        size: Output image side length in pixels.

    Returns:
        uint8 numpy array of shape (size, size, 3).
    """
    arr = (tensor_chw.permute(1, 2, 0).detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    return np.array(PILImg.fromarray(arr).resize((size, size), PILImg.NEAREST))


def main() -> None:
    """Runs the FGSM example and saves the visualization."""
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # Load image
    img_pil = PILImg.open(args.img_path).convert('RGB')
    x_clean = transforms.ToTensor()(img_pil).unsqueeze(0).to(device)

    # Load classifier
    clf = get_net('cifar10', device, 'wrn-34-10')
    clf = load_model(clf, args.clf_weights, device)
    clf.eval()
    print('Classifier loaded.')

    # Load painter
    actor, renderer = load_painter(args.actor_weights, args.renderer_weights, device)
    print('Painter loaded.')

    # FGSM attack
    epsilon = args.epsilon / 255.0
    y = torch.tensor([args.true_class], device=device)
    x_adv = attack_batch(clf, x_clean, 'fgsm', epsilon, attack_nb_iter=1,
                         targeted=False, y_classes_targeted=y, norm='linf')
    print(f'Perturbation L-inf: {(x_adv - x_clean).abs().max().item():.4f}')

    # Paint
    paints_clean = paint_images(x_clean, OUTPUT_EVERY, device, actor, renderer, add_original=True)
    paints_adv   = paint_images(x_adv,   OUTPUT_EVERY, device, actor, renderer, add_original=True)
    steps = paints_clean.shape[1]

    # Classify
    normalize = transforms.Normalize(mean=CIFAR10Consts.MEAN, std=CIFAR10Consts.STD)
    probs_clean = classify_paints(paints_clean, clf, normalize, steps)
    probs_adv   = classify_paints(paints_adv,   clf, normalize, steps)

    step_labels = [str(s) for s in OUTPUT_EVERY] + ['original']
    true_name = CLASS_NAMES[args.true_class]
    print('Clean preds:      ', [CLASS_NAMES[p.argmax()] for p in probs_clean])
    print('Adversarial preds:', [CLASS_NAMES[p.argmax()] for p in probs_adv])

    # Build figure
    n_sel = len(SELECTED_STEPS_IDX)
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f'FGSM Untargeted L-inf (ε={args.epsilon}/255)  |  True class: {true_name}',
        fontsize=14, fontweight='bold'
    )

    # Rows 1-2: image grids (clean / adversarial)
    for row, (paints, probs, label) in enumerate([
        (paints_clean, probs_clean, 'Clean'),
        (paints_adv,   probs_adv,   f'Adversarial (FGSM ε={args.epsilon}/255)')
    ]):
        for col, si in enumerate(SELECTED_STEPS_IDX):
            ax = fig.add_subplot(4, n_sel, row * n_sel + col + 1)
            ax.imshow(to_display(paints[0, si]))
            pred = CLASS_NAMES[probs[si].argmax()]
            conf = probs[si].max()
            color = 'green' if pred == true_name else 'red'
            ax.set_title(f'{label}\nstep={step_labels[si]}\n→{pred} ({conf:.2f})',
                         fontsize=7, color=color)
            ax.axis('off')

    x_axis = list(range(steps))

    # Row 3: true class confidence trajectory
    ax3 = fig.add_subplot(4, 1, 3)
    ax3.plot(x_axis, probs_clean[:, args.true_class], 'b-o', markersize=5,
             label=f'Clean — {true_name} (true)')
    ax3.plot(x_axis, probs_adv[:,  args.true_class], 'r-o', markersize=5,
             label=f'Adv — {true_name} (true)')
    ax3.set_xticks(x_axis)
    ax3.set_xticklabels(step_labels, rotation=45, fontsize=7)
    ax3.set_ylabel('Confidence')
    ax3.set_title(f'Confidence of true class ({true_name}) across paint steps')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)

    # Row 4: adversarial top class confidence trajectory
    top_adv_class = int(probs_adv[0].argmax())
    ax4 = fig.add_subplot(4, 1, 4)
    ax4.plot(x_axis, probs_clean[:, top_adv_class], 'b--s', markersize=5,
             label=f'Clean — {CLASS_NAMES[top_adv_class]}')
    ax4.plot(x_axis, probs_adv[:,  top_adv_class], 'r--s', markersize=5,
             label=f'Adv — {CLASS_NAMES[top_adv_class]} (fooled class)')
    ax4.set_xticks(x_axis)
    ax4.set_xticklabels(step_labels, rotation=45, fontsize=7)
    ax4.set_ylabel('Confidence')
    ax4.set_title(f'Confidence of fooled class ({CLASS_NAMES[top_adv_class]}) across paint steps')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(args.save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved to {args.save_path}')


if __name__ == '__main__':
    main()
