import torch
from util.consts import IMAGENET_7_LABELS
import matplotlib.pyplot as plt


def classify(x, probs, acual_index, targeted_index):
    # Ensure probs is a list of probabilities
    if isinstance(probs, torch.Tensor):
        probs = probs.squeeze().tolist()

    # Data for the bar chart
    top_7 = list(sorted(range(len(probs)), key=lambda i: probs[i]))[::-1]
    colors = ['blue'] * len(top_7)
    idx_actual = top_7.index(acual_index)
    colors[idx_actual] = 'green'
    idx_targeted = top_7.index(targeted_index)
    colors[idx_targeted] = 'red'
    labels = [IMAGENET_7_LABELS[idx] for idx in top_7]
    probs_top_5 = [probs[t] for t in top_7]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))

    ax1.bar(labels, probs_top_5, color=colors)
    ax1.set_title('Softmax Predictions')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Probabilities')

    ax1.set_xticklabels(labels, rotation=90, fontsize=8)  # Rotate labels by 45 degrees


    # Image processing
    if isinstance(x, torch.Tensor):
        x = x.to('cpu')
        if x.ndim == 3:
            x = x.permute(1, 2, 0)  # Change (C, H, W) to (H, W, C)

    # Image
    ax2.imshow(x)
    ax2.axis('off')  # Hide the axis
    ax2.set_title('Input')

    plt.tight_layout()
    plt.show()


def plot_lines(probs, output_every):
    # Plotting
    plt.figure(figsize=(10, 2.5))
    for class_index in range(probs.shape[1]):
        plt.plot(output_every, probs.cpu().detach().numpy()[:, class_index], label=IMAGENET_7_LABELS[class_index])
        plt.scatter(output_every, probs.cpu().detach().numpy()[:, class_index], s=10)

    plt.xlabel('Paint Step')
    plt.ylabel('Confidence')
    plt.title('Probabilities for Each Class Across Paint Steps')
    # plt.legend()
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(False)
    plt.show()

