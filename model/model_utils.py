import os
import re
import torch
import torch.nn as nn
from torchvision import models
from model.painter_surrogate import PainterSurrogate_
torch.manual_seed(42)


def load_model(model, path, device):
    state_dict = torch.load(path, map_location=torch.device(device))
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    return model


def load_painter_surrogate(models_folder, device, output_every=None):
    output_every_names = [f'model_t{oe}.pth' for oe in output_every]
    models_names = os.listdir(models_folder)
    pattern = re.compile(r'model_t(\d+)\.pth')
    def sort_key(filename):
        match = pattern.match(filename)
        if match:
            return int(match.group(1))
        else:
            return float('inf')  # Return infinity if the pattern doesn't match

    sorted_list = sorted(models_names, key=sort_key)
    surrogate_list = []
    for name in sorted_list:
        if output_every:
            if name in output_every_names:
                model_path = os.path.join(models_folder, name)
            else:
                continue
        else:
            model_path = os.path.join(models_folder, name)

        # Create the model
        encoder = models.resnet18(weights='IMAGENET1K_V1')
        encoder = nn.Sequential(*list(encoder.children())[:-3])
        painter_surr = PainterSurrogate_(encoder)
        painter_surr = painter_surr.to(device)
        model = load_model(painter_surr, model_path, device)
        surrogate_list.append(model)

    return surrogate_list