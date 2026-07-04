from robustbench.utils import load_model
from robustbench.eval import benchmark
from robustbench.data import load_imagenet
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# Load a model from the model zoo
model = load_model(model_name='Rebuffi2021Fixing_70_16_cutmix_extra',
                   dataset='cifar10',
                   threat_model='Linf')

# Evaluate the Linf robustness of the model using AutoAttack
clean_acc, robust_acc = benchmark(model,
                                  dataset='cifar10',
                                  threat_model='Linf',
                                  device=device,
                                  eps=0.03)

print(clean_acc, robust_acc)