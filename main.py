import os
import warnings

from pcld.utils.consts import NUM_OF_HYPHENS
from pcld.utils.integrative import parse_args
from pcld.experiments.experiment_navigator import apply_experiment
import torch

torch.manual_seed(42)

if __name__ == '__main__':
    print('Starting PCLD Service...')
    warnings.filterwarnings("ignore")

    # ---------------------- Parse arguments ---------------------- #
    args = parse_args()

    # ---------------------- Get device (gpu / cpu) ---------------------- #
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.device(device)
    print(f'device: {device}')

    # ---------------------- Run experiment ----------------------
    print('-' * NUM_OF_HYPHENS)
    print(f'Run {args.experiment_name}...')
    apply_experiment(args=args, device=device)
    print(f'Finished executing experiment {args.experiment_name} via PCLD Service')

    os._exit(0)
