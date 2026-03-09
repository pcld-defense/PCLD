# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PCLD (Painter-Classifier-Decisioner) is an adversarial robustness defense framework for image classification. It couples stroke-based rendering with a decisioner trained on classifier confidence trajectories across paint steps. The defense pipeline renders multiple intermediate canvases at increasing stroke counts, classifies each, and uses a decisioner to pick the final prediction.

## Setup

### Install dependencies
```bash
pip install -r requirements.txt
```

### Environment variables (`.env` file)
The project uses `python-dotenv` to load path configuration. The `.env` at the project root is already configured with absolute paths:
```
RESOURCES_DIR=/home/idanbib/PCLD/code/resources/
RESOURCES_DATASETS_DIR=/home/idanbib/PCLD/data/
RESOURCES_RESULTS_DIR=/home/idanbib/PCLD/results
RESOURCES_MODELS_DIR=/home/idanbib/PCLD/models
ACTOR_WEIGHTS_PATH=/home/idanbib/PCLD/models/painter_actor/actor.pkl
RENDERER_WEIGHTS_PATH=/home/idanbib/PCLD/models/painter_renderer/renderer.pkl
```

Key path locations:
- **Models** (pretrained + trained): `/home/idanbib/PCLD/models/`
- **Datasets**: `/home/idanbib/PCLD/data/`
- **Results** (CSV/Parquet/HDF5): `/home/idanbib/PCLD/results/`

### Download pretrained models
Download the full `models/` folder from [Google Drive](https://drive.google.com/drive/folders/1wydFD78BNzktSY162IYZ5AJMrPE2O43D?usp=drive_link) and place it at `/home/idanbib/PCLD/models/`. Alternatively, run `experiments_stuff.py` to programmatically download individual model files via gdown.

## Python Environment

Always use the project's virtual environment:
```bash
/home/idanbib/PCLD/code/.venv/bin/python main.py ...
```

Before running any experiment, verify GPU availability:
```bash
/home/idanbib/PCLD/code/.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

## Running Experiments

All experiments are launched via `main.py` with `--experiment_type`:

```bash
# 1. Generate painted dataset (B_p) for classifier training
python main.py --experiment_type paint_dataset --experiment_suff bp --dataset subset_of_imagenet --batch_size 10 --output_every 50,100,200,300,400,500,600,700,950,1200,1700,2200,3200,4200,5200

# 2. Train classifier on painted images
python main.py --experiment_type train_classifier --experiment_suff bp --dataset paint_dataset_bp_subset_of_imagenet --batch_size 10 --max_epochs 13 --find_best_epoch 0

# 3. Attack PCL (Painter-Classifier, no decisioner) — generates decisioner training data
python main.py --experiment_type attack_pcl --experiment_suff bp_fgsm_untargeted --dataset subset_of_imagenet --batch_size 10 --classifier_experiment train_classifier_bp --attack fgsm --attack_direction untargeted --attack_train 1 --epsilons 0|3|6|9|12

# 4. Train decisioner on PCL attack output
python main.py --experiment_type train_decisioner --experiment_suff conv_fgsm_untargeted --dataset attack_pcl_bp_fgsm_untargeted --batch_size 10 --decisioner_architechture conv

# 5. Attack full PCLD pipeline (adaptive BPDA+EOT attack)
python main.py --experiment_type attack_pcld --experiment_suff pgd10_targeted --dataset subset_of_imagenet --batch_size 10 --classifier_experiment train_classifier_bp --decisioner_experiment train_decisioner_conv_fgsm_untargeted --attack pgd --attack_direction targeted --attack_nb_iter 10 --epsilons 4|8
```

Outputs:
- Painted datasets → `/home/idanbib/PCLD/data/<experiment_name>/`
- Trained models → `/home/idanbib/PCLD/models/<experiment_name>/`
- Attack results (CSV + Parquet + HDF5) → `/home/idanbib/PCLD/results/<experiment_name>/`

## Architecture

### Three-Component Pipeline

```
Input Image → [Painter] → Canvases at t steps → [Classifier] → Confidence trajectories → [Decisioner] → Final label
```

**Painter** (`painter/`): Stroke-based neural renderer using two pretrained models:
- `ActorResNet` (ResNet18 backbone) — predicts stroke parameters (5 strokes × 13 params = 65 outputs)
- `RendererFCN` — renders individual strokes as alpha masks
- Two-phase painting: global pass (full image, Phase 1), then patched local pass (5×5 grid, Phase 2)
- `paint()` / `paint_images()` in `painter/painter_utils.py` — main painting functions, output shape `(B, Steps, 3, H, W)`

**Classifier** (`model/`): Fine-tuned ResNet18 trained on painted images (from step 2 above).

**Decisioner** (`model/decisioner.py`): Two architectures:
- `Decisioner1DConv` — 1D conv over the sequence of softmax outputs across paint steps
- `DecisionerFC` — flattened MLP over all softmax outputs

### Key Model Wrappers (`model/pcld_bpda.py`)

| Class | Purpose |
|-------|---------|
| `PCLD` | Full pipeline: `BPDAPainter → Classifier → Decisioner` (for adaptive attack) |
| `PCL`  | Painter → Classifier only (no decisioner) |
| `CLD`  | Classifier → Decisioner on pre-painted inputs (for naïve/baseline attack) |
| `BPDAPainter` | Wraps the non-differentiable painter with BPDA gradient approximation |
| `BPDAPainterLayer` | Custom `torch.autograd.Function` implementing BPDA; backward uses the surrogate painter's gradient |

### BPDA Gradient Approximation

The painter is non-differentiable (rendering with a neural renderer + no backprop). BPDA uses a surrogate painter (`PainterSurrogate_` in `painter/painter_surrogate.py`) — one ResNet18-based model per paint step — to approximate gradients during backprop. Surrogates are stored in `resources/models/train_surrogate_painter/model_t<step>.pth`.

### Experiment Dispatch

`main.py` → `util/integrative.py:parse_args()` → `experiment/experiment_navigator.py:apply_experiment()` → routes to one of:
- `experiment/paint_dataset.py`
- `experiment/train_classifier.py`
- `experiment/attack_pcl.py`
- `experiment/train_decisioner.py`
- `experiment/attack_pcld.py`

### Key `util/` Modules

- `consts.py` — loads `.env` paths, `IMAGENETConsts`, `CIFAR10Consts`, `PainterConsts` (MAX_STEP=80, WIDTH=128, DIVIDE=5)
- `datasets.py` — `ImageFolderWithPaths`, `get_loaders()`, `transform_dataset()`
- `attacks.py` — `attack_batch()` (FGSM/PGD/AutoAttack), `attacker()` (full attack loop, saves HDF5 + Parquet)
- `models.py` — `load_model()` (handles DataParallel prefix stripping), `trainer_decisioner()`, `process_epoch_clf()`

### Default Paint Steps

The standard `output_every` schedule: `50,100,200,300,400,500,600,700,950,1200,1700,2200,3200,4200,5200` (15 steps + original image = 16 total paint steps fed to the decisioner).

## Code Style

- Use Python type hints in function signatures for all parameters and return values. Do **not** repeat types inside docstrings.
- Document all functions and classes with **Google-style docstrings** (summary line, then `Args:`, `Returns:`, `Raises:` sections as needed).

```python
# Correct
def paint_images(x: torch.Tensor, output_every: list[int], device: str,
                 actor: nn.Module, renderer: nn.Module,
                 add_original: bool = True) -> torch.Tensor:
    """Paint a batch of images and optionally append the originals.

    Args:
        x: Input image batch.
        output_every: Stroke-count checkpoints at which to save canvases.
        device: Target device string.
        actor: ActorResNet stroke-prediction model.
        renderer: RendererFCN stroke-rendering model.
        add_original: If True, append the original image as the last step.

    Returns:
        Float tensor of shape (B, Steps[+1], 3, H, W) in [0, 1].
    """
    ...
```

## Run Naming Conventions

Experiment names are constructed as `<experiment_type>_<architecture/suff>_<dataset>_<variant>`. Match the patterns observed in `/home/idanbib/PCLD/results/`:

| Pattern | Example |
|---------|---------|
| `train_classifier_<arch>_<dataset>` | `train_classifier_wrn34_cifar10` |
| `train_classifier_<arch>_<dataset>_<variant>` | `train_classifier_wrn34_cifar10_paints`, `train_classifier_wrn34_cifar10_augmented` |
| `attack_pcl_<arch>_<attack>_<direction>_<norm>` | `attack_pcl_wrn34-10-standard_fgsm_untargeted_linf` |
| `eval_clf_<arch>_<dataset>` | `eval_clf_wrn34-10-standard_cifar10` |
| `eval_rb_<model>_<variant>` | `eval_rb_carmon2019_paints` |

The `--experiment_suff` CLI flag appends a suffix to the experiment type, forming the full name used for output directories.

## Important Notes

- Epsilon values in CLI args are integers (e.g., `--epsilons 3|9`) and converted to floats internally (`epsilon / 255.0`) for attacks
- Multi-GPU is supported via `torch.nn.DataParallel`; `load_model()` strips the `module.` prefix from DataParallel checkpoints
- The `attack_train` flag in `attack_pcld` controls whether train/val splits are also attacked (used for generating decisioner training data)
- Targeted attacks randomly select a target class `(y + randint(1, 6)) % n_classes`; `targeted_jumps_allowed=6` for targeted, `1` for untargeted