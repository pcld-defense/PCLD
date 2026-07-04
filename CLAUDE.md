# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PCLD (Painter-Classifier-Decisioner) is an adversarial robustness defense framework for image classification. It couples stroke-based rendering with a decisioner trained on classifier confidence trajectories across paint steps. The defense pipeline renders multiple intermediate canvases at increasing stroke counts, classifies each, and uses a decisioner to pick the final prediction.

The codebase is a config-driven experiment framework: an installable package under `src/pcld/`, Hydra YAML configs under `configs/`, and thin CLIs under `scripts/`. See `README.md` for the architecture diagrams and `VERIFICATION.md` for the behaviour-equivalence procedure.

## Setup

### Install dependencies (pinned) + create the venv
```bash
make setup-cuda        # Linux/cluster: .venv + CUDA 12.1 torch 2.5.1 + `pip install -e .`
make setup             # CPU-only variant
# Windows: powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1 -Cuda
```
All dependency versions are pinned in `pyproject.toml` (torch/torchvision/robustbench included). `requirements.txt` just points at `pip install -e .`.

### Environment variables (`.env` file)
`python-dotenv` loads path configuration. Copy `.env.example` → `.env` and set:
```
RESOURCES_DIR, RESOURCES_DATASETS_DIR, RESOURCES_RESULTS_DIR, RESOURCES_MODELS_DIR,
ACTOR_WEIGHTS_PATH, RENDERER_WEIGHTS_PATH   (+ HF_TOKEN for dataset downloads)
```
`.env` is gitignored. To run against a different machine, override the `RESOURCES_*` env vars (they win over `.env` because `load_dotenv` does not override existing env vars).

### Download pretrained models
Download the `models/` folder from [Google Drive](https://drive.google.com/drive/folders/1wydFD78BNzktSY162IYZ5AJMrPE2O43D?usp=drive_link) into `RESOURCES_MODELS_DIR`, or run `python scripts/download_models.py` (gdown).

### GPU check
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

## Running Experiments

Experiments are config-driven through Hydra. Prefer `scripts/run.py`; the legacy `python main.py --experiment_type ...` CLI still works (same code path).

```bash
# Example presets (configs/experiment/)
python scripts/run.py experiment=smoke_test          # fast FGSM sanity check
python scripts/run.py experiment=paper_pcld_pgd10    # adaptive targeted PGD-10 vs PCLD
python scripts/sweep.py experiment=rb_sweep          # multi-model RobustBench comparison table

# Override any leaf on the CLI
python scripts/run.py experiment=paper_pcld_pgd10 batch_size=4 attack.epsilons='[4,8]'

# Compose ad-hoc without a preset
python scripts/run.py experiment_type=attack_pcl dataset=cifar10 attack=fgsm
```

`experiment_type` is one of: `paint_dataset`, `train_classifier`, `eval_classifier`, `attack_pcl`, `train_decisioner`, `attack_pcld`, `train_surrogate_painter`.

Each run seeds all RNGs (`cfg.seed`), snapshots the resolved config to `<run_dir>/config_snapshot.yaml`, and writes results to `RESOURCES_RESULTS_DIR/<experiment_name>/`. Painted datasets go to `RESOURCES_DATASETS_DIR`, trained models to `RESOURCES_MODELS_DIR`.

### Config system (`configs/`)
```
configs/
  config.yaml            # root; composes the groups below + optional experiment preset
  dataset/               # subset_of_imagenet, cifar10, ...   (@package dataset)
  model/                 # pcld_bp, robustbench, ...           (@package model)
  attack/                # pgd, fgsm, aa                       (@package attack)
  experiment/            # full presets (# @package _global_, override any group)
```
`pcld/utils/config.py:config_to_namespace()` flattens the composed config tree (depth-agnostic) into the same `argparse.Namespace` the experiment entry points consume — this is what keeps the config layer behaviour-preserving. Defaults in `_DEFAULTS` mirror the old `parse_args()` defaults 1:1. `output_every`/`epsilons` still accept delimited strings; `dataset.name` aliases to `args.dataset`.

## Architecture

### Three-Component Pipeline
```
Input Image → [Painter] → Canvases at t steps → [Classifier] → Confidence trajectories → [Decisioner] → Final label
```

**Painter** (`src/pcld/painter/`): stroke-based neural renderer using two pretrained models:
- `ActorResNet` (ResNet18 backbone) — predicts stroke parameters (5 strokes × 13 params = 65 outputs)
- `RendererFCN` — renders individual strokes as alpha masks
- `paint_images()` in `src/pcld/painter/painter_utils.py` — main painting function, output `(B, Steps, 3, H, W)` in [0, 1]

**Classifier** (`src/pcld/models/`): built by `classifier.py:get_net()` from the `CLASSIFIER_REGISTRY` in `registry.py` (wrn / timm / robustbench families). Wrapped in `NormalizedModel` so attacks operate in [0, 1] pixel space (RobustBench convention). **`n_classes` is derived from the dataset's class folders** (a 7-class subset builds a 7-class head); it falls back to the full size for `dataset_type` (imagenet=1000, cifar10=10) only when no count is passed. RobustBench models keep their fixed pretrained head.

**Decisioner** (`src/pcld/models/decisioner.py`): `Decisioner1DConv` (1D conv over the paint-step softmax sequence) or `DecisionerFC` (flattened MLP). Also `DecisionerStepAttention`.

### Key Model Wrappers (`src/pcld/attacks/pcld_bpda.py`)
| Class | Purpose |
|-------|---------|
| `PCLD` | Full pipeline `BPDAPainter → Classifier → Decisioner` (adaptive attack) |
| `PCL`  | Painter → Classifier only (generates decisioner training data) |
| `CLD`  | Classifier → Decisioner on pre-painted inputs (naïve baseline) |
| `BPDAPainter` | Wraps the non-differentiable painter with BPDA gradient approximation |
| `BPDAPainterLayer` | `torch.autograd.Function` implementing BPDA; backward uses the surrogate painter's JVP |

### BPDA Gradient Approximation
The painter is non-differentiable. BPDA (Athalye et al. 2018) uses a surrogate painter (`PainterSurrogate` in `src/pcld/painter/painter_surrogate.py`) — one ResNet18-based model per paint step — to compute a Jacobian-vector product during backprop. Surrogates are stored at `RESOURCES_MODELS_DIR/train_surrogate_painter/model_t<step>.pth`.

### Registries (`src/pcld/utils/registry.py`)
Decorator-based `Registry[T]`. New components register without editing the runner:
- **Attacks** (`src/pcld/attacks/registry.py`): `ATTACKS` — `fgsm`, `pgd`, `aa`; `attack_batch()` dispatches through it.
- **Datasets** (`src/pcld/data/registry.py`): `DATASETS` — `ensure_dataset()` is a no-op when the folder exists, else downloads (CIFAR-10 from HuggingFace). ImageNet/subset raise with setup instructions.
- **Models**: the dict-based `CLASSIFIER_REGISTRY` in `src/pcld/models/registry.py` (add a `ClassifierConfig` entry — see README "Adding a New Classifier Architecture").

### Experiment Dispatch
`scripts/run.py` (or `main.py`) → `config_to_namespace` / `parse_args` → `src/pcld/experiments/experiment_navigator.py:apply_experiment()` → one of `src/pcld/experiments/{paint_dataset,train_classifier,eval_classifier,attack_pcl,train_decisioner,attack_pcld,train_surrogate_painter}.py`.

### Key package modules
- `src/pcld/utils/consts.py` — loads `.env` paths, `IMAGENETConsts`, `CIFAR10Consts`, `PainterConsts` (currently `MAX_STEP=40`, `WIDTH=128`, `DIVIDE=1`)
- `src/pcld/utils/config.py` — Hydra config → Namespace adapter; `src/pcld/utils/seeding.py` — `seed_everything()`
- `src/pcld/data/datasets.py` — `ImageFolderWithPaths`, `get_loaders()` (calls `ensure_dataset`), `transform_dataset()`
- `src/pcld/attacks/attacks.py` — `attack_batch()`, `pgd_with_multi_step_loss()` (APGD schedule, restarts, custom loss), `attacker()` (full loop, saves Parquet/CSV)
- `src/pcld/models/train_utils.py` — `load_model()` (strips DataParallel `module.` prefix), `trainer_decisioner()`, `process_epoch_clf()`
- `src/pcld/eval/metrics.py` — `robust_accuracy()`, `summarize_run()`, `emit_table()` (CSV + LaTeX comparison tables)

### Default Paint Steps
`output_every` = `50,100,200,300,400,500,600,700,950,1200,1700,2200,3200,4200,5200` (15 steps + original image = 16 total paint steps fed to the decisioner).

## Testing

```bash
PYTHONPATH=src pytest tests/ -q
```
`tests/` runs without GPU. Torch-dependent tests self-skip if torch/cleverhans/robustbench are absent. Notable: `test_attack_equivalence.py` proves the registry dispatch is bit-identical to a direct attack call; `test_config.py` checks every preset flattens to the expected Namespace; `test_classifier_nclasses.py` checks the head follows `n_classes`.

## Code Style

- Python type hints in signatures for all params and return values. Do **not** repeat types inside docstrings.
- **Google-style docstrings** on all functions/classes (summary, then `Args:`/`Returns:`/`Raises:`).

```python
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

## Important Notes

- Epsilon CLI/config values are integers (e.g. `epsilons: [3, 9]`), converted to floats internally (`epsilon / 255.0`).
- `n_classes` is dataset-driven everywhere (`get_net(..., n_classes=...)`); don't reintroduce hardcoded class counts.
- Multi-GPU via `torch.nn.DataParallel`; `load_model()` strips the `module.` prefix.
- Targeted attacks pick a target class `(y + randint(1, targeted_jumps_allowed+1)) % n_classes`; use `targeted_jumps_allowed=6` for ImageNet, `1` for CIFAR-10.
- Behaviour-preservation is a hard requirement for this refactor: config → same Namespace, attack dispatch is numerically identical, module moves are inert. Prove equivalence (see `VERIFICATION.md`) before claiming a change is behaviour-preserving.
- The branch `refactor/experiment-framework` holds the framework; `claude/refactor_rc` is the pre-refactor baseline used for equivalence checks.
