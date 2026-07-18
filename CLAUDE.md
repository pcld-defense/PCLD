# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PCLD (Painter-Classifier-Decisioner) is an adversarial robustness defense framework for image classification. It couples stroke-based rendering with a decisioner trained on classifier confidence trajectories across paint steps. The defense pipeline renders multiple intermediate canvases at increasing stroke counts, classifies each, and uses a decisioner to pick the final prediction.

The codebase is a config-driven experiment framework: an installable package under `src/pcld/`, Hydra YAML configs under `configs/`, and thin CLIs under `scripts/`. See `README.md` for the architecture diagrams and `VERIFICATION.md` for the behaviour-equivalence procedure.

## Setup

See the `setup-environment` skill for installing dependencies, configuring `.env`, downloading pretrained models, and checking GPU access.

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

## Important Notes

- Epsilon CLI/config values are integers (e.g. `epsilons: [3, 9]`), converted to floats internally (`epsilon / 255.0`).
- `n_classes` is dataset-driven everywhere (`get_net(..., n_classes=...)`); don't reintroduce hardcoded class counts.
- Multi-GPU via `torch.nn.DataParallel`; `load_model()` strips the `module.` prefix.
- Targeted attacks pick a target class `(y + randint(1, targeted_jumps_allowed+1)) % n_classes`; use `targeted_jumps_allowed=6` for ImageNet, `1` for CIFAR-10.
- Behaviour-preservation is a hard requirement for this refactor: config → same Namespace, attack dispatch is numerically identical, module moves are inert. Prove equivalence (see `VERIFICATION.md`) before claiming a change is behaviour-preserving.
- The branch `refactor/experiment-framework` holds the framework; `claude/refactor_rc` is the pre-refactor baseline used for equivalence checks.
