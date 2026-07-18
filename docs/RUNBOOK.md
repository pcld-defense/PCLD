# PCLD Cluster Runbook

Execution guide for the heavy GPU runs that cannot run locally. It maps the
research playbook's cluster steps (0.5 -> 3, runs R00-R07) to exact commands
against the current framework. Every flag / preset / registry key below was
verified against the code in this branch (`refactor/experiment-framework`).

Conventions used in the commands:

- `python` is assumed to be >= 3.10 inside the project venv (see section 0).
- `<PLACEHOLDER>` marks a genuinely site-specific value you must fill in.
- Hydra override syntax: `group=name` selects a config group, `group.leaf=val`
  overrides a leaf inside it, and **`+key=val` is required to add a key that is
  not already in the composed tree** (notably `+data_source=` and `+num_samples=`
  on ad-hoc runs — presets that already set them do not need the `+`).

> Scope note (read before R05): every attack experiment (`attack_classifier`,
> `gradient_battery`, `attack_pcl`, `attack_pcld`) loads its evaluation images
> through `build_eval_loaders`, so `+data_source=robustbench +num_samples=1000`
> gives all of them the identical fixed test prefix and writes
> `rb_prefix_fingerprint.json` into the run dir. With the robustbench source only
> the `test` split is produced (train/val entries in `splits` are skipped).
> Painting/training flows (`paint_dataset`, `train_*`) still use the folder
> dataset.

---

## 0. Environment

| Requirement | Value |
|-------------|-------|
| Python | >= 3.10 |
| Install | `pip install -e .` (deps pinned in `pyproject.toml`, incl. torch/torchvision/robustbench) |
| GPU | CUDA GPU; multi-GPU is used automatically via `torch.nn.DataParallel` |
| RNG | every run seeds Python/NumPy/torch from `seed` (default 42); add `deterministic=true` for cuDNN-deterministic kernels |

```bash
# From the repo root, in a fresh venv:
pip install -e .
cp .env.example .env      # then edit .env (see below)
```

### `.env` keys (all required)

```
RESOURCES_DIR=<root of the resources tree>
RESOURCES_DATASETS_DIR=<datasets root>
RESOURCES_RESULTS_DIR=<results root>       # every run writes results/<experiment_name>/
RESOURCES_MODELS_DIR=<models root>         # classifier/decisioner/surrogate checkpoints
ACTOR_WEIGHTS_PATH=<path to actor.pkl>
RENDERER_WEIGHTS_PATH=<path to renderer.pkl>
HF_TOKEN=<HuggingFace token, for CIFAR auto-download>
```

### Painter weights

```bash
python scripts/download_models.py     # fetches actor.pkl / renderer.pkl (+ legacy surrogates) via gdown
```

Note: the legacy surrogate checkpoints pulled here were trained on the **broken
4-step trajectory** and are INVALID for the fixed 16-step pipeline — retrain them
(see "Pipeline rebuild"). RobustBench backbone weights auto-download on first use
via `robustbench.utils.load_model`.

### RobustBench version / zoo-ID check (run this first on the cluster)

The new backbones require a RobustBench recent enough to expose the Wang2023
WRN-70-16, Bartoldson2024, Liu2023 Swin-L zoo entries and `load_cifar100`. The
exact zoo IDs are asserted here for the first time — if any raises, upgrade
RobustBench (`pip install -U git+https://github.com/RobustBench/robustbench.git`).

```bash
python - <<'PY'
from robustbench.utils import load_model
from robustbench.data import load_cifar10, load_cifar100, load_imagenet   # import must succeed
checks = [
    ('Rebuffi2021Fixing_28_10_cutmix_ddpm', 'cifar10',  'Linf'),
    ('Wang2023Better_WRN-28-10',            'cifar10',  'Linf'),
    ('Wang2023Better_WRN-70-16',            'cifar10',  'Linf'),
    ('Rebuffi2021Fixing_28_10_cutmix_ddpm', 'cifar10',  'L2'),
    ('Wang2023Better_WRN-70-16',            'cifar10',  'L2'),
    ('Bartoldson2024Adversarial_WRN-94-16', 'cifar10',  'Linf'),
    ('Wang2023Better_WRN-70-16',            'cifar100', 'Linf'),
    ('Liu2023Comprehensive_Swin-L',         'imagenet', 'Linf'),
    ('Standard_R50',                        'imagenet', 'Linf'),
]
for name, ds, tm in checks:
    try:
        load_model(model_name=name, dataset=ds, threat_model=tm,
                   model_dir='<RESOURCES_MODELS_DIR>/robustbench')
        print('OK  ', name, ds, tm)
    except Exception as e:
        print('FAIL', name, ds, tm, '->', type(e).__name__, e)
PY
```

---

## 0.5 Backbone download & clean-accuracy check (playbook 0.5)

Evaluate each backbone with `attack_classifier` at `epsilons=[0]` (clean pass)
on the fixed RobustBench n=1000 prefix, and confirm clean accuracy is within
**+/-0.5%** of the published value. The epsilon-0 rows are written to
`RESOURCES_RESULTS_DIR/<experiment_name>/test_eps0_<norm>_results.parquet`; the
prefix fingerprint lands in the same dir as `rb_prefix_fingerprint.json`.

```bash
# Template — one backbone. Swap <MODEL>, <DATASET>, and (for L2) attack_norm.
python scripts/run.py \
  experiment_type=attack_classifier \
  model=robustbench model.model_type=<MODEL> \
  dataset=<DATASET> \
  attack=pgd attack.epsilons=[0] \
  +data_source=robustbench +num_samples=1000 \
  experiment_name=clean_check_<MODEL>
```

| Backbone (`model_type`) | dataset | Published clean acc (%) |
|-------------------------|---------|--------------------------|
| `wrn-28-10-wang2023`    | cifar10  | 92.44 |
| `wrn-70-16-wang2023`    | cifar10  | 93.25 |
| `wrn-28-10-rebuffi2021` | cifar10  | 87.33 |
| `wrn-94-16-bartoldson2024` | cifar10 | 93.68 |
| `wrn-70-16-wang2023`    | cifar100 | (RobustBench C100 clean — re-verify) |
| `swin-l-liu2023`        | imagenet | 78.6 |

> All published numbers are **from the design doc / RobustBench — re-verify at
> run time**. L2 backbones (`wrn-28-10-rebuffi2021-l2`, `wrn-70-16-wang2023-l2`)
> share weights with their Linf counterparts and have the same clean accuracy;
> add `attack.attack_norm=l2` when evaluating them.

---

## R00 Determinism (playbook 0.2 / 1.2)

Prove the painter is bit-identical across repeats and batch layouts on the real
pretrained weights and the fixed (80, 5) trajectory.

```bash
python scripts/verify_determinism.py --repeats 5 --out determinism_R00.json
```

The JSON artifact must show **zero** deltas:

```json
"max_abs_delta": {
  "single_repeats": 0.0,
  "batch_repeats": 0.0,
  "single_vs_batch_informational": <any value — informational only>
}
```

`single_repeats` / `batch_repeats` must be exactly `0.0` (the script exits
non-zero otherwise). `single_vs_batch_informational` is reported but not gated
(batched kernels may legally differ). `config.fired_checkpoints` should list all
15 default checkpoints firing.

### Cross-machine fingerprint check

Any `attack_classifier` / `gradient_battery` run with `data_source=robustbench`
writes `rb_prefix_fingerprint.json` (sha256 of `x` and `y`) into its run dir.
Run the same config on the laptop and the cluster and diff:

```bash
diff <(python -c "import json;d=json.load(open('laptop/rb_prefix_fingerprint.json'));print(d['x_sha256'],d['y_sha256'])") \
     <(python -c "import json;d=json.load(open('cluster/rb_prefix_fingerprint.json'));print(d['x_sha256'],d['y_sha256'])")
```

Identical `x_sha256` / `y_sha256` proves both machines evaluated byte-identical
data in identical order.

---

## Pipeline rebuild (playbook 1.1) — REQUIRED before R01/R05-R07

The old decisioner and surrogate checkpoints were trained on the broken 4-step
trajectory and are **INVALID**. On the fixed (80, 5) trajectory all 16 paint
steps fire, so retrain the whole chain. Do this **per dataset/backbone** the
PCLD headline will use.

> Warning: the fixed painter is ~50x slower than the broken config (Phase 2 now
> renders 25 patch canvases). **Benchmark first** so you can size the cluster
> job:
> ```bash
> python benchmarks/benchmark_painter.py
> ```

Sequence (each stage reads the previous stage's output):

```bash
# 1. Paint the raw dataset on the fixed 16-step trajectory.
python scripts/run.py experiment_type=paint_dataset \
  dataset=<DATASET> painter_max_step=80 painter_divide=5 \
  experiment_name=paint_<DATASET>_16step

# 2. Train per-step surrogates (one per checkpoint in output_every -> 15 models
#    saved as model_t<step>.pth under RESOURCES_MODELS_DIR/train_surrogate_painter/).
python scripts/run.py experiment_type=train_surrogate_painter \
  dataset=<DATASET> painter_max_step=80 painter_divide=5 \
  experiment_name=train_surrogate_painter

# 3. Generate decisioner training trajectories (PCL BPDA attack -> parquet).
python scripts/run.py experiment_type=attack_pcl \
  dataset=<DATASET> model=pcld_bp \
  model.classifier_experiment=<CLF_EXPERIMENT> \
  attack=fgsm attack.epsilons=[3,6,9] \
  experiment_name=<DATASET>          # attack_pcl writes to results/<experiment_name>/

# 4. Train the decisioner on those trajectories (reads results/<dataset>/*.parquet).
python scripts/run.py experiment_type=train_decisioner \
  dataset=<DATASET> model=pcld_bp model.decisioner_architechture=conv \
  experiment_name=<DEC_EXPERIMENT>
```

`train_decisioner` reads the parquet trajectories from
`RESOURCES_RESULTS_DIR/<args.dataset>/`, so keep the `attack_pcl`
`experiment_name` equal to the dataset folder name (matching the legacy layout).
Note the arg is spelled `decisioner_architechture` (`conv` | `fc`); a
`decisioner_experiment` name containing `conv` also selects the conv head at
load time in `build_pcld`.

---

## R01 GATE — gradient-validity battery

**Hard gate.** The six-check battery must pass **6/6** before any robustness
number is trusted. If it fails, STOP and fix the pipeline.

```bash
python scripts/run.py experiment=gradient_battery \
  dataset=<DATASET> \
  model.classifier_experiment=<CLF_EXPERIMENT> \
  model.decisioner_experiment=<DEC_EXPERIMENT> \
  +data_source=robustbench num_samples=<N e.g. 512> \
  experiment_name=R01_gate_<DATASET>
```

(`num_samples` is already a global key in this preset, so no `+` is needed for
it; `data_source` is not, so it needs `+`.)

Artifacts written to `RESOURCES_RESULTS_DIR/<experiment_name>/`:

| Artifact | Content |
|----------|---------|
| `gradient_battery.json` | per-check results + `gate_passed` verdict + `n_passed`/`n_checks` |
| `loss_curve.csv` | PGD loss vs iteration (check 2/3) |
| `eps_sweep.csv` | robust accuracy per epsilon incl. the unbounded eps=255 row (check 5/6) |

The six checks: (1) FGSM sign test, (2) loss-vs-iteration increases,
(3) monotonicity, (4) finite-difference sign agreement, (5) eps-to-zero,
(6) unbounded eps=255 -> ~0 accuracy. **Rule:** `gate_passed: true` (6/6) or STOP.

---

## R02 / R03 Harness validation (playbook 2.1)

Reproduce the competitors' published AutoAttack numbers **within 1.5%** to prove
the attack harness is calibrated, before trusting PCLD numbers. Standalone
`attack_classifier` (no painter/decisioner) on the fixed n=1000 prefix.

```bash
# R02 — Rebuffi2021 Linf, eps 8/255
python scripts/run.py experiment_type=attack_classifier \
  dataset=cifar10 model=robustbench model.model_type=wrn-28-10-rebuffi2021 \
  attack=aa attack.attack_norm=linf attack.epsilons=[8] \
  +data_source=robustbench +num_samples=1000 \
  experiment_name=R02_rebuffi_linf

# R03 — Wang2023 WRN-70-16 L2, eps 0.5
python scripts/run.py experiment_type=attack_classifier \
  dataset=cifar10 model=robustbench model.model_type=wrn-70-16-wang2023-l2 \
  attack=aa attack.attack_norm=l2 attack.epsilons=[0.5] \
  +data_source=robustbench +num_samples=1000 \
  experiment_name=R03_wang_l2
```

Acceptance: robust accuracy (from `test_eps8_linf_results.parquet` /
`test_eps0.5_l2_results.parquet` via `pcld.eval.metrics.robust_accuracy`) within
**1.5%** of the published AutoAttack number for that model (re-verify the target
against the current RobustBench leaderboard).

Use `attack=aa_rand` for the AutoAttack `rand` (EOT) ensemble; report the lower
of `standard` vs `rand`.

---

## R05 / R06 / R07 Headline PCLD runs (playbook 2.2)

The headline claim requires **>= 3 adaptive attacks run to convergence**, with
the reported number being the **worst** adaptive result, and the AutoAttack
number being the **min of standard and rand**. Run the full matrix below against
the full PCLD pipeline (`attack_pcld`).

Prereqs: R01 gate PASSED for this dataset, and the pipeline rebuilt
(surrogates + classifier + decisioner on the 16-step trajectory).

### R05 — CIFAR-10 Linf (eps 8/255), one command per matrix row

Shared prefix (fill the two `<...>` experiment names from the rebuild):

```
COMMON="dataset=cifar10 model=pcld_bp \
  model.classifier_experiment=<CLF_EXPERIMENT> \
  model.decisioner_experiment=<DEC_EXPERIMENT> \
  attack.attack_norm=linf attack.epsilons=[8] \
  +data_source=robustbench +num_samples=1000"
```

| Row | Attack | Key overrides |
|-----|--------|---------------|
| Naive baseline | CLD (no painter) | `attack=pgd attack.run_naive_attack=1` |
| PGD-BPDA learned | adaptive, learned surrogate | `attack=pgd model.surrogate_type=learned attack.attack_nb_iter=100` |
| PGD-BPDA straight-through | adaptive, identity surrogate | `attack=pgd model.surrogate_type=straight_through attack.attack_nb_iter=100` |
| BPDA + EOT | gradient averaging | `attack=pgd model.surrogate_type=learned attack.eot_samples=10 attack.attack_nb_iter=100` |
| Decisioner-aware | multi-step loss | `attack=pgd model.surrogate_type=learned attack.multi_step_loss_weight=0.5 attack.attack_nb_iter=100` |
| AA standard | AutoAttack | `attack=aa` |
| AA rand | AutoAttack (EOT) | `attack=aa_rand` |

```bash
# Example: PGD-BPDA learned row
python scripts/run.py experiment_type=attack_pcld $COMMON \
  attack=pgd attack.attack_direction=untargeted \
  model.surrogate_type=learned attack.attack_nb_iter=100 \
  attack.targeted_jumps_allowed=1 \
  experiment_name=R05_c10_linf_pgd_bpda_learned

# Example: BPDA+EOT row
python scripts/run.py experiment_type=attack_pcld $COMMON \
  attack=pgd model.surrogate_type=learned \
  attack.eot_samples=10 attack.attack_nb_iter=100 \
  experiment_name=R05_c10_linf_bpda_eot

# Example: AA standard row (report min(standard, rand))
python scripts/run.py experiment_type=attack_pcld $COMMON \
  attack=aa experiment_name=R05_c10_linf_aa_standard
python scripts/run.py experiment_type=attack_pcld $COMMON \
  attack=aa_rand experiment_name=R05_c10_linf_aa_rand
```

Headline = **worst** of the adaptive rows and **min** of the two AA rows.
Number to beat: **Bartoldson 73.71** (CIFAR-10 Linf leaderboard).

> Reminder: add `+data_source=robustbench +num_samples=1000` to `$COMMON` so every
> R05 row attacks the identical fixed RB test prefix; check that the
> `rb_prefix_fingerprint.json` written to each run dir matches across rows and
> machines. (The decisioner's *training* trajectories still come from the painted
> folder dataset — only the evaluation input switches to the prefix.)

### R06 — CIFAR-10 L2 (delta from R05)

Swap the norm, epsilon, and backbones (L2-trained classifier trajectory /
decisioner from a rebuild done under L2):

```
attack.attack_norm=l2 attack.epsilons=[0.5]
```

Numbers to beat (L2 leaderboard): **Wang 84.9 / Rebuffi 78.7**.

### R07 — CIFAR-100 Linf (delta from R05)

```
dataset=cifar100 attack.attack_norm=linf attack.epsilons=[8]
```

with the CIFAR-100 rebuild (paint + surrogates + `wrn-70-16-wang2023`-backed
classifier + decisioner). Number to beat: **Wang2023 42.66**.

### 7-day-job practices (playbook 2.2)

- **Register the run before launch** (log to your tracker: run id, backbone, n,
  seeds, attack, epsilon, norm, git SHA).
- **Checkpoint** long jobs; `attacker()` writes one parquet per epsilon as it
  goes, so a killed job keeps completed epsilons.
- **Fixed batch layout** across every compared run (`batch_size` constant) so the
  RNG-driven targeted-jump / EOT draws line up.
- **Log backbone, n, seeds, attack** in the run record; the resolved config is
  also snapshotted to `config_snapshot.yaml` per run and `args.json` is saved
  next to results.
- **Results to Parquet** — `save_parquet=1` is the default; leave it on.

---

## R04 DiffPure re-attack (parallel)

Out of scope for this repo. DiffPure needs its own diffusion-purification
harness; drive it from the external DiffPure codebase under **equal attack
strength** to the PCLD adaptive attacks (same eps, same iteration budget, BPDA
where applicable). Requirement from the plan: bridge the reported ~70.6 vs the
~45 that DiffPure achieves under a correctly-strengthened adaptive attack. Point
the re-attack at the same fixed n=1000 prefix (fingerprint-matched) so the
comparison is apples-to-apples.

---

## ImageNet phase

Mirror R00 -> R02 on Swin-L first, then the headline. Prerequisites and caveats:

- Backbone: `swin-l-liu2023` (`Liu2023Comprehensive_Swin-L`), Linf **eps 4/255**.
- `robustbench.data.load_imagenet` needs the **ImageNet val set on disk** under
  `RESOURCES_DATASETS_DIR/robustbench` — RobustBench does not download it.
- Painter caveat: `PainterConsts.WIDTH=128` canvas; at 224px inputs this means
  canvas upscaling, and the **surrogates must be retrained at ImageNet
  resolution** (the CIFAR surrogates do not transfer).

```bash
# ImageNet harness validation (standalone classifier, RB prefix, eps 4):
python scripts/run.py experiment_type=attack_classifier \
  model=robustbench model.model_type=swin-l-liu2023 \
  dataset.dataset_type=imagenet \
  attack=aa attack.attack_norm=linf attack.epsilons=[4] \
  +data_source=robustbench +num_samples=1000 \
  experiment_name=imagenet_swinl_aa
```

Number to beat: **MeanSparse 62.12** (ImageNet Linf leaderboard).

---

## Multi-model comparison sweeps

For the competitor comparison tables, `scripts/sweep.py` runs `attack_classifier`
per model and emits `comparison.csv` / `comparison.tex`:

```bash
python scripts/sweep.py experiment=rb_sweep            # CIFAR-10 Linf (5 models)
python scripts/sweep.py experiment=rb_sweep_l2         # CIFAR-10 L2  (2 models)
python scripts/sweep.py experiment=rb_sweep_cifar100   # CIFAR-100 Linf
python scripts/sweep.py experiment=rb_sweep_imagenet   # ImageNet Linf (Swin-L + R50)
```

Add a competitor = add one line to the preset's `comparison.models` list.

---

## Per-run checklist (playbook)

Before you trust any headline number, confirm, per run:

- [ ] **Register** the run (id, backbone, n, seeds, attack, eps, norm, git SHA).
- [ ] **Fingerprint check** — `rb_prefix_fingerprint.json` sha256 matches the
      reference machine (classifier/battery runs); for PCLD, confirm the painted
      folder is the intended fixed set.
- [ ] **Config snapshot** — `config_snapshot.yaml` present in the run dir.
- [ ] **Parquet** — per-epsilon `*_results.parquet` written (`save_parquet=1`).
- [ ] **Seeds logged** — `seed` recorded; `deterministic=true` if bit-repro is
      required.
- [ ] **R01 gate PASSED** (6/6) for this dataset/pipeline before any claim.
