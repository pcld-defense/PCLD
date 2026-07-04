# CHANGES — Strongest Adaptive Attack Improvements

This document describes every change made to the codebase to strengthen the
adaptive attack against the PCLD painter defence. Each section explains what
the bug or gap was, why it mattered, and what the fix does.

---

## Phase 0 — Pre-requisite Bug Fixes

These were blockers: the attack was silently producing wrong results before
any new features were added.

### 1. Fix BPDA backward pass — `model/pcld_bpda.py`

**Bug:**
```python
# OLD — wrong: multiplies upstream gradient by canvas pixel values (meaningless)
with torch.no_grad():
    approx_grad = grad_approx_net(input)          # canvas tensor, not a gradient
new_grad_input = (grad_output * approx_grad).mean(dim=1)
```
`approx_grad` here is the surrogate's *output canvas* (pixel values in [0, 1]),
not a Jacobian. Multiplying `dL/d(canvas)` by canvas pixels has no geometric
meaning. This produced noise masquerading as gradients — every downstream
improvement (APGD, restarts, multi-step loss) was built on a broken foundation.

**Fix:**
```python
# NEW — correct BPDA: Jacobian-vector product through the surrogate
x_for_grad = input.detach().requires_grad_(True)
with torch.enable_grad():
    surrogate_out = grad_approx_net(x_for_grad)
new_grad_input = torch.autograd.grad(
    surrogate_out, x_for_grad, grad_outputs=grad_output)[0]
```
This is the correct implementation of BPDA (Athalye et al., ICML 2018): replace
the non-differentiable painter's Jacobian with the surrogate's Jacobian in
the backward pass via a proper JVP. `torch.autograd.grad` computes exactly
`dL/dx ≈ dL/d(canvas) · d(surrogate)/dx` via the chain rule.

### 2. Add `BPDAPainterLayer.reset()` classmethod — `model/pcld_bpda.py`

**Bug:**
`_stored_non_diff_layer` and `_stored_grad_output` are class-level attributes.
They persisted across batches: when a new batch started, the PGD's first
forward pass used stale canvases painted from the previous batch's final
adversarial example. This contaminated every batch after the first.

The same problem occurred across random restarts: restart N started with
canvases from restart N-1's final iterate, so the "fresh" random start was
actually seeded with wrong information.

**Fix:**
Added `BPDAPainterLayer.reset()` classmethod that sets both class variables
to `None`. This is called:
- At the start of each batch in `attack_batch()` (before the attack begins)
- At the start of each restart in `pgd_with_multi_step_loss()`

### 3. Fix three existing bugs in `experiment/attack_pcld.py`

**Bug A — `load_painter(device)`:**
The function signature requires three arguments `(actor_path, renderer_path, device)`
but the call passed only `device`. This would raise `TypeError` at runtime.
Fixed to: `load_painter(ACTOR_WEIGHTS_PATH, RENDERER_WEIGHTS_PATH, device)` and
added the required imports.

**Bug B — `get_loaders(dataset, train_transform, test_transform, batch_size)`:**
The `get_loaders()` API takes `(dataset, splits, transform_dict, batch_size)`.
The old call passed two transforms directly, matching an obsolete signature.
Fixed to use `args.splits` and a `transform_dict` dict, matching `attack_pcl.py`.

**Bug C — Missing args in `attacker()` call for the test split:**
```python
# OLD — missing norm, save_parquet, targeted_jumps_allowed
attacker(..., loaders['test'][1], 'test', epsilon, targeted, ...)
```
The call for the test split was missing `norm`, `save_parquet`, and
`targeted_jumps_allowed`. These silently fell back to defaults, meaning
test results were saved with wrong norms, wrong parquet settings, and
wrong target label generation.

**Bug D — `args.attack_train` AttributeError:**
`attack_pcld.py` referenced `args.attack_train` which was never added to
`parse_args()`. This would raise `AttributeError` whenever the script ran.
Fixed by removing the `attack_train` logic and replacing it with a loop over
`args.splits` (consistent with `attack_pcl.py`).

---

## Phase 1 — Custom PGD Loop with APGD and Random Restarts

**File:** `util/attacks.py`

### New function: `pgd_with_multi_step_loss()`

Replaces the CleverHans `projected_gradient_descent` call for the `'pgd'`
attack path. Three improvements over the previous implementation:

#### a) APGD adaptive step-size schedule (`use_apgd=True`)

**Why the old PGD failed:** Fixed step size `alpha = epsilon / nb_iter` is too
small when the loss surface is flat and too large when approaching a sharp
boundary. With BPDA's noisy gradient approximation, a fixed step often either
diverges or stalls.

**Fix:** When `use_apgd=True`, the step size starts at
`alpha_0 = 2 * epsilon / nb_iter` (the Croce & Hein 2020 convention) and is
halved whenever the loss fails to improve for
`checkfreq = max(nb_iter // 10, 10)` consecutive steps. This matches the
APGD-CE schedule from AutoAttack (Croce & Hein, ICML 2020).

Note: the initial step size is `2 * epsilon / nb_iter`, not `epsilon / 4` as
some implementations use. The `epsilon / 4` starting point is too aggressive
for BPDA where gradient approximation error is large.

#### b) Random restarts (`nb_restarts > 1`)

**Why one restart fails:** A single PGD trajectory can get stuck in a local
optimum, especially near the non-differentiable painter boundary. Using only
one restart severely underestimates the attack's true capability.

**Fix:** For each restart, the adversarial example is initialised from a
uniform random start `x + U(-ε, ε)`. After all restarts, the adversarial
example with the **highest loss** (evaluated via `model.eval()` with
`torch.no_grad()`) is kept. This is the standard protocol from Tramer et al.
(NeurIPS 2020 checklist).

For PCL's `paints_inference` path (where `y` is repeated to `B*Steps`), the
per-sample loss is averaged across steps before comparing restarts, so the
best-restart comparison is always per-image.

#### c) Custom `loss_fn` interface

**Why needed:** CleverHans internally computes CE on the model's final output.
For PCLD, this means the gradient must survive the decisioner, 15 paint steps,
and the classifier — signal can vanish. A custom `loss_fn` closure lets us
inject auxiliary losses without modifying the model's forward signature
(changing `forward()` to return tuples would break inference, DataParallel,
and all existing call sites).

When `loss_fn=None`, the default CE on `model(x_adv)` is used (backward
compatible).

### Updated `attack_batch()` signature

Added three new keyword arguments with defaults:
- `loss_fn=None` — custom loss callable, passed through to `pgd_with_multi_step_loss`
- `nb_restarts=1` — number of PGD restarts
- `use_apgd=False` — enable APGD step schedule

`BPDAPainterLayer.reset()` is now called at the top of the `'pgd'` branch to
clear stale batch state before every new batch's attack.

---

## Phase 2 — Multi-Step Intermediate Loss for PCLD

**Files:** `experiment/attack_pcld.py`, `experiment/attack_pcl.py`

### PCLD multi-step loss closure — `experiment/attack_pcld.py`

**Why PCLD needs explicit intermediate loss:**
In PCLD, the decisioner collapses 15 paint-step confidence vectors into a
single prediction. The gradient of `CE(decisioner_output, y)` must survive
backprop through the decisioner, all 15 classifiers, and 15 BPDA surrogate
steps before reaching the input image. This chain produces vanishing gradients
even when BPDA is correct.

**Fix:**
When `--multi_step_loss_weight > 0`, a closure is built that computes:
```
L = CE(decisioner_final, y) + λ · CE(all_canvas_logits, y_repeated)
```
The second term adds a direct gradient path from the loss to each paint step's
canvas independently, bypassing the decisioner. This is the "intermediate loss"
technique from DiffAttack (NeurIPS 2023) and ILA (ICCV 2019).

The closure captures `bpda_painter`, `clf`, and `decisioner` **before**
`torch.nn.DataParallel` wrapping. With DataParallel, the multi-GPU forward
pass distributes the batch across GPUs, but `BPDAPainterLayer`'s class-level
state is not thread-safe across GPU replicas. Using pre-DataParallel module
references ensures the loss closure runs on a single device.

Recommended `λ` range: 0.1–0.5. Start with 0.2.

### PCL — no multi-step closure needed — `experiment/attack_pcl.py`

PCL outputs `(B*Steps, n_classes)` — all steps flat in the batch dimension.
The CE loss computed by PGD is already implicitly multi-step: each step's
logits contribute equally to the total loss. Adding an explicit multi-step
term would double-count every step's gradient. `loss_fn=None` is passed
explicitly to make this intent clear.

---

## Phase 3 — New CLI Flags

**File:** `util/integrative.py`

Four new flags added to `parse_args()`:

| Flag | Short | Type | Default | Purpose |
|---|---|---|---|---|
| `--attack_nb_restarts` | `-anr` | `int` | `1` | Number of PGD random restarts |
| `--multi_step_loss_weight` | `-msl` | `float` | `0.0` | Intermediate loss weight λ (0 = disabled) |
| `--eot_samples` | `-eot` | `int` | `1` | EOT gradient averaging samples (1 = no EOT) |
| `--use_apgd` | `-apgd` | `int` | `0` | 1 = APGD step schedule, 0 = fixed step |

**Important:** All boolean flags use `type=int` (not `type=bool`). Python's
argparse with `type=bool` evaluates `bool('False') == True`, silently
ignoring `--flag False` on the command line. Using `type=int` with `0/1`
is consistent with all other boolean flags in this codebase
(`run_naive_attack`, `save_parquet`, etc.).

`--eot_samples` is exposed but not yet wired into the PGD inner loop. It is
reserved for a future EOT averaging wrapper inside `pgd_with_multi_step_loss`.
The painter is deterministic so EOT has lower priority than restarts and
multi-step loss, but it can reduce surrogate gradient approximation noise.

---

## Phase 4 — Gradient Masking Diagnostics

**File:** `util/evaluations.py`

### `diagnose_gradient_masking()`

A standalone diagnostic function that should be run on a single batch
**before** committing to a full attack run. It detects obfuscated gradients
(Carlini et al. 2019; Tramer et al. NeurIPS 2020) via three tests:

1. **Sign test** — computes `∇L` at the clean input and checks that
   `L(x + ε·sign(∇L)) > L(x)`. If the gradient correctly points uphill,
   a single FGSM step must increase the loss. Failure means gradients are
   pointing in the wrong direction (surrogate mismatch or masking).

2. **Loss-vs-iteration curve** — runs PGD for `nb_iter` steps, logging loss
   at each step. Saved to CSV if `output_csv` is provided. A healthy attack
   shows monotonically increasing loss.

3. **Monotonicity check** — compares the mean loss in the first half of the
   PGD trajectory against the second half. If second half < first half,
   gradients are likely masked.

All tests run with `model.eval()` to avoid BatchNorm training-mode artefacts.

### `evaluate_surrogate_quality()`

Runs both the real painter and the surrogate on the same input batch and
computes per-step MSE and SSIM. Steps with MSE > `mse_threshold` (default
0.05) are flagged and a warning is printed recommending retraining with
perceptual loss (LPIPS).

This should be run before any adaptive attack to verify that the surrogates
provide accurate enough gradient approximations. A surrogate with MSE > 0.05
will produce JVPs that are too far from the true Jacobian to be useful.

The function handles the size mismatch between surrogate outputs (28×28 for
CIFAR-10 step surrogates before the resize fix) and real painter outputs via
`F.interpolate`, matching the fix already in `PainterSurrogate.forward()`.

---

## Verification Protocol

Run these checks in order before interpreting attack results:

```bash
# 1. Check surrogate quality (MSE < 0.05 for all steps)
from util.evaluations import evaluate_surrogate_quality
evaluate_surrogate_quality(painter_surrogate, paint_images, x_batch,
                           output_every, device, actor, renderer)

# 2. Sign test (loss must increase after FGSM step)
from util.evaluations import diagnose_gradient_masking
results = diagnose_gradient_masking(pcld_model, x_batch, y_batch,
                                    epsilon=8/255, nb_iter=50,
                                    output_csv='results/loss_curve.csv')
assert results['sign_test_passed'], 'Gradient masking detected!'
assert results['monotonicity_passed'], 'Loss curve non-monotonic!'

# 3. Run FGSM — should be weaker than PGD-20
# 4. Run PGD-20 — should be weaker than PGD-50
# 5. Compare multi_step_loss_weight in {0, 0.1, 0.2, 0.5}
```

---

## Recommended First Attack Config

```bash
# Full PCLD adaptive attack — multi-step loss + APGD + 3 restarts
python main.py \
  --experiment_type attack_pcld \
  --experiment_name pcld_pgd20_targeted_multistep \
  --dataset subset_of_imagenet \
  --dataset_type imagenet \
  --splits test \
  --batch_size 4 \
  --classifier_experiment train_classifier_bp \
  --decisioner_experiment train_decisioner_conv_fgsm_untargeted \
  --attack pgd \
  --attack_direction targeted \
  --attack_nb_iter 20 \
  --attack_nb_restarts 3 \
  --multi_step_loss_weight 0.2 \
  --use_apgd 1 \
  --epsilons 4|8

# PCL attack — APGD + restarts (no multi-step closure, already implicit)
python main.py \
  --experiment_type attack_pcl \
  --experiment_name pcl_pgd20_untargeted_apgd \
  --dataset subset_of_imagenet \
  --dataset_type imagenet \
  --splits val \
  --batch_size 4 \
  --classifier_experiment train_classifier_bp \
  --attack pgd \
  --attack_direction untargeted \
  --attack_nb_iter 20 \
  --attack_nb_restarts 3 \
  --use_apgd 1 \
  --epsilons 4|8
```

---

## Files Changed

| File | Change summary |
|---|---|
| `model/pcld_bpda.py` | Fix backward JVP; add `reset()` classmethod; update docstrings |
| `util/attacks.py` | New `pgd_with_multi_step_loss()`; updated `attack_batch()` with `loss_fn`, `nb_restarts`, `use_apgd`; updated `attacker()` with same new params |
| `experiment/attack_pcld.py` | Fix 4 pre-existing bugs; add PCLD multi-step loss closure; pass new args to `attacker()` |
| `experiment/attack_pcl.py` | Pass `nb_restarts`, `use_apgd`, `loss_fn=None` through to `attacker()` |
| `util/integrative.py` | Add `--attack_nb_restarts`, `--multi_step_loss_weight`, `--eot_samples`, `--use_apgd` flags |
| `util/evaluations.py` | Add `diagnose_gradient_masking()`; add `evaluate_surrogate_quality()` |
