# Prodigy+ (Schedule-Free) in ai-toolkit

> Reference guide for using and reasoning about the **Prodigy+ Schedule-Free**
> optimizer added to this fork. Written to be precise enough that an agent can
> both configure a training run and modify the integration code.
>
> Scope note: this guide is **only about `prodigy_plus`**. The repo also ships
> upstream optimizers `prodigyopt` (standard Prodigy) and `prodigy8bit` — those
> are original ostris/ai-toolkit code and are out of scope here.

## 1. TL;DR

`prodigy_plus` is an adaptive, **schedule-free** optimizer
(`prodigy-plus-schedule-free` package, class `ProdigyPlusScheduleFree`). It
auto-tunes the learning rate and manages its own LR schedule, so you do not
hand-tune `lr` and you do not use a decaying scheduler.

Accepted config aliases (all map to the same branch): `prodigy_plus`,
`prodigyplus`, `prodigy_plus_schedulefree`, `prodigyplusschedulefree`,
`prodigy+`.

## 2. The three rules you must not get wrong

1. **`lr: 1.0`.** Prodigy+ adapts the LR internally. The configured `lr` is
   propagated to every parameter group, and Prodigy treats a per-group `lr` as a
   *multiplier* on its adapted step. Setting `lr: 1e-4` (e.g. copied from an
   AdamW config) silently scales the effective LR down by 1e‑4 and training
   barely moves. Always use `lr: 1.0`. The UI sets it automatically, and the
   backend normalizes stale Adam-style parameter-group values from older jobs.
   Tune `optimizer_params.d_coef` (`0.5`–`2.0`) instead, never `lr`.
2. **`lr_scheduler: "constant"`.** Prodigy+ is schedule-free; a decaying
   scheduler (`cosine`, `linear`, …) fights its internal schedule. The trainer
   prints a warning if `prodigy_plus` is paired with a non-constant scheduler.
   `constant` is the repo default, so usually you just don't override it.
3. **EMA is auto-disabled.** Schedule-free already keeps an internal weight
   average; layering ai-toolkit's EMA on top corrupts the averaged weights during
   save/sample. If `ema_config.use_ema: true` is set with `prodigy_plus`, the
   trainer logs a warning and disables EMA. Leave `use_ema: false`.

## 3. Minimal config

```yaml
train:
  optimizer: "prodigy_plus"
  lr: 1.0                  # prodigy adapts the LR; keep at 1.0
  lr_scheduler: "constant" # schedule-free; do not decay
  optimizer_params:        # all optional; values below are the trainer defaults
    weight_decay: 0.01
    d_coef: 1.0
  ema_config:
    use_ema: false         # auto-disabled for schedule-free anyway
```

Full runnable example: `config/examples/train_lora_flux_24gb_prodigy_plus.yaml`.

## 4. `optimizer_params` reference

Everything in `optimizer_params` is forwarded verbatim to
`ProdigyPlusScheduleFree(...)`, so any constructor argument is settable.

Trainer-applied defaults (injected via `setdefault`, so your config overrides
them): `betas=[0.9, 0.999]`, `weight_decay=0.01`, `d_coef=1.0`.

| Param | Default | Meaning |
|---|---|---|
| `d_coef` | `1.0` | Scales the adapted step size. Main tuning knob (`0.5`–`2.0`). |
| `weight_decay` | `0.01` | Decoupled weight decay. |
| `betas` | `[0.9, 0.999]` | Adam moment decay rates. |
| `prodigy_steps` | `0` | `>0` freezes the adapted LR after N steps (stops `d` growth). |
| `use_cautious` | `false` | Cautious (sign-aligned) updates; can reach quality earlier. |
| `schedulefree_c` | `0` | Schedule-free averaging window (`0` = optimizer default). |
| `use_orthograd` | `false` | Orthogonal-gradient regularization. |

(`ProdigyPlusScheduleFree` has more flags — `use_stableadamw`, `use_adopt`,
`use_grams`, `factored`, `stochastic_rounding`, etc. — all overridable.)

## 5. How schedule-free changes the training lifecycle

A schedule-free optimizer holds the model on the raw **train-mode** weights
during optimization and only materializes the **averaged (eval-mode)** weights
when `optimizer.eval()` is called. ai-toolkit handles this automatically:

- **Sampling** (`sample()`): switches to eval weights before generating, back to
  train after — samples reflect inference quality.
- **Saving** (`save()`): switches to eval weights before writing the
  `.safetensors` and `optimizer.pt`, back to train after — the exported LoRA is
  the averaged weights.
- **Resume**: on restart the LoRA weights (eval/averaged) are reloaded from the
  latest `.safetensors`, the optimizer state is loaded from `optimizer.pt`, then
  `.train()` reconstructs the raw train-mode weights. Verified exact at the
  resume point.

These toggles have **zero effect on the training trajectory** (`eval()`→`train()`
round-trips `p.data` exactly), so periodic sampling/saving does not perturb
training. `accelerate` wraps the optimizer in `AcceleratedOptimizer`, which
forwards `.train()`/`.eval()`, so the toggles work after `accelerator.prepare`.

## 6. Where it is wired

`toolkit/optimizer.py`
- `get_optimizer()` — the `prodigy_plus` branch is placed before the generic
  `startswith("prodigy")` branch so it is not shadowed. It normalizes stale
  parameter-group LRs and calls `optimizer.train()` after construction.
- `optimizer_requires_eval_mode(optimizer_type)` is the single source of truth
  for whether an optimizer needs eval/train toggling.

`jobs/process/BaseSDTrainProcess.py`
- `self._optimizer_is_schedule_free` tracks the lifecycle requirement.
- `_optimizer_eval()` / `_optimizer_train()` guard the mode transitions.
- Sampling and saving switch to averaged weights, then restore train weights.
- Resume calls `.train()` after loading optimizer state.
- `setup_ema()` disables a second EMA, and scheduler setup warns when the
  configured scheduler is not constant.

`ui/src/app/jobs/new/SimpleJob.tsx` — the optimizer dropdown includes Prodigy+
and selects `lr=1.0`, `weight_decay=0.01`, and `use_ema=false` with it.

Dependency: tested `prodigy-plus-schedule-free==2.0.1` in
`requirements_base.txt`.

## 7. Edge cases / behavior notes

- **Switching optimizer type on resume** (e.g. AdamW → Prodigy+): the old
  `optimizer.pt` fails to load into the new optimizer; the trainer catches the
  error, logs it, and starts a fresh optimizer from the loaded weights. No
  corruption.
- **Multi-GPU**: only the main process saves/samples and therefore toggles
  eval/train; the toggle is balanced within `save()`/`sample()` with no training
  step in between, so all ranks are in train mode before the next step.
- **Stale UI jobs**: if an older job supplies an Adam-style per-group LR below
  `0.1`, the backend raises it to the Prodigy+ relative LR and prints a warning.
- **bf16**: `ProdigyPlusScheduleFree` uses stochastic rounding by default, making
  bf16 runs slightly non-deterministic (expected, beneficial).
- **Trainer scope**: the schedule-free save/sample/resume lifecycle is wired in
  `BaseSDTrainProcess` (`sd_trainer`, including Krea2). Specialized VAE, ESRGAN,
  and critic training processes should not select Prodigy+ until they gain the
  same lifecycle hooks.

## 8. Quick agent checklist

When a user asks to "use Prodigy+ in ai-toolkit":
- set `optimizer: prodigy_plus`, `lr: 1.0`, `lr_scheduler: constant`,
  `ema_config.use_ema: false`;
- tune via `optimizer_params.d_coef`, not `lr`.
