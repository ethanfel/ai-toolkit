# Krea 2 training on the upstream-based fork

This branch uses ai-toolkit's upstream-native Krea 2 implementation and keeps
this fork's Prodigy+ support and one-click model downloads.

## Model downloads

Use the official public Raw checkpoint for training:

```yaml
model:
  name_or_path: "krea/Krea-2-Raw"
  arch: krea2
```

On first use, ai-toolkit downloads `raw.safetensors` from
[`krea/Krea-2-Raw`](https://huggingface.co/krea/Krea-2-Raw), the Qwen3-VL text
encoder from `Qwen/Qwen3-VL-4B-Instruct`, and the VAE from `Qwen/Qwen-Image`.
The normal Hugging Face cache is reused on later runs. `HF_TOKEN` is forwarded
when one is configured.

The upstream loader also accepts:

- `krea/Krea-2-Turbo` (automatically selects `turbo.safetensors`);
- a local `.safetensors` file;
- a local directory containing one `.safetensors` file; or
- a custom Hub repository plus
  `model.model_kwargs.checkpoint_filename`.

The old fork used `arch: krea_2` with a converted Diffusers directory or
`ethanfel/Krea-2-Base-Diffusers`. The `krea_2` name remains a backend alias so
it cannot silently fall back to an unrelated model, but the old checkpoint
layout is deliberately rejected with a migration error. The native and legacy
layouts have different module names, so silently resuming an old output could
apply incompatible LoRA state.

To migrate, copy the job, give it a **new output name**, and change both fields:

```yaml
model:
  name_or_path: "krea/Krea-2-Raw"
  arch: krea2
```

Do not point the native implementation at a locally converted Diffusers
directory. Existing LoRAs need a separate key-conversion step before reuse.

## Ready-made configs

- `config/examples/train_lora_krea2_32gb.yaml`: qfloat8 transformer/text
  encoder and Prodigy+ for a ~32 GB GPU (`low_vram: true` for smaller cards).
- `config/examples/train_lora_krea2_96gb.yaml`: full bf16 for a large-memory GPU.
- `config/examples/train_lora_krea2_edit_96gb.yaml`: image-edit LoRA training
  with Prodigy+; its comments show both stock and Identity Edit v1.2 profiles.

Edit the dataset path, output name, captions, and samples, then run:

```bash
python run.py config/examples/train_lora_krea2_32gb.yaml
```

Krea 2 resolutions must be divisible by 16 (VAE scale 8 x patch size 2). Raw is
the undistilled training checkpoint and relies on CFG for previews, so a real
negative prompt, guidance around 3.5, and the full sampling schedule produce
more representative samples. A LoRA trained on Raw can be used with Turbo.

## Identity Edit v1.2 compatibility

Krea2 edit conditioning has two incompatible contracts. The default
`ai_toolkit_t0` profile keeps current upstream behavior. To train or continue a
[`conradlocke/krea2-identity-edit`](https://huggingface.co/conradlocke/krea2-identity-edit)
v1.2-compatible adapter, opt in explicitly:

```yaml
model:
  name_or_path: "krea/Krea-2-Raw"
  arch: krea2
  model_kwargs:
    edit: true
    edit_profile: identity_edit_v12
    kv_cache: false
    vlm_longest_side: 768
```

That profile uses the recovered Identity Edit contract: bare consecutive Qwen
vision blocks, clean references before the noisy target, the current target
timestep for every token span, target-only loss, v1.2 pixel-space FIT, and
centered stride-1 reference positions. Reference images stay in their supplied
order; for two-reference training use scene/base first and identity/subject
second. Keep `full_size_control_images: true` and
`cache_text_embeddings: false` so FIT and per-example Qwen grounding are not
bypassed.

The profile deliberately rejects `kv_cache: true`: Identity Edit references use
the current timestep and full joint attention, so their features are not
step-invariant. Its official inference-only prompt weights, masks, schedules,
and `ref_boost` are not part of this trainer.

The public nodes expose the conditioning graph and geometry, but the original
optimizer, learning rate, stage sizes, and Qwen resize-jitter distribution were
not published. This implementation therefore uses a deterministic 768-pixel
longest-side cap by default; override `vlm_longest_side` only as an intentional
experiment.

## Prodigy+

The preserved optimizer name is `prodigy_plus`:

```yaml
train:
  optimizer: prodigy_plus
  lr: 1.0
  lr_scheduler: constant
  optimizer_params:
    weight_decay: 0.01
    d_coef: 1.0
  ema_config:
    use_ema: false
```

Prodigy+ adapts its own step size and maintains schedule-free averaged weights.
The trainer switches to averaged weights for samples and saves, restores train
weights afterward, and reconstructs them after resume. It also disables a
second EMA and warns about non-constant schedulers. Selecting Prodigy+ in the UI
sets the required LR, weight decay, and EMA values; the backend additionally
normalizes stale Adam-style parameter-group LRs from older jobs.

See [`docs/prodigy_plus.md`](prodigy_plus.md) for the detailed optimizer guide.

## Docker

The standard Compose file already persists the Hugging Face cache. The GHCR
Compose file also forwards `HF_TOKEN` when it is present. The Dockerfile accepts
`GIT_REPO` and `GIT_COMMIT` build arguments, so a fork image can be built from
this branch.

The Docker dependency layers come from the local build context while source is
cloned from Git. Therefore the checkout must be clean, the selected commit must
match `HEAD`, and that exact commit must already be pushed. The helper script
enforces those conditions:

```bash
./build_and_push_docker_ghcr_krea2
```
