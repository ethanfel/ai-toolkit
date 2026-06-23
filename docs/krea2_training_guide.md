# Training a Krea 2 LoRA with ai-toolkit (`krea2` branch)

This branch adds a `krea_2` model architecture so you can train LoRAs on **Krea 2**, the
flow-matching MMDiT text-to-image model from [Krea](https://krea.ai). This guide explains how to
get the weights, write a training YAML, and run it.

> **RAW vs Turbo.** Krea 2 ships as two checkpoints:
> - **Krea 2 RAW / Base** — undistilled base model. **This is what you train on** (fine-tuning / LoRA).
> - **Krea 2 Turbo** — 8-step distilled model for fast inference.
>
> Train your LoRA on **RAW/Base**, then load it onto **Turbo** for fast, high-quality generation.
> (RAW is not meant for direct inference — its own samples look rough; that's expected.)

---

## 1. Getting the model weights (read this first)

There is **no ready-to-use diffusers repo on the Hub** for Krea 2:
- The official `krea/Krea-2-Raw` / `krea/Krea-2-Turbo` repos are **gated/private** (you must be granted access and use an HF token).
- The public community repo `CalamitousFelicitousness/Krea-2-Base-Diffusers` is in the **original krea (`mmdit.py`) weight naming**, which diffusers' `Krea2Transformer2DModel` **cannot load directly** — it must be converted first.

So you have two options.

### Option A — Use a hosted, already-converted repo (enables auto-download) ✅ recommended

If a **converted** model is on the Hub, just put its repo id in `name_or_path` and ai-toolkit
**auto-downloads it** on first run (diffusers `from_pretrained` handles the download + caching):

```yaml
model:
  name_or_path: "your-hf-username/Krea-2-Base-Diffusers-aitk"   # a CONVERTED repo
  arch: "krea_2"
```

To create such a repo, convert once (Option B) and `hf upload` the output folder to your account.
After that, anyone using your config/branch/Docker image gets auto-download with no manual steps.

> ⚠️ Do **not** point `name_or_path` at `CalamitousFelicitousness/Krea-2-Base-Diffusers` directly —
> it will download but fail to load (wrong weight key naming). It must be converted first.

### Option B — Convert locally from the public community repo

One-time conversion (needs ~26 GB download + ~26 GB for the output):

```bash
# 1. download the community repo (original-naming weights + correct vae/text_encoder/tokenizer)
hf download CalamitousFelicitousness/Krea-2-Base-Diffusers \
    --local-dir /path/to/krea2-base-src

# 2. convert the transformer into diffusers layout
python scripts/convert_krea2_community_to_diffusers.py \
    --src /path/to/krea2-base-src \
    --dst /path/to/krea2-base-diffusers
```

Then point `name_or_path` at the `--dst` folder:

```yaml
model:
  name_or_path: "/path/to/krea2-base-diffusers"
  arch: "krea_2"
```

> The converter renames the transformer keys, reshapes the per-block modulation table, and drops the
> two final-layer `up`/`down` weights that diffusers' Krea2 port doesn't implement (this matches what
> the official diffusers weights would be). The `vae`, `text_encoder`, `tokenizer` and `scheduler`
> subfolders are already correct and are reused as-is.

### Does it auto-download?

- **Local path** (Option B output) → no download, loads from disk.
- **Hub repo id of a converted model** (Option A) → **yes, auto-downloads** and caches under
  `~/.cache/huggingface` (or `$HF_HOME`). For a gated repo, set `HF_TOKEN` / run `hf auth login` first.

---

## 2. Pick a config for your VRAM

Two ready-made examples are in `config/examples/`:

| GPU VRAM | Example | Settings |
|---|---|---|
| ~24–48 GB (e.g. RTX 4090/5090) | `train_lora_krea2_32gb.yaml` | `quantize: true` (qfloat8) + `quantize_te: true` |
| ~80–96 GB (e.g. RTX 6000 Pro) | `train_lora_krea2_96gb.yaml` | `quantize: false` (full bf16) |

The transformer is **12.8B params**: full bf16 needs a big card; qfloat8 quantization loads in **~17 GB**
and trains comfortably on a 32 GB card.

Copy one and edit it (next section).

---

## 3. Writing the training YAML — field by field

A minimal, annotated config:

```yaml
---
job: extension
config:
  name: "my_krea2_lora_v1"          # output folder + filename
  process:
    - type: 'sd_trainer'
      training_folder: "output"
      device: cuda:0
      # trigger_word: "p3r5on"      # optional; appended to captions (don't use with cached text embeds)

      network:
        type: "lora"
        linear: 16                  # LoRA rank. 16–32 typical; raise for high-frequency styles
        linear_alpha: 16            # usually == rank

      save:
        dtype: float16
        save_every: 250
        max_step_saves_to_keep: 4

      datasets:
        - folder_path: "/path/to/images"   # images + same-name .txt captions
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          cache_latents_to_disk: true
          resolution: [ 512, 768, 1024 ]   # multi-res bucketing; must be multiples of 16

      train:
        batch_size: 1
        cache_text_embeddings: true        # recommended; frees the text encoder after caching
        steps: 2000                        # 500–4000 typical
        gradient_accumulation: 1
        train_unet: true
        train_text_encoder: false          # text encoder training not supported
        gradient_checkpointing: true       # keep on unless you have tons of VRAM
        noise_scheduler: "flowmatch"       # Krea 2 is flow-matching
        optimizer: "adamw8bit"             # or prodigy_plus (available on this branch)
        lr: 1e-4                           # 3e-4–7e-4 with constant schedule also works well
        dtype: bf16

      model:
        name_or_path: "your-username/Krea-2-Base-Diffusers-aitk"  # converted repo OR local dir
        arch: "krea_2"                     # <-- selects this architecture
        # --- VRAM: uncomment the quant block on <48 GB cards ---
        quantize: true
        qtype: "qfloat8"
        quantize_te: true
        qtype_te: "qfloat8"
        low_vram: false                    # set true only if you OOM at high res/batch (slower)

      sample:
        sampler: "flowmatch"               # must match train.noise_scheduler
        sample_every: 250
        width: 1024
        height: 1024
        guidance_scale: 4                  # standard CFG scale (this branch converts it to Krea's convention)
        sample_steps: 25
        seed: 42
        walk_seed: true
        prompts:
          - "a photo of [trigger] in a field of flowers"
          - "a cinematic portrait of [trigger], studio lighting"
meta:
  name: "[name]"
  version: '1.0'
```

### Krea 2-specific notes

- **`arch: "krea_2"`** is the only thing that selects this model — no UI/dropdown change needed.
- **Resolutions must be multiples of 16** (vae_scale_factor 8 × patch 2). 512/768/1024 all qualify.
- **`guidance_scale`** in `sample` is a *standard* CFG scale; the trainer converts it to Krea 2's
  `cond + (scale−1)·(cond−uncond)` convention internally, so use it like any other model (≈3–5).
- **Training samples on RAW look rough** — that's normal. Judge quality by loading the LoRA on Turbo.
- **Captioning a style:** describe what you *don't* want baked in, omit the stylistic parts you *do*
  want learned, and add a descriptive trigger phrase (e.g. `"... hand-drawn children's book illustration"`)
  rather than a rare token. For a subject/character, a trigger word is fine.
- **Capacity:** the default rank 16–32 on the full linear set fits most styles. For long runs on
  high-frequency styles, raise the rank.

---

## 4. Running training

```bash
python run.py config/examples/train_lora_krea2_32gb.yaml
```

Or with the dedicated Docker image (ships the code + the Krea2-capable diffusers):

```bash
docker run --gpus all --rm \
  -e HF_TOKEN=$HF_TOKEN \
  -v /path/to/data:/data \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/ethanfel/ai-toolkit:krea2 \
  python run.py /data/my_krea2_config.yaml
```

The image contains the code and a Krea2-capable diffusers, but **not the weights** — they still
download (Option A repo) or must be mounted (Option B local folder).

---

## 5. Using your trained LoRA

Train on RAW, generate on **Turbo** (8 steps, no CFG):

```python
import torch
from diffusers import Krea2Pipeline

pipe = Krea2Pipeline.from_pretrained("your-username/Krea-2-Turbo-Diffusers", torch_dtype=torch.bfloat16).to("cuda")
pipe.load_lora_weights("output/my_krea2_lora_v1/my_krea2_lora_v1.safetensors")

image = pipe(
    prompt="a photo of p3r5on in a field of flowers",
    height=1024, width=1024,
    num_inference_steps=8, guidance_scale=0.0, mu=1.15,   # Turbo recipe
    generator=torch.Generator("cuda").manual_seed(0),
).images[0]
image.save("out.png")
```

The saved LoRA uses the `diffusion_model.` key prefix, so it loads into diffusers' `Krea2Pipeline`
and ComfyUI-style loaders.

> **Turbo also needs conversion.** Like Base, the public `CalamitousFelicitousness/Krea-2-Turbo-Diffusers`
> repo is in original-krea naming. Run the same converter on it (`--src <turbo download> --dst <turbo out>`)
> and use the converted folder/repo id above. (Inference uses plain diffusers, not ai-toolkit.)
