# Krea 2 LoRA Training Support — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a new `krea_2` model architecture to ai-toolkit so users can train LoRAs on Krea 2 (Base), on a dedicated `krea2` git branch with its own Docker image, keeping `main`/`prodigy`/`krea2` separate.

**Architecture:** Krea 2 is a flow-matching MMDiT latent-diffusion model that is architecturally a close cousin of Qwen-Image: it shares the `AutoencoderKLQwenImage` VAE, a `FlowMatchEulerDiscreteScheduler`, and a Qwen-VL family text encoder. We therefore add a new `Krea2Model(BaseModel)` class adapted from the existing `QwenImageModel`, swapping in `diffusers.Krea2Transformer2DModel` + `transformers.Qwen3VLModel` and adjusting the transformer call (RoPE `position_ids` + `encoder_attention_mask` instead of qwen's `img_shapes`/`txt_seq_lens`). It plugs into ai-toolkit's pluggable model registry (`arch = "krea_2"`), so no UI change is needed. The Krea2 transformer/pipeline only exist in recent diffusers `main`, so this branch bumps the pinned diffusers commit.

**Tech Stack:** Python, PyTorch, diffusers (bumped to a `main` commit with `transformer_krea2.py`), transformers 5.5.3 (`Qwen3VLModel`), ai-toolkit `BaseModel` framework, RTX 6000 Pro (~96 GB → full bf16, no quantization required).

**Target weights:** `CalamitousFelicitousness/Krea-2-Base-Diffusers` (diffusers layout: `transformer/`, `vae/`, `text_encoder/`, `tokenizer/`, `scheduler/`).

**Python interpreter for all commands:** `/media/p5/miniforge3/envs/ai_toolkit/bin/python` (the repo has no in-repo venv; the base PATH python lacks deps).

---

## Background the implementer needs

ai-toolkit registers models as subclasses of `toolkit/models/base_model.py::BaseModel` with a class attribute `arch = "<name>"`. The factory `toolkit/util/get_model.py::get_model_class()` matches `config.arch` to a class. Extension models live in `extensions_built_in/diffusion_models/<name>/` and are exported via `extensions_built_in/diffusion_models/__init__.py`'s `AI_TOOLKIT_MODELS` list. The UI discovers models dynamically — **no UI/dropdown edit is required**; a YAML `model.arch: "krea_2"` is enough.

The closest existing template is `extensions_built_in/diffusion_models/qwen_image/qwen_image.py` (487 lines). Read it in full before starting — Krea2 reuses ~80% of it verbatim (VAE encode/decode, save_model, lora key conversion, scheduler pattern, structure).

### Krea2 vs Qwen-Image — the only real differences

| Concern | Qwen-Image | Krea2 | Action |
|---|---|---|---|
| VAE | `AutoencoderKLQwenImage` | `AutoencoderKLQwenImage` | **identical — copy verbatim** |
| Scheduler | `FlowMatchEulerDiscreteScheduler` | `FlowMatchEulerDiscreteScheduler` (different config values, see Krea2 `scheduler/scheduler_config.json`) | copy pattern, use Krea2 config dict |
| Text encoder | `Qwen2_5_VLForConditionalGeneration` | `Qwen3VLModel` (`transformers.Qwen3VLModel`, in transformers 5.5.3) | swap class |
| Tokenizer | `Qwen2Tokenizer` | `Qwen2Tokenizer` | identical |
| Transformer | `QwenImageTransformer2DModel` | `Krea2Transformer2DModel` (diffusers `main` only) | swap class + change forward call |
| Pipeline | `QwenImagePipeline` | `Krea2Pipeline` (diffusers `main` only) | swap class |
| Prompt embeds | 2D `(B, seq, dim)` | **stacked** `(B, seq, num_layers, 2560)` | reuse `Krea2Pipeline.encode_prompt` — carries shape automatically |
| Transformer forward | `img_shapes`, `txt_seq_lens`, `encoder_hidden_states_mask`, `guidance`, `timestep/1000` | `position_ids`, `encoder_attention_mask`, `timestep` (scale TBD) | **rewrite `get_noise_prediction`** |
| Latent packing | view/permute/reshape, patch 2 | same | copy verbatim |

### Krea2 `Krea2Transformer2DModel.forward` (diffusers main) — verbatim signature

```python
def forward(
    self,
    hidden_states: torch.Tensor,          # packed latents (B, img_seq, in_channels=64)
    encoder_hidden_states: torch.Tensor,  # stacked text embeds (B, txt_seq, num_layers, 2560)
    timestep: torch.Tensor,
    position_ids: torch.Tensor,           # (txt_seq + img_seq, 3) RoPE coords
    encoder_attention_mask: torch.Tensor | None = None,
    return_dict: bool = True,
):
```

### Krea2Pipeline helpers to mirror (from diffusers `pipelines/krea2/pipeline_krea2.py`)

```python
# position ids: text tokens at origin, image tokens carry (0, h, w)
@staticmethod
def prepare_position_ids(text_seq_len, grid_height, grid_width, device):
    text_ids = torch.zeros(text_seq_len, 3, device=device)
    image_ids = torch.zeros(grid_height, grid_width, 3, device=device)
    image_ids[..., 1] = torch.arange(grid_height, device=device)[:, None]
    image_ids[..., 2] = torch.arange(grid_width, device=device)[None, :]
    image_ids = image_ids.reshape(grid_height * grid_width, 3)
    return torch.cat([text_ids, image_ids], dim=0)

# transformer call site in the pipeline denoising loop:
noise_pred = self.transformer(
    hidden_states=latents,
    encoder_hidden_states=prompt_embeds,
    timestep=timestep,
    position_ids=position_ids,
    encoder_attention_mask=prompt_embeds_mask,
    return_dict=False,
)[0]
```

Latent pack/unpack is identical to qwen (patch `p=2`): `view(B,C,H//p,p,W//p,p) → permute(0,2,4,1,3,5) → reshape(B,(H//p)*(W//p),C*p*p)`.

---

## Risks & open questions (resolve in Phase 0 before building)

1. **Config-schema mismatch (HIGH):** The conversion repo's `transformer/config.json` uses old keys (`features`, `heads`, `kvheads`, `layers`, `patch`, `tdim`, `txtdim`, …) while diffusers-main `Krea2Transformer2DModel.__init__` uses `in_channels`, `num_layers`, `attention_head_dim`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size`, `timestep_embed_dim`, `text_hidden_dim`, …. Values map 1:1 and the diffusers **defaults equal the Krea2-Large values**, but `from_pretrained` may error on unknown keys or silently use defaults that mismatch FFN sizes (`multiplier: 4` vs `intermediate_size`/`text_intermediate_size`). **Must verify weights load with no shape errors.** If it fails: pin diffusers to the exact commit the conversion targeted, OR re-convert from the bucket `raw.safetensors` with diffusers' conversion script, OR find/produce a conversion whose config matches diffusers main.
2. **diffusers bump (HIGH):** Pinned commit `dc8d9032…` lacks `diffusers.utils.torch_utils.maybe_adjust_dtype_for_device` (imported by `transformer_krea2.py`). Must bump diffusers `main`. Bumping risks regressions in other ai-toolkit models — but this is isolated on the `krea2` branch/image, so blast radius is contained. Smoke-test at least flux/qwen_image load after the bump (load-only, no train).
3. **timestep scale (MED):** qwen passes `timestep/1000`; the Krea2 pipeline passes `timestep` directly. Confirm whether `Krea2Transformer2DModel` expects 0–1 sigma-scaled or 0–1000. Mirror exactly what `pipeline_krea2.py` feeds the transformer.
4. **position_ids batching (MED):** `prepare_position_ids` builds one `(seq,3)` tensor; confirm whether the transformer wants it unbatched (shared) or per-batch `(B,seq,3)`, and how variable text lengths within a batch are handled (ai-toolkit pads). Mirror the pipeline.
5. **LoRA target modules (MED):** `target_lora_modules = ["Krea2Transformer2DModel"]` and `get_transformer_block_names()` must match real module names. Inspect `named_modules()` of the loaded transformer; `_no_split_modules = ["Krea2TransformerBlock", "Krea2TextFusionBlock", "Krea2FinalLayer"]`. Confirm the block-list attribute name (likely `transformer_blocks`).
6. **bucket divisibility (LOW):** qwen returns `16*2`. Krea2 uses the same VAE; confirm `vae_scale_factor` and set `vae_scale_factor * patch_size`. Default to qwen's value, verify against `pipeline_krea2.py` `_unpack_latents`.
7. **LoRA save key convention (LOW):** qwen maps `transformer.` ↔ `diffusion_model.`. Keep the same unless Krea2 ComfyUI/inference expects a different prefix.

---

## Phase 0 — De-risk: branch, diffusers bump, weight-load verification

> Do this FIRST. If 0.3 fails, stop and resolve the config-schema risk before writing any model code.

### Task 0.1: Branch is already created

`krea2` branched off `prodigy`. Confirm:

Run: `git -C /media/p5/ai-toolkit branch --show-current`
Expected: `krea2`

### Task 0.2: Bump diffusers to a main commit that has Krea2

**Files:** Modify `requirements_base.txt:3`

**Step 1:** Find a recent diffusers `main` commit SHA that contains `src/diffusers/models/transformers/transformer_krea2.py` and `src/diffusers/pipelines/krea2/` (check https://github.com/huggingface/diffusers/commits/main). Record the SHA.

**Step 2:** Edit `requirements_base.txt` line 3:
```
git+https://github.com/huggingface/diffusers.git@<NEW_SHA_WITH_KREA2>
```

**Step 3:** Install into the env:
Run: `/media/p5/miniforge3/envs/ai_toolkit/bin/python -m pip install -U "git+https://github.com/huggingface/diffusers.git@<NEW_SHA_WITH_KREA2>"`

**Step 4:** Verify the classes import:
Run:
```
/media/p5/miniforge3/envs/ai_toolkit/bin/python -c "import diffusers; print(diffusers.__version__); from diffusers import Krea2Transformer2DModel, Krea2Pipeline, AutoencoderKLQwenImage; from transformers import Qwen3VLModel, Qwen2Tokenizer; print('all imports OK')"
```
Expected: prints a version and `all imports OK`.

**Step 5:** Regression smoke — confirm an existing model class still imports/loads its diffusers deps:
Run: `/media/p5/miniforge3/envs/ai_toolkit/bin/python -c "from diffusers import QwenImageTransformer2DModel, FluxTransformer2DModel; print('existing diffusers classes OK')"`
Expected: `existing diffusers classes OK`

**Step 6: Commit**
```bash
git add requirements_base.txt
git commit -m "krea2: bump diffusers to a main commit with Krea2 transformer/pipeline"
```

### Task 0.3: Verify Krea2 weights load with the bumped diffusers (CRITICAL GATE)

**Files:** Create `scratch/krea2_load_check.py` (throwaway, delete after; not committed)

**Step 1:** Write the load check:
```python
import torch
from diffusers import Krea2Transformer2DModel, AutoencoderKLQwenImage
from transformers import Qwen3VLModel, Qwen2Tokenizer
REPO = "CalamitousFelicitousness/Krea-2-Base-Diffusers"
print("loading transformer...")
t = Krea2Transformer2DModel.from_pretrained(REPO, subfolder="transformer", torch_dtype=torch.bfloat16)
print("transformer OK:", type(t).__name__, "| config:", dict(t.config))
print("loading vae..."); v = AutoencoderKLQwenImage.from_pretrained(REPO, subfolder="vae", torch_dtype=torch.bfloat16); print("vae OK")
print("loading text encoder..."); te = Qwen3VLModel.from_pretrained(REPO, subfolder="text_encoder", torch_dtype=torch.bfloat16); print("te OK:", type(te).__name__)
print("loading tokenizer..."); tok = Qwen2Tokenizer.from_pretrained(REPO, subfolder="tokenizer"); print("tok OK")
# inspect block names for LoRA targeting
blocks = [n for n,_ in t.named_modules() if n.count(".")<=1][:40]
print("top modules:", blocks)
```

**Step 2:** Run it (downloads ~26 GB once; needs network + disk):
Run: `cd /media/p5/ai-toolkit && /media/p5/miniforge3/envs/ai_toolkit/bin/python scratch/krea2_load_check.py`
Expected: all four `OK` lines, NO shape/size-mismatch errors, NO "unexpected/missing keys" warnings about the transformer. Record the printed config and top module names.

**Step 3 (GATE):** If it loads cleanly → proceed to Phase 1. If it errors on config keys or weight shapes → resolve Risk #1 (try pinning diffusers to the conversion's target commit, or re-convert from the `krea-community/krea-2` bucket `raw.safetensors`) and re-run before continuing.

**Step 4:** Record findings (timestep scale, block names, vae_scale_factor) in this plan's "Resolved facts" section at the bottom, then `rm scratch/krea2_load_check.py`.

---

## Phase 1 — Model class skeleton + registration (no weights yet)

### Task 1.1: Create the package + class skeleton

**Files:**
- Create: `extensions_built_in/diffusion_models/krea_2/__init__.py`
- Create: `extensions_built_in/diffusion_models/krea_2/krea_2_model.py`

**Step 1:** `__init__.py`:
```python
from .krea_2_model import Krea2Model
```

**Step 2:** `krea_2_model.py` minimal skeleton (copy imports from qwen_image, swap classes):
```python
import os
from typing import List, Optional
import torch, yaml
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.models.base_model import BaseModel
from toolkit.basic import flush
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler
from toolkit.accelerator import unwrap_model
from diffusers import Krea2Pipeline, Krea2Transformer2DModel, AutoencoderKLQwenImage
from transformers import Qwen3VLModel, Qwen2Tokenizer

# from Krea2 scheduler/scheduler_config.json
scheduler_config = {
    "base_image_seq_len": 256, "base_shift": 0.5, "invert_sigmas": False,
    "max_image_seq_len": 6400, "max_shift": 1.15, "num_train_timesteps": 1000,
    "shift": 1.0, "shift_terminal": None, "stochastic_sampling": False,
    "time_shift_type": "exponential", "use_dynamic_shifting": True,
}

class Krea2Model(BaseModel):
    arch = "krea_2"

    def __init__(self, device, model_config: ModelConfig, dtype="bf16",
                 custom_pipeline=None, noise_scheduler=None, **kwargs):
        super().__init__(device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs)
        self.is_flow_matching = True
        self.is_transformer = True
        self.target_lora_modules = ["Krea2Transformer2DModel"]  # verify in Phase 0.3

    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        return 16 * 2  # verify vae_scale_factor * patch_size in Phase 0.3

    def get_base_model_version(self):
        return "krea_2"
```

**Step 3:** Register in `extensions_built_in/diffusion_models/__init__.py`: add `from .krea_2 import Krea2Model` with the other imports and add `Krea2Model` to the `AI_TOOLKIT_MODELS` list.

**Step 4:** Verify dispatch (no weights, no GPU needed):
Run:
```
cd /media/p5/ai-toolkit && /media/p5/miniforge3/envs/ai_toolkit/bin/python -c "from toolkit.util.get_model import get_model_class; from toolkit.config_modules import ModelConfig; print(get_model_class(ModelConfig(name_or_path='x', arch='krea_2')))"
```
Expected: `<class '...Krea2Model'>`

**Step 5: Commit**
```bash
git add extensions_built_in/diffusion_models/krea_2 extensions_built_in/diffusion_models/__init__.py
git commit -m "krea2: register Krea2Model arch skeleton"
```

---

## Phase 2 — `load_model()`

**Files:** Modify `extensions_built_in/diffusion_models/krea_2/krea_2_model.py`

**Step 1:** Implement `load_model()` by copying qwen_image's `load_model` and applying: transformer → `Krea2Transformer2DModel`, text encoder → `Qwen3VLModel` (no `.model.visual = None` unless Krea2's encoder exposes a visual tower — verify and drop it if present to save VRAM), pipeline → `Krea2Pipeline`. Keep the `quantize`/`low_vram`/`layer_offloading` branches (harmless; user runs full bf16 so they're inert). VAE load + `self.vae/self.text_encoder/self.tokenizer/self.model/self.pipeline` assignment identical to qwen.

Key skeleton (abbreviated; fill from qwen template):
```python
def load_model(self):
    dtype = self.torch_dtype
    model_path = self.model_config.name_or_path
    base = self.model_config.extras_name_or_path or model_path
    transformer = Krea2Transformer2DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=dtype)
    tokenizer = Qwen2Tokenizer.from_pretrained(base, subfolder="tokenizer")
    text_encoder = Qwen3VLModel.from_pretrained(base, subfolder="text_encoder", torch_dtype=dtype)
    vae = AutoencoderKLQwenImage.from_pretrained(base, subfolder="vae", torch_dtype=dtype)
    self.noise_scheduler = Krea2Model.get_train_scheduler()
    pipe = Krea2Pipeline(scheduler=self.noise_scheduler, text_encoder=None, tokenizer=tokenizer, vae=vae, transformer=None)
    pipe.text_encoder = text_encoder; pipe.transformer = transformer
    pipe.transformer.to(self.device_torch)
    text_encoder.to(self.device_torch, dtype=dtype); text_encoder.requires_grad_(False); text_encoder.eval()
    self.vae = vae; self.text_encoder = [text_encoder]; self.tokenizer = [tokenizer]
    self.model = pipe.transformer; self.pipeline = pipe
```

**Step 2:** Verify load on the GPU box:
Run a 1-off: instantiate `Krea2Model` via the trainer config path or a small script that calls `.load_model()` and prints `type(self.model)`, `type(self.vae)`, `type(self.text_encoder[0])`.
Expected: Krea2Transformer2DModel / AutoencoderKLQwenImage / Qwen3VLModel, no errors.

**Step 3: Commit** `git commit -am "krea2: implement load_model"`

---

## Phase 3 — `get_prompt_embeds()` (reuse pipeline encode_prompt)

**Files:** Modify `krea_2_model.py`

**Step 1:** Implement, mirroring qwen but tolerant of Krea2's stacked embeds:
```python
def get_prompt_embeds(self, prompt) -> PromptEmbeds:
    if self.pipeline.text_encoder.device != self.device_torch:
        self.pipeline.text_encoder.to(self.device_torch)
    prompt_embeds, prompt_embeds_mask = self.pipeline.encode_prompt(
        prompt, device=self.device_torch, num_images_per_prompt=1)
    if prompt_embeds_mask is None:
        prompt_embeds_mask = torch.ones(prompt_embeds.shape[:2], device=prompt_embeds.device, dtype=torch.int64)
    pe = PromptEmbeds(prompt_embeds); pe.attention_mask = prompt_embeds_mask
    return pe

def get_model_has_grad(self): return False
def get_te_has_grad(self): return False
```

**Step 2:** Verify shapes:
Run a script: `pe = model.get_prompt_embeds(["a cat"])`; print `pe.text_embeds.shape` and `pe.attention_mask.shape`.
Expected: text_embeds rank 4 `(1, seq, num_layers, 2560)` (or whatever 0.3 recorded), mask `(1, seq)`. No error.

**Step 3: Commit** `git commit -am "krea2: implement get_prompt_embeds"`

---

## Phase 4 — `get_noise_prediction()` (the core delta)

**Files:** Modify `krea_2_model.py`

**Step 1:** Implement, packing latents like qwen but calling the transformer the Krea2 way (position_ids + encoder_attention_mask). Mirror `pipeline_krea2.py` exactly for `position_ids`, the timestep scale, and arg names:
```python
def get_noise_prediction(self, latent_model_input, timestep, text_embeddings: PromptEmbeds, **kwargs):
    self.model.to(self.device_torch)
    B, C, H, W = latent_model_input.shape
    ps = 2  # confirm self.transformer.config exposes patch; else hardcode 2
    x = latent_model_input.view(B, C, H//ps, ps, W//ps, ps).permute(0,2,4,1,3,5).reshape(B, (H//ps)*(W//ps), C*ps*ps)
    gh, gw = H//ps, W//ps
    enc = text_embeddings.text_embeds.to(self.device_torch, self.torch_dtype)
    mask = text_embeddings.attention_mask.to(self.device_torch, dtype=torch.int64)
    txt_seq = enc.shape[1]
    position_ids = self.pipeline.prepare_position_ids(txt_seq, gh, gw, self.device_torch)  # mirror pipeline batching
    noise_pred = self.transformer(
        hidden_states=x.to(self.device_torch, self.torch_dtype).detach(),
        encoder_hidden_states=enc.detach(),
        timestep=(timestep / 1000).detach(),   # CONFIRM scale vs pipeline in 0.3
        position_ids=position_ids,
        encoder_attention_mask=mask.detach(),
        return_dict=False,
    )[0]
    # unpack
    np_ = noise_pred.view(B, gh, gw, C, ps, ps).permute(0,3,1,4,2,5).reshape(B, C, H, W)
    return np_
```

**Step 2 (CRITICAL):** Numerical cross-check against the reference pipeline. Write a throwaway script that: builds a fixed latent + prompt, runs ONE transformer step via (a) the diffusers `Krea2Pipeline` internals and (b) `model.get_noise_prediction`, and asserts the outputs match (allclose, atol ~1e-3 in bf16). This catches position_ids/timestep/packing mistakes. Fix until they match.
Expected: max abs diff < 1e-3.

**Step 3:** Also implement `get_loss_target` (flow matching, identical to qwen):
```python
def get_loss_target(self, *a, **k):
    return (k.get("noise") - k.get("batch").latents).detach()
```

**Step 4: Commit** `git commit -am "krea2: implement get_noise_prediction (position_ids + flow-matching target)"`

---

## Phase 5 — VAE encode/decode + block/save helpers (copy from qwen)

**Files:** Modify `krea_2_model.py`

**Step 1:** Copy verbatim from qwen_image (same VAE, identical normalization with `unsqueeze(2)` frame dim): `encode_images()`, `decode_latents()`, `get_transformer_block_names()` (return `["transformer_blocks"]` — verify against 0.3 module names), `save_model()`, `convert_lora_weights_before_save/load()`.

**Step 2:** Verify VAE round-trip: encode a test image tensor → decode → shape matches input; values finite.
Expected: no error, output shape == input shape.

**Step 3: Commit** `git commit -am "krea2: vae encode/decode, save_model, lora key conversion"`

---

## Phase 6 — Generation pipeline + single-image sampling

**Files:** Modify `krea_2_model.py`

**Step 1:** Implement `get_generation_pipeline()` (build `Krea2Pipeline` from unwrapped components, like qwen) and `generate_single_image()` (mirror qwen's call but with Krea2Pipeline's expected kwargs — check whether it takes `prompt_embeds`/`prompt_embeds_mask`/`negative_*` and `true_cfg_scale` like qwen, or different names; adapt from `pipeline_krea2.__call__`).

**Step 2:** Verify: generate one 1024×1024 image from a prompt via the model's sampling path; confirm a valid PIL image is returned (visually sane). Save to `output/krea2_smoke.png`.
Expected: a coherent image (proves text-encoder + transformer + vae wired correctly end-to-end for inference).

**Step 3: Commit** `git commit -am "krea2: generation pipeline + single image sampling"`

---

## Phase 7 — Example config

**Files:** Create `config/examples/train_lora_krea2_96gb.yaml`

**Step 1:** Base it on `config/examples/train_lora_qwen_image_24gb.yaml`, set:
```yaml
model:
  name_or_path: "CalamitousFelicitousness/Krea-2-Base-Diffusers"
  arch: "krea_2"
  quantize: false          # RTX 6000 Pro 96GB → full bf16
train:
  optimizer: "adamw8bit"   # or prodigy_plus (this branch has it)
  lr: 1e-4
  dtype: bf16
  noise_scheduler: "flowmatch"
sample:
  sampler: "flowmatch"
```
Keep network=lora linear 16/16, datasets/resolution as in the qwen example.

**Step 2:** Validate YAML parses and dispatches:
Run: `/media/p5/miniforge3/envs/ai_toolkit/bin/python -c "import yaml; d=yaml.safe_load(open('config/examples/train_lora_krea2_96gb.yaml')); print(d['config']['process'][0]['model']['arch'])"`
Expected: `krea_2`

**Step 3: Commit** `git commit -am "krea2: example LoRA training config"`

---

## Phase 8 — End-to-end training smoke test (RTX 6000 Pro)

**Step 1:** Point the example config at a tiny real dataset (5–10 images), `steps: 20`, `sample_every: 10`, `save_every: 20`.

**Step 2:** Run a real short training:
Run: `cd /media/p5/ai-toolkit && /media/p5/miniforge3/envs/ai_toolkit/bin/python run.py config/examples/train_lora_krea2_96gb.yaml`
Expected: loss is finite and trends down, samples render, a `.safetensors` LoRA saves without error, VRAM fits in 96 GB.

**Step 3:** Confirm the saved LoRA loads back (resume) without key errors.

**Step 4: Commit** any fixes: `git commit -am "krea2: fixes from end-to-end smoke test"`

---

## Phase 9 — Dedicated Docker image for the krea2 branch

**Files:** Modify `build_and_push_docker_ghcr` (or add `build_and_push_docker_ghcr_krea2`); the Dockerfile already supports `GIT_REPO`/`GIT_COMMIT` build args.

**Step 1:** Create a krea2 build script that builds from this branch and tags a distinct image so main/prodigy/krea2 stay separate, e.g. `ghcr.io/ethanfel/ai-toolkit:krea2` (and `:krea2-<version>`), passing `--build-arg GIT_COMMIT=krea2`. Because the krea2 branch bumps `requirements_base.txt` diffusers, the image's `pip install -r requirements.txt` picks up the new diffusers automatically (Dockerfile copies local requirements). Build context uses the existing `.dockerignore` from prodigy.

**Step 2:** Push the krea2 branch first (the Dockerfile clones it): `git push -u origin krea2`.

**Step 3:** Build + push the image; then verify inside the image:
Run: `docker run --rm -w /app/ai-toolkit --entrypoint python ghcr.io/ethanfel/ai-toolkit:krea2 -c "from toolkit.util.get_model import get_model_class; from toolkit.config_modules import ModelConfig; from diffusers import Krea2Transformer2DModel; print(get_model_class(ModelConfig(name_or_path='x', arch='krea_2')))"`
Expected: prints `Krea2Model` (proves arch registered + diffusers krea2 present in the image).

**Step 4: Commit** `git commit -am "krea2: dedicated GHCR docker build for krea2 branch"` and push.

---

## Resolved facts (fill in during Phase 0.3)

- diffusers commit used: `__________`
- Krea2 transformer loads from conversion repo cleanly? `__________`
- timestep scale fed to transformer (raw vs /1000): `__________`
- prompt_embeds shape: `__________`
- transformer block container module name (for LoRA): `__________`
- vae_scale_factor / bucket divisibility: `__________`
- Qwen3VL has a visual tower to drop? `__________`

---

## Notes
- **No UI change** is needed — ai-toolkit discovers models dynamically; `arch: "krea_2"` in YAML is enough.
- This branch deliberately diverges from `prodigy`/`main` on the diffusers pin; keep that isolation (separate image tag).
- If the community conversion proves unreliable (Risk #1), the fallback is converting from the official `krea-community/krea-2` bucket (`raw.safetensors` + `mmdit.py`) using diffusers' krea2 conversion script — a larger sub-task to scope only if needed.
