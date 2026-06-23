import os
from typing import TYPE_CHECKING, List, Optional

import torch
import yaml
from PIL import Image

from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.models.base_model import BaseModel
from toolkit.basic import flush
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
from toolkit.accelerator import unwrap_model
from optimum.quanto import freeze
from toolkit.util.quantize import quantize, get_qtype, quantize_model
from toolkit.memory_management import MemoryManager

from diffusers import (
    Krea2Pipeline,
    Krea2Transformer2DModel,
    AutoencoderKLQwenImage,
)
from transformers import Qwen3VLModel, Qwen2Tokenizer

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO

# From the Krea 2 `scheduler/scheduler_config.json`. Note shift_terminal is null (unlike qwen's 0.02)
# and max_image_seq_len/max_shift differ.
scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "invert_sigmas": False,
    "max_image_seq_len": 6400,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}


class Krea2Model(BaseModel):
    """Krea 2 (K2) — a flow-matching single-stream MMDiT text-to-image model.

    Architecturally a close cousin of Qwen-Image (shared AutoencoderKLQwenImage VAE,
    FlowMatchEulerDiscreteScheduler, Qwen-family text encoder), so VAE encode/decode, save_model and
    LoRA key conversion are reused verbatim. The transformer is called differently than qwen: it takes
    RoPE `position_ids` + a boolean `encoder_attention_mask` instead of qwen's `img_shapes`/`txt_seq_lens`,
    and the text embeds are 4D `(B, seq, num_text_layers, 2560)`.
    """

    arch = "krea_2"
    # text-only conditioning, so the Qwen3-VL visual tower is dropped by default to save memory
    _keep_visual = False

    def __init__(
        self,
        device,
        model_config: ModelConfig,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(
            device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs
        )
        self.is_flow_matching = True
        self.is_transformer = True
        # ai-toolkit targets every nn.Linear inside the named module class for LoRA.
        self.target_lora_modules = ["Krea2Transformer2DModel"]

    # static method to get the noise scheduler
    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        # vae_scale_factor (AutoencoderKLQwenImage = 2**len(temperal_downsample) = 8) * patch_size (2)
        return 8 * 2

    def get_base_model_version(self):
        return "krea_2"

    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading Krea 2 model")
        model_path = self.model_config.name_or_path

        self.noise_scheduler = Krea2Model.get_train_scheduler()

        # Build the whole pipeline via from_pretrained. This is required (unlike qwen's empty-pipe
        # pattern) because Krea2's encode_prompt reads self.pipeline.text_encoder_select_layers, which
        # is only configured correctly when the pipeline config is loaded. Components are loaded on CPU
        # first, then quantized / offloaded / moved per the model config (12.8B transformer needs
        # quantization to fit a ~24-32 GB card; full bf16 only fits a big card).
        self.print_and_status_update("Loading pipeline (transformer + text encoder + vae)")
        pipe: Krea2Pipeline = Krea2Pipeline.from_pretrained(model_path, torch_dtype=dtype)
        pipe.scheduler = self.noise_scheduler
        transformer = pipe.transformer
        text_encoder = pipe.text_encoder

        # drop the Qwen3-VL visual tower; Krea2 conditions on text only and never invokes it
        if not self._keep_visual and hasattr(text_encoder, "visual"):
            text_encoder.visual = None
            flush()

        # ---- transformer: quantize / layer-offload / device placement ----
        if self.model_config.quantize:
            self.print_and_status_update("Quantizing transformer")
            quantize_model(self, transformer)
            flush()

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_transformer_percent > 0
        ):
            MemoryManager.attach(
                transformer,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_transformer_percent,
            )

        if self.low_vram:
            self.print_and_status_update("Keeping transformer on CPU (low_vram)")
            transformer.to("cpu")
        else:
            transformer.to(self.device_torch)
        flush()

        # ---- text encoder: device, then quantize / layer-offload ----
        self.print_and_status_update("Preparing text encoder")
        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_text_encoder_percent > 0
        ):
            MemoryManager.attach(
                text_encoder,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_text_encoder_percent,
            )

        text_encoder.to(self.device_torch, dtype=dtype)
        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing text encoder")
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype_te))
            freeze(text_encoder)
            flush()

        text_encoder.requires_grad_(False)
        text_encoder.eval()
        pipe.vae.requires_grad_(False)
        pipe.vae.eval()
        flush()

        # reattach (visual-dropped / quantized) components to the pipe
        pipe.transformer = transformer
        pipe.text_encoder = text_encoder

        # save it to the model class
        self.vae = pipe.vae
        self.text_encoder = [text_encoder]  # list of text encoders
        self.tokenizer = [pipe.tokenizer]  # list of tokenizers
        self.model = transformer
        self.pipeline = pipe
        self.print_and_status_update("Model Loaded")

    def get_generation_pipeline(self):
        scheduler = Krea2Model.get_train_scheduler()

        pipeline = Krea2Pipeline(
            scheduler=scheduler,
            text_encoder=unwrap_model(self.text_encoder[0]),
            tokenizer=self.tokenizer[0],
            vae=unwrap_model(self.vae),
            transformer=unwrap_model(self.transformer),
        )
        pipeline = pipeline.to(self.device_torch)
        return pipeline

    def generate_single_image(
        self,
        pipeline: Krea2Pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        self.model.to(self.device_torch, dtype=self.torch_dtype)
        if gen_config.ctrl_img is not None:
            raise NotImplementedError(
                "Control image generation is not supported in Krea 2 model... yet"
            )

        flush_between_steps = self.model_config.low_vram

        def callback_on_step_end(pipe, i, t, callback_kwargs):
            if flush_between_steps:
                flush()
            return {"latents": callback_kwargs["latents"]}

        sc = self.get_bucket_divisibility()
        gen_config.width = int(gen_config.width // sc * sc)
        gen_config.height = int(gen_config.height // sc * sc)

        # Krea 2 guidance convention is `cond + guidance_scale * (cond - uncond)`, which equals standard
        # CFG with scale `1 + guidance_scale`. ai-toolkit's gen_config.guidance_scale is a standard CFG
        # scale (as for the sibling qwen model), so subtract 1 to keep the same perceptual meaning.
        # guidance_scale <= 0 disables CFG (skips the negative pass).
        krea_guidance_scale = max(float(gen_config.guidance_scale) - 1.0, 0.0)

        img = pipeline(
            prompt_embeds=conditional_embeds.text_embeds,
            prompt_embeds_mask=conditional_embeds.attention_mask.to(
                self.device_torch, dtype=torch.bool
            ),
            negative_prompt_embeds=unconditional_embeds.text_embeds,
            negative_prompt_embeds_mask=unconditional_embeds.attention_mask.to(
                self.device_torch, dtype=torch.bool
            ),
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=krea_guidance_scale,
            latents=gen_config.latents,
            generator=generator,
            callback_on_step_end=callback_on_step_end,
            **extra,
        ).images[0]
        return img

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        **kwargs,
    ):
        self.model.to(self.device_torch)
        batch_size, num_channels_latents, height, width = latent_model_input.shape

        # Krea2Transformer2DModel has no patch_size in its config (in_channels=64=16*2*2); hardcode 2.
        ps = 2

        # pack image tokens -> (B, (H//ps)*(W//ps), C*ps*ps)
        x = latent_model_input.view(
            batch_size, num_channels_latents, height // ps, ps, width // ps, ps
        )
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(
            batch_size, (height // ps) * (width // ps), num_channels_latents * (ps * ps)
        )

        grid_h, grid_w = height // ps, width // ps

        enc = text_embeddings.text_embeds.to(self.device_torch, self.torch_dtype)
        # Krea2 feeds encoder_attention_mask straight into SDPA, so it MUST be bool (not int64).
        mask = text_embeddings.attention_mask.to(self.device_torch, dtype=torch.bool)
        txt_seq = enc.shape[1]

        # position_ids: unbatched (txt_seq + grid_h*grid_w, 3), shared across the batch (mirrors the pipeline).
        position_ids = Krea2Pipeline.prepare_position_ids(
            txt_seq, grid_h, grid_w, self.device_torch
        )

        noise_pred = self.transformer(
            hidden_states=x.to(self.device_torch, self.torch_dtype).detach(),
            encoder_hidden_states=enc.detach(),
            timestep=(timestep / 1000).detach(),  # pipeline feeds t / num_train_timesteps (=1000)
            position_ids=position_ids,
            encoder_attention_mask=mask.detach(),
            return_dict=False,
        )[0]

        # unpack -> (B, C, H, W)
        noise_pred = noise_pred.view(
            batch_size, grid_h, grid_w, num_channels_latents, ps, ps
        )
        noise_pred = noise_pred.permute(0, 3, 1, 4, 2, 5)
        noise_pred = noise_pred.reshape(batch_size, num_channels_latents, height, width)
        return noise_pred

    def get_prompt_embeds(self, prompt: str, control_images=None) -> PromptEmbeds:
        if self.pipeline.text_encoder.device != self.device_torch:
            self.pipeline.text_encoder.to(self.device_torch)

        prompt_embeds, prompt_embeds_mask = self.pipeline.encode_prompt(
            prompt,
            device=self.device_torch,
            num_images_per_prompt=1,
        )
        if prompt_embeds_mask is None:
            prompt_embeds_mask = torch.ones(
                prompt_embeds.shape[:2], device=prompt_embeds.device, dtype=torch.bool
            )
        pe = PromptEmbeds(prompt_embeds)
        pe.attention_mask = prompt_embeds_mask
        return pe

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def save_model(self, output_path, meta, save_dtype):
        # only save the transformer
        transformer: Krea2Transformer2DModel = unwrap_model(self.model)
        transformer.save_pretrained(
            save_directory=os.path.join(output_path, "transformer"),
            safe_serialization=True,
        )

        meta_path = os.path.join(output_path, "aitk_meta.yaml")
        with open(meta_path, "w") as f:
            yaml.dump(meta, f)

    def get_loss_target(self, *args, **kwargs):
        # flow matching target: noise - latents
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        return (noise - batch.latents).detach()

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["transformer_blocks"]

    def convert_lora_weights_before_save(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("transformer.", "diffusion_model.")
            new_sd[new_key] = value
        return new_sd

    def convert_lora_weights_before_load(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("diffusion_model.", "transformer.")
            new_sd[new_key] = value
        return new_sd

    def encode_images(self, image_list: List[torch.Tensor], device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        # Move vae to device if on cpu
        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)
        self.vae.eval()
        self.vae.requires_grad_(False)
        image_list = [image.to(device, dtype=dtype) for image in image_list]
        images = torch.stack(image_list).to(device, dtype=dtype)
        # AutoencoderKLQwenImage is a video VAE, so add a frame dim
        images = images.unsqueeze(2)
        latents = self.vae.encode(images).latent_dist.sample()

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)

        latents = (latents - latents_mean) * latents_std
        latents = latents.to(device, dtype=dtype)

        latents = latents.squeeze(2)  # remove the frame count dimension

        return latents

    def decode_latents(self, latents: torch.Tensor, device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)

        latents = latents.to(device, dtype=dtype)

        # add frame count dim for the video vae
        latents = latents.unsqueeze(2)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = latents * latents_std + latents_mean

        images = self.vae.decode(latents).sample

        images = images.squeeze(2)  # remove the frame count dimension

        return images.to(device, dtype=dtype)
