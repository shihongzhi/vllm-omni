# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# Bria Fibo-2.0 Dev text-to-image pipeline, ported from the HuggingFace
# diffusers ``BriaFibo2Pipeline`` (diffusers PR #14991).
#
# Stage 1 scope (matching the parity baseline): single request, 1024-class
# text-to-image, CFG as two independent forwards on the fixed null caption.
# The edit (context-latents) path is not wired yet; the transformer port
# already carries its scaffolding.

import inspect
import json
import os
from collections.abc import Iterable
from typing import Any, ClassVar

import torch
import torch.nn as nn
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_flux2 import (
    DistributedAutoencoderKLFlux2,
)
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import prefetch_subfolders
from vllm_omni.diffusion.models.bria_fibo2.fibo2_transformer import (
    Fibo2Transformer2DModel,
)
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# The null caption Fibo 2.0 was trained with for classifier-free guidance:
# every key of the structured caption, every value empty
DEFAULT_NEGATIVE_PROMPT = (
    '{"short_description":"","objects":[],"background_setting":"","lighting":{},'
    '"aesthetics":{},"photographic_characteristics":{},"style_medium":"",'
    '"context":"","artistic_style":""}'
)

# The chat turn each prompt is wrapped in for Qwen3-VL
PROMPT_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

# Qwen3-VL layers behind the transformer's six text inputs: the Perceiver's
# bundle first, then one per injection block
TEXT_ENCODER_OUT_LAYERS = ((9, 20, 31), (5, 16, 27), (7, 18, 29), (9, 20, 31), (11, 22, 33), (13, 24, 35))

# Keys dropped from the structured caption before serialization (aesthetics
# and preference scores are sampler-side hints, not caption content)
_DROP_CAPTION_KEYS = ("aesthetic_score", "preference_score")


def parse_prompt(prompt: str) -> str:
    """Clean a structured JSON caption the way Fibo 2.0 was trained with:
    drop empty values and score keys, serialize compactly. Non-JSON prompts
    pass through unchanged."""
    try:
        caption = json.loads(prompt)
    except json.JSONDecodeError:
        return prompt
    if not isinstance(caption, dict):
        return prompt

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if k not in _DROP_CAPTION_KEYS and clean(v) != ""}
        if isinstance(value, list):
            return [clean(v) for v in value if clean(v) != ""]
        return value

    caption = {k: v for k, v in clean(caption).items() if v != ""}
    return json.dumps(caption, separators=(",", ":"), ensure_ascii=False)


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_timesteps(scheduler, num_inference_steps, device, sigmas=None, **kwargs):
    """Calls the scheduler's ``set_timesteps`` and retrieves the schedule."""
    accepts_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
    if sigmas is not None and not accepts_sigmas:
        raise ValueError(
            f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom "
            f"sigmas schedules."
        )
    if sigmas is not None:
        kwargs["sigmas"] = sigmas
    scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    timesteps = scheduler.timesteps
    num_inference_steps = len(timesteps)
    return timesteps, num_inference_steps


def get_post_process_func(od_config: OmniDiffusionConfig):
    image_processor = VaeImageProcessor(vae_scale_factor=16, do_convert_rgb=True)

    def post_process_func(images: torch.Tensor):
        return image_processor.postprocess(images, output_type="pil")

    return post_process_func


class BriaFibo2Pipeline(nn.Module, ProgressBarMixin, DiffusionPipelineProfilerMixin, SupportsComponentDiscovery):
    """Bria Fibo-2.0 Dev text-to-image pipeline.

    Native vLLM-Omni implementation of the diffusers ``BriaFibo2Pipeline``
    T2I path: Qwen3-VL caption encoding (six bundles of three hidden layers,
    per-token RMS-normalized), flow-matching Euler denoising with CFG as two
    independent forwards, and the FLUX.2 VAE with its BatchNorm
    de-normalization.
    """

    supports_request_batch = False

    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        self.od_config = od_config
        self._execution_device = get_local_device()

        # text_encoder and vae load their weights eagerly via from_pretrained;
        # only the DiT weights flow through the engine's weight loader.
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=od_config.revision,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]

        model = od_config.model
        local_files_only = os.path.exists(model)

        # transformers v5 multi-worker subfolder race; prefetch first.
        fibo_subfolders = ["scheduler", "text_encoder", "processor", "vae"]
        prefetch_subfolders(
            model,
            fibo_subfolders,
            local_files_only=local_files_only,
            revision=od_config.revision,
        )

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model,
            subfolder="scheduler",
            local_files_only=local_files_only,
            revision=od_config.revision,
        )
        self.text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            model,
            subfolder="text_encoder",
            local_files_only=local_files_only,
            torch_dtype=od_config.dtype,
            revision=od_config.revision,
        ).to(self._execution_device)
        self.processor = Qwen3VLProcessor.from_pretrained(
            model,
            subfolder="processor",
            local_files_only=local_files_only,
            revision=od_config.revision,
        )
        self.vae = DistributedAutoencoderKLFlux2.from_pretrained(
            model,
            subfolder="vae",
            local_files_only=local_files_only,
            revision=od_config.revision,
        ).to(self._execution_device)
        self.transformer = Fibo2Transformer2DModel(quant_config=od_config.quantization_config)

        self.tokenizer_max_length = 4096
        self.vae_scale_factor = 8
        self.default_sample_size = 128
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2, do_convert_rgb=True)

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    # ------------------------------------------------------------------
    # Prompt encoding: port of diffusers' _get_qwen_prompt_embeds
    # ------------------------------------------------------------------
    def _encode_prompt_bundle(
        self, prompt: str | list[str], device: torch.device, max_sequence_length: int
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        prompt = [prompt] if isinstance(prompt, str) else prompt
        text = [PROMPT_TEMPLATE.format(p) for p in prompt]
        text_inputs = self.processor(
            text=text,
            padding="longest",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        text_inputs = text_inputs.to(device)

        with torch.no_grad():
            hidden_states = self.text_encoder(**text_inputs, output_hidden_states=True).hidden_states

        # Each layer is scaled to unit RMS per token in float32, then every
        # bundle concatenates its three layers
        dtype = self.text_encoder.dtype
        normalized = {}
        for layer in sorted({layer for layers in TEXT_ENCODER_OUT_LAYERS for layer in layers}):
            states = hidden_states[layer].float()
            normalized[layer] = (states * torch.rsqrt(states.pow(2).mean(dim=-1, keepdim=True) + 1e-6)).to(dtype)
        bundles = [torch.cat([normalized[layer] for layer in layers], dim=-1) for layers in TEXT_ENCODER_OUT_LAYERS]
        embeds = torch.stack(bundles, dim=1)

        mask = text_inputs.attention_mask.bool()
        if mask.all():
            # Nothing is padded: attention runs without a mask, which every
            # attention backend supports
            mask = None
        return embeds, mask

    def encode_prompt(
        self,
        prompt: str | list[str],
        device: torch.device | None = None,
        num_images_per_prompt: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = device or self._execution_device
        embeds, mask = self._encode_prompt_bundle(prompt, device, self.tokenizer_max_length)
        embeds = embeds.to(device).repeat_interleave(num_images_per_prompt, dim=0)
        if mask is not None:
            mask = mask.to(device).repeat_interleave(num_images_per_prompt, dim=0)
        return embeds, mask

    # ------------------------------------------------------------------
    # Latents / timesteps
    # ------------------------------------------------------------------
    def prepare_latents(self, batch_size, num_channels_latents, height, width, device, generator):
        # 1024 -> 128: Fibo keeps latents in fp32 for the whole trajectory
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        latents = randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)
        return latents

    def _patchify_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """VAE-side 2x2 regrouping: (B, C, H, W) -> (B, 4C, H/2, W/2).
        This is the FLUX.2 VAE layout op, not the DiT tokenization."""
        batch_size, channels, height, width = latents.shape
        latents = latents.view(batch_size, channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        return latents.reshape(batch_size, channels * 4, height // 2, width // 2)

    def _unpatchify_latents(self, latents: torch.Tensor) -> torch.Tensor:
        batch_size, channels4, height, width = latents.shape
        latents = latents.reshape(batch_size, channels4 // 4, 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        return latents.reshape(batch_size, channels4 // 4, height * 2, width * 2)

    def _denormalize_bn(self, latents: torch.Tensor) -> torch.Tensor:
        """FLUX.2 VAE latents come out of the DiT BatchNorm-normalized; undo
        with the running statistics before decoding (pipeline step 7)."""
        lat = self._patchify_latents(latents)
        bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(lat.device, lat.dtype)
        bn_std = torch.sqrt(self.vae.bn.running_var + self.vae.config.batch_norm_eps).view(1, -1, 1, 1)
        lat = lat * bn_std.to(lat.device, lat.dtype) + bn_mean
        return self._unpatchify_latents(lat)

    # ------------------------------------------------------------------
    # Forward: single-request T2I
    # ------------------------------------------------------------------
    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        if len(req.prompts) != 1:
            raise ValueError("BriaFibo2Pipeline currently supports exactly one prompt per request")

        raw_prompt = req.prompts[0]
        prompt = raw_prompt if isinstance(raw_prompt, str) else (raw_prompt.get("prompt") or "")
        prompt = parse_prompt(prompt)

        sp = req.sampling_params
        height = sp.height or 1024
        width = sp.width or 1024
        num_inference_steps = sp.num_inference_steps or 30
        guidance_scale = sp.guidance_scale if sp.guidance_scale is not None else 5.0
        generator = sp.generator
        seed = getattr(sp, "seed", None)
        if generator is None and seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed)
        output_type = sp.output_type or "pil"

        if height % (self.vae_scale_factor * 2) or width % (self.vae_scale_factor * 2):
            raise ValueError(f"Height/width must be multiples of {self.vae_scale_factor * 2} (got {height}x{width}).")

        device = self._execution_device
        dtype = self.transformer.dtype

        # 1. Encode the caption and the fixed null caption
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(prompt, device)
        negative_embeds, negative_mask = self.encode_prompt(DEFAULT_NEGATIVE_PROMPT, device)

        # 2. Initial noise latents (fp32 for the whole trajectory)
        latents = self.prepare_latents(1, self.transformer.in_channels, height, width, device, generator)

        # 3. Timesteps: sigmas linspace(1, 1/steps) with the resolution shift
        patch_size = self.transformer.patch_size
        image_seq_len = (latents.shape[2] // patch_size) * (latents.shape[3] // patch_size)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        sigmas = torch.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas.tolist(), mu=mu
        )

        # 4. Denoise: two independent CFG forwards per step (positive then
        # null caption), merged as uncond + g * (cond - uncond). Inference
        # throughout: the DiT holds no grad graph (see the transformer port
        # notes) and the VAE decode must not build one either.
        with torch.no_grad(), self.progress_bar(total=num_inference_steps) as progress_bar:
            for t in timesteps:
                timestep = t.expand(latents.shape[0]) / 1000
                latent_model_input = latents.to(dtype)
                noise_pred_cond = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_embeds_mask,
                    context_latents=None,
                )[0].float()
                noise_pred_uncond = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_embeds,
                    encoder_attention_mask=negative_mask,
                    context_latents=None,
                )[0].float()
                guided = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents = self.scheduler.step(guided, t, latents, return_dict=False)[0]
                progress_bar.update()

            # 5. Decode through the FLUX.2 VAE
            if output_type == "latent":
                image = latents
            else:
                lat = self._denormalize_bn(latents)
                image = self.vae.decode(lat.to(self.vae.dtype), return_dict=False)[0]
                image = self.image_processor.postprocess(image, output_type=output_type)

        return DiffusionOutput(output=image, stage_durations=getattr(self, "stage_durations", None))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        loaded_weights = loader.load_weights(weights)
        # Components already loaded eagerly in __init__; record their params so
        # the engine's strict completeness check passes.
        loaded_weights |= {f"vae.{name}" for name, _ in self.vae.named_parameters()}
        loaded_weights |= {f"text_encoder.{name}" for name, _ in self.text_encoder.named_parameters()}
        return loaded_weights
