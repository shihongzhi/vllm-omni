# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Audio8 TTS codec decoder (Stage 1).

Consumes codebook-major flat codes from Stage 0 and decodes waveforms via
the ported ArkttsCodec (``codec.pth`` shipped inside the Audio8 model dir).
Analogous to ``MossTTSCodecDecoder`` / ``FishSpeechDACDecoder``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from torch.nn.utils.parametrize import remove_parametrizations
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.audio8_tts.audio8_tts_codec import (
    load_arktts_codec,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


class Audio8CodecDecoder(nn.Module):
    """Stage-1 ArkttsCodec decoder for Audio8 TTS (GenerationModelRunner)."""

    input_modalities = "audio"

    have_multimodal_outputs: bool = True
    has_preprocess: bool = False
    has_postprocess: bool = False
    enable_update_additional_information: bool = True
    requires_raw_input_tokens: bool = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model
        hf_cfg = vllm_config.model_config.hf_config
        self._num_codebooks = int(getattr(hf_cfg, "num_codebooks", 10))
        self._output_sample_rate = int(getattr(hf_cfg, "codec_sample_rate", 44100))
        self._codec: nn.Module | None = None
        self._sr_tensor = torch.tensor(self._output_sample_rate, dtype=torch.int32)

    # ------------------------------------------------------------------
    # vLLM stubs (no AR loop on this stage)
    # ------------------------------------------------------------------

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> None:
        return None

    # ------------------------------------------------------------------
    # Codec loading + decode
    # ------------------------------------------------------------------

    def _bake_weight_norm(self, module: nn.Module) -> int:
        """Remove weight-norm parametrizations once weights are loaded."""
        baked = 0
        for mod in module.modules():
            parametrizations = getattr(mod, "parametrizations", None)
            if not parametrizations:
                continue
            for name in list(parametrizations.keys()):
                remove_parametrizations(mod, name, leave_parametrized=True)
                baked += 1
        return baked

    def _ensure_codec_loaded(self) -> nn.Module:
        if self._codec is not None:
            return self._codec
        codec = load_arktts_codec(self.model_path)
        # Decode never touches the encoder path; drop it before GPU transfer.
        del codec.encoder
        device = self.vllm_config.device_config.device
        baked = self._bake_weight_norm(codec)
        codec = codec.to(device=device, dtype=torch.float32)
        codec.eval()
        # Bypass nn.Module registration: a plain attribute assignment would
        # fold every codec parameter into named_parameters() and the loader
        # would then demand ``_codec.*`` from the checkpoint.
        object.__setattr__(self, "_codec", codec)
        logger.info(
            "ArkttsCodec loaded from %s (device=%s, fp32, sample_rate=%d, baked=%d)",
            f"{self.model_path}/codec.pth",
            device,
            self._output_sample_rate,
            baked,
        )
        return codec

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Drain the Stage-0-style weight iterator; real weights come from codec.pth."""
        for _ in weights:
            pass
        self._ensure_codec_loaded()
        return set()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        """Decode per-request code stacks into waveforms.

        ``input_ids`` is the concatenation of every request's codebook-major
        flat codes; ``kwargs["seq_token_counts"]`` provides the per-request
        slice boundaries (shared code2wav contract).
        """
        sr_tensor = self._sr_tensor
        empty = torch.zeros((0,), dtype=torch.float32)
        codec = self._ensure_codec_loaded()
        device = next(codec.parameters()).device
        q = self._num_codebooks

        audios: list[torch.Tensor] = []
        srs: list[torch.Tensor] = []

        if input_ids is None or input_ids.numel() == 0:
            n = max(len(runtime_additional_information or []), 1)
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [empty] * n, "sr": [sr_tensor] * n},
            )

        ids_flat = input_ids.reshape(-1).to(dtype=torch.long)
        token_counts = kwargs.get("seq_token_counts")
        if not token_counts:
            raise RuntimeError(
                "Audio8CodecDecoder requires seq_token_counts to split concatenated codec tokens per request."
            )
        offsets = [0]
        for count in token_counts:
            offsets.append(offsets[-1] + int(count))

        for i in range(len(token_counts)):
            seg = ids_flat[offsets[i] : offsets[i + 1]].to(device)
            if seg.numel() == 0 or seg.numel() % q != 0:
                logger.warning(
                    "request %d: invalid code length %d (needs multiple of %d); emitting silence.",
                    i,
                    int(seg.numel()),
                    q,
                )
                audios.append(empty)
                srs.append(sr_tensor)
                continue
            codes_k_t = seg.reshape(q, -1)
            waveform = codec.decode(codes_k_t.unsqueeze(0))[0, 0].float().cpu()
            audios.append(waveform)
            srs.append(sr_tensor)

        return OmniOutput(text_hidden_states=None, multimodal_outputs={"model_outputs": audios, "sr": srs})
