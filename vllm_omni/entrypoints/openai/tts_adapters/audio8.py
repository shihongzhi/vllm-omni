# SPDX-License-Identifier: Apache-2.0
"""Audio8 TTS (arktts) serving adapter.

Zero-shot TTS from ``input`` alone; voice cloning when ``ref_audio`` +
``ref_text`` are provided (the reference waveform is encoded to codec codes
server-side with the checkpoint's own ArkttsCodec encoder).
"""

import asyncio
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import (
    ARTTSAdapter,
    PreparedRequest,
    apply_max_new_tokens,
    conditioning_cache_salt,
)

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

logger = init_logger(__name__)

# Qwen-family <|endoftext|>: placeholder width only — the engine's preprocess
# replaces these with real embeddings built from ``additional_information``.
_AUDIO8_PAD_TOKEN_ID = 151643


@register_tts_adapter
class Audio8TTSAdapter(ARTTSAdapter):
    stage_keys = frozenset({"audio8_tts_ar", "audio8_codec"})
    model_archs = frozenset({"ArkttsModel", "Audio8CodecDecoder"})
    name = "audio8_tts"

    def __init__(self, ctx: Any) -> None:
        super().__init__(ctx)
        self._codec: Any | None = None
        self._codec_model_dir: str | None = None
        self._ref_codes_cache: dict[str, torch.Tensor] = {}

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        server = self.ctx.server
        err = server._apply_uploaded_speaker(request)
        if err:
            return err
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"

        if request.ref_audio is not None:
            fmt_err = server._validate_ref_audio_format(request.ref_audio)
            if fmt_err:
                return fmt_err
            if not request.ref_text or not request.ref_text.strip():
                return "Audio8 voice cloning requires 'ref_text' (transcript of the reference audio)"

        if request.max_new_tokens is not None:
            if request.max_new_tokens < self.max_new_tokens_min:
                return f"max_new_tokens must be >= {self.max_new_tokens_min}"
            if request.max_new_tokens > self.max_new_tokens_max:
                return f"max_new_tokens cannot exceed {self.max_new_tokens_max}"
        return None

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        from vllm_omni.model_executor.models.audio8_tts.audio8_tts_ar import (
            estimate_audio8_prompt_len,
        )

        server = self.ctx.server
        model_dir = server.model_config.model
        target_text = request.input.strip()
        tts_params: dict[str, Any] = {"target_text": target_text}

        ref_frames = 0
        ref_text = None
        if request.ref_audio is not None:
            wav_list, sr = await server._resolve_ref_audio(request.ref_audio)
            ref_codes = await asyncio.to_thread(self._encode_ref_audio, model_dir, wav_list, sr)
            ref_frames = int(ref_codes.shape[1])
            ref_text = request.ref_text.strip()
            tts_params["ref_codes"] = ref_codes
            tts_params["ref_text"] = ref_text
            logger.info("Audio8 clone conditioning: ref_frames=%d sr=%d", ref_frames, sr)

        prompt_len = estimate_audio8_prompt_len(
            model_dir,
            target_text,
            ref_text=ref_text,
            ref_frames=ref_frames,
        )
        prompt: dict[str, Any] = {
            "prompt_token_ids": [_AUDIO8_PAD_TOKEN_ID] * prompt_len,
            "additional_information": tts_params,
        }
        prompt["cache_salt"] = conditioning_cache_salt(request, tts_params)
        return PreparedRequest(prompt=prompt, tts_params=tts_params, model_type=self.name)

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: "OpenAICreateSpeechRequest",
        prompt: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> list:
        sampling_params_list = apply_max_new_tokens(sampling_params_list, request)
        seed = getattr(request, "seed", None)
        if seed is not None:
            sampling_params_list[0].seed = seed
        return sampling_params_list

    def _load_codec_frame_rate(self) -> float | None:
        hf_cfg = getattr(self.ctx.engine_client.model_config, "hf_config", None)
        sample_rate = int(getattr(hf_cfg, "codec_sample_rate", 44100))
        samples_per_frame = int(getattr(hf_cfg, "codec_frame_length", 2048))
        rate = sample_rate / samples_per_frame if samples_per_frame > 0 else None
        return rate or None

    # ------------------------------------------------------------------
    # Reference-audio encoding (server side, lazily-loaded codec encoder)
    # ------------------------------------------------------------------

    def _encode_ref_audio(self, model_dir: str, wav_list: list[float], sr: int) -> torch.Tensor:
        """Encode mono reference waveform to [num_codebooks, T] long codes."""
        wav = torch.from_numpy(np.asarray(wav_list, dtype=np.float32)).reshape(-1)
        hf_cfg = self.ctx.server.model_config.hf_config
        target_sr = int(getattr(hf_cfg, "codec_sample_rate", 44100))
        if int(sr) != target_sr:
            from torchaudio.functional import resample

            wav = resample(wav, int(sr), target_sr)

        codec = self._get_codec(model_dir)
        codes, lengths = codec.encode(wav.reshape(1, 1, -1))
        frames = int(lengths[0])
        return codes[0, :, :frames].to(torch.long).contiguous()

    def _get_codec(self, model_dir: str) -> Any:
        if self._codec is not None and self._codec_model_dir == model_dir:
            return self._codec
        from vllm_omni.model_executor.models.audio8_tts.audio8_tts_codec import (
            load_arktts_codec,
        )

        codec = load_arktts_codec(model_dir)
        codec.eval()
        self._codec = codec
        self._codec_model_dir = model_dir
        return codec
