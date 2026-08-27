# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""E2E offline inference tests for Audio8 TTS (ArkttsModel, two-stage).

Stage 0 (audio8_tts_ar) is a dual-AR backbone emitting 10-codebook frame
stacks; stage 1 (audio8_codec) decodes them to 44.1 kHz waveform through the
ArkttsCodec. Requests mirror examples/offline_inference/text_to_speech/
audio8/end2end.py: placeholder prompt ids sized by
``estimate_audio8_prompt_len`` plus ``additional_information`` carrying the
target text; the engine's preprocess builds the real embeddings.
"""

from __future__ import annotations

import pytest
import torch
from vllm import SamplingParams

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config

MODEL = "AutoArk-AI/Audio8-TTS-Preview-0.6b"
SAMPLE_RATE = 44_100

_STAGE0_PARAMS = SamplingParams(
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    max_tokens=1536,
    seed=42,
    detokenize=False,
)
_STAGE1_PARAMS = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    top_k=-1,
    max_tokens=65536,
    seed=42,
    detokenize=False,
)
_SAMPLING = [_STAGE0_PARAMS, _STAGE1_PARAMS]


def _get_test_config() -> str:
    """Derive a CI-friendly config from audio8_tts.yaml."""
    return modify_stage_config(
        get_deploy_config_path("audio8_tts.yaml"),
        updates={
            "stages": {
                0: {
                    "max_num_seqs": 1,
                },
            },
        },
    )


def _try_resolve_model_dir() -> str | None:
    """Resolve the checkpoint to a local dir (stage-1 reads codec.pth from it)."""
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(MODEL, local_files_only=True)
    except Exception:  # noqa: BLE001 - converted to a skip marker below
        return None


_MODEL_DIR = _try_resolve_model_dir()

pytestmark = [
    pytest.mark.slow,
    pytest.mark.tts,
    pytest.mark.parametrize(
        "omni_runner",
        [(_MODEL_DIR, _get_test_config(), {"trust_remote_code": True})],
        indirect=True,
    ),
]


def _build_request(text: str) -> dict:
    """Placeholder ids + real payload; the engine's preprocess builds embeds."""
    from vllm_omni.model_executor.models.audio8_tts.audio8_tts_ar import (
        estimate_audio8_prompt_len,
    )

    prompt_len = estimate_audio8_prompt_len(_MODEL_DIR, text)
    return {
        "prompt_token_ids": [151643] * prompt_len,
        "additional_information": {"target_text": text},
    }


def _collect_audio(omni_runner: OmniRunner, request: dict) -> tuple[torch.Tensor, int]:
    chunks: list[torch.Tensor] = []
    sr_final = SAMPLE_RATE
    for out in omni_runner.omni.generate(request, _SAMPLING):
        mm = getattr(out.outputs[0], "multimodal_output", None) if out.outputs else None
        if mm is None or not hasattr(mm, "get"):
            continue
        audio = mm.get("audio")
        if audio is None:
            audio = mm.get("model_outputs")
        if audio is None:
            continue
        sr = mm.get("sr")
        if sr is not None:
            if isinstance(sr, (list, tuple)):
                sr = sr[-1]
            sr_final = int(sr.item() if hasattr(sr, "item") else sr)
        if isinstance(audio, list):
            audio = torch.cat(
                [t.reshape(-1) for t in audio if isinstance(t, torch.Tensor) and t.numel() > 0],
                dim=0,
            )
        if isinstance(audio, torch.Tensor) and audio.numel() > 0:
            chunks.append(audio.reshape(-1).float().cpu())
    if not chunks:
        raise AssertionError("No audio output received from generate()")
    return torch.cat(chunks, dim=0), sr_final


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_audio8_tts_english(omni_runner: OmniRunner, run_level: str) -> None:
    """Zero-shot TTS: English text produces non-empty non-silent 44.1 kHz audio.

    At ``core_model`` level the runner loads DUMMY weights (structural smoke
    only); real-speech assertions apply from ``advanced_model`` up.
    """
    if run_level not in {"advanced_model", "full_model"}:
        pytest.skip("Audio8 speech quality assertions require real weights (advanced_model+)")
    req = _build_request("Hello, this is an Audio8 text to speech engine test.")
    audio, sr = _collect_audio(omni_runner, req)

    assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE} Hz, got {sr}"
    assert audio.numel() > 0, "Audio tensor is empty"
    peak = float(audio.abs().max())
    assert peak > 1e-3, f"Audio is silent (peak={peak:.6f})"
    duration = audio.numel() / sr
    assert 0.5 <= duration <= 30.0, f"Unplausible duration {duration:.2f}s"
