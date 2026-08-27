# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Step 3 verification: ported ArkttsCodec decode vs the baseline environment.

The reference waveforms live next to the Step 2 dumps in
``AUDIO8_TTS_BASELINE_DIR`` (``ref_decoded_cpu_f32.pt``, produced by running
the checkpoint's own ``modeling_arktts_codec`` under the pinned baseline env,
CPU float32). Golden codes come from ``golden_<variant>.npy``. Tests are
skipped when either artifact is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.audio8_tts.audio8_tts_codec import (
    ArkttsCodecStreamer,
    decode_audio,
    load_arktts_codec,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

VARIANTS = ["no_ref", "clone"]


@pytest.fixture(scope="session")
def codec(audio8_model_dir):
    return load_arktts_codec(audio8_model_dir)


@pytest.fixture(scope="session")
def ref_waveforms(audio8_ref_dir):
    path = audio8_ref_dir / "ref_decoded_cpu_f32.pt"
    if not path.is_file():
        pytest.skip(f"baseline decoded waveforms missing: {path}")
    return torch.load(path, weights_only=True)


def _golden_codes(audio8_ref_dir, variant: str) -> torch.Tensor:
    codes = torch.from_numpy(np.load(audio8_ref_dir / f"golden_{variant}.npy").astype(np.int64))
    return codes.unsqueeze(0)


@pytest.mark.parametrize("variant", VARIANTS)
def test_decode_matches_baseline(codec, ref_waveforms, audio8_ref_dir, variant):
    codes = _golden_codes(audio8_ref_dir, variant)
    waveform, lengths = decode_audio(codec, codes)
    reference = ref_waveforms[variant]

    assert lengths.tolist() == [reference.numel()]
    assert int(codes.shape[-1]) * codec.samples_per_frame == reference.numel()
    assert torch.isfinite(waveform).all()
    max_diff = (waveform - reference[None]).abs().max().item()
    correlation = torch.dot(waveform.reshape(-1), reference.reshape(-1)) / (waveform.norm() * reference.norm() + 1e-12)
    # Same weights/architecture/kernels as the baseline dump: expect near-bit
    # agreement; corr guards against any silent layout mismatch.
    assert max_diff < 1e-4, f"decoded waveform drifted from baseline: max|diff|={max_diff}"
    assert float(correlation) > 0.99999


def test_streaming_matches_one_shot(codec, audio8_ref_dir):
    """Pushed-chunk streaming must reproduce one-shot decode exactly."""
    codes = _golden_codes(audio8_ref_dir, "clone")[0]  # [10, T]
    one_shot = codec.decode(codes.unsqueeze(0))[0, 0].float()
    expected_numel = int(codes.shape[-1]) * codec.samples_per_frame

    streamer = ArkttsCodecStreamer(codec)
    pieces: list[torch.Tensor] = []
    start = 0
    for size in (11,) * 6:  # small chunks maximise prefix-recompute stress
        chunk = codes[:, start : start + size]
        streamer.push(chunk)
        start += int(chunk.shape[1])
        pieces.append(streamer.read())
    assert start < codes.shape[-1], "streaming test must leave a remainder"
    streamer.push(codes[:, start:])
    pieces.append(streamer.read())
    pieces.append(streamer.finish())

    streamed = torch.cat([p for p in pieces if p.numel()], dim=-1)
    assert streamed.numel() == expected_numel
    max_diff = (streamed - one_shot[:expected_numel]).abs().max().item()
    assert max_diff < 1e-5, f"streamed waveform differs from one-shot: max|diff|={max_diff}"


def test_decode_audio_stops_at_pad_sentinel(codec, audio8_ref_dir):
    codes = _golden_codes(audio8_ref_dir, "clone").clone()
    codes[0, :, -3:] = -1  # sentinel padding after frame T-3
    _, lengths = decode_audio(codec, codes)
    assert lengths.tolist() == [(int(codes.shape[-1]) - 3) * codec.samples_per_frame]


def test_decode_rejects_wrong_codebook_count(codec):
    bad = torch.zeros(1, 7, 10, dtype=torch.long)
    with pytest.raises(ValueError, match=r"\[B, 10, T\]"):
        decode_audio(codec, bad)


def test_codec_metadata(codec):
    assert codec.sample_rate == 44100
    assert codec.samples_per_frame == 2048
