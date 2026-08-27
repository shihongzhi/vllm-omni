"""Offline inference demo for Audio8 TTS via vLLM Omni.

Generates speech from text using the AutoArk-AI/Audio8-TTS-Preview-0.6b
checkpoint (text → AR codes → ArkttsCodec waveform @44.1 kHz).

Usage:
    python end2end.py --model /path/to/Audio8-TTS-Preview-0.6b \
        --text "Hello, this is a test."
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import soundfile as sf
import torch

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from vllm_omni import Omni
from vllm_omni.model_executor.models.audio8_tts.audio8_tts_ar import estimate_audio8_prompt_len

logger = logging.getLogger(__name__)


def build_prompt(model_dir: str, text: str) -> dict:
    """Placeholder ids + real payload; the engine's preprocess builds embeds."""
    prompt_len = estimate_audio8_prompt_len(model_dir, text)
    return {
        "prompt_token_ids": [151643] * prompt_len,  # pad token placeholders
        "additional_information": {"target_text": text},
    }


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = args.model
    inputs = [build_prompt(model_dir, args.text)]

    omni = Omni(model=model_dir, deploy_config=args.deploy_config)

    t_start = time.perf_counter()
    saved = 0
    for request_output in omni.generate(inputs):
        if request_output is None or not request_output.outputs:
            continue
        mm = request_output.outputs[0].multimodal_output
        if mm is None or not hasattr(mm, "get"):
            continue
        # Final audio ships as either a flat waveform tensor ("audio") or a
        # per-request list of waveform tensors ("model_outputs"); mm is a
        # dict or a MultimodalPayload (Mapping) — never index it directly.
        audio_chunks = mm.get("audio")
        if audio_chunks is None:
            audio_chunks = mm.get("model_outputs")
        if audio_chunks is None:
            continue
        sr_raw = mm.get("sr")
        if sr_raw is None:
            continue
        if isinstance(sr_raw, (list, tuple)):
            sr_raw = sr_raw[-1]
        sr = int(sr_raw.item() if hasattr(sr_raw, "item") else sr_raw)
        wav = torch.cat(audio_chunks, dim=-1) if isinstance(audio_chunks, list) else audio_chunks
        out_wav = os.path.join(args.output_dir, f"output_{request_output.request_id}.wav")
        sf.write(out_wav, wav.float().cpu().numpy().flatten(), samplerate=sr, format="WAV")

        duration = wav.numel() / max(sr, 1)
        wall = time.perf_counter() - t_start
        rtf = wall / max(duration, 1e-9)
        peak = float(torch.as_tensor(wav).abs().max())
        logger.info(
            "saved %s | sr=%d | %.2fs audio | wall=%.1fs | RTF=%.2f | peak=%.3f %s",
            out_wav,
            sr,
            duration,
            wall,
            rtf,
            peak,
            "(SILENT!)" if peak < 1e-3 else "",
        )
        saved += 1
    if not saved:
        raise RuntimeError("no audio output produced")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to the Audio8 TTS checkpoint dir")
    parser.add_argument(
        "--text", default="Hello, this is a golden baseline test for the Audio8 text to speech model integration."
    )
    parser.add_argument("--output-dir", default="output_audio8")
    parser.add_argument("--deploy-config", default=None)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main(parser.parse_args())
