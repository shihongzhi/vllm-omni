# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage input processors: Audio8 TTS AR (Stage 0) → codec decoder (Stage 1).

Two transfer routes, mirroring the MOSS-TTS processors:

* ``ar2codec_async_chunk`` (Stage 0 ``async_chunk_process_next_stage_input_func``)
  buffers the per-step accumulated code snapshots and flushes the whole clip
  to Stage 1 on finish (the MOSS raw-codec contract, without delay
  de-interleaving).
* ``ar2codec`` (Stage 1 ``sync_process_input_func``, ``async_chunk: false``)
  collects every finished Stage-0 request, reads its aggregated
  ``codes["audio"]`` frame stack [T, K] from the completion-level
  ``multimodal_output``, transposes to codebook-major [K, T] and flattens
  into Stage-1 ``input_ids``.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.inputs import TokensPrompt as OmniTokensPrompt
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct

logger = init_logger(__name__)


def _extract_audio_codes(stage_output: Any) -> torch.Tensor | None:
    """Pull the aggregated [T, K] frame stack from a Stage-0 request output."""
    direct = getattr(stage_output, "multimodal_outputs", None)
    candidates: list[Any] = [direct, getattr(stage_output, "multimodal_output", None)]
    for completion in getattr(stage_output, "outputs", None) or []:
        candidates.append(getattr(completion, "multimodal_output", None))
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        codes_dict = candidate.get("codes")
        if isinstance(codes_dict, dict):
            ac = codes_dict.get("audio")
            if isinstance(ac, torch.Tensor) and ac.numel() > 0:
                return ac
    return None


def ar2codec(
    source_outputs: Any,
    prompt: Any = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[OmniTokensPrompt]:
    """Pack each finished Stage-0 clip into one Stage-1 token prompt.

    ``codes["audio"]`` arrives as [T, K] (frames × codebooks); it is
    transposed to codebook-major [K, T] and flattened so Stage-1 receives
    ``input_ids`` of length K*T (the reverse reshape happens in the
    Audio8CodecDecoder).
    """
    outputs = source_outputs if isinstance(source_outputs, list) else [source_outputs]
    results: list[OmniTokensPrompt] = []
    for stage_output in outputs:
        if getattr(stage_output, "finished", True) is False:
            continue
        audio_codes = _extract_audio_codes(stage_output)
        if audio_codes is None:
            logger.warning("ar2codec: no audio codes for a Stage-0 output; emitting silence.")
            results.append(OmniTokensPrompt(prompt_token_ids=[]))
            continue

        codes_k_t = audio_codes.transpose(0, 1).contiguous()  # [K, T]
        flat = codes_k_t.reshape(-1).tolist()
        mm = {"codes": {"audio": codes_k_t}} if requires_multimodal_data else None
        results.append(
            OmniTokensPrompt(
                prompt_token_ids=flat,
                multi_modal_data=mm,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Streaming (async chunk): called each time Stage 0 emits a new chunk
# ---------------------------------------------------------------------------


def ar2codec_async_chunk(
    transfer_manager: Any,
    multimodal_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Forward AR code frames to the codec stage (async_chunk producer).

    The AR model's ``make_omni_output`` publishes the full per-request
    accumulated ``[T, K]`` snapshot every step, so only the *new* tail rows
    are appended to the per-request buffer (otherwise history would be
    duplicated quadratically).

    Chunked decode is not wired for ArkttsCodec yet, so the buffer flushes
    once on ``is_finished`` — the codec stage then sees the whole clip as a
    single request, matching the offline decode loop.
    """
    external_req_id = getattr(request, "external_req_id", None)
    req_id = str(external_req_id if external_req_id is not None else getattr(request, "request_id", id(request)))

    if not hasattr(transfer_manager, "_audio8_codes_buffer"):
        transfer_manager._audio8_codes_buffer = {}
    if not hasattr(transfer_manager, "_audio8_flushed_reqs"):
        transfer_manager._audio8_flushed_reqs = set()
    if req_id in transfer_manager._audio8_flushed_reqs:
        # Exactly-once flush: the AR model keeps publishing the accumulated
        # snapshot after finish, so a later save_async would re-buffer the
        # whole clip and ship a duplicate chunk that clobbers the codec
        # request's prompt. The flush chunk already carried the finish marker.
        return None
    state = transfer_manager._audio8_codes_buffer
    if req_id not in state:
        state[req_id] = {"accumulated": None, "total_emitted": 0}
    req_state = state[req_id]

    if isinstance(multimodal_output, dict):
        codes_dict = multimodal_output.get("codes", {}) or {}
        snapshot = codes_dict.get("audio")
        if isinstance(snapshot, torch.Tensor) and snapshot.numel() > 0:
            snapshot_cpu = snapshot.detach().to("cpu", torch.long).contiguous()
            if snapshot_cpu.ndim == 1:
                snapshot_cpu = snapshot_cpu.reshape(1, -1)
            if snapshot_cpu.ndim != 2:
                raise ValueError(f"Audio8 codec frames must be 2-D, got {tuple(snapshot_cpu.shape)}")
            prev_t = 0 if req_state["accumulated"] is None else int(req_state["accumulated"].shape[0])
            new_rows = snapshot_cpu[prev_t:]
            if new_rows.numel() > 0:
                if req_state["accumulated"] is None:
                    req_state["accumulated"] = new_rows
                else:
                    req_state["accumulated"] = torch.cat([req_state["accumulated"], new_rows], dim=0)

    acc = req_state["accumulated"]
    if acc is None or acc.numel() == 0:
        if not is_finished:
            return None
        del state[req_id]
        transfer_manager._audio8_flushed_reqs.add(req_id)
        # Flush sentinel: the codec request must still complete cleanly.
        return OmniPayloadStruct(
            codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
            meta=MetaStruct(
                req_id=[req_id],
                left_context_size=0,
                codec_chunk_frames=0,
                codec_left_context_frames=0,
                stream_finished=torch.tensor(True, dtype=torch.bool),
                finished=torch.tensor(True, dtype=torch.bool),
            ),
            request_id=req_id,
        )

    if not is_finished:
        return None

    del state[req_id]
    transfer_manager._audio8_flushed_reqs.add(req_id)
    t_acc = int(acc.shape[0])
    codec_flat = acc.transpose(0, 1).contiguous().reshape(-1)  # [K * T]
    return OmniPayloadStruct(
        codes=CodesStruct(audio=codec_flat),
        meta=MetaStruct(
            req_id=[req_id],
            left_context_size=0,
            codec_chunk_frames=t_acc,
            codec_left_context_frames=0,
            code_flat_numel=int(codec_flat.numel()),
            stream_finished=torch.tensor(True, dtype=torch.bool),
            finished=torch.tensor(True, dtype=torch.bool),
        ),
        request_id=req_id,
    )
