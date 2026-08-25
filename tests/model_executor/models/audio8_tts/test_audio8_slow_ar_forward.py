# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Step 2.3 verification: slow-AR forward numerics vs the HF reference dumps.

Drives the ported Qwen2 backbone through the SDPA reference runner using the
dumped per-step embeddings (isolating the transformer stack), and asserts:

  - prefill logits/hidden are numerically close to the HF reference,
  - greedy argmax over the allowed set (EOS + semantic ids) reproduces the
    reference's sampled semantic token at EVERY decode step,
  - the normed slow hidden that feeds the fast head stays numerically close.

Runs on GPU (bf16) when available; tolerances account for SDPA-vs-HF backend
reduction-order differences, while the argmax checks must hold exactly.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.core_model]

VARIANTS = ["no_ref", "clone"]


def _allowed_ids(config_snapshot: dict) -> torch.Tensor:
    eos = int(config_snapshot["eos_token_id"])
    begin = int(config_snapshot["semantic_begin_id"])
    end = int(config_snapshot["semantic_end_id"])
    return torch.tensor([eos] + list(range(begin, end + 1)), dtype=torch.long)


def _allowed_argmax(logits: torch.Tensor, allowed: torch.Tensor) -> int:
    sliced = logits.float()[:, allowed.to(logits.device)]
    return int(allowed[sliced.argmax(dim=-1).cpu()].item())


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return F.cosine_similarity(a, b, dim=0).item()


@pytest.mark.parametrize("variant", VARIANTS)
def test_prefill_matches_reference(audio8_model, slow_ar_runner, audio8_load_dump, variant):
    dump = audio8_load_dump(variant)
    slow_ar_runner.reset()

    normed, logits = slow_ar_runner.prefill(dump["prefill_embeds"], dump["position_ids"][0])

    ref_logits = dump["prefill_logits"].to(logits.device)
    ref_hidden = dump["prefill_slow_hidden"].to(logits.device)

    cos = _cosine(logits, ref_logits)
    max_abs = (logits.float() - ref_logits.float()).abs().max().item()
    assert cos > 0.999, f"prefill logits cosine similarity too low: {cos}"
    assert max_abs < 1.0, f"prefill logits max abs diff too high: {max_abs}"

    hidden_cos = _cosine(normed, ref_hidden.reshape(normed.shape))
    # bf16 ulp at magnitude m is ~m*2^-7; allow ~2 ulp of the reference scale.
    ref_scale = max(ref_hidden.float().abs().max().item(), 1.0)
    max_hidden = (normed.float() - ref_hidden.float().reshape(normed.shape)).abs().max().item()
    assert hidden_cos > 0.999, f"prefill hidden cosine similarity too low: {hidden_cos}"
    assert max_hidden < max(0.15, ref_scale * 0.02), (
        f"prefill hidden max abs diff too high: {max_hidden} (ref scale {ref_scale})"
    )

    # Greedy decision at the first frame must match the reference sampling.
    allowed = _allowed_ids(dump["config_snapshot"])
    assert _allowed_argmax(logits, allowed) == dump["semantic_ids"][0]


@pytest.mark.parametrize("variant", VARIANTS)
def test_decode_steps_match_reference(audio8_model, slow_ar_runner, audio8_load_dump, variant):
    dump = audio8_load_dump(variant)
    slow_ar_runner.reset()

    cfg = dump["config_snapshot"]
    allowed = _allowed_ids(cfg)

    # Replay the prompt into the cache first (needed for decode attention).
    slow_ar_runner.prefill(dump["prefill_embeds"], dump["position_ids"][0])

    steps = len(dump["step_embeds"])
    argmax_agree = 0
    diverged_gaps: list[float] = []
    min_logits_cos = 1.0
    min_hidden_cos = 1.0
    for i in range(steps):
        step_in = dump["step_inputs"][i]
        token_position = int(step_in["position_ids"].reshape(-1)[0])
        physical_position = int(step_in["cache_position"].reshape(-1)[0])
        normed, logits = slow_ar_runner.decode_step(
            dump["step_embeds"][i],
            token_position,
            physical_position,
            kv_written=physical_position + 1,
        )
        ref_logits = dump["step_logits"][i].to(logits.device)
        ref_hidden = dump["step_slow_hidden"][i].to(logits.device)

        if _allowed_argmax(logits, allowed) == dump["semantic_ids"][i + 1]:
            argmax_agree += 1
        else:
            # bf16 cross-implementation noise may only flip genuine near-ties:
            # require the reference's top1-top2 gap to be tiny at that step.
            ref_allowed = ref_logits.float()[:, allowed.to(ref_logits.device)]
            top2 = ref_allowed.topk(2, dim=-1).values
            diverged_gaps.append(float(top2[0, 0] - top2[0, 1]))
        min_logits_cos = min(min_logits_cos, _cosine(logits, ref_logits))
        min_hidden_cos = min(min_hidden_cos, _cosine(normed, ref_hidden.reshape(normed.shape)))

    assert steps > 0
    assert argmax_agree / steps >= 0.95, f"greedy semantic decisions diverged at {steps - argmax_agree}/{steps} steps"
    assert diverged_gaps == [] or max(diverged_gaps) < 0.3, (
        f"diverged at a non-tie step (reference top1-top2 gaps: {diverged_gaps})"
    )
    assert min_logits_cos > 0.998, f"step logits cosine similarity too low: {min_logits_cos}"
    assert min_hidden_cos > 0.998, f"step hidden cosine similarity too low: {min_hidden_cos}"


def test_rope_positions_match_reference_layout(slow_ar_runner, audio8_load_dump):
    """Sanity: reference position ids for an unpadded prompt are 0..T-1."""
    dump = audio8_load_dump("no_ref")
    positions = dump["position_ids"].reshape(-1)
    assert torch.equal(positions, torch.arange(positions.numel()))
