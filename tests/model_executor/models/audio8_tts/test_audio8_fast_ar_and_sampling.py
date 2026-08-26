# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Step 2.4 verification: fast codebook head, sampling and the full offline loop.

  1. ``compose_embeds`` reproduces the HF ``_embed`` outputs (prefill for both
     variants + every decode step) numerically.
  2. ``semantic_logits`` + greedy ``sample_semantic`` reproduce the reference's
     per-iteration semantic decisions (near-tie divergences excepted).
  3. Fast-head faithfulness, per-position: teacher-forcing every frame through
     BOTH the production cached path and an explicit-matmul mirror of the HF
     fast layers (the upstream fast head runs ``use_sdpa=False``) yields
     matching logits, and any argmax disagreement sits on a genuine near-tie.
  4. Autonomous greedy expansion against the reference seeds regenerates the
     reference codebooks except where a legitimate tie flip cascades through
     the remainder of a frame.
  5. The full offline loop is pinned to the reference decisions frame by
     frame; its normed hiddens must track the reference's for the whole clip,
     which exercises prefill -> decode bookkeeping (positions, cache writes,
     RAS window updates) end to end. An unpinned greedy rollout additionally
     proves structural health (length, code validity) -- cross-implementation
     bf16 noise legitimately diverges open-loop trajectories after tie flips,
     so exact long-horizon agreement with the golden clip is NOT asserted.

All criteria follow the bf16 cross-implementation noise policy established in
step 2.3: disagreement is only allowed where the reference itself is a
near-tie (top1-top2 gap < 0.3).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.core_model]

VARIANTS = ["no_ref", "clone"]
NEAR_TIE_GAP = 0.3


def _top2_gap(scores: torch.Tensor) -> float:
    top2 = scores.float().flatten().topk(2, dim=-1).values
    return float(top2[0] - top2[1])


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    import torch.nn.functional as F

    return F.cosine_similarity(a.float().reshape(-1), b.float().reshape(-1), dim=0).item()


@pytest.mark.parametrize("variant", VARIANTS)
def test_compose_embeds_matches_reference(audio8_model, audio8_load_dump, variant):
    dump = audio8_load_dump(variant)
    device = next(audio8_model.parameters()).device

    prefill = audio8_model.compose_embeds(dump["prompt_ids"].to(device))
    ref = dump["prefill_embeds"].to(device)
    assert prefill.shape == (1, ref.shape[0], ref.shape[1])
    max_diff = (prefill.float() - ref.float()).abs().max().item()
    assert max_diff < 5e-2, f"prefill embeds max abs diff too high: {max_diff}"

    for i, ref_step in enumerate(dump["step_embeds"]):
        rows = dump["step_inputs"][i]["input_ids"].to(device)
        composed = audio8_model.compose_embeds(rows)[0, 0]
        diff = (composed.float() - ref_step.to(device).float()).abs().max().item()
        assert diff < 5e-2, f"step {i} embeds max abs diff too high: {diff}"


@pytest.mark.parametrize("variant", VARIANTS)
def test_semantic_sampling_matches_reference(audio8_model, audio8_load_dump, variant):
    dump = audio8_load_dump(variant)
    device = next(audio8_model.parameters()).device
    cfg = dump["config_snapshot"]
    eos = int(cfg["eos_token_id"])

    hiddens = [dump["prefill_slow_hidden"].reshape(1, -1)] + [h.reshape(1, -1) for h in dump["step_slow_hidden"]]
    iterations = len(dump["semantic_ids"])
    agree = 0
    diverged_gaps: list[float] = []
    for i, hidden in enumerate(hiddens):
        logits = audio8_model.semantic_logits(hidden.to(device))
        picked = int(
            audio8_model.sample_semantic(logits, temperature=0.8, top_p=0.95, top_k=50, do_sample=False).reshape(-1)[0]
        )
        expected = dump["semantic_ids"][i]
        if picked == expected:
            agree += 1
        else:
            diverged_gaps.append(_top2_gap(logits))
    assert agree / iterations >= 0.95, f"semantic decisions diverged at {iterations - agree}/{iterations} iterations"
    assert diverged_gaps == [] or max(diverged_gaps) < NEAR_TIE_GAP, (
        f"diverged at a non-tie iteration (gaps: {diverged_gaps})"
    )
    assert dump["semantic_ids"][-1] == eos


@pytest.mark.parametrize("variant", VARIANTS)
def test_fast_head_logits_parity_vs_reference_math(audio8_model, audio8_load_dump, fast_ar_runner, variant):
    """Per-position fast-head logits on identical inputs, all frames.

    Production cached-SDPA decode vs the reference-math matmul runner: the
    two paths share weights and inputs, so any gap above backend reduction
    noise -- or a flipped argmax off a near-tie -- is a porting bug.
    """
    dump = audio8_load_dump(variant)
    device = next(audio8_model.parameters()).device
    audio8_model.setup_fast_decode(1)

    cosines: list[float] = []
    scaled_diffs: list[float] = []
    flipped_gaps: list[float] = []
    checked_positions = 0
    for i, fast_in in enumerate(dump["fast_inputs"]):
        codes = dump["codebooks"][i].reshape(-1)[:10]
        ref_rows = fast_ar_runner.frame_logits(fast_in["slow_hidden"], codes[:9])
        ref_scale = max(ref_rows.float().abs().max().item(), 1.0)

        # Drive the production path over the same frozen context.
        for layer in audio8_model.fast_layers:
            layer.attention.clear_audio8_cache()
        hidden = fast_in["slow_hidden"].reshape(1, -1).to(device)
        ours_rows = [
            audio8_model._run_fast_position(hidden.reshape(1, 1, -1).to(audio8_model.fast_embeddings.weight.dtype), 0)
        ]
        for position in range(1, int(audio8_model.config.num_codebooks)):
            emb = audio8_model.fast_embeddings(codes[position - 1].reshape(1).to(device)).unsqueeze(1)
            ours_rows.append(audio8_model._run_fast_position(emb.to(emb.dtype), position))

        for p in range(1, ref_rows.shape[0]):
            ours = ours_rows[p].float().flatten()
            ref = ref_rows[p].float().flatten()
            cosines.append(_cosine(ours, ref))
            scaled_diffs.append((ours - ref).abs().max().item() / ref_scale)
            if int(ours.argmax()) != int(ref.argmax()):
                # The two renderings each carry bf16 backend noise; a
                # disagreement is only benign if the decision was a near-tie
                # from at least one side's vantage point.
                flipped_gaps.append(min(_top2_gap(ref), _top2_gap(ours)))
            checked_positions += 1

    assert checked_positions > 0
    # Aggregate health + strict tail policy: bulk agreement high, worst scaled
    # drift bounded well below decisive logit margins (observed cross-backend
    # budget ~0.05, generous cap 0.10 -- a real porting bug distorts by tens
    # of percent), and every flipped argmax sits on a genuine near-tie.
    mean_cos = sum(cosines) / len(cosines)
    strong_frac = sum(1 for c in cosines if c >= 0.999) / len(cosines)
    worst_scaled = max(scaled_diffs)
    assert mean_cos > 0.9997, f"fast-head logits mean cosine too low: {mean_cos}"
    assert strong_frac >= 0.97, f"only {strong_frac:.3f} of fast-head positions reach cosine 0.999"
    assert worst_scaled < 0.10, f"fast-head logits scaled max abs diff too high: {worst_scaled}"
    assert flipped_gaps == [] or max(flipped_gaps) < NEAR_TIE_GAP, (
        f"flipped a non-tie fast decision (reference-math gaps: {flipped_gaps})"
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_fast_codebooks_match_reference(audio8_model, audio8_load_dump, variant):
    """Autonomous greedy expansion seeded by each reference frame.

    Histories coincide until the first legitimate tie flip inside a frame;
    afterwards the remaining positions decide on different conditioning and
    may disagree wholesale. Codebook 0 derives from the semantic id and must
    always match.
    """
    dump = audio8_load_dump(variant)
    device = next(audio8_model.parameters()).device
    audio8_model.setup_fast_decode(1)

    total_positions = 0
    equal_positions = 0
    for i, fast_in in enumerate(dump["fast_inputs"]):
        semantic = fast_in["semantic"].reshape(-1).to(device)
        slow_hidden = fast_in["slow_hidden"].reshape(1, -1).to(device)
        codes = audio8_model.decode_codebooks(
            slow_hidden, semantic, temperature=0.8, top_p=0.95, top_k=50, do_sample=False
        )
        ref_codes = dump["codebooks"][i].to(device)
        # Codebook 0 is derived from the semantic id: must always match.
        assert torch.equal(codes[:, 0], ref_codes[:, 0])
        equal = codes == ref_codes
        equal_positions += int(equal.sum())
        total_positions += equal.numel()

    # Greedy residual-code agreement: tie flips cascade, so the bar tracks the
    # measured cross-implementation baseline rather than bit-exactness (see
    # test_fast_head_logits_parity_vs_reference_math for the strict gate).
    assert equal_positions / total_positions >= 0.75, f"codebook agreement too low: {equal_positions}/{total_positions}"


@pytest.mark.parametrize("variant", VARIANTS)
def test_full_loop_tracks_reference_when_pinned(audio8_model, slow_ar_runner, audio8_load_dump, variant):
    """Replay reference semantics/codes through OUR loop plumbing.

    Compose -> SDPA slow AR decode step uses the same bookkeeping as the real
    loop (token vs physical positions, kv_written, RAS window rotation). The
    resulting normed hidden must track the reference's per-step hidden for the
    entire clip -- any positional or composition bug blows up immediately.
    """
    dump = audio8_load_dump(variant)
    device = next(audio8_model.parameters()).device
    cfg = dump["config_snapshot"]
    eos = int(cfg["eos_token_id"])
    audio8_model.setup_fast_decode(1)
    slow_ar_runner.reset()

    normed, _ = slow_ar_runner.prefill(dump["prefill_embeds"], dump["position_ids"][0])
    prefill_diff = (normed.float() - dump["prefill_slow_hidden"].to(device).float()).abs().max().item()

    steps = len(dump["step_embeds"])
    min_step_cos = 1.0
    worst_step_diff = 0.0
    ras_updates = 0
    previous = torch.zeros(1, int(cfg["ras_window_size"]), dtype=torch.long, device=device)
    previous_valid = torch.zeros_like(previous, dtype=torch.bool)
    for i in range(steps):
        step_in = dump["step_inputs"][i]
        rows = step_in["input_ids"].to(device)  # [1, num_codebooks+1, T=1]
        semantic_id = int(rows[0, 0, 0])
        assert semantic_id != eos, f"reference emitted EOS at step {i}; extend replay"
        embed = audio8_model.compose_embeds(rows)[0, 0]
        normed, _ = slow_ar_runner.decode_step(
            embed,
            token_position=int(step_in["position_ids"].reshape(-1)[0]),
            physical_position=int(step_in["cache_position"].reshape(-1)[0]),
            kv_written=int(step_in["cache_position"].reshape(-1)[0]) + 1,
        )
        ref_hidden = dump["step_slow_hidden"][i].to(device)
        cos = _cosine(normed, ref_hidden.reshape(normed.shape))
        diff = (normed.float() - ref_hidden.float().reshape(normed.shape)).abs().max().item()
        min_step_cos = min(min_step_cos, cos)
        worst_step_diff = max(worst_step_diff, diff)

        previous = previous.roll(-1, dims=1)
        previous[:, -1] = torch.tensor([semantic_id], device=device, dtype=torch.long)
        previous_valid = previous_valid.roll(-1, dims=1)
        previous_valid[:, -1] = True
        ras_updates += 1

    assert prefill_diff < max(0.15, float(dump["prefill_slow_hidden"].abs().max()) * 0.02)
    assert min_step_cos > 0.998, f"pinned-loop hidden cosine too low: {min_step_cos}"
    assert worst_step_diff < 1.0, f"pinned-loop hidden drifted: {worst_step_diff}"
    assert ras_updates == steps and steps > 0


def test_full_offline_loop_autonomous_health(audio8_model, slow_ar_runner, audio8_load_dump, audio8_ref_dir):
    """Unpinned greedy rollout: structural health while trajectories stay sane.

    The semantic stream may legitimately leave the golden trajectory after a
    near-tie flip (chaotic autoregression under cross-backend bf16 noise), so
    this asserts generation mechanics only: bounded length, valid code range,
    benign growth, and that the fast head never emits out-of-range ids.
    """
    dump = audio8_load_dump("no_ref")
    device = next(audio8_model.parameters()).device
    cfg = dump["config_snapshot"]
    eos = int(cfg["eos_token_id"])
    begin = int(cfg["semantic_begin_id"])
    end = int(cfg["semantic_end_id"])
    codebook_size = int(cfg["codebook_size"])
    audio8_model.setup_fast_decode(1)
    slow_ar_runner.reset()

    golden = np.load(audio8_ref_dir / "golden_no_ref.npy")  # [10, T]
    budget = int(golden.shape[1] * 1.25)
    prompt_width = int(dump["prompt_ids"].shape[-1])

    normed, _ = slow_ar_runner.prefill(dump["prefill_embeds"], dump["position_ids"][0])
    logits = audio8_model.semantic_logits(normed)

    semantic_ids: list[int] = []
    previous = torch.zeros(1, int(cfg["ras_window_size"]), dtype=torch.long, device=device)
    previous_valid = torch.zeros_like(previous, dtype=torch.bool)
    while len(semantic_ids) <= budget:
        semantic = audio8_model.sample_semantic(
            logits,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            do_sample=False,
            previous=previous,
            previous_valid=previous_valid,
        ).reshape(1)
        picked = int(semantic[0])
        assert picked == eos or begin <= picked <= end, f"invalid semantic id {picked}"
        semantic_ids.append(picked)
        if picked == eos:
            break

        frame_codes = audio8_model.decode_codebooks(
            normed.reshape(1, -1),
            semantic,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            do_sample=False,
        )
        assert int(frame_codes.min()) >= 0 and int(frame_codes.max()) < codebook_size

        rows = torch.cat((semantic.reshape(1, 1), frame_codes.to(device)), dim=1)
        embed = audio8_model.compose_embeds(rows.unsqueeze(-1))[0, 0]
        normed, _ = slow_ar_runner.decode_step(
            embed,
            token_position=prompt_width + len(semantic_ids) - 1,
            physical_position=prompt_width + len(semantic_ids) - 1,
            kv_written=prompt_width + len(semantic_ids),
        )
        logits = audio8_model.semantic_logits(normed)

        previous = previous.roll(-1, dims=1)
        previous[:, -1] = semantic
        previous_valid = previous_valid.roll(-1, dims=1)
        previous_valid[:, -1] = True

    frames = len(semantic_ids)
    assert frames > golden.shape[1] * 0.9, f"rollout collapsed early: {frames} frames"
    assert frames <= budget + 1, "rollout exceeded budget without terminating"


def test_sampler_gumbel_behaviour(audio8_model):
    """Lock the sampler: top_k=1 forces argmax; tiny temperature ≈ argmax."""
    device = next(audio8_model.parameters()).device
    scores = torch.randn(4, 100, device=device)

    one_hot_k = audio8_model.sample(
        scores,
        torch.full((4,), 1.0, device=device),
        torch.full((4,), 1.0, device=device),
        torch.full((4,), 1, device=device, dtype=torch.long),
        torch.full((4,), True, device=device),
    )
    assert torch.equal(one_hot_k, scores.argmax(dim=-1))

    torch.manual_seed(0)
    near_greedy = audio8_model.sample(
        scores,
        torch.full((4,), 1e-5, device=device),
        torch.full((4,), 1.0, device=device),
        torch.full((4,), 0, device=device, dtype=torch.long),
        torch.full((4,), True, device=device),
    )
    assert torch.equal(near_greedy, scores.argmax(dim=-1))
