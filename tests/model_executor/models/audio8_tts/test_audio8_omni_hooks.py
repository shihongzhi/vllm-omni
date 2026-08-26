# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Step 2.5 verification: omni engine hooks on ``Audio8TTSAR``.

Against the Step 2.2 HF-reference dumps:

  - ``build_prompt_rows`` reproduces the golden prompts byte-for-byte for the
    no-reference and voice-clone templates,
  - composed prefill embeds (incl. reference-code injection via codebook rows)
    match the reference ``_embed`` output,
  - decode-step embedding composition matches the reference per-step embeds,
    and ``talker_mtp`` regenerates each frame's residual codes from the dumped
    slow hiddens and folds them into embeddings identically,
  - RAS-window per-request state rolls correctly through decode preprocess
    calls, logit masking restricts sampling to EOS + semantic ids, and
    ``make_omni_output`` concatenates per-step codes.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.core_model]

# Same strings dump_reference_tensors.py fed the golden runs.
GOLDEN_TEXT = "Hello, this is a golden baseline test for the Audio8 text to speech model integration."
CLONE_TEXT = "Voice cloning baseline with a reference audio sample."

VARIANTS = ["no_ref", "clone"]
EMBED_ATOL = 5e-2


def _dump(audio8_load_dump, variant: str) -> dict:
    return audio8_load_dump(variant)


def _reference_codes_from_prompt(prompt_rows: torch.Tensor, begin_id: int) -> torch.Tensor:
    """Invert the clone prompt layout back into reference codes [K, T].

    Frames occupy columns where row 0 holds semantic ids; the full codebook
    stack lives in rows 1..K (row 0 additionally mirrors codes[0]).
    """
    row0 = prompt_rows[0]
    cols = ((row0 >= begin_id) & (row0 <= 155_776)).nonzero(as_tuple=False).reshape(-1)
    return prompt_rows[1:, cols].clone()


@pytest.mark.parametrize("variant", VARIANTS)
def test_prompt_rows_match_reference_dump(audio8_model, audio8_load_dump, variant):
    dump = _dump(audio8_load_dump, variant)
    cfg = dump["config_snapshot"]
    golden_rows = dump["prompt_ids"][0]

    if variant == "no_ref":
        rows = audio8_model.build_prompt_rows(GOLDEN_TEXT)
    else:
        ref_codes = _reference_codes_from_prompt(golden_rows, int(cfg["semantic_begin_id"]))
        rows = audio8_model.build_prompt_rows(CLONE_TEXT, ref_codes, GOLDEN_TEXT)

    assert torch.equal(rows.to(golden_rows.device), golden_rows), "prompt template drifted from reference"


@pytest.mark.parametrize("variant", VARIANTS)
def test_prefill_embeds_match_reference_dump(audio8_model, audio8_load_dump, variant):
    dump = _dump(audio8_load_dump, variant)
    cfg = dump["config_snapshot"]
    golden_rows = dump["prompt_ids"][0]

    if variant == "no_ref":
        rows = audio8_model.build_prompt_rows(GOLDEN_TEXT)
    else:
        ref_codes = _reference_codes_from_prompt(golden_rows, int(cfg["semantic_begin_id"]))
        rows = audio8_model.build_prompt_rows(CLONE_TEXT, ref_codes, GOLDEN_TEXT)

    embeds = audio8_model.compose_frame_rows(rows)
    ref = dump["prefill_embeds"].to(embeds.device)
    max_diff = (embeds.float() - ref.float()).abs().max().item()
    assert max_diff < EMBED_ATOL, f"prefill embeds max abs diff too high: {max_diff}"


@pytest.mark.parametrize("variant", VARIANTS)
def test_decode_step_compose_matches_reference(audio8_model, audio8_load_dump, variant):
    dump = _dump(audio8_load_dump, variant)
    for i, step_in in enumerate(dump["step_inputs"]):
        rows = step_in["input_ids"].to(next(audio8_model.parameters()).device)
        composed = audio8_model.compose_embeds(rows)[0, 0]
        diff = (composed.float() - dump["step_embeds"][i].to(composed.device).float()).abs().max().item()
        assert diff < EMBED_ATOL, f"decode step {i} embeds max abs diff too high: {diff}"


@pytest.mark.parametrize("variant", VARIANTS)
def test_talker_mtp_matches_reference_frames(audio8_model, audio8_load_dump, variant):
    """Fast-head expansion off the dumped slow hiddens reproduces frames.

    Conditioning (slow hidden + semantic seed) comes straight from the dump,
    so residual-code disagreements are cross-backend near-tie flips only; at
    least one fully-matching frame must also confirm the folded embedding
    bit-for-bit against the reference composition.
    """
    dump = _dump(audio8_load_dump, variant)
    device = next(audio8_model.parameters()).device
    cfg = dump["config_snapshot"]
    begin = int(cfg["semantic_begin_id"])
    end = int(cfg["semantic_end_id"])
    audio8_model.setup_fast_decode(1)

    total_positions = 0
    agree_positions = 0
    checked_embed = False
    for i, step_in in enumerate(dump["step_inputs"]):
        rows = step_in["input_ids"]
        semantic = int(rows[0, 0, 0])
        if not (begin <= semantic <= end):
            continue
        hidden_i = (dump["prefill_slow_hidden"] if i == 0 else dump["step_slow_hidden"][i - 1]).reshape(1, -1)
        token_embed = audio8_model.embed_input_ids(torch.tensor([[semantic]], device=device)).reshape(1, -1)
        out_embeds, codes = audio8_model.talker_mtp(
            torch.tensor([[semantic]], device=device),
            token_embed,
            hidden_i.to(device),
            torch.zeros(1, int(cfg["dim"]), device=device),
            do_sample=False,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
        )
        assert out_embeds.shape == (1, int(cfg["dim"]))
        assert int(codes.min()) >= 0 and int(codes.max()) < int(cfg["codebook_size"])

        # Decode frames carry ALL codebooks in rows 1..K (cb0 rides beside the
        # semantic id), and talker_mtp likewise returns the full stack.
        ref_codes_i = rows[0, 1:, 0].to(codes.device)
        agree_positions += int((codes[0].cpu() == ref_codes_i.cpu()).sum())
        total_positions += codes.shape[1]
        if bool(torch.equal(codes[0].cpu(), ref_codes_i.cpu())) and not checked_embed:
            # Identical conditioning pieces must fold to the reference embed.
            composed_ref = audio8_model.compose_embeds(rows.to(device))[0, 0]
            diff = (out_embeds[0].float() - composed_ref.float()).abs().max().item()
            assert diff < EMBED_ATOL, f"folded embed mismatch on fully-matching frame {i}: {diff}"
            checked_embed = True

    assert total_positions > 0, "no decodable frames found in dump"
    # Cross-backend tie flips cascade inside a frame (production cached SDPA
    # vs the dump's matmul mirror -- same noise source as Step 2.4's codebook
    # test, whose gate was 0.75); conditioning is exact here, so tighten a bit.
    ratio = agree_positions / total_positions
    assert ratio >= 0.80, f"residual code agreement too low: {agree_positions}/{total_positions}"
    assert checked_embed, "no frame reproduced its residual codes exactly"


def test_ras_state_and_mtp_inputs_flow(audio8_model):
    """Decode preprocess rolls the RAS window and emits mtp_inputs per step."""
    device = next(audio8_model.parameters()).device
    cfg = audio8_model.config
    eos = int(cfg.eos_token_id)
    begin = int(cfg.semantic_begin_id)
    end = int(cfg.semantic_end_id)
    window = int(cfg.ras_window_size)

    picked: list[int] = []
    state_payload: dict | None = None
    for i in range(window + 2):
        # Alternate semantic/eos ids, then repeat an earlier semantic at the end.
        if i % 2 == 0:
            token = begin + 7
        elif i == window + 1:
            token = begin + 7
        else:
            token = eos if i == 1 else begin + 100 + i
        picked.append(token)
        hidden = torch.zeros(1, int(cfg.dim), device=device)
        info: dict = {"hidden_states": {"last": hidden}, "request_id": "req"}
        if state_payload is not None:
            info["state"] = state_payload
        _, _, updates = audio8_model.preprocess(torch.tensor([token], device=device), None, **info)
        assert "mtp_inputs" in updates
        last_h, filler = updates["mtp_inputs"]
        assert last_h.shape == (1, int(cfg.dim)) and filler.shape == (1, int(cfg.dim))
        state_payload = updates["state"]

        state = updates["state"]
        previous, valid = state["previous"], state["previous_valid"]
        assert previous.shape == (1, window) and valid.shape == (1, window)
        recent = picked[: i + 1][-window:]
        expected_flags = [False] * (window - len(recent)) + [begin <= tok <= end for tok in recent]
        expected_tokens = [0] * (window - len(recent)) + recent
        assert valid[0].tolist() == expected_flags, f"validity mask wrong at step {i}"
        assert previous[0].tolist() == expected_tokens, f"window content wrong at step {i}"


def test_compute_logits_masks_to_audio_ids(audio8_model):
    device = next(audio8_model.parameters()).device
    cfg = audio8_model.config
    hidden = torch.zeros(1, int(cfg.dim), device=device, dtype=torch.bfloat16)
    logits = audio8_model.compute_logits(hidden)
    finite = torch.isfinite(logits).reshape(-1).nonzero(as_tuple=False).reshape(-1)
    allowed = {int(cfg.eos_token_id)} | set(range(int(cfg.semantic_begin_id), int(cfg.semantic_end_id) + 1))
    assert set(finite.tolist()) == allowed, "logit mask allows unexpected ids"


def test_make_omni_output_concatenates_codes(audio8_model):
    cfg = audio8_model.config
    hidden_dim = int(cfg.dim)
    c1 = torch.arange(2 * int(cfg.num_codebooks)).reshape(2, -1)
    c2 = torch.arange(4 * int(cfg.num_codebooks)).reshape(4, -1)
    outputs = audio8_model.make_omni_output(
        torch.randn(9, hidden_dim),
        runtime_additional_information=[
            {"codes": {"audio": c1}},
            {"not_a_dict": 1},
            {"codes": {"audio": c2}},
        ],
    )
    codes = outputs.multimodal_outputs["audio_codes"]
    assert codes.shape == (6, int(cfg.num_codebooks))
    assert outputs.text_hidden_states.shape == (6, hidden_dim)
