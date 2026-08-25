# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Step 2.2 verification: integrity of the HF-reference tensor dumps.

The dumps are produced by ``dump_reference_tensors.py`` in the baseline
environment (transformers 4.57) and live outside the repo.  Point
``AUDIO8_TTS_BASELINE_DIR`` at the directory holding
``ref_tensors_<variant>.pt`` and ``golden_<variant>.npy`` (skipped otherwise).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

VARIANTS = ["no_ref", "clone"]


def _baseline_dir() -> Path:
    raw = os.environ.get("AUDIO8_TTS_BASELINE_DIR")
    if not raw:
        pytest.skip("AUDIO8_TTS_BASELINE_DIR not set (local baseline artifacts required)")
    path = Path(raw)
    if not path.is_dir():
        pytest.skip(f"{path} does not exist")
    return path


def _load(variant: str) -> tuple[dict, np.ndarray]:
    base = _baseline_dir()
    dump = torch.load(base / f"ref_tensors_{variant}.pt", weights_only=True)
    golden = np.load(base / f"golden_{variant}.npy")
    return dump, golden


@pytest.mark.parametrize("variant", VARIANTS)
def test_dump_shapes_and_lengths(variant):
    dump, golden = _load(variant)
    num_codebooks, hidden = 10, 896
    eos = int(dump["config_snapshot"]["eos_token_id"])

    prompt_ids = dump["prompt_ids"]
    prompt_width = prompt_ids.shape[-1]
    assert prompt_ids.shape == (1, num_codebooks + 1, prompt_width)
    assert dump["prompt_mask"].shape == (1, prompt_width)
    assert bool(dump["prompt_mask"].all()), "single-item prompt must be unpadded"
    assert dump["prefill_embeds"].shape == (prompt_width, hidden)
    assert dump["prefill_embeds"].dtype == torch.bfloat16
    assert dump["prefill_logits"].shape == (1, 155_776)
    assert dump["prefill_slow_hidden"].shape == (1, 1, hidden)

    semantic_ids = dump["semantic_ids"]
    valid_frames = sum(1 for s in semantic_ids if s != eos)
    assert semantic_ids[-1] == eos, "generation must end with EOS"
    assert valid_frames == golden.shape[1]

    # The EOS iteration samples semantic+codebooks but runs no further step.
    steps = len(dump["step_inputs"])
    assert steps == len(semantic_ids) - 1
    assert len(dump["step_logits"]) == steps
    assert len(dump["step_slow_hidden"]) == steps
    assert len(dump["step_embeds"]) == steps
    for i in range(steps):
        assert dump["step_inputs"][i]["input_ids"].shape == (1, num_codebooks + 1, 1)
        assert dump["step_logits"][i].shape == (1, 155_776)
        assert dump["step_slow_hidden"][i].shape == (1, 1, hidden)


@pytest.mark.parametrize("variant", VARIANTS)
def test_dump_codes_match_golden(variant):
    dump, golden = _load(variant)
    eos = int(dump["config_snapshot"]["eos_token_id"])
    begin = int(dump["config_snapshot"]["semantic_begin_id"])

    semantic_ids = dump["semantic_ids"]
    valid_frames = sum(1 for s in semantic_ids if s != eos)
    stacked = torch.cat(dump["codebooks"], dim=0)[:valid_frames]  # [T, 10]

    assert np.array_equal(stacked.numpy().T, golden), "dumped codes must equal golden npy"

    # Codebook 0 is the semantic code; rows must agree with sampled tokens.
    semantic_codes = np.array([s - begin for s in semantic_ids[:valid_frames]])
    assert np.array_equal(stacked[:, 0].numpy(), semantic_codes)
    assert semantic_codes.min() >= 0 and semantic_codes.max() < 4096


@pytest.mark.parametrize("variant", VARIANTS)
def test_fast_inputs_align_with_slow_hidden(variant):
    """The fast head consumes the normed slow hidden of the same iteration."""
    dump, _ = _load(variant)
    semantic_ids = dump["semantic_ids"]

    for i, fast_in in enumerate(dump["fast_inputs"]):
        if i == 0:
            reference = dump["prefill_slow_hidden"]
        else:
            reference = dump["step_slow_hidden"][i - 1]
        assert torch.equal(fast_in["slow_hidden"], reference)
        assert int(fast_in["semantic"].reshape(-1)[0]) == semantic_ids[i]
