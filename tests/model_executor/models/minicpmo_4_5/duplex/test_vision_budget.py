# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import base64

import pytest

from vllm_omni.model_executor.models.minicpmo_4_5.duplex.policy import MiniCPMO45DuplexPolicy
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.runtime import (
    _duplex_vision_tokens,
    duplex_scheduler_token_budget,
)
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0 import MiniCPMO45Stage0DuplexRuntime

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _pcm_1s() -> str:
    return base64.b64encode(b"\x00" * (16_000 * 4)).decode("ascii")


@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_vision_budget_is_linear_in_frame_count(count):
    payload = {"video_frames": ["Zg=="] * count}
    assert _duplex_vision_tokens(payload) == count * MiniCPMO45DuplexPolicy.VISION_TOKENS_PER_FRAME


@pytest.mark.parametrize("count", [1, 2, 3])
def test_scheduler_vision_budget_matches_stage0_blocks_per_frame(count):
    """Reserved slots must equal the blocks Stage0 actually builds per frame.

    A stacked pair used to reserve HD slices ((3 + 1) * 66) while the duplex
    adapter encodes one 448-normalized tile per frame, so surplus slots
    entered the KV as pad embeddings. The two sides must move together when
    HD slicing lands.
    """
    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    stage0_blocks = sum(runtime._official_max_slice_nums(count))

    assert stage0_blocks == count
    assert stage0_blocks * MiniCPMO45DuplexPolicy.VISION_TOKENS_PER_FRAME == _duplex_vision_tokens(
        {"video_frames": ["Zg=="] * count}
    )


def test_scheduler_token_budget_counts_stacked_pair_as_two_frames():
    payload = {
        "audio": _pcm_1s(),
        "format": "pcm_f32le",
        "sample_rate_hz": 16_000,
        "video_frames": ["Zg==", "Zg=="],
    }

    # 1 unit * (2 unit tokens + 10 audio tokens) + 2 * 66 vision tokens.
    assert duplex_scheduler_token_budget(payload) == 12 + 2 * MiniCPMO45DuplexPolicy.VISION_TOKENS_PER_FRAME
