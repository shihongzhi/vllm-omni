# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Step 2.1 verification: Audio8 AR model skeleton + weight loading.

Asserts, against the real ``AutoArk-AI/Audio8-TTS-Preview-0.6b`` checkpoint:

  1. every checkpoint tensor is consumed by ``load_weights`` (exact set match),
  2. the instantiated parameter count equals the published 601,159,424,
  3. selected weights are bitwise identical to the checkpoint after the
     wqkv→qkv_proj / w1+w3→gate_up_proj remapping,
  4. unknown tensor names are rejected loudly.

Skips when the checkpoint is not present in the local HF cache.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import load_file

from vllm_omni.model_executor.models.audio8_tts.audio8_tts_ar import Audio8TTSAR
from vllm_omni.model_executor.models.audio8_tts.configuration_audio8 import Audio8TTSConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

MODEL_REPO = "AutoArk-AI/Audio8-TTS-Preview-0.6b"
EXPECTED_TOTAL_PARAMS = 601_159_424
EXPECTED_NUM_TENSORS = 226


@pytest.fixture(scope="module")
def model_dir():
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            MODEL_REPO,
            local_files_only=True,
            allow_patterns=["config.json", "model.safetensors"],
        )
    except Exception as exc:  # pragma: no cover - depends on local cache
        pytest.skip(f"Audio8 checkpoint not available in local HF cache: {exc}")
    return path


@pytest.fixture(scope="module")
def built_model(model_dir):
    """Build + load the AR model once per module.

    vLLM's Attention registers layer names in a process-global registry, so a
    model with engine attention layers can only be constructed once per
    process (the engine does exactly that).  Fake the TP group here instead of
    using the function-scoped ``init_fake_tp_group`` fixture.
    """
    from vllm.distributed import parallel_state

    mock_tp = MagicMock()
    mock_tp.world_size = 1
    mock_tp.rank_in_group = 0
    old_tp = parallel_state._TP
    parallel_state._TP = mock_tp
    try:
        config = Audio8TTSConfig.from_pretrained(model_dir)
        model = Audio8TTSAR(config).to(torch.bfloat16)
        checkpoint = load_file(f"{model_dir}/model.safetensors")
        model.load_weights(checkpoint.items())
        model.eval()
        yield model, checkpoint
    finally:
        parallel_state._TP = old_tp


def test_all_checkpoint_tensors_consumed(built_model):
    model, checkpoint = built_model
    consumed = model.load_weights(checkpoint.items())
    assert consumed == set(checkpoint.keys())
    assert len(consumed) == EXPECTED_NUM_TENSORS


def test_total_parameter_count(built_model):
    model, _ = built_model
    total = sum(p.numel() for p in model.parameters())
    assert total == EXPECTED_TOTAL_PARAMS


def test_weights_bitwise_equal(built_model):
    model, ckpt = built_model

    def check(ckpt_name: str, param: torch.Tensor):
        expected = ckpt[ckpt_name]
        assert param.dtype == expected.dtype, ckpt_name
        assert param.shape == expected.shape, ckpt_name
        assert torch.equal(param, expected.to(param.dtype)), ckpt_name

    check("embeddings.weight", model.embed_tokens.weight)
    check("codebook_embeddings.weight", model.codebook_embeddings.weight)
    check("norm.weight", model.norm.weight)

    # Slow layer 0: fused QKV weight+bias reassembled from q/k/v shards.
    check("layers.0.attention.wqkv.weight", model.layers[0].self_attn.qkv_proj.weight)
    check("layers.0.attention.wqkv.bias", model.layers[0].self_attn.qkv_proj.bias)
    check("layers.0.attention.wo.weight", model.layers[0].self_attn.o_proj.weight)
    check("layers.11.attention_norm.weight", model.layers[11].input_layernorm.weight)
    check("layers.11.ffn_norm.weight", model.layers[11].post_attention_layernorm.weight)

    # Merged gate_up: gate (w1) is the first shard, up (w3) the second.
    gate_up = model.layers[23].gate_up_proj.weight
    check("layers.23.feed_forward.w1.weight", gate_up[: ckpt["layers.23.feed_forward.w1.weight"].shape[0]])
    check("layers.23.feed_forward.w3.weight", gate_up[ckpt["layers.23.feed_forward.w1.weight"].shape[0] :])
    check("layers.23.feed_forward.w2.weight", model.layers[23].down_proj.weight)

    # Fast head loads 1:1.
    check("fast_embeddings.weight", model.fast_embeddings.weight)
    check("fast_layers.0.attention.wqkv.weight", model.fast_layers[0].attention.wqkv.weight)
    check("fast_layers.3.feed_forward.w2.weight", model.fast_layers[3].feed_forward.w2.weight)
    check("fast_norm.weight", model.fast_norm.weight)
    check("fast_output.weight", model.fast_output.weight)


def test_unknown_weight_name_rejected(built_model):
    model, _ = built_model
    with pytest.raises(KeyError):
        model.load_weights([("totally.bogus.weight", torch.zeros(1))])
