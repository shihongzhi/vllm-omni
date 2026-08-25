# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Step 2.1 verification: Audio8 AR model skeleton + weight loading.

Asserts, against the real ``AutoArk-AI/Audio8-TTS-Preview-0.6b`` checkpoint:

  1. every checkpoint tensor is consumed by ``load_weights`` (exact set match),
  2. the instantiated parameter count equals the published 601,159,424,
  3. selected weights are bitwise identical to the checkpoint after the
     wqkv→qkv_proj / w1+w3→gate_up_proj remapping,
  4. unknown tensor names are rejected loudly.

Model construction lives in ``conftest.audio8_model`` (session-scoped).
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.core_model]

EXPECTED_TOTAL_PARAMS = 601_159_424
EXPECTED_NUM_TENSORS = 226


def test_all_checkpoint_tensors_consumed(audio8_model, audio8_checkpoint):
    consumed = audio8_model.load_weights(audio8_checkpoint.items())
    assert consumed == set(audio8_checkpoint.keys())
    assert len(consumed) == EXPECTED_NUM_TENSORS


def test_total_parameter_count(audio8_model):
    total = sum(p.numel() for p in audio8_model.parameters())
    assert total == EXPECTED_TOTAL_PARAMS


def test_weights_bitwise_equal(audio8_model, audio8_checkpoint):
    model, ckpt = audio8_model, audio8_checkpoint

    def check(ckpt_name: str, param: torch.Tensor):
        expected = ckpt[ckpt_name]
        actual = param.detach().cpu()
        assert actual.dtype == expected.dtype, ckpt_name
        assert actual.shape == expected.shape, ckpt_name
        assert torch.equal(actual, expected.to(actual.dtype)), ckpt_name

    backbone = model.model
    check("embeddings.weight", backbone.embed_tokens.weight)
    check("norm.weight", backbone.norm.weight)
    check("codebook_embeddings.weight", model.codebook_embeddings.weight)

    # Slow layer 0: fused QKV weight+bias reassembled from q/k/v shards.
    check("layers.0.attention.wqkv.weight", backbone.layers[0].self_attn.qkv_proj.weight)
    check("layers.0.attention.wqkv.bias", backbone.layers[0].self_attn.qkv_proj.bias)
    check("layers.0.attention.wo.weight", backbone.layers[0].self_attn.o_proj.weight)
    check("layers.11.attention_norm.weight", backbone.layers[11].input_layernorm.weight)
    check("layers.11.ffn_norm.weight", backbone.layers[11].post_attention_layernorm.weight)

    # Merged gate_up: gate (w1) is the first shard, up (w3) the second.
    gate_up = backbone.layers[23].mlp.gate_up_proj.weight
    check(
        "layers.23.feed_forward.w1.weight",
        gate_up[: ckpt["layers.23.feed_forward.w1.weight"].shape[0]],
    )
    check(
        "layers.23.feed_forward.w3.weight",
        gate_up[ckpt["layers.23.feed_forward.w1.weight"].shape[0] :],
    )
    check("layers.23.feed_forward.w2.weight", backbone.layers[23].mlp.down_proj.weight)

    # Fast head loads 1:1.
    check("fast_embeddings.weight", model.fast_embeddings.weight)
    check(
        "fast_layers.0.attention.wqkv.weight",
        model.fast_layers[0].attention.wqkv.weight,
    )
    check(
        "fast_layers.3.feed_forward.w2.weight",
        model.fast_layers[3].feed_forward.w2.weight,
    )
    check("fast_norm.weight", model.fast_norm.weight)
    check("fast_output.weight", model.fast_output.weight)


def test_rope_is_interleaved_and_bf16_truncated(audio8_model):
    """Audio8 uses GPT-J style RoPE with a bf16 table, unlike stock Qwen2."""
    from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

    rotary = audio8_model.model.layers[0].self_attn.rotary_emb
    assert isinstance(rotary, RotaryEmbedding)
    assert rotary.is_neox_style is False
    # cos_sin_cache round-trips through bf16, so it must equal its bf16 cast.
    cache = rotary.cos_sin_cache
    assert torch.equal(cache, cache.to(torch.bfloat16).to(cache.dtype))


def test_unknown_weight_name_rejected(audio8_model):
    with pytest.raises(KeyError):
        audio8_model.load_weights([("totally.bogus.weight", torch.zeros(1))])
