# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared fixtures for Audio8 TTS model tests.

``audio8_model`` is session-scoped on purpose: vLLM's Attention registers
layer names in a process-global registry, so the model (with engine
attention layers) can only be constructed once per test session -- the
engine does exactly that per worker process.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

MODEL_REPO = "AutoArk-AI/Audio8-TTS-Preview-0.6b"


def register_audio8_in_vllm() -> None:
    """Register the arktts config class and ArkttsModel architecture.

    Same mechanism the engine uses for omni models; needed here so
    ``ModelConfig`` accepts the checkpoint's architecture.
    """
    from transformers import AutoConfig
    from vllm.model_executor.models.registry import ModelRegistry

    from vllm_omni.model_executor.models.audio8_tts.configuration_audio8 import (
        Audio8TTSConfig,
    )

    try:
        AutoConfig.register("arktts", Audio8TTSConfig)
    except ValueError:
        pass  # already registered
    ModelRegistry.register_model(
        "ArkttsModel",
        "vllm_omni.model_executor.models.audio8_tts.audio8_tts_ar:Audio8TTSAR",
    )


def fake_single_process_groups() -> tuple[MagicMock, MagicMock]:

    mock_tp = MagicMock()
    mock_tp.world_size = 1
    mock_tp.rank_in_group = 0
    mock_pp = MagicMock()
    mock_pp.world_size = 1
    mock_pp.rank_in_group = 0
    mock_pp.is_last_rank = True
    mock_pp.is_first_rank = True
    return mock_tp, mock_pp


@pytest.fixture(scope="session")
def audio8_model_dir():
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(MODEL_REPO, local_files_only=True)
    except Exception as exc:  # pragma: no cover - depends on local cache
        pytest.skip(f"Audio8 checkpoint not available in local HF cache: {exc}")


@pytest.fixture(scope="session")
def audio8_model(audio8_model_dir):
    """Build + load the Audio8 AR model once, on GPU when available."""
    from safetensors.torch import load_file
    from vllm.config import ModelConfig, VllmConfig
    from vllm.distributed import parallel_state

    from vllm_omni.model_executor.models.audio8_tts.audio8_tts_ar import Audio8TTSAR

    register_audio8_in_vllm()
    mock_tp, mock_pp = fake_single_process_groups()
    old_tp, old_pp = parallel_state._TP, parallel_state._PP
    parallel_state._TP = mock_tp
    parallel_state._PP = mock_pp
    try:
        model_config = ModelConfig(model=audio8_model_dir)
        vllm_config = VllmConfig(model_config=model_config)
        model = Audio8TTSAR(vllm_config=vllm_config).to(torch.bfloat16)
        checkpoint = load_file(f"{audio8_model_dir}/model.safetensors")
        model.load_weights(checkpoint.items())
        del checkpoint
        model.eval()
        if torch.cuda.is_available():
            model = model.to("cuda")
        yield model
    finally:
        parallel_state._TP = old_tp
        parallel_state._PP = old_pp


@pytest.fixture(scope="session")
def audio8_checkpoint(audio8_model_dir):
    """Freshly loaded checkpoint dict (tests mutate nothing, but keep it local)."""
    from safetensors.torch import load_file

    return load_file(f"{audio8_model_dir}/model.safetensors")


@pytest.fixture(scope="session")
def audio8_ref_dir():
    raw = os.environ.get("AUDIO8_TTS_BASELINE_DIR")
    if not raw or not Path(raw).is_dir():
        pytest.skip("AUDIO8_TTS_BASELINE_DIR not set (local baseline artifacts required)")
    return Path(raw)


@pytest.fixture(scope="session")
def audio8_load_dump(audio8_ref_dir):
    def load(variant: str) -> dict:
        return torch.load(audio8_ref_dir / f"ref_tensors_{variant}.pt", weights_only=True)

    return load


def _rms_norm(norm_module, x: torch.Tensor) -> torch.Tensor:
    """Manual RMSNorm matching HF ArkttsRMSNorm/Qwen2 math exactly.

    vLLM's RMSNorm CustomOp (ir.ops.rms_norm) behaves differently outside the
    engine context, so the reference runner applies the math directly.
    """
    normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + norm_module.variance_epsilon)
    return normalized.to(x.dtype) * norm_module.weight


class SlowARSDPARunner:
    """Reference-math forward for the Qwen2 backbone of ``Audio8TTSAR``.

    Reuses the loaded modules (qkv/o projections, RoPE, norms, MLPs) but runs
    the attention core with ``F.scaled_dot_product_attention`` against an
    explicit per-layer KV cache -- no engine attention metadata required.
    Used by the numeric-parity tests; the production path is the engine's
    paged attention.
    """

    def __init__(self, ar_model) -> None:
        import torch.nn.functional as F  # noqa: F401  (kept for callers)

        self.ar = ar_model
        cfg = ar_model.config
        self.cfg = cfg
        self.num_heads = int(cfg.n_head)
        self.num_kv_heads = int(cfg.n_local_heads)
        self.head_dim = int(cfg.head_dim)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        device = next(ar_model.parameters()).device
        self.device = device
        self.dtype = next(ar_model.parameters()).dtype
        n_layers = int(cfg.n_layer)
        # [layers, batch=1, kv_heads, max_pos, head_dim]
        cache_shape = (n_layers, 1, self.num_kv_heads, int(cfg.max_seq_len), self.head_dim)
        self.k_cache = torch.zeros(cache_shape, dtype=self.dtype, device=device)
        self.v_cache = torch.zeros_like(self.k_cache)

    def _attention(
        self,
        layer_idx: int,
        layer,
        x: torch.Tensor,
        positions: torch.Tensor,
        cache_position: torch.Tensor,
        kv_written: int,
    ):
        """x: [tokens, hidden]; positions/cache_position: [tokens] (physical)."""
        import torch.nn.functional as F

        tokens = x.shape[0]
        qkv, _ = layer.self_attn.qkv_proj(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = layer.self_attn.rotary_emb(positions, q, k)
        q = q.view(tokens, self.num_heads, self.head_dim).transpose(0, 1)  # [H, T, D]
        k = k.view(tokens, self.num_kv_heads, self.head_dim).transpose(0, 1)  # [KV, T, D]
        v = v.view(tokens, self.num_kv_heads, self.head_dim).transpose(0, 1)

        # Single advanced index keeps its position: result is [KV, T, D].
        self.k_cache[layer_idx, 0, :, cache_position, :] = k
        self.v_cache[layer_idx, 0, :, cache_position, :] = v

        keys = self.k_cache[layer_idx, 0, :, :kv_written, :]  # [KV, S, D]
        values = self.v_cache[layer_idx, 0, :, :kv_written, :]
        q4 = q.unsqueeze(0)  # [1, H, T, D]
        k4 = keys.unsqueeze(0)  # [1, KV, S, D]
        v4 = values.unsqueeze(0)
        if tokens > 1 and kv_written == tokens:
            attn = F.scaled_dot_product_attention(
                q4, k4, v4, scale=self.head_dim**-0.5, is_causal=True, enable_gqa=True
            )
        else:
            # Mirror the HF reference: decode steps run under the MATH backend.
            from torch.nn.attention import SDPBackend, sdpa_kernel

            pos_q = cache_position.view(tokens, 1)
            valid = torch.arange(kv_written, device=self.device).view(1, -1) <= pos_q
            with sdpa_kernel(SDPBackend.MATH):
                attn = F.scaled_dot_product_attention(
                    q4,
                    k4,
                    v4,
                    scale=self.head_dim**-0.5,
                    attn_mask=valid.unsqueeze(0).unsqueeze(0),
                    enable_gqa=True,
                )
        attn = attn.squeeze(0).transpose(0, 1).reshape(tokens, -1)
        out, _ = layer.self_attn.o_proj(attn)
        return out

    def _forward_layers(
        self, embeds: torch.Tensor, positions: torch.Tensor, cache_position: torch.Tensor, kv_written: int
    ) -> torch.Tensor:
        """embeds: [tokens, hidden] already composed; returns hidden [tokens, hidden]."""
        import torch.nn.functional as F

        hidden = embeds
        for li, layer in enumerate(self.ar.model.layers):
            residual = hidden
            h = _rms_norm(layer.input_layernorm, hidden)
            h = self._attention(li, layer, h, positions, cache_position, kv_written)
            hidden = residual + h
            residual = hidden
            h = _rms_norm(layer.post_attention_layernorm, hidden)
            gate_up, _ = layer.mlp.gate_up_proj(h)
            gate, up = gate_up.chunk(2, dim=-1)
            h, _ = layer.mlp.down_proj(F.silu(gate) * up)
            hidden = residual + h
        return hidden

    def prefill(self, prefill_embeds: torch.Tensor, position_ids: torch.Tensor):
        """prefill_embeds: [T, H]; position_ids: [T]. Returns (normed_last [1, H], logits [1, vocab])."""
        import torch.nn.functional as F

        t = prefill_embeds.shape[0]
        embeds = prefill_embeds.to(self.device, self.dtype)
        positions = position_ids.reshape(-1).to(self.device, torch.long)
        cache_position = torch.arange(t, device=self.device)
        hidden = self._forward_layers(embeds, positions, cache_position, t)
        last = hidden[-1:].to(self.dtype)
        normed = _rms_norm(self.ar.model.norm, last)
        logits = F.linear(normed, self.ar.model.embed_tokens.weight)
        return normed, logits

    def decode_step(self, step_embed: torch.Tensor, token_position: int, physical_position: int, kv_written: int):
        """step_embed: [H]. Returns (normed [1, H], logits [1, vocab])."""
        import torch.nn.functional as F

        embeds = step_embed.to(self.device, self.dtype).unsqueeze(0)
        positions = torch.tensor([token_position], device=self.device, dtype=torch.long)
        cache_position = torch.tensor([physical_position], device=self.device)
        hidden = self._forward_layers(embeds, positions, cache_position, kv_written)
        normed = _rms_norm(self.ar.model.norm, hidden)
        logits = F.linear(normed, self.ar.model.embed_tokens.weight)
        return normed, logits

    def reset(self) -> None:
        self.k_cache.zero_()
        self.v_cache.zero_()


@pytest.fixture(scope="session")
def slow_ar_runner(audio8_model):
    return SlowARSDPARunner(audio8_model)
