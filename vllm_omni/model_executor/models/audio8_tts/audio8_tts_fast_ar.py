"""Audio8 TTS fast codebook head (pure PyTorch).

A tiny fixed-length AR transformer that expands each semantic token from the
slow AR into ``num_codebooks`` residual codebook codes.  Ported from the
reference SGLang implementation (``Audio8_TTS/sglang_omni/.../sglang_model.py``):

  - Module names intentionally match the checkpoint (``attention.wqkv``,
    ``feed_forward.w1/w2/w3``, ``attention_norm``, ``ffn_norm``) so fast-AR
    weights load 1:1 without remapping.
  - Per-layer KV cache with a fixed ``num_codebooks + 1`` slots lives outside
    any paged allocator; it is cleared at the start of every frame.
  - Position 0 consumes the (normed) slow-AR hidden state and only fills the
    cache; positions 1..num_codebooks-1 consume ``fast_embeddings`` of the
    previous code and emit logits for the next one.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Only positions 0..num_codebooks-1 are ever addressed; caches are allocated
# with num_codebooks+1 slots to keep buffer arithmetic forgiving.


class Audio8FastRMSNorm(nn.Module):
    """RMSNorm that preserves arbitrary trailing shapes (float32 compute)."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, value: Tensor) -> Tensor:
        shape = value.shape
        normalized = value.float().reshape(-1, shape[-1])
        normalized = normalized * torch.rsqrt(normalized.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * normalized).to(value.dtype).view(shape)


def build_fast_rope_table(length: int, head_dim: int, base: float, device: torch.device) -> Tensor:
    """Precompute interleaved (GPT-J style) RoPE as [length, head_dim/2, 2]."""
    frequencies = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    phases = torch.outer(torch.arange(length, device=device), frequencies)
    values = torch.polar(torch.ones_like(phases), phases)
    return torch.stack((values.real, values.imag), dim=-1).to(torch.bfloat16)


def apply_fast_rope(value: Tensor, rope_values: Tensor) -> Tensor:
    """Apply interleaved RoPE. ``value``: [B, heads, seq, head_dim]."""
    shaped = value.float().reshape(*value.shape[:-1], -1, 2)
    rope_values = rope_values[None, :, None]
    output = torch.stack(
        (
            shaped[..., 0] * rope_values[..., 0] - shaped[..., 1] * rope_values[..., 1],
            shaped[..., 1] * rope_values[..., 0] + shaped[..., 0] * rope_values[..., 1],
        ),
        dim=-1,
    )
    return output.flatten(3).to(value.dtype)


class Audio8FastKVCache(nn.Module):
    """Fixed-size per-layer KV cache for the codebook loop."""

    def __init__(
        self,
        max_batch_size: int,
        max_sequence_length: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        shape = (max_batch_size, max_sequence_length, num_kv_heads, head_dim)
        self.register_buffer("key", torch.zeros(shape, device=device, dtype=dtype), persistent=False)
        self.register_buffer("value", torch.zeros(shape, device=device, dtype=dtype), persistent=False)

    def for_batch(self, batch_size: int) -> tuple[Tensor, Tensor]:
        return self.key[:batch_size], self.value[:batch_size]

    def clear(self) -> None:
        self.key.zero_()
        self.value.zero_()


class Audio8FastAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        total = (config.fast_n_head + 2 * config.fast_n_local_heads) * config.fast_head_dim
        self.wqkv = nn.Linear(config.fast_dim, total, bias=config.fast_attention_qkv_bias)
        self.wo = nn.Linear(
            config.fast_n_head * config.fast_head_dim,
            config.fast_dim,
            bias=config.fast_attention_o_bias,
        )
        self.n_head = config.fast_n_head
        self.n_local_heads = config.fast_n_local_heads
        self.head_dim = config.fast_head_dim
        self.qk_norm = config.fast_attention_qk_norm
        self.audio8_cache: Audio8FastKVCache | None = None
        if self.qk_norm:
            self.q_norm = Audio8FastRMSNorm(self.head_dim, config.norm_eps)
            self.k_norm = Audio8FastRMSNorm(self.head_dim, config.norm_eps)

    def setup_audio8_cache(
        self,
        max_batch_size: int,
        max_sequence_length: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.audio8_cache = Audio8FastKVCache(
            max_batch_size,
            max_sequence_length,
            self.n_local_heads,
            self.head_dim,
            device=device,
            dtype=dtype,
        )

    def clear_audio8_cache(self) -> None:
        if self.audio8_cache is None:
            raise RuntimeError("Audio8 fast KV cache has not been initialized")
        self.audio8_cache.clear()

    def forward(self, value: Tensor, rope_values: Tensor) -> Tensor:
        """Full-sequence (prefill-style) forward for tests / teacher forcing."""
        batch, length, _ = value.shape
        q_size = self.n_head * self.head_dim
        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(value).split((q_size, kv_size, kv_size), dim=-1)
        q = q.view(batch, length, self.n_head, self.head_dim)
        k = k.view(batch, length, self.n_local_heads, self.head_dim)
        v = v.view(batch, length, self.n_local_heads, self.head_dim)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = apply_fast_rope(q, rope_values).transpose(1, 2)
        k = apply_fast_rope(k, rope_values).transpose(1, 2)
        v = v.transpose(1, 2)
        repeat = self.n_head // self.n_local_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
        scores = q @ k.transpose(-2, -1) / (self.head_dim**0.5)
        causal = torch.ones(length, length, dtype=torch.bool, device=value.device).tril()
        scores = scores.masked_fill(~causal[None, None], -float("inf"))
        output = torch.softmax(scores, dim=-1) @ v
        return self.wo(output.transpose(1, 2).contiguous().view(batch, length, q_size))

    def forward_audio8_cached(
        self,
        value: Tensor,
        rope_values: Tensor,
        cache_positions: Tensor,
    ) -> Tensor:
        """One-token decode step against the fixed per-layer KV cache."""
        batch, length, _ = value.shape
        if length != 1:
            raise ValueError("Audio8 fast KV-cache decode expects one token per step")
        q_size = self.n_head * self.head_dim
        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(value).split((q_size, kv_size, kv_size), dim=-1)
        q = q.view(batch, length, self.n_head, self.head_dim)
        k = k.view(batch, length, self.n_local_heads, self.head_dim)
        v = v.view(batch, length, self.n_local_heads, self.head_dim)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = apply_fast_rope(q, rope_values)
        k = apply_fast_rope(k, rope_values)
        if self.audio8_cache is None:
            raise RuntimeError("Audio8 fast KV cache has not been initialized")
        key_cache, value_cache = self.audio8_cache.for_batch(batch)

        # Write k/v at cache_positions, then attend with a prefix mask over the
        # whole fixed-size cache (unwritten slots hold zeros and are masked out)
        # -- mirrors the reference SDPA fallback; no host sync needed.
        batch_indices = torch.arange(batch, device=value.device)
        key_cache[batch_indices, cache_positions] = k[:, 0]
        value_cache[batch_indices, cache_positions] = v[:, 0]
        slots = key_cache.shape[1]
        repeat = self.n_head // self.n_local_heads
        keys = key_cache.repeat_interleave(repeat, dim=2).transpose(1, 2)  # [B, H, S, D]
        values = value_cache.repeat_interleave(repeat, dim=2).transpose(1, 2)
        query = q.transpose(1, 2)  # [B, H, 1, D]
        valid = torch.arange(slots, device=value.device)[None, :] <= cache_positions[:, None]
        output = F.scaled_dot_product_attention(
            query,
            keys,
            values,
            scale=self.head_dim**-0.5,
            attn_mask=valid[:, None, None, :],
        )
        return self.wo(output.transpose(1, 2).contiguous().view(batch, length, q_size))


class Audio8FastFeedForward(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.fast_dim, config.fast_intermediate_size, bias=False)
        self.w2 = nn.Linear(config.fast_intermediate_size, config.fast_dim, bias=False)
        self.w3 = nn.Linear(config.fast_dim, config.fast_intermediate_size, bias=False)

    def forward(self, value: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(value)) * self.w3(value))


class Audio8FastDecoderLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.attention = Audio8FastAttention(config)
        self.feed_forward = Audio8FastFeedForward(config)
        self.ffn_norm = Audio8FastRMSNorm(config.fast_dim, config.norm_eps)
        self.attention_norm = Audio8FastRMSNorm(config.fast_dim, config.norm_eps)

    def forward(self, value: Tensor, rope_values: Tensor) -> Tensor:
        hidden = value + self.attention(self.attention_norm(value), rope_values)
        return hidden + self.feed_forward(self.ffn_norm(hidden))

    def forward_audio8_cached(
        self,
        value: Tensor,
        rope_values: Tensor,
        cache_positions: Tensor,
    ) -> Tensor:
        hidden = value + self.attention.forward_audio8_cached(
            self.attention_norm(value),
            rope_values,
            cache_positions,
        )
        return hidden + self.feed_forward(self.ffn_norm(hidden))
