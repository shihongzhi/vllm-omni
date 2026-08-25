"""Audio8 TTS dual-AR model (``ArkttsModel``) for vLLM-Omni.

Slow AR (24-layer GQA transformer over text+semantic tokens) built from vLLM
primitives so it runs on the engine's paged attention path; fast codebook head
(:mod:`.audio8_tts_fast_ar`) runs side-band with its own fixed KV cache.

Structure ported from the reference SGLang implementation
(``Audio8_TTS/sglang_omni/.../sglang_model.py``); numerics anchored to the
HF remote-code ``modeling_arktts.py``.

Weight name mapping (slow AR):
  ``embeddings.weight`` → ``embed_tokens.weight``
  ``attention.wqkv.{weight,bias}`` → ``self_attn.qkv_proj.{weight,bias}`` (q/k/v split)
  ``attention.wo.{weight,bias}`` → ``self_attn.o_proj.{weight,bias}``
  ``attention_norm`` → ``input_layernorm``, ``ffn_norm`` → ``post_attention_layernorm``
  ``feed_forward.w1/w3`` → ``gate_up_proj`` shards, ``w2`` → ``down_proj``
Fast-AR modules keep their checkpoint names and load 1:1.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from .audio8_tts_fast_ar import (
    Audio8FastDecoderLayer,
    Audio8FastRMSNorm,
    build_fast_rope_table,
)

# Checkpoint-side names that are recomputed at runtime, never stored.
_NON_PERSISTENT_WEIGHTS = {"freqs_cis", "fast_freqs_cis"}


class Audio8Attention(nn.Module):
    """Slow-AR self-attention on the engine's paged-attention path."""

    def __init__(self, config: Any, layer_id: int, *, prefix: str = "") -> None:
        super().__init__()
        self.q_size = config.n_head * config.head_dim
        self.kv_size = config.n_local_heads * config.head_dim
        self.head_dim = config.head_dim
        self.qkv_proj = QKVParallelLinear(
            config.dim,
            config.head_dim,
            config.n_head,
            config.n_local_heads,
            bias=config.attention_qkv_bias,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            config.n_head * config.head_dim,
            config.dim,
            bias=config.attention_o_bias,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            config.head_dim,
            max_position=config.max_seq_len,
            is_neox_style=False,  # Audio8 uses interleaved (GPT-J) RoPE
            rope_parameters={"rope_theta": config.rope_base, "rope_type": "default"},
        )
        self.attn = Attention(
            config.n_head,
            config.head_dim,
            config.head_dim**-0.5,
            num_kv_heads=config.n_local_heads,
            prefix=f"{prefix}.attn",
        )
        self.qk_norm = config.attention_qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(config.head_dim, eps=config.norm_eps)
            self.k_norm = RMSNorm(config.head_dim, eps=config.norm_eps)


class Audio8DecoderLayer(nn.Module):
    def __init__(self, config: Any, layer_id: int, *, prefix: str = "") -> None:
        super().__init__()
        self.self_attn = Audio8Attention(config, layer_id, prefix=f"{prefix}.self_attn")
        self.gate_up_proj = MergedColumnParallelLinear(
            config.dim,
            [config.intermediate_size, config.intermediate_size],
            bias=False,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.dim,
            bias=False,
            prefix=f"{prefix}.down_proj",
        )
        self.input_layernorm = RMSNorm(config.dim, eps=config.norm_eps)
        self.post_attention_layernorm = RMSNorm(config.dim, eps=config.norm_eps)


class Audio8TTSAR(nn.Module):
    """Dual-AR Audio8 model: slow AR (paged attention) + fast codebook head."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = int(config.vocab_size)
        self.hidden_size = int(config.dim)
        self.num_layers = int(config.n_layer)
        self.tie_word_embeddings = bool(config.tie_word_embeddings)

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.dim,
            prefix="embed_tokens",
        )
        # Multi-codebook conditioning table: code i of frame f contributes
        # codebook_embeddings(code + i * codebook_size).
        self.codebook_embeddings = nn.Embedding(config.codebook_size * config.num_codebooks, config.dim)
        self.layers = nn.ModuleList(
            Audio8DecoderLayer(config, idx, prefix=f"layers.{idx}") for idx in range(config.n_layer)
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        # Fast codebook head. fast_project_in is Identity when dims match
        # (the 0.6b checkpoint has fast_dim == dim and no projection weights).
        self.fast_project_in = (
            nn.Linear(config.dim, config.fast_dim) if config.fast_dim != config.dim else nn.Identity()
        )
        self.fast_embeddings = nn.Embedding(config.codebook_size, config.fast_dim)
        self.fast_layers = nn.ModuleList(Audio8FastDecoderLayer(config) for _ in range(config.n_fast_layer))
        self.fast_norm = Audio8FastRMSNorm(config.fast_dim, config.norm_eps)
        self.fast_output = nn.Linear(config.fast_dim, config.codebook_size, bias=False)

    def build_fast_rope(self, device: torch.device) -> Tensor:
        return build_fast_rope_table(
            self.config.num_codebooks,
            self.config.fast_head_dim,
            self.config.rope_base,
            device,
        )

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        """Load checkpoint weights. Returns the set of consumed source names."""
        params = dict(self.named_parameters())
        consumed: set[str] = set()
        for name, loaded_weight in weights:
            if name in _NON_PERSISTENT_WEIGHTS:
                consumed.add(name)
                continue
            if self._load_slow_weight(name, loaded_weight, params):
                consumed.add(name)
                continue
            target_name = "embed_tokens.weight" if name == "embeddings.weight" else name
            target = params.get(target_name)
            if target is None:
                raise KeyError(f"Unexpected Audio8 weight in checkpoint: {name}")
            loader = getattr(target, "weight_loader", default_weight_loader)
            loader(target, loaded_weight)
            consumed.add(name)
        return consumed

    def _load_slow_weight(
        self,
        name: str,
        loaded_weight: Tensor,
        params: dict[str, nn.Parameter],
    ) -> bool:
        if not name.startswith("layers."):
            return False
        remap: dict[str, Any] = {
            "attention.wqkv.weight": None,
            "attention.wqkv.bias": None,
            "attention.wo.weight": "self_attn.o_proj.weight",
            "attention.wo.bias": "self_attn.o_proj.bias",
            "attention.q_norm.weight": "self_attn.q_norm.weight",
            "attention.k_norm.weight": "self_attn.k_norm.weight",
            "attention_norm.weight": "input_layernorm.weight",
            "ffn_norm.weight": "post_attention_layernorm.weight",
            "feed_forward.w1.weight": ("gate_up_proj.weight", 0),
            "feed_forward.w3.weight": ("gate_up_proj.weight", 1),
            "feed_forward.w2.weight": "down_proj.weight",
        }
        for source_suffix, target in remap.items():
            if not name.endswith(source_suffix):
                continue
            prefix = name[: -len(source_suffix)]
            if target is None:
                self._load_fused_qkv(
                    prefix,
                    loaded_weight,
                    params,
                    is_bias=source_suffix.endswith("bias"),
                )
                return True
            if isinstance(target, tuple):
                target_suffix, shard_id = target
            else:
                target_suffix, shard_id = target, None
            parameter = params[prefix + target_suffix]
            if shard_id is None:
                loader = getattr(parameter, "weight_loader", default_weight_loader)
                loader(parameter, loaded_weight)
            else:
                parameter.weight_loader(parameter, loaded_weight, shard_id)
            return True
        return False

    def _load_fused_qkv(
        self,
        prefix: str,
        loaded_weight: Tensor,
        params: dict[str, nn.Parameter],
        *,
        is_bias: bool,
    ) -> None:
        target_name = prefix + "self_attn.qkv_proj." + ("bias" if is_bias else "weight")
        parameter = params[target_name]
        layer = self.layers[int(prefix.split(".")[1])]
        q, k, v = loaded_weight.split(
            [layer.self_attn.q_size, layer.self_attn.kv_size, layer.self_attn.kv_size],
            dim=0,
        )
        for shard_id, weight in (("q", q), ("k", k), ("v", v)):
            parameter.weight_loader(parameter, weight, shard_id)
