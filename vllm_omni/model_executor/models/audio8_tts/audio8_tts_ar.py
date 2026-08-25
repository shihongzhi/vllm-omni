"""Audio8 TTS dual-AR model (``ArkttsModel``) for vLLM-Omni.

The slow AR reuses vLLM's ``Qwen2Model`` as its backbone: Audio8's attention
matches Qwen2 exactly (fused QKV with bias, o_proj without bias, no q/k norm,
head_dim 64 via 896/14) except for the RoPE style, which is rebuilt as
interleaved (GPT-J) in :meth:`Audio8TTSAR._fix_rope_style` -- the same
approach Fish Speech uses on top of ``Qwen3Model``.

The fast codebook head (:mod:`.audio8_tts_fast_ar`) runs side-band in pure
torch with its own fixed KV cache.  Structure ported from the reference SGLang
implementation; numerics anchored to the HF remote-code ``modeling_arktts.py``.

Weight name mapping (slow AR):
  ``embeddings.weight`` → ``model.embed_tokens.weight``
  ``norm.weight`` → ``model.norm.weight``
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
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.utils import maybe_prefix

from .audio8_tts_fast_ar import (
    Audio8FastDecoderLayer,
    Audio8FastRMSNorm,
    build_fast_rope_table,
)

logger = init_logger(__name__)

# Checkpoint-side names that are recomputed at runtime, never stored.
_NON_PERSISTENT_WEIGHTS = {"freqs_cis", "fast_freqs_cis"}

# The HF reference precomputes the RoPE table in bf16 (modeling_arktts.py
# ``_precompute_rope``).  vLLM builds f32 cos/sin caches; truncating them to
# bf16 and back keeps engine numerics aligned with the reference (same fix as
# Fish Speech, without which greedy decode can diverge).
_ROPE_CACHE_TRUNCATE_DTYPE = torch.bfloat16


class Audio8TTSAR(nn.Module):
    """Dual-AR Audio8 model: slow AR (Qwen2 backbone) + fast codebook head."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        config: Any = vllm_config.model_config.hf_config
        self.config = config
        self.vocab_size = int(config.vocab_size)
        self.hidden_size = int(config.dim)
        self.num_layers = int(config.n_layer)
        self.tie_word_embeddings = bool(config.tie_word_embeddings)

        self.model = Qwen2Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self._fix_rope_style()

        # Multi-codebook conditioning table: code i of frame f contributes
        # codebook_embeddings(code + i * codebook_size).
        self.codebook_embeddings = nn.Embedding(config.codebook_size * config.num_codebooks, config.dim)

        # Fast codebook head. fast_project_in is Identity when dims match
        # (the 0.6b checkpoint has fast_dim == dim and no projection weights).
        self.fast_project_in = (
            nn.Linear(config.dim, config.fast_dim) if config.fast_dim != config.dim else nn.Identity()
        )
        self.fast_embeddings = nn.Embedding(config.codebook_size, config.fast_dim)
        self.fast_layers = nn.ModuleList(Audio8FastDecoderLayer(config) for _ in range(config.n_fast_layer))
        self.fast_norm = Audio8FastRMSNorm(config.fast_dim, config.norm_eps)
        self.fast_output = nn.Linear(config.fast_dim, config.codebook_size, bias=False)

    @property
    def embed_tokens(self) -> nn.Module:
        return self.model.embed_tokens

    def _fix_rope_style(self) -> None:
        """Rebuild each layer's RoPE as interleaved (GPT-J) style.

        Audio8 was trained with interleaved RoPE, but vLLM's Qwen2 attention
        defaults to NeoX style.
        """
        for layer in self.model.layers:
            attn = layer.self_attn
            attn.rotary_emb = get_rope(
                head_size=attn.head_dim,
                max_position=self.config.max_seq_len,
                is_neox_style=False,
                rope_parameters={
                    "rope_theta": self.config.rope_base,
                    "rope_type": "default",
                },
            )

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
            if name == "norm.weight":
                self._load_direct("model.norm.weight", loaded_weight, params)
                consumed.add(name)
                continue
            if self._load_slow_weight(name, loaded_weight, params):
                consumed.add(name)
                continue
            target_name = "model.embed_tokens.weight" if name == "embeddings.weight" else name
            target = params.get(target_name)
            if target is None:
                raise KeyError(f"Unexpected Audio8 weight in checkpoint: {name}")
            loader = getattr(target, "weight_loader", default_weight_loader)
            loader(target, loaded_weight)
            consumed.add(name)

        # Align RoPE cos/sin caches with the reference's bf16 table; see
        # _ROPE_CACHE_TRUNCATE_DTYPE.
        truncated = 0
        for module in self.modules():
            if hasattr(module, "cos_sin_cache") and isinstance(module.cos_sin_cache, torch.Tensor):
                cache = module.cos_sin_cache
                module.cos_sin_cache = cache.to(_ROPE_CACHE_TRUNCATE_DTYPE).to(cache.dtype)
                truncated += 1
        if truncated:
            logger.debug("Truncated %d RoPE cos/sin caches to bf16", truncated)

        return consumed

    def _load_direct(
        self,
        target_name: str,
        loaded_weight: Tensor,
        params: dict[str, nn.Parameter],
    ) -> None:
        parameter = params[target_name]
        loader = getattr(parameter, "weight_loader", default_weight_loader)
        loader(parameter, loaded_weight)

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
            "feed_forward.w1.weight": ("mlp.gate_up_proj.weight", 0),
            "feed_forward.w3.weight": ("mlp.gate_up_proj.weight", 1),
            "feed_forward.w2.weight": "mlp.down_proj.weight",
        }
        for source_suffix, target in remap.items():
            if not name.endswith(source_suffix):
                continue
            prefix = "model." + name[: -len(source_suffix)]
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
        layer = self.model.layers[int(prefix.split(".")[2])]
        q, k, v = loaded_weight.split(
            [layer.self_attn.q_size, layer.self_attn.kv_size, layer.self_attn.kv_size],
            dim=0,
        )
        for shard_id, weight in (("q", q), ("k", k), ("v", v)):
            parameter.weight_loader(parameter, weight, shard_id)
