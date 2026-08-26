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

        self.register_buffer(
            "_codebook_offsets",
            torch.arange(config.num_codebooks) * config.codebook_size,
            persistent=False,
        )
        self.register_buffer(
            "_fast_rope",
            build_fast_rope_table(config.num_codebooks, config.fast_head_dim, config.rope_base, "cpu"),
            persistent=False,
        )

    @property
    def embed_tokens(self) -> nn.Module:
        return self.model.embed_tokens

    # ------------------------------------------------------------------
    # Decode-side composition / sampling / fast codebook head
    # ------------------------------------------------------------------

    def setup_fast_decode(self, max_batch_size: int) -> None:
        """Allocate the fast head's fixed KV caches (rope table already exists)."""
        device = self.codebook_embeddings.weight.device
        dtype = self.codebook_embeddings.weight.dtype
        cfg = self.config
        for layer in self.fast_layers:
            layer.attention.setup_audio8_cache(
                max_batch_size,
                cfg.num_codebooks + 1,
                device=device,
                dtype=dtype,
            )

    def compose_embeds(self, input_ids: Tensor) -> Tensor:
        """HF ``_embed`` equivalent: rows [B, num_codebooks+1, T] → [B, T, H].

        Row 0 carries vocab-space token ids; rows 1..N carry codebook codes.
        Codebook embeddings are summed in only where row 0 is a semantic id.
        """
        cfg = self.config
        semantic = input_ids[:, 0]
        base = self.embed_tokens(semantic)
        codes = input_ids[:, 1:] + self._codebook_offsets[None, :, None]
        codebook_sum = self.codebook_embeddings(codes).sum(dim=1).to(base.dtype)
        mask = (semantic >= cfg.semantic_begin_id) & (semantic <= cfg.semantic_end_id)
        return base + torch.where(mask.unsqueeze(-1), codebook_sum, torch.zeros_like(base))

    def semantic_logits(self, hidden_states: Tensor) -> Tensor:
        """[B, H] normed hidden → [B, codebook_size + 1] over [EOS, semantic range]."""
        weight = self.embed_tokens.weight
        cfg = self.config
        eos_weight = weight[cfg.eos_token_id : cfg.eos_token_id + 1]
        semantic_weight = weight[cfg.semantic_begin_id : cfg.semantic_end_id + 1]
        return torch.cat(
            (
                torch.nn.functional.linear(hidden_states, eos_weight),
                torch.nn.functional.linear(hidden_states, semantic_weight),
            ),
            dim=-1,
        )

    @staticmethod
    def sample(
        scores: Tensor,
        temperature: Tensor,
        top_p: Tensor,
        top_k: Tensor,
        do_sample: Tensor,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Batched top-k/top-p filter + Gumbel-style sampling (argmax of p/-log(u))."""
        sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        cumulative = torch.softmax(sorted_scores, dim=-1).cumsum(dim=-1)
        positions = torch.arange(scores.shape[-1], device=scores.device)[None]
        remove_sorted = (cumulative > top_p[:, None]) | (positions >= top_k[:, None])
        remove_sorted[:, 0] = False
        remove = torch.zeros_like(remove_sorted).scatter(1, sorted_indices, remove_sorted)
        filtered = scores.masked_fill(remove, -float("inf"))
        filtered = filtered / temperature[:, None].clamp_min(1e-5)
        greedy = filtered.argmax(dim=-1)
        probabilities = torch.softmax(filtered, dim=-1)
        random = torch.rand(
            probabilities.shape,
            dtype=probabilities.dtype,
            device=probabilities.device,
            generator=generator,
        ).clamp_min(torch.finfo(probabilities.dtype).tiny)
        sampled = torch.argmax(probabilities / (-torch.log(random)), dim=-1)
        return torch.where(do_sample, sampled, greedy)

    def sample_semantic(
        self,
        logits: Tensor,
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
        previous: Tensor | None = None,
        previous_valid: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample semantic token ids (vocab space) with RAS anti-repetition."""
        cfg = self.config
        batch = logits.shape[0]
        scores = logits.float()
        t = torch.full((batch,), float(temperature), device=logits.device)
        p = torch.full((batch,), float(top_p), device=logits.device)
        k = torch.full((batch,), int(top_k), device=logits.device, dtype=torch.long)
        s = torch.full((batch,), bool(do_sample), device=logits.device)
        normal_index = self.sample(scores, t, p, k, s, generator)
        ras_t = torch.full_like(t, cfg.ras_temperature)
        ras_p = torch.full_like(p, cfg.ras_top_p)
        high_index = self.sample(scores, ras_t, ras_p, k, s, generator)

        begin = cfg.semantic_begin_id
        normal = torch.where(normal_index == 0, cfg.eos_token_id, begin + normal_index - 1)
        high = torch.where(high_index == 0, cfg.eos_token_id, begin + high_index - 1)
        if previous is None or previous_valid is None:
            return normal
        repeated = ((previous[:batch] == normal[:, None]) & previous_valid[:batch]).any(dim=1)
        is_semantic = (normal >= begin) & (normal <= cfg.semantic_end_id)
        return torch.where(repeated & is_semantic, high, normal)

    def _run_fast_position(self, hidden: Tensor, position: int) -> Tensor:
        """One fast-head position against the fixed per-layer KV caches."""
        batch = hidden.shape[0]
        pos = torch.tensor([position], device=hidden.device)
        rope_values = self._fast_rope[pos]
        cache_positions = pos.expand(batch).to(torch.int32)
        for layer in self.fast_layers:
            hidden = layer.forward_audio8_cached(hidden, rope_values, cache_positions)
        return self.fast_output(self.fast_norm(hidden))[:, -1]

    @torch.inference_mode()
    def decode_codebooks(
        self,
        slow_hidden: Tensor,
        semantic: Tensor,
        *,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        do_sample: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Expand sampled semantic ids into full codebook frames.

        Position 0 consumes the (normed) slow hidden and only seeds the KV
        cache; codebook 0 is the semantic code itself; positions 1..N-1
        produce the residual codes.  Returns (codes [B, num_codebooks],
        fast_logits_first [B, codebook_size]) -- the first position's scores
        are unused by the reference too.
        """
        cfg = self.config
        batch = slow_hidden.shape[0]
        device = slow_hidden.device
        current = (semantic - cfg.semantic_begin_id).clamp(0, cfg.codebook_size - 1)
        codes = [current]
        for layer in self.fast_layers:
            layer.attention.clear_audio8_cache()
        hidden = self.fast_project_in(slow_hidden.reshape(batch, -1)).unsqueeze(1)
        self._run_fast_position(hidden.to(self.fast_embeddings.weight.dtype), 0)

        t = torch.full((batch,), float(temperature), device=device)
        p = torch.full((batch,), float(top_p), device=device)
        k = torch.full((batch,), int(top_k), device=device, dtype=torch.long)
        s = torch.full((batch,), bool(do_sample), device=device)
        for position in range(1, cfg.num_codebooks):
            embed = self.fast_embeddings(current).unsqueeze(1)
            scores = self._run_fast_position(embed, position).float()
            current = self.sample(scores, t, p, k, s, generator)
            codes.append(current)
        return torch.stack(codes, dim=1)

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
