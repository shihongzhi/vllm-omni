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

import os
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
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

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

# Prompt pieces, reproduced byte-for-byte from the reference processor
# (dump_reference_tensors prompts decode to exactly these pieces; the clone
# segments also match Audio8_TTS/onnx_runtime/arktts_runtime/prompt.py).
# IMPORTANT: pieces are encoded SEPARATELY and concatenated -- encoding one
# long string would merge a preceding segment's trailing "\n" into the next
# segment's "\n\n" token, shifting every downstream id.
_PROMPT_NO_REF_PARTS = (
    "<|im_start|>system\n",
    "convert the provided text to speech<|im_end|>\n",
    "<|im_start|>user\n",
    "{text}",
    "<|im_end|>\n",
    "<|im_start|>assistant\n<|voice|>",
)
_PROMPT_CLONE_PREFIX = (
    "<|im_start|>system\n",
    "convert the provided text to speech reference to the following:\n\nText:\n",
    "<|speaker:0|>{ref_text}",
    "\n\nSpeech:\n",
)
_PROMPT_CLONE_SUFFIX = (
    "<|im_end|>\n",
    "<|im_start|>user\n",
    "{target}",
    "<|im_end|>\n",
    "<|im_start|>assistant\n<|voice|>",
)


class Audio8TTSAR(nn.Module):
    """Dual-AR Audio8 model: slow AR (Qwen2 backbone) + fast codebook head."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        config: Any = vllm_config.model_config.hf_config
        self.config = config
        self.model_path = vllm_config.model_config.model
        self.vocab_size = int(config.vocab_size)
        self.hidden_size = int(config.dim)
        self.num_layers = int(config.n_layer)
        self.tie_word_embeddings = bool(config.tie_word_embeddings)

        # Omni engine hook flags (see gpu_ar_model_runner / fish_speech).
        self.have_multimodal_outputs = True
        self.has_preprocess = True
        self.has_postprocess = True
        self.mtp_hidden_size = self.hidden_size
        self.talker_mtp_output_key = ("codes", "audio")
        self.gpu_resident_buffer_keys: set[tuple[str, str]] = {
            ("hidden_states", "last"),
            ("state", "previous"),
            ("state", "previous_valid"),
        }
        self.talker_mtp_graph_safe = True

        self.model = Qwen2Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self._fix_rope_style()

        # Multi-codebook conditioning table: code i of frame f contributes
        # codebook_embeddings(code + i * codebook_size).
        self.codebook_embeddings = nn.Embedding(config.codebook_size * config.num_codebooks, config.dim)

        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

        # Constant logit mask: allow only the EOS token + semantic ids, exactly
        # the choice set of the reference sampler.  Without it the engine could
        # sample arbitrary text tokens as "semantic" decisions.
        semantic_mask = torch.zeros((self.vocab_size,), dtype=torch.bool)
        lo = int(config.semantic_begin_id)
        hi = min(int(config.semantic_end_id) + 1, self.vocab_size)
        if hi > lo:
            semantic_mask[lo:hi] = True
        eos = int(config.eos_token_id)
        if eos < self.vocab_size:
            semantic_mask[eos] = True
        self.register_buffer("_semantic_allowed_mask", semantic_mask, persistent=False)

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

        self._tokenizer = None  # lazy ``tokenizers.Tokenizer``

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

    # ------------------------------------------------------------------
    # Omni engine hooks (vLLM AR scheduler; mirrors the Fish Speech model)
    # ------------------------------------------------------------------

    def embed_input_ids(self, input_ids: Tensor, **_: Any) -> Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: Tensor | None,
        positions: Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: Tensor | None = None,
        **_: Any,
    ) -> Tensor | IntermediateTensors:
        # Prefill/decode embeds are composed by ``preprocess`` (rows carry
        # codebook conditioning), so the backbone runs on inputs_embeds.
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> Tensor | None:
        """Full-vocab logits from the tied embedding, restricted to EOS +
        semantic ids (the reference sampler's whole choice set)."""
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None
        weight = self.embed_tokens.weight
        logits = torch.nn.functional.linear(hidden_states.to(weight.dtype), weight)
        return logits.masked_fill(~self._semantic_allowed_mask, float("-inf"))

    def postprocess(self, hidden_states: Tensor, **_: Any) -> dict[str, Any]:
        """Stash the last normed hidden row; it feeds both semantic sampling
        (indirectly, via the next step's logits) and the fast codebook head."""
        if hidden_states.numel() == 0:
            return {}
        last = hidden_states[-1, :].detach().contiguous()
        return {"hidden_states": {"last": last.reshape(1, -1)}}

    def make_omni_output(self, model_outputs: Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        hidden = model_outputs
        info_dicts = kwargs.get("model_intermediate_buffer")
        if info_dicts is None:
            info_dicts = kwargs.get("runtime_additional_information") or []

        frames: list[Tensor] = []
        for info in info_dicts:
            if not isinstance(info, dict):
                continue
            ac = info.get("codes", {}).get("audio")
            if isinstance(ac, Tensor):
                frames.append(ac.reshape(ac.shape[0], -1))
        if not frames:
            logger.debug("make_omni_output: no audio codes in info dicts (len=%d)", len(info_dicts))
            return OmniOutput(text_hidden_states=hidden, multimodal_outputs={})

        audio_codes = torch.cat(frames, dim=0)
        span_len = int(audio_codes.shape[0])
        mm: dict[str, Tensor] = {"audio_codes": audio_codes}
        return OmniOutput(text_hidden_states=hidden[:span_len], multimodal_outputs=mm)

    def preprocess(
        self,
        input_ids: Tensor,
        input_embeds: Tensor | None,
        **info_dict: Any,
    ) -> tuple[Tensor, Tensor, dict[str, Any]]:
        """Per-request driver: compose prompt/frame embeddings and per-request
        anti-repetition (RAS) window bookkeeping."""
        extra = info_dict.get("additional_information")
        if isinstance(extra, dict):
            merged = {k: v for k, v in info_dict.items() if k != "additional_information"}
            for k, v in extra.items():
                merged.setdefault(k, v)
            info_dict = merged

        cfg = self.config
        span_len = int(input_ids.shape[0])
        if span_len <= 0:
            return input_ids, input_embeds if input_embeds is not None else self.embed_input_ids(input_ids), {}

        if span_len > 1:
            # ---- Prefill ----
            buf = (info_dict.get("embed") or {}).get("prefill")
            is_first_prefill = not isinstance(buf, Tensor) or buf.ndim != 2
            dev = input_ids.device
            pad_id = int(cfg.pad_token_id)

            def _take(chunk_buf: Tensor, offset: int) -> tuple[Tensor, int]:
                total = int(chunk_buf.shape[0])
                start = max(0, min(offset, total))
                end = max(0, min(offset + span_len, total))
                take = chunk_buf[start:end].to(device=dev, dtype=torch.bfloat16, non_blocking=True)
                missing = span_len - int(take.shape[0])
                if missing > 0:
                    pad = self.embed_input_ids(torch.tensor([pad_id], device=dev)).reshape(1, -1)
                    take = torch.cat([take, pad.expand(missing, -1)], dim=0)
                return take, offset + span_len

            if is_first_prefill:
                ref_codes = info_dict.get("ref_codes")
                ref_text = info_dict.get("ref_text")
                has_ref = isinstance(ref_codes, Tensor) and ref_codes.numel() > 0
                target_text = info_dict.get("target_text", info_dict.get("text"))
                if not isinstance(target_text, str):
                    raise ValueError("Audio8 prefill requires additional_information['target_text']")
                if has_ref and not isinstance(ref_text, str):
                    raise ValueError("Audio8 voice clone requires additional_information['ref_text']")

                rows = self.build_prompt_rows(target_text, ref_codes if has_ref else None, ref_text)
                prompt_embeds_all = self.compose_frame_rows(rows)  # [W, H] bf16
                chunk_buf = prompt_embeds_all.detach().to("cpu", torch.bfloat16).contiguous()
                if not chunk_buf.is_pinned():
                    chunk_buf = chunk_buf.pin_memory()
                total_prompt_len = int(chunk_buf.shape[0])
                offset = 0
            else:
                meta = info_dict.get("meta") or {}
                chunk_buf = buf
                offset = int(meta.get("prefill_offset", 0) or 0)
                total_prompt_len = int(chunk_buf.shape[0])

            take, next_offset = _take(chunk_buf, offset)
            # Prefill contributes no audio codes; zero rows keep every request's
            # codes.buffer indexed like its decode-step payloads.
            updates: dict[str, Any] = {
                "embed": {"prefill": chunk_buf if next_offset < total_prompt_len else None},
                "meta": {"prefill_offset": next_offset},
                "codes": {
                    "audio": torch.zeros((total_prompt_len, int(cfg.num_codebooks)), device=dev, dtype=torch.long)
                },
            }
            out_ids = input_ids.clone()
            out_ids.fill_(int(cfg.pad_token_id))
            return out_ids, take, updates

        # ---- Decode (span_len == 1) ----
        dev = input_ids.device
        hs = info_dict.get("hidden_states") or {}
        last_hidden = hs.get("last")
        if not isinstance(last_hidden, Tensor):
            # First decode step right after prefill: plain embedding, no fast
            # head this round (mtp_inputs unset disables talker_mtp).
            logger.warning("Audio8 preprocess decode: hidden_states.last missing (keys=%s)", list(info_dict.keys()))
            embeds = self.embed_input_ids(input_ids.reshape(-1)[:1]).reshape(1, -1)
            return input_ids, embeds.to(dtype=torch.bfloat16), {}

        previous, previous_valid = self._ras_window(info_dict, dev)
        begin = int(cfg.semantic_begin_id)
        end = int(cfg.semantic_end_id)
        token = int(input_ids.reshape(-1)[0])
        rolled_prev, rolled_valid = previous.clone(), previous_valid.clone()
        rolled_prev = rolled_prev.roll(-1, dims=1)
        rolled_prev[:, -1] = token
        rolled_valid = rolled_valid.roll(-1, dims=1)
        rolled_valid[:, -1] = begin <= token <= end

        token_embed = self.embed_input_ids(torch.tensor([[token]], device=dev)).reshape(1, -1)
        info_update = {
            "mtp_inputs": (
                last_hidden.to(device=dev, dtype=torch.bfloat16).reshape(1, -1),
                torch.zeros(1, int(cfg.dim), device=dev, dtype=torch.bfloat16),
            ),
            "state": {"previous": rolled_prev, "previous_valid": rolled_valid},
        }
        return input_ids, token_embed.to(torch.bfloat16), info_update

    def _ras_window(self, info_dict: dict[str, Any], device: torch.device) -> tuple[Tensor, Tensor]:
        """Per-request RAS window from the intermediate buffer (init if absent)."""
        raw = info_dict.get("state") or {}
        previous, previous_valid = raw.get("previous"), raw.get("previous_valid")
        if isinstance(previous, Tensor) and isinstance(previous_valid, Tensor):
            return previous, previous_valid
        prev = torch.zeros(1, int(self.config.ras_window_size), dtype=torch.long, device=device)
        return prev, torch.zeros_like(prev, dtype=torch.bool)

    @torch.inference_mode()
    def talker_mtp(
        self,
        input_ids: Tensor,
        input_embeds: Tensor,
        last_talker_hidden: Tensor,
        text_step: Tensor,
        seed: int | None = None,
        generators: list[torch.Generator] | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor]:
        """Fast-head frame expansion for one decode step (graph-safe: pure).

        Given the engine-sampled semantic id and last step's normed slow
        hidden, expands the residual codebooks and folds them into the step's
        input embedding.  Codebook 0 rides on the semantic token id itself
        (matching the reference ``_embed``), so only codes[:, 1:] add terms.
        Returns (inputs_embeds [B, H], codes [B, num_codebooks]).
        """
        del text_step
        bsz = int(input_ids.shape[0])
        dev = input_embeds.device
        cfg = self.config
        ids = input_ids.reshape(bsz)
        past_hidden = last_talker_hidden.reshape(bsz, -1).to(dtype=torch.bfloat16, device=dev)
        do_sample = kwargs.get("do_sample")
        temperature = kwargs.get("temperature")
        top_k = kwargs.get("top_k")
        top_p = kwargs.get("top_p")
        generator = kwargs.get("generator")
        codes = self.decode_codebooks(
            past_hidden,
            ids,
            temperature=0.8 if temperature is None else float(temperature),
            top_p=0.95 if top_p is None else float(top_p),
            top_k=50 if top_k is None else int(top_k),
            do_sample=True if do_sample is None else bool(do_sample),
            generator=generator,
        )

        base = input_embeds.reshape(bsz, -1).to(torch.bfloat16)
        # compose_embeds semantics: generated code j rides stack row j+1, and
        # stack row r shifts by _codebook_offsets[r]; folding the full stack
        # therefore replays ALL ten offsets against the ten codes.
        residuals = codes + self._codebook_offsets.to(codes.device)
        codebook_sum = self.codebook_embeddings(residuals).sum(dim=1).to(base.dtype)
        begin, end = int(cfg.semantic_begin_id), int(cfg.semantic_end_id)
        semantic_mask = (ids >= begin) & (ids <= end)
        composed = torch.where(semantic_mask.unsqueeze(-1), base + codebook_sum, base)
        return composed, codes.to(torch.long)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from tokenizers import Tokenizer

            path = os.path.join(self.model_path, "tokenizer.json")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"tokenizer.json not found under {self.model_path}")
            self._tokenizer = Tokenizer.from_file(path)
        return self._tokenizer

    def _encode(self, text: str) -> list[int]:
        return list(self._get_tokenizer().encode(text, add_special_tokens=False).ids)

    def build_prompt_rows(
        self,
        target_text: str,
        ref_codes: Tensor | None = None,
        ref_text: str | None = None,
    ) -> Tensor:
        """Reference-format multi-row prompt [num_codebooks+1, W].

        Row 0 holds vocab-space token ids; with reference conditioning, frames
        occupy columns after the head: row 0 carries ``codes[0]+begin`` and
        rows 1..N carry the residual codebooks' codes. Mirrors the golden
        prompts captured in Step 2.2 (``dump_reference_tensors``).
        """
        cfg = self.config
        if ref_codes is None:
            row0 = self._encode_parts(_PROMPT_NO_REF_PARTS, text=target_text)
            rows = torch.zeros(int(cfg.num_codebooks) + 1, len(row0), dtype=torch.long)
            rows[0] = torch.tensor(row0, dtype=torch.long)
            return rows

        assert isinstance(ref_text, str), "voice cloning requires reference text"
        codes = ref_codes.detach().to("cpu", torch.long)
        if codes.ndim != 2 or int(codes.shape[0]) != int(cfg.num_codebooks) or codes.shape[1] == 0:
            raise ValueError(f"reference codes must have shape [{cfg.num_codebooks}, T>0], got {tuple(codes.shape)}")
        head = self._encode_parts(_PROMPT_CLONE_PREFIX, ref_text=ref_text)
        tail = self._encode_parts(_PROMPT_CLONE_SUFFIX, target=target_text)
        semantic_ids = (codes[0] + int(cfg.semantic_begin_id)).tolist()
        width = len(head) + len(semantic_ids) + len(tail)
        rows = torch.zeros(int(cfg.num_codebooks) + 1, width, dtype=torch.long)
        rows[0, : len(head)] = torch.tensor(head, dtype=torch.long)
        rows[0, len(head) : len(head) + len(semantic_ids)] = torch.tensor(semantic_ids, dtype=torch.long)
        rows[0, len(head) + len(semantic_ids) :] = torch.tensor(tail, dtype=torch.long)
        # Rows 1..K hold the FULL codebook stack (codebook 0 rides in row 1 in
        # addition to steering row 0's semantic ids), matching both the golden
        # prompts and the decode-step input layout consumed by compose_embeds.
        rows[1:, len(head) : len(head) + codes.shape[1]] = codes
        return rows

    def _encode_parts(self, parts: tuple[str, ...], **values: str) -> list[int]:
        ids: list[int] = []
        for part in parts:
            ids.extend(self._encode(part.format(**values)))
        return ids

    def compose_frame_rows(self, rows: Tensor) -> Tensor:
        """[num_codebooks+1, W] prompt rows -> prefill embeds [W, H] (bf16)."""
        embeds = self.compose_embeds(rows.unsqueeze(0).to(self.codebook_embeddings.weight.device))
        return embeds[0]

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
            # The checkpoint stores the tied embedding as ``embeddings.weight``.
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
