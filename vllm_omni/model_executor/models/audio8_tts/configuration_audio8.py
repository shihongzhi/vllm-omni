"""Configuration for Audio8 TTS (``ArkttsModel``).

Field names follow the original Audio8 checkpoint ``config.json``
(``dim``, ``n_head``, ``n_layer``, ...).  Standard Transformers attribute
names are also populated so generic vLLM code paths (e.g. KV cache shape
detection) can read ``hidden_size`` / ``num_key_value_heads`` etc.
"""

from __future__ import annotations

from transformers import PretrainedConfig


class Audio8TTSConfig(PretrainedConfig):
    model_type = "arktts"

    def __init__(
        self,
        dim: int = 896,
        n_layer: int = 24,
        n_head: int = 14,
        n_local_heads: int = 2,
        head_dim: int = 64,
        rope_base: float = 1_000_000.0,
        max_seq_len: int = 2048,
        norm_eps: float = 1e-6,
        intermediate_size: int = 4864,
        vocab_size: int = 155_776,
        tie_word_embeddings: bool = True,
        attention_qkv_bias: bool = True,
        attention_o_bias: bool = False,
        attention_qk_norm: bool = False,
        codebook_size: int = 4096,
        num_codebooks: int = 10,
        semantic_begin_id: int = 151_678,
        semantic_end_id: int = 155_773,
        eos_token_id: int = 151_645,
        pad_token_id: int = 151_643,
        bos_token_id: int | None = None,
        n_fast_layer: int = 4,
        fast_dim: int = 896,
        fast_head_dim: int = 64,
        fast_n_head: int = 14,
        fast_n_local_heads: int = 2,
        fast_intermediate_size: int = 4864,
        fast_attention_qkv_bias: bool = False,
        fast_attention_o_bias: bool = False,
        fast_attention_qk_norm: bool = False,
        norm_fastlayer_input: bool = True,
        ras_window_size: int = 10,
        ras_temperature: float = 1.0,
        ras_top_p: float = 0.9,
        codec_sample_rate: int = 44_100,
        codec_frame_size: int = 2048,
        codec_filename: str = "codec.pth",
        **kwargs,
    ):
        self.dim = dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_local_heads = n_local_heads
        self.head_dim = head_dim
        self.rope_base = rope_base
        self.max_seq_len = max_seq_len
        self.norm_eps = norm_eps
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings
        self.attention_qkv_bias = attention_qkv_bias
        self.attention_o_bias = attention_o_bias
        self.attention_qk_norm = attention_qk_norm
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.semantic_begin_id = semantic_begin_id
        self.semantic_end_id = semantic_end_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.n_fast_layer = n_fast_layer
        self.fast_dim = fast_dim
        self.fast_head_dim = fast_head_dim
        self.fast_n_head = fast_n_head
        self.fast_n_local_heads = fast_n_local_heads
        self.fast_intermediate_size = fast_intermediate_size
        self.fast_attention_qkv_bias = fast_attention_qkv_bias
        self.fast_attention_o_bias = fast_attention_o_bias
        self.fast_attention_qk_norm = fast_attention_qk_norm
        self.norm_fastlayer_input = norm_fastlayer_input
        self.ras_window_size = ras_window_size
        self.ras_temperature = ras_temperature
        self.ras_top_p = ras_top_p
        self.codec_sample_rate = codec_sample_rate
        self.codec_frame_size = codec_frame_size
        self.codec_filename = codec_filename

        # Standard Transformers/vLLM aliases.
        self.hidden_size = dim
        self.num_hidden_layers = n_layer
        self.num_attention_heads = n_head
        self.num_key_value_heads = n_local_heads
        self.max_position_embeddings = max_seq_len
        self.rope_theta = rope_base
        self.rms_norm_eps = norm_eps
        self.hidden_act = "silu"

        super().__init__(**kwargs)
