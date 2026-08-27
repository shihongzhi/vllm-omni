# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Arktts audio codec (``codec.pth``) port for Audio8 TTS.

Ported nearly verbatim from the checkpoint's remote code
(``modeling_arktts_codec.py`` in the AutoArk-AI/Audio8-TTS repo) so that
``load_state_dict(strict=True)`` keys align one-to-one with ``codec.pth``
and decoded waveforms stay numerically identical to the HF/baseline
implementation.

Frame/codebook layout shared with the AR model stacks:

  - frame period 2048 samples @44.1kHz (quantizer ×4 upsample, decoder ×512),
  - codebook 0: semantic book, size ``semantic codebook`` (4096),
  - codebooks 1..9: residual books (1024 each, clamped inside decode).

Streaming: every sub-module is strictly time-causal (windowed attention
blocks the future, convolutions are causal), so decoding a prefix of the
code sequence reproduces the corresponding prefix of the full decode -- the
only wrinkle is tail behaviour whose padding depends on the total length.
:class:`ArkttsCodecStreamer` exploits this: it holds back a few frames near
the moving end, confirms everything before them, and flushes exactly on
``finish()``, so the concatenated output matches one-shot decoding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import weight_norm as legacy_weight_norm
from torch.nn.utils.parametrizations import weight_norm


def _rope(length: int, head_dim: int, base: float, device=None) -> Tensor:
    frequencies = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    phases = torch.outer(torch.arange(length, device=device), frequencies)
    values = torch.polar(torch.ones_like(phases), phases)
    return torch.stack((values.real, values.imag), dim=-1).to(torch.bfloat16)


def _apply_rope(x: Tensor, values: Tensor) -> Tensor:
    shaped = x.float().reshape(*x.shape[:-1], -1, 2)
    values = values.view(1, shaped.shape[1], 1, shaped.shape[3], 2)
    output = torch.stack(
        (
            shaped[..., 0] * values[..., 0] - shaped[..., 1] * values[..., 1],
            shaped[..., 1] * values[..., 0] + shaped[..., 0] * values[..., 1],
        ),
        dim=-1,
    )
    return output.flatten(3).to(x.dtype)


class ArkttsCodecRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return output.to(x.dtype) * self.weight


class ArkttsCodecLayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-2, inplace: bool = False):
        super().__init__()
        self.inplace = bool(inplace)
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


@dataclass
class ArkttsCodecTransformerConfig:
    n_layer: int
    n_head: int
    dim: int
    intermediate_size: int
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    dropout_rate: float = 0.1
    attn_dropout_rate: float = 0.1
    channels_first: bool = True

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head


class ArkttsCodecAttention(nn.Module):
    def __init__(self, config: ArkttsCodecTransformerConfig):
        super().__init__()
        total = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        self.wqkv = nn.Linear(config.dim, total, bias=False)
        self.wo = nn.Linear(config.head_dim * config.n_head, config.dim, bias=False)
        self.n_head = config.n_head
        self.n_local_heads = config.n_local_heads
        self.head_dim = config.head_dim
        self.attn_dropout_rate = config.attn_dropout_rate

    def forward(self, x, rope_values, mask):
        batch, length, _ = x.shape
        query_size = self.n_head * self.head_dim
        kv_size = self.n_local_heads * self.head_dim
        query, key, value = self.wqkv(x).split((query_size, kv_size, kv_size), dim=-1)
        query = query.view(batch, length, self.n_head, self.head_dim)
        key = key.view(batch, length, self.n_local_heads, self.head_dim)
        value = value.view(batch, length, self.n_local_heads, self.head_dim)
        query = _apply_rope(query, rope_values).transpose(1, 2)
        key = _apply_rope(key, rope_values).transpose(1, 2)
        value = value.transpose(1, 2)
        repeat = self.n_head // self.n_local_heads
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=self.attn_dropout_rate if self.training else 0.0,
        )
        output = output.transpose(1, 2).contiguous().view(batch, length, query_size)
        return self.wo(output)


class ArkttsCodecFeedForward(nn.Module):
    def __init__(self, config: ArkttsCodecTransformerConfig):
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


class ArkttsCodecTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = ArkttsCodecAttention(config)
        self.feed_forward = ArkttsCodecFeedForward(config)
        self.ffn_norm = ArkttsCodecRMSNorm(config.dim, config.norm_eps)
        self.attention_norm = ArkttsCodecRMSNorm(config.dim, config.norm_eps)
        self.attention_layer_scale = ArkttsCodecLayerScale(config.dim, inplace=True)
        self.ffn_layer_scale = ArkttsCodecLayerScale(config.dim, inplace=True)

    def forward(self, x, rope_values, mask):
        hidden = x + self.attention_layer_scale(self.attention(self.attention_norm(x), rope_values, mask))
        return hidden + self.ffn_layer_scale(self.feed_forward(self.ffn_norm(hidden)))


class ArkttsCodecWindowTransformer(nn.Module):
    def __init__(self, config, input_dim: int, window_size: int | None, causal: bool = True):
        super().__init__()
        self.layers = nn.ModuleList([ArkttsCodecTransformerBlock(config) for _ in range(config.n_layer)])
        self.norm = ArkttsCodecRMSNorm(config.dim, config.norm_eps)
        self.window_size = window_size
        self.causal = causal
        self.channels_first = config.channels_first
        self.input_proj = nn.Linear(input_dim, config.dim) if input_dim != config.dim else nn.Identity()
        self.output_proj = nn.Linear(config.dim, input_dim) if input_dim != config.dim else nn.Identity()
        self.look_ahead_conv = nn.Identity()
        self.head_dim = config.head_dim
        self.rope_base = config.rope_base

    def forward(self, x, x_lens=None):
        del x_lens
        if self.channels_first:
            x = x.transpose(1, 2)
        x = self.look_ahead_conv(self.input_proj(x))
        length = x.shape[1]
        row = torch.arange(length, device=x.device)[:, None]
        column = torch.arange(length, device=x.device)[None, :]
        mask = column <= row
        if self.window_size is not None:
            mask &= column >= (row - self.window_size + 1).clamp_min(0)
        mask = mask[None, None]
        rope_values = _rope(length, self.head_dim, self.rope_base, x.device)
        for layer in self.layers:
            x = layer(x, rope_values, mask)
        x = self.output_proj(self.norm(x))
        return x.transpose(1, 2) if self.channels_first else x


def _extra_padding(x, kernel_size: int, stride: int, padding_total: int = 0) -> int:
    length = x.shape[-1]
    frames = (length - kernel_size + padding_total) / stride + 1
    ideal = (math.ceil(frames) - 1) * stride + kernel_size - padding_total
    return ideal - length


class ArkttsCausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, groups=groups)
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.padding = self.kernel_size - self.stride

    def forward(self, x):
        right = _extra_padding(x, self.kernel_size, self.stride, self.padding)
        return self.conv(F.pad(x, (self.padding, right))).contiguous()

    def apply_weight_norm(self):
        self.conv = weight_norm(self.conv)
        return self


class ArkttsCausalConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation)
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x):
        x = self.conv(x)
        crop = self.kernel_size - self.stride
        return x[..., : x.shape[-1] - crop].contiguous() if crop else x.contiguous()

    def apply_weight_norm(self):
        self.conv = weight_norm(self.conv)
        return self


def _causal_wn_conv(*args, **kwargs):
    # weight_norm must wrap the ALREADY-constructed causal module so state-dict
    # keys become conv.parametrizations.weight.original{0,1}, matching the
    # checkpoint.
    return ArkttsCausalConv1d(*args, **kwargs).apply_weight_norm()


def _causal_wn_transpose(*args, **kwargs):
    return ArkttsCausalConvTranspose1d(*args, **kwargs).apply_weight_norm()


class ArkttsSnake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return _arktts_snake(x, self.alpha)


@torch.jit.script
def _arktts_snake(x: Tensor, alpha: Tensor) -> Tensor:
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    return x.reshape(shape)


class ArkttsResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int):
        super().__init__()
        self.block = nn.Sequential(
            ArkttsSnake1d(dim),
            _causal_wn_conv(dim, dim, kernel_size=7, dilation=dilation),
            ArkttsSnake1d(dim),
            _causal_wn_conv(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        output = self.block(x)
        difference = x.shape[-1] - output.shape[-1]
        if difference > 0:
            x = x[..., :-difference]
        return x + output


class ArkttsEncoderBlock(nn.Module):
    def __init__(self, dim: int, stride: int, transformer_layers: int):
        super().__init__()
        modules = [
            ArkttsResidualUnit(dim // 2, 1),
            ArkttsResidualUnit(dim // 2, 3),
            ArkttsResidualUnit(dim // 2, 9),
            ArkttsSnake1d(dim // 2),
            _causal_wn_conv(dim // 2, dim, kernel_size=2 * stride, stride=stride),
        ]
        if transformer_layers:
            config = ArkttsCodecTransformerConfig(
                n_layer=transformer_layers,
                n_head=dim // 64,
                dim=dim,
                intermediate_size=dim * 3,
            )
            modules.append(ArkttsCodecWindowTransformer(config, dim, window_size=512))
        else:
            modules.append(nn.Identity())
        self.block = nn.Sequential(*modules)

    def forward(self, x):
        return self.block(x)


class ArkttsEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        dim = 64
        modules = [_causal_wn_conv(1, dim, kernel_size=7)]
        for stride, transformer_layers in zip((2, 4, 8, 8), (0, 0, 0, 4)):
            dim *= 2
            modules.append(ArkttsEncoderBlock(dim, stride, transformer_layers))
        modules.extend((ArkttsSnake1d(dim), _causal_wn_conv(dim, 1024, kernel_size=3)))
        self.block = nn.Sequential(*modules)

    def forward(self, x):
        return self.block(x)


class ArkttsDecoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            ArkttsSnake1d(input_dim),
            _causal_wn_transpose(input_dim, output_dim, kernel_size=2 * stride, stride=stride),
            ArkttsResidualUnit(output_dim, 1),
            ArkttsResidualUnit(output_dim, 3),
            ArkttsResidualUnit(output_dim, 9),
        )

    def forward(self, x):
        return self.block(x)


class ArkttsDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        channels = 1536
        modules = [_causal_wn_conv(1024, channels, kernel_size=7)]
        for index, stride in enumerate((8, 8, 4, 2)):
            input_dim = channels // (2**index)
            output_dim = channels // (2 ** (index + 1))
            modules.append(ArkttsDecoderBlock(input_dim, output_dim, stride))
        modules.extend((ArkttsSnake1d(output_dim), _causal_wn_conv(output_dim, 1, kernel_size=7), nn.Tanh()))
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class ArkttsVectorQuantizer(nn.Module):
    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.codebook_dim = int(codebook_dim)
        self.in_proj = legacy_weight_norm(nn.Conv1d(input_dim, codebook_dim, kernel_size=1))
        self.out_proj = legacy_weight_norm(nn.Conv1d(codebook_dim, input_dim, kernel_size=1))
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def decode_code(self, indices):
        return F.embedding(indices, self.codebook.weight).transpose(1, 2)

    def decode_latents(self, latents):
        batch, _, length = latents.shape
        flattened = latents.transpose(1, 2).reshape(batch * length, -1)
        flattened = F.normalize(flattened)
        codebook = F.normalize(self.codebook.weight)
        distances = (
            flattened.pow(2).sum(1, keepdim=True)
            - 2 * flattened @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = (-distances).argmax(1).view(batch, length)
        return self.decode_code(indices), indices

    def forward(self, z):
        projected = self.in_proj(z)
        quantized, indices = self.decode_latents(projected)
        quantized_st = projected + (quantized - projected).detach()
        return self.out_proj(quantized_st), indices, projected


class ArkttsResidualQuantizer(nn.Module):
    def __init__(self, input_dim: int, n_codebooks: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.n_codebooks = int(n_codebooks)
        self.codebook_size = int(codebook_size)
        self.quantizers = nn.ModuleList(
            [ArkttsVectorQuantizer(input_dim, codebook_size, codebook_dim) for _ in range(n_codebooks)]
        )

    def forward(self, z):
        quantized_sum = 0.0
        residual = z
        codes = []
        for quantizer in self.quantizers:
            quantized, indices, _ = quantizer(residual)
            quantized_sum = quantized_sum + quantized
            residual = residual - quantized
            codes.append(indices)
        return quantized_sum, torch.stack(codes, dim=1)

    def from_codes(self, codes):
        output = 0.0
        for index in range(codes.shape[1]):
            projected = self.quantizers[index].decode_code(codes[:, index])
            output = output + self.quantizers[index].out_proj(projected)
        return output


class ArkttsConvNeXtBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = ArkttsCausalConv1d(dim, dim, kernel_size=7, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x):
        residual = x
        x = self.dwconv(x).permute(0, 2, 1)
        x = self.pwconv2(self.act(self.pwconv1(self.norm(x))))
        x = (self.gamma * x).permute(0, 2, 1)
        return residual + x


class ArkttsDownsampleQuantizer(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.semantic_quantizer = ArkttsResidualQuantizer(1024, 1, 4096, 8)
        self.quantizer = ArkttsResidualQuantizer(1024, 9, 1024, 8)
        self.downsample = nn.Sequential(
            nn.Sequential(
                ArkttsCausalConv1d(1024, 1024, kernel_size=2, stride=2),
                ArkttsConvNeXtBlock(1024),
            ),
            nn.Sequential(
                ArkttsCausalConv1d(1024, 1024, kernel_size=2, stride=2),
                ArkttsConvNeXtBlock(1024),
            ),
        )
        self.upsample = nn.Sequential(
            nn.Sequential(
                ArkttsCausalConvTranspose1d(1024, 1024, kernel_size=2, stride=2),
                ArkttsConvNeXtBlock(1024),
            ),
            nn.Sequential(
                ArkttsCausalConvTranspose1d(1024, 1024, kernel_size=2, stride=2),
                ArkttsConvNeXtBlock(1024),
            ),
        )
        pre_transformer_config = ArkttsCodecTransformerConfig(
            n_layer=8,
            n_head=16,
            dim=1024,
            intermediate_size=3072,
        )
        post_transformer_config = ArkttsCodecTransformerConfig(
            n_layer=int(getattr(config, "codec_post_n_layer", 8)),
            n_head=int(getattr(config, "codec_post_n_head", 16)),
            n_local_heads=int(getattr(config, "codec_post_n_local_heads", 8)),
            dim=1024,
            intermediate_size=int(getattr(config, "codec_post_intermediate_size", 1216)),
        )
        self.pre_module = ArkttsCodecWindowTransformer(pre_transformer_config, 1024, window_size=128)
        self.post_module = ArkttsCodecWindowTransformer(post_transformer_config, 1024, window_size=128)
        self.semantic_predictor_module = nn.Identity()

    def forward(self, z):
        original_length = z.shape[-1]
        z = self.pre_module(self.downsample(z))
        semantic, semantic_codes = self.semantic_quantizer(z)
        residual, residual_codes = self.quantizer(z - semantic)
        z = self.upsample(self.post_module(semantic + residual))
        difference = original_length - z.shape[-1]
        if difference > 0:
            z = F.pad(z, (difference, 0))
        elif difference < 0:
            z = z[..., -difference:]
        return z, torch.cat((semantic_codes, residual_codes), dim=1)

    def decode(self, indices):
        indices = indices.clone()
        indices[:, 0].clamp_(0, self.semantic_quantizer.codebook_size - 1)
        indices[:, 1:].clamp_(0, self.quantizer.codebook_size - 1)
        semantic = self.semantic_quantizer.from_codes(indices[:, :1])
        residual = self.quantizer.from_codes(indices[:, 1:])
        return self.upsample(self.post_module(semantic + residual))


class ArkttsCodec(nn.Module):
    """Full codec: frame=2048 samples @44.1kHz; decode maps [B,10,T] codes."""

    sample_rate = 44100
    hop_length = 512
    frame_length = 2048
    samples_per_frame = hop_length * 4  # quantizer x4 then decoder x512

    def __init__(self, config=None):
        super().__init__()
        self.encoder = ArkttsEncoder()
        self.quantizer = ArkttsDownsampleQuantizer(config)
        self.decoder = ArkttsDecoder()

    @torch.inference_mode()
    def encode(self, audio, audio_lengths=None):
        if audio.ndim == 2:
            audio = audio[:, None]
        if audio.ndim != 3 or audio.shape[1] != 1:
            raise ValueError("audio must have shape [B, 1, samples]")
        original_length = audio.shape[-1]
        right = math.ceil(original_length / self.frame_length) * self.frame_length - original_length
        if right:
            audio = F.pad(audio, (0, right))
        if audio_lengths is None:
            audio_lengths = torch.full((audio.shape[0],), original_length, device=audio.device, dtype=torch.long)
        encoded = self.encoder(audio)
        _, codes = self.quantizer(encoded)
        code_lengths = torch.ceil(audio_lengths.float() / self.frame_length).long()
        max_codes = codes.shape[-1]
        padded = torch.full_like(codes, -1)
        for index, length in enumerate(code_lengths.tolist()):
            padded[index, :, : min(length, max_codes)] = codes[index, :, : min(length, max_codes)]
        return padded, code_lengths.clamp_max(max_codes)

    @torch.inference_mode()
    def decode(self, codes):
        return self.decoder(self.quantizer.decode(codes.long()))


@torch.no_grad()
def load_arktts_codec(model_dir: str | Path, config: Any | None = None, *, eval_mode: bool = True) -> ArkttsCodec:
    """Build the codec and load ``<model_dir>/codec.pth`` (strict keys).

    Mirrors the reference ``ArkttsModel.load_codec`` sanitising rules so both
    vendored checkpoint variants load identically. Runs float32 on whichever
    device the tensors land (CUDA callers may ``.to(device=..., dtype=...)``
    afterwards like the reference does).
    """
    path = Path(model_dir) / "codec.pth"
    if not path.is_file():
        raise FileNotFoundError(f"codec checkpoint not found: {path}")
    codec = ArkttsCodec(config)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    if any("generator." in key for key in state):
        state = {key.replace("generator.", ""): value for key, value in state.items() if "generator." in key}
    state = {key: value for key, value in state.items() if not key.endswith(("freqs_cis", "causal_mask"))}
    codec.load_state_dict(state, strict=True)
    if eval_mode:
        codec.eval()
    return codec


@torch.inference_mode()
def decode_audio(codec: ArkttsCodec, codes: Tensor) -> tuple[Tensor, Tensor]:
    """Reference ``ArkttsModel.decode_audio`` semantics.

    codes: [B, num_codebooks, T] or [num_codebooks, T]; frames where any
    codebook holds the pad sentinel -1 stop the clip per item. Returns
    (waveform [B, Nmax] float32, lengths [B]).
    """
    if codes.ndim == 2:
        codes = codes.unsqueeze(0)
    downsample = codec.quantizer
    num_codebooks = downsample.semantic_quantizer.n_codebooks + downsample.quantizer.n_codebooks
    if codes.ndim != 3 or codes.shape[1] != num_codebooks:
        raise ValueError(f"codes must have shape [B, {num_codebooks}, T]")
    waveforms = []
    for item in codes:
        valid = (item >= 0).all(dim=0)
        length = int(valid.sum().item())
        if length == 0:
            waveform = torch.empty(0, device=codes.device, dtype=torch.float32)
        else:
            waveform = codec.decode(item[:, :length].unsqueeze(0))[0, 0].float()
        waveforms.append(waveform)
    max_length = max((w.numel() for w in waveforms), default=0)
    padded = torch.zeros((len(waveforms), max_length), dtype=torch.float32, device=codes.device)
    for index, waveform in enumerate(waveforms):
        padded[index, : waveform.numel()] = waveform
    lengths = torch.tensor([w.numel() for w in waveforms], dtype=torch.long, device=codes.device)
    return padded, lengths


class ArkttsCodecStreamer:
    """Chunk-push streaming decode.

    Push code frames incrementally; ``read()`` returns every sample that has
    become final except a small hold-back zone before the moving end (tail
    padding depends on total length, so the newest frames are not yet final);
    ``finish()`` flushes exactly the remainder. Concatenating all returned
    chunks reproduces one-shot ``decode`` truncated to ``frames * 2048``
    samples.
    """

    HOLD_BACK_FRAMES = 4  # newer frames whose tail effects are not final yet

    def __init__(self, codec: ArkttsCodec):
        self.codec = codec
        self._chunks: list[Tensor] = []
        self._total_frames = 0
        self._emitted = 0

    def push(self, codes_chunk: Tensor) -> None:
        chunk = codes_chunk.detach()
        if chunk.ndim == 1:
            chunk = chunk[None, :]
        if chunk.ndim != 2 or chunk.shape[0] < 2:
            raise ValueError("codes chunk must be [num_codebooks, T]")
        self._chunks.append(chunk.to(device=self._device(), dtype=torch.long))
        self._total_frames += int(chunk.shape[1])

    def _device(self):
        return next(self.codec.parameters()).device

    def _stack(self) -> Tensor:
        return torch.cat(self._chunks, dim=-1)

    def _confirmed_samples(self, decoded_numel: int) -> int:
        floor = max(0, self._total_frames - self.HOLD_BACK_FRAMES) * self.codec.samples_per_frame
        return min(floor, decoded_numel)

    def read(self) -> Tensor:
        if self._total_frames == 0 or self._total_frames <= self.HOLD_BACK_FRAMES:
            return torch.empty(0)
        decoded = self.codec.decode(self._stack().unsqueeze(0))[0, 0].detach().float()
        cut = self._confirmed_samples(decoded.numel())
        if cut <= self._emitted:
            return torch.empty(0)
        out = decoded[self._emitted : cut].clone()
        self._emitted = cut
        return out

    def finish(self) -> Tensor:
        if self._total_frames == 0:
            return torch.empty(0)
        decoded = self.codec.decode(self._stack().unsqueeze(0))[0, 0].detach().float()
        cut = min(self._total_frames * self.codec.samples_per_frame, decoded.numel())
        out = decoded[max(self._emitted, 0) : cut].clone()
        self._emitted = cut
        return out


__all__ = [
    "ArkttsCodec",
    "ArkttsCodecStreamer",
    "decode_audio",
    "load_arktts_codec",
]
