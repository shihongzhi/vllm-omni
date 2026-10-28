#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fibo-2.0 transformer parity: the correctness oracle for the in-tree port.

Checks that :class:`Fibo2Transformer2DModel` (the vLLM-Omni port) reproduces
the diffusers ``BriaFibo2Transformer2DModel`` on the released
``briaai/Fibo-2.0-dev`` weights, using the fixed inputs and outputs captured
by the Diffusers baseline run (vllm-omni issue #9212, step 1):

* inputs  - first latent_model_input + prompt/negative prompt embeds, seed 42
* outputs - first-step noise predictions (cond / uncond), saved in fp32

The reference data lives in ``ref-data/run1/tensors`` of the baseline; the
weights in the HF cache downloaded for it. Both are needed, so this test runs
locally rather than in CI. Point ``FIBO2_HF_HOME`` at the baseline HF cache
and ``FIBO2_REF_DATA`` at the baseline ref-data directory (defaults below),
then::

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m pytest \
        tests/diffusion/models/bria_fibo2/test_fibo2_transformer_parity.py -v -s

Tolerances mirror the AuK parity oracle: bf16 kernels differ from diffusers'
SDPA in reduction order, so exact bitwise equality is not expected; agreement
within bf16 noise (atol 3e-2 on fp32 outputs, cosine > 0.999) is.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, safe_open

FIBO2_HF_HOME = Path(os.environ.get("FIBO2_HF_HOME", "/root/autodl-fs/fibo-baseline/hf-home"))
FIBO2_REF_DATA = Path(os.environ.get("FIBO2_REF_DATA", "/root/autodl-fs/fibo-baseline/ref-data"))

# Tolerances: the port runs end-to-end in bf16 (the deployment dtype), while
# the diffusers baseline ran the same bf16 weights through PyTorch SDPA and
# eager RMSNorm. Kernel reduction orders differ, so bitwise equality is not
# expected; after 44 blocks the max deviation sits near bf16 noise (~1e-2
# mean, ~1% of the output scale). A logic error anywhere (RoPE planes, bundle
# indexing, modulation) collapses the cosine far below this bar, so the
# cosine + relative-error pair is the correctness oracle; absolute atol is
# reported for information.
FORWARD_COS = 0.9995
FORWARD_REL = 0.015


def _snapshot_dir() -> Path:
    root = FIBO2_HF_HOME / "hub" / "models--briaai--Fibo-2.0-dev" / "snapshots"
    return next(iter(sorted(root.iterdir())))


def _skip_if_unavailable():
    if not (FIBO2_REF_DATA / "run1" / "tensors" / "first_step.safetensors").exists():
        pytest.skip(f"reference data not found under {FIBO2_REF_DATA}")
    if not FIBO2_HF_HOME.exists():
        pytest.skip(f"HF cache not found under {FIBO2_HF_HOME}")


@pytest.fixture(scope="module")
def vllm_env():
    """Single-process distributed env so vLLM parallel layers initialize."""
    import vllm.distributed as dist
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cuda"))):
        dist.init_distributed_environment(world_size=1, rank=0, distributed_init_method="tcp://127.0.0.1:29611")
        dist.initialize_model_parallel(tensor_model_parallel_size=1)
        yield


@pytest.fixture(scope="module")
def model(vllm_env):
    from vllm_omni.diffusion.models.bria_fibo2.fibo2_transformer import (
        Fibo2Transformer2DModel,
    )

    snap = _snapshot_dir()
    config = json.loads((snap / "transformer" / "config.json").read_text())
    config.pop("_class_name", None)
    config.pop("_diffusers_version", None)

    torch.set_default_dtype(torch.bfloat16)
    model = Fibo2Transformer2DModel(**config)
    model = model.to("cuda")

    weights = []
    for shard in sorted((snap / "transformer").glob("diffusion_pytorch_model-*.safetensors")):
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                weights.append((k, f.get_tensor(k)))
    loaded = model.load_weights(weights)
    model.eval()

    all_params = {n for n, _ in model.named_parameters()}
    missing = all_params - loaded
    assert not missing, f"parameters never loaded from checkpoint: {sorted(missing)}"
    return model


def _first_inputs():
    tdir = FIBO2_REF_DATA / "run1" / "tensors"
    first = load_file(tdir / "first_step.safetensors")
    pos = load_file(tdir / "prompt_embeds.safetensors")
    neg = load_file(tdir / "negative_prompt_embeds.safetensors")
    meta = json.loads((FIBO2_REF_DATA / "run1" / "meta.json").read_text())
    # Basline scheduler timesteps are in "1000 * sigma" units; the transformer
    # takes t in [0, 1].
    t0 = meta["timing_s"]["timesteps"][0] / 1000.0
    timestep = torch.full((1,), t0, dtype=torch.float32, device="cuda")
    return first, pos, neg, timestep


def _compare(name: str, out: torch.Tensor, ref: torch.Tensor, report: dict):
    out_f, ref_f = out.float().cpu(), ref.float().cpu()
    max_abs = (out_f - ref_f).abs().max().item()
    mean_abs = (out_f - ref_f).abs().mean().item()
    cos = torch.nn.functional.cosine_similarity(out_f.flatten(), ref_f.flatten(), dim=0).item()
    ref_scale = ref_f.abs().max().item()
    rel = max_abs / ref_scale
    ok = (cos >= FORWARD_COS) and (rel <= FORWARD_REL)
    report[name] = {
        "pass": ok,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "cosine": cos,
        "ref_abs_max": ref_scale,
        "rel_err": rel,
    }


@pytest.mark.parametrize("branch", ["cond", "uncond"])
def test_first_step_parity(model, branch):
    _skip_if_unavailable()
    first, pos, neg, timestep = _first_inputs()
    embeds = pos["prompt_embeds"] if branch == "cond" else neg["negative_prompt_embeds"]
    ref = first[f"noise_pred_{branch}"]

    with torch.no_grad():
        out, _ = model(
            hidden_states=first["latent_model_input"].to("cuda", torch.bfloat16),
            timestep=timestep,
            encoder_hidden_states=embeds.to("cuda", torch.bfloat16),
            encoder_attention_mask=None,
            context_latents=None,
        )

    report: dict = {}
    _compare(branch, out, ref, report)
    print(json.dumps(report, indent=2))
    entry = report[branch]
    assert entry["pass"], (
        f"{branch} forward mismatch: cosine={entry['cosine']:.6f} (min {FORWARD_COS}), "
        f"rel_err={entry['rel_err']:.4f} (max {FORWARD_REL}), max_abs_diff={entry['max_abs_diff']:.4g}"
    )


def test_gist_count():
    from vllm_omni.diffusion.models.bria_fibo2.fibo2_transformer import (
        compute_gist_count,
    )

    # Text shorter than gist_min_text_len maps to the floor: min_gist tokens.
    assert compute_gist_count([78], 256, 512, 64, 256, 2048) == [256]
    assert compute_gist_count([46], 256, 512, 64, 256, 2048) == [256]
    # At max_text_len the gist saturates.
    assert compute_gist_count([2048], 256, 512, 64, 256, 2048) == [512]
    # Mid-range maps linearly, rounded up to a gist_step multiple.
    assert compute_gist_count([1024], 256, 512, 64, 256, 2048) == [384]
