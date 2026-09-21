# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavioral-equivalence tests for the split denoising-loop helpers.

``_run_denoising_loop()`` owns a request-local :class:`SenseNovaDenoiseState`
and delegates to ``_run_single_denoise_step()`` /
``_apply_denoise_step_update()`` / ``_cleanup_denoise_caches()`` /
``_build_diffusion_output()``. These tests pin the helpers to the
pre-refactor monolithic loop (kept verbatim below as the reference) by
running both on identical stubbed inputs and comparing every per-step
tensor, the final pixels, and the cache cleanup set — including the
exception and abort paths.
"""

import types

import numpy as np
import pytest
import torch

import vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 as pipe_mod
from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import (
    SenseNovaDenoiseState,
    SenseNovaU1Pipeline,
    _patchify,
    _to_pil,
    _unpatchify,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

PATCH = 2
MERGE = 2
H = W = 16
GRID = H // PATCH
DIM = 16
BATCH = 2
STEPS = 3


class _StubEmbedder:
    """Deterministic stand-in for timestep/noise-scale embedders: [N] -> [N, DIM]."""

    def __init__(self, dim: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.w = torch.randn(dim, generator=g)
        self.b = torch.randn(dim, generator=g)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.w + self.b


class _Recorder:
    """Records every model call and replays identical noise across runs."""

    def __init__(self):
        self.extract_calls: list[torch.Tensor] = []
        self.denoise_calls: list[dict] = []
        self.cleaned: list[object] = []
        self.gen = torch.Generator().manual_seed(4242)

    def extract_feature(self, image_input, gen_model, grid_hw):
        self.extract_calls.append(image_input.clone())
        return torch.randn(image_input.shape[0], DIM, generator=self.gen)

    def denoise(self, image_prediction, state, t, z, image_embeds, caches, p, step_i, is_it2i):
        self.denoise_calls.append(
            {
                "image_prediction": image_prediction.clone(),
                "t": t.clone(),
                "z": z.clone(),
                "image_embeds": image_embeds.clone(),
                "step_i": step_i,
                "is_it2i": is_it2i,
            }
        )
        return torch.randn(z.shape, generator=self.gen)


def _make_setup():
    pipe = object.__new__(SenseNovaU1Pipeline)
    pipe.patch_size = PATCH
    pipe.merge_size = MERGE
    pipe.fm_modules = {
        "timestep_embedder": _StubEmbedder(DIM, 1),
        "noise_scale_embedder": _StubEmbedder(DIM, 2),
    }
    pipe.model_cfg = types.SimpleNamespace(
        add_noise_scale_embedding=True,
        noise_scale_max_value=2.0,
    )

    g = torch.Generator().manual_seed(1234)
    state = SenseNovaDenoiseState(
        image_prediction=torch.randn(BATCH, 3, H, W, generator=g),
        timesteps=1.0 - torch.arange(STEPS + 1, dtype=torch.float32) / STEPS,
        grid_h=GRID,
        grid_w=GRID,
        grid_hw=torch.tensor([[GRID, GRID]]),
        token_h=GRID,
        token_w=GRID,
        noise_scale=0.5,
        caches={
            "cond": object(),
            "uncond": object(),
            "img_cond": {"nested": True},  # dict entries must be skipped by cleanup
        },
    )
    p = types.SimpleNamespace(batch_size=BATCH, num_steps=STEPS, image_size=[H, W])
    return pipe, state, p


def _install_recorder(pipe):
    recorder = _Recorder()
    pipe._extract_feature = recorder.extract_feature
    pipe._denoise = recorder.denoise
    return recorder


def _reference_denoising_loop(pipe, state, p, think_text="", is_it2i=False):
    """The pre-refactor monolithic loop, copied verbatim from the old code.

    Reads the same values through the pre-refactor plain-namespace locals
    (``ns`` / ``caches``) that ``forward()`` used to own.
    """
    clear_flash_kv_cache = pipe_mod.clear_flash_kv_cache

    ns = types.SimpleNamespace(
        image_prediction=state.image_prediction,
        timesteps=state.timesteps,
        grid_h=state.grid_h,
        grid_w=state.grid_w,
        grid_hw=state.grid_hw,
        token_h=state.token_h,
        token_w=state.token_w,
        noise_scale=state.noise_scale,
    )
    caches = state.caches

    merge_size = pipe.merge_size
    image_prediction = ns.image_prediction

    for step_i in range(p.num_steps):
        t = ns.timesteps[step_i]
        t_next = ns.timesteps[step_i + 1]

        z = _patchify(image_prediction, pipe.patch_size * merge_size)
        image_input = _patchify(image_prediction, pipe.patch_size, channel_first=True)
        image_embeds = pipe._extract_feature(
            image_input.view(p.batch_size * ns.grid_h * ns.grid_w, -1),
            gen_model=True,
            grid_hw=ns.grid_hw,
        ).view(p.batch_size, ns.token_h * ns.token_w, -1)

        t_expanded = t.expand(p.batch_size * ns.token_h * ns.token_w)
        timestep_embeddings = pipe.fm_modules["timestep_embedder"](t_expanded).view(
            p.batch_size,
            ns.token_h * ns.token_w,
            -1,
        )
        if pipe.model_cfg.add_noise_scale_embedding:
            ns_tensor = torch.full_like(
                t_expanded, ns.noise_scale / pipe.model_cfg.noise_scale_max_value
            )
            ns_emb = pipe.fm_modules["noise_scale_embedder"](ns_tensor).view(
                p.batch_size,
                ns.token_h * ns.token_w,
                -1,
            )
            timestep_embeddings = timestep_embeddings + ns_emb
        image_embeds = image_embeds + timestep_embeddings

        v_pred = pipe._denoise(image_prediction, ns, t, z, image_embeds, caches, p, step_i, is_it2i)
        z = z + (t_next - t) * v_pred
        image_prediction = _unpatchify(
            z, pipe.patch_size * merge_size, p.image_size[1], p.image_size[0]
        )

    for key in ("cond", "uncond", "img_cond"):
        if key in caches and not isinstance(caches[key], dict):
            clear_flash_kv_cache(caches[key])

    return image_prediction, _to_pil(image_prediction)


def _run_reference(monkeypatch, think_text, is_it2i):
    pipe, state, p = _make_setup()
    recorder = _install_recorder(pipe)
    monkeypatch.setattr(
        pipe_mod, "clear_flash_kv_cache", lambda cache: recorder.cleaned.append(cache)
    )
    final_state, images = _reference_denoising_loop(pipe, state, p, think_text, is_it2i)
    return recorder, final_state, images, state.caches


def _run_current(monkeypatch, think_text, is_it2i):
    pipe, state, p = _make_setup()
    recorder = _install_recorder(pipe)
    monkeypatch.setattr(
        pipe_mod, "clear_flash_kv_cache", lambda cache: recorder.cleaned.append(cache)
    )
    output = pipe._run_denoising_loop(state, p, think_text, is_it2i)
    return recorder, output, state.caches


@pytest.mark.parametrize("think_text,is_it2i", [("", False), ("reasoning...", False), ("it2i", True)])
def test_split_loop_matches_reference(think_text, is_it2i, monkeypatch):
    ref_recorder, ref_state, ref_images, ref_caches = _run_reference(monkeypatch, think_text, is_it2i)
    new_recorder, new_output, new_caches = _run_current(monkeypatch, think_text, is_it2i)

    # Every denoise forward saw bit-identical inputs, in the same order.
    assert len(ref_recorder.denoise_calls) == len(new_recorder.denoise_calls) == STEPS
    for ref_call, new_call in zip(ref_recorder.denoise_calls, new_recorder.denoise_calls):
        assert ref_call["step_i"] == new_call["step_i"]
        assert ref_call["is_it2i"] == new_call["is_it2i"]
        assert torch.equal(ref_call["t"], new_call["t"])
        assert torch.equal(ref_call["image_prediction"], new_call["image_prediction"])
        assert torch.equal(ref_call["z"], new_call["z"])
        assert torch.equal(ref_call["image_embeds"], new_call["image_embeds"])

    # Feature extraction saw bit-identical inputs, in the same order.
    assert len(ref_recorder.extract_calls) == len(new_recorder.extract_calls) == STEPS
    for ref_in, new_in in zip(ref_recorder.extract_calls, new_recorder.extract_calls):
        assert torch.equal(ref_in, new_in)

    # Final pixels are bit-identical.
    new_img = new_output.output["payload"]["image"]
    assert np.array_equal(np.asarray(new_img), np.asarray(ref_images[0]))

    # think_text reaches the metadata unchanged.
    metadata = new_output.output["metadata"]
    if think_text:
        assert metadata["text"] == {"think_text": think_text}
    else:
        assert metadata == {}

    # Cleanup released exactly the non-dict cache entries, in key order.
    assert ref_recorder.cleaned == [ref_caches["cond"], ref_caches["uncond"]]
    assert new_recorder.cleaned == [new_caches["cond"], new_caches["uncond"]]


def test_single_step_helper_does_not_mutate_image_state():
    pipe, state, p = _make_setup()
    recorder = _install_recorder(pipe)

    image_before = state.image_prediction.clone()
    pipe._run_single_denoise_step(state.image_prediction, state, p, 0, is_it2i=False)

    assert torch.equal(state.image_prediction, image_before)
    assert len(recorder.denoise_calls) == 1


def test_cleanup_runs_on_exception_path(monkeypatch):
    pipe, state, p = _make_setup()
    recorder = _install_recorder(pipe)
    monkeypatch.setattr(
        pipe_mod, "clear_flash_kv_cache", lambda cache: recorder.cleaned.append(cache)
    )

    def failing_denoise(*args, **kwargs):
        if recorder.denoise_calls:
            raise RuntimeError("denoise exploded mid-loop")
        return recorder.denoise(*args, **kwargs)

    pipe._denoise = failing_denoise

    with pytest.raises(RuntimeError, match="denoise exploded"):
        pipe._run_denoising_loop(state, p, is_it2i=False)

    # Step 0 completed, step 1 raised — caches must still be released.
    assert len(recorder.denoise_calls) == 1
    assert recorder.cleaned == [state.caches["cond"], state.caches["uncond"]]


def test_cleanup_after_partial_steps_covers_abort_path(monkeypatch):
    """Abort between steps: the caller holds the request-local state and
    releases the caches externally, yielding the same release set as normal
    completion."""
    pipe, state, p = _make_setup()
    recorder = _install_recorder(pipe)
    released = []
    monkeypatch.setattr(pipe_mod, "clear_flash_kv_cache", lambda cache: released.append(cache))

    pipe._run_single_denoise_step(state.image_prediction, state, p, 0, is_it2i=False)
    pipe._cleanup_denoise_caches(state.caches)

    assert len(recorder.denoise_calls) == 1
    assert released == [state.caches["cond"], state.caches["uncond"]]


def test_init_noise_and_schedule_builds_request_state():
    pipe = object.__new__(SenseNovaU1Pipeline)
    pipe.patch_size = PATCH
    pipe.merge_size = MERGE
    pipe.device = torch.device("cpu")
    pipe.od_config = types.SimpleNamespace(dtype=torch.float32)
    pipe.model_cfg = types.SimpleNamespace(
        noise_scale=1.0,
        noise_scale_mode=None,
        noise_scale_max_value=2.0,
    )
    p = types.SimpleNamespace(
        batch_size=BATCH, num_steps=STEPS, image_size=[W, H], seed=7, timestep_shift=3.0
    )

    state = pipe._init_noise_and_schedule(p)

    assert isinstance(state, SenseNovaDenoiseState)
    assert state.caches == {}
    assert state.image_prediction.shape == (BATCH, 3, H, W)
    assert state.image_prediction.dtype == torch.float32
    assert state.timesteps.shape == (STEPS + 1,)
    assert (state.grid_h, state.grid_w) == (GRID, GRID)
    assert (state.token_h, state.token_w) == (H // (PATCH * MERGE), W // (PATCH * MERGE))
    assert state.grid_hw.tolist() == [[GRID, GRID]] * BATCH
    assert state.noise_scale == 1.0

    # Same seed -> identical request-local noise.
    replay = pipe._init_noise_and_schedule(p)
    assert torch.equal(state.image_prediction, replay.image_prediction)


def test_cleanup_skips_dict_valued_caches(monkeypatch):
    pipe, state, p = _make_setup()
    released = []
    monkeypatch.setattr(pipe_mod, "clear_flash_kv_cache", lambda cache: released.append(cache))

    pipe._cleanup_denoise_caches(state.caches)

    # "cond" and "uncond" are released; "img_cond" is a dict and must be skipped.
    assert released == [state.caches["cond"], state.caches["uncond"]]
