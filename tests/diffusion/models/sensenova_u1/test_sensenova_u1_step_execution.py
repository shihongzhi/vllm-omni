# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Step-execution tests for the SenseNova-U1.5 pipeline.

``prepare_encode`` / ``denoise_step`` / ``step_scheduler`` / ``post_decode``
split the request-level forward into runner-driven stages. These tests pin:

- the step-driven path to the full-request path bit-for-bit (same seed, same
  stubbed model, same recorder replay) across the t2i/it2i/think variants;
- the ``StepRequestState`` contract: latents/timesteps placement, the
  num_steps-visible schedule, ``total_steps``, and the ``extra`` layout;
- the multi-state layout ``step_scheduler`` consumes (per-request row slices)
  and the text-request routing of the pre-process func.
"""

import types

import numpy as np
import pytest
import torch

import vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 as pipe_mod
from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import (
    _STEP_DENOISE_STATE,
    _STEP_IS_IT2I,
    _STEP_PARAMS,
    _STEP_THINK_TEXT,
    SenseNovaDenoiseState,
    SenseNovaU1Pipeline,
)
from vllm_omni.diffusion.worker.utils import StepRequestState

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


def _track_cleanup(monkeypatch, recorder):
    monkeypatch.setattr(pipe_mod, "clear_flash_kv_cache", lambda cache: recorder.cleaned.append(cache))


def _build_step_request(sn_state, p, think_text="", is_it2i=False):
    """Assemble the StepRequestState exactly as prepare_encode lays it out."""
    req = StepRequestState(request_id="req-1", sampling=types.SimpleNamespace(), prompt="a cat")
    req.latents = sn_state.image_prediction
    req.timesteps = sn_state.timesteps[:-1]
    req.extra[_STEP_DENOISE_STATE] = sn_state
    req.extra[_STEP_PARAMS] = p
    req.extra[_STEP_THINK_TEXT] = think_text
    req.extra[_STEP_IS_IT2I] = is_it2i
    return req


def test_pipeline_declares_step_execution():
    from vllm_omni.diffusion.models.interface import SupportsStepExecution

    assert SenseNovaU1Pipeline.supports_step_execution is True
    assert isinstance(SenseNovaU1Pipeline, SupportsStepExecution)


@pytest.mark.parametrize("think_text,is_it2i", [("", False), ("reasoning...", False), ("it2i", True)])
def test_step_execution_matches_full_request_path(think_text, is_it2i, monkeypatch):
    """Driving the four step methods must reproduce the full request path bit-for-bit."""
    pipe_full, sn_full, p_full = _make_setup()
    rec_full = _install_recorder(pipe_full)
    _track_cleanup(monkeypatch, rec_full)
    full_out = pipe_full._run_denoising_loop(sn_full, p_full, think_text, is_it2i)

    pipe_step, sn_step, p_step = _make_setup()
    rec_step = _install_recorder(pipe_step)
    _track_cleanup(monkeypatch, rec_step)
    req = _build_step_request(sn_step, p_step, think_text, is_it2i)
    # Cache references must be captured before post_decode clears the dicts.
    cond_step, uncond_step = sn_step.caches["cond"], sn_step.caches["uncond"]
    input_batch = types.SimpleNamespace(states=(req,))

    while not req.denoise_completed:
        v_pred = pipe_step.denoise_step(input_batch)
        pipe_step.step_scheduler(req, v_pred)
    step_out = pipe_step.post_decode(req)

    # Every denoise forward saw bit-identical inputs, in the same order.
    assert len(rec_step.denoise_calls) == len(rec_full.denoise_calls) == STEPS
    for step_call, full_call in zip(rec_step.denoise_calls, rec_full.denoise_calls):
        assert step_call["step_i"] == full_call["step_i"]
        assert step_call["is_it2i"] == full_call["is_it2i"]
        assert torch.equal(step_call["t"], full_call["t"])
        assert torch.equal(step_call["image_prediction"], full_call["image_prediction"])
        assert torch.equal(step_call["z"], full_call["z"])
        assert torch.equal(step_call["image_embeds"], full_call["image_embeds"])

    # Feature extraction saw bit-identical inputs, in the same order.
    assert len(rec_step.extract_calls) == len(rec_full.extract_calls) == STEPS
    for step_in, full_in in zip(rec_step.extract_calls, rec_full.extract_calls):
        assert torch.equal(step_in, full_in)

    # Final pixels are bit-identical.
    step_img = step_out.output["payload"]["image"]
    full_img = full_out.output["payload"]["image"]
    assert np.array_equal(np.asarray(step_img), np.asarray(full_img))

    # think_text reaches the metadata unchanged.
    metadata = step_out.output["metadata"]
    if think_text:
        assert metadata["text"] == {"think_text": think_text}
    else:
        assert metadata == {}

    # The request retired after exactly num_steps scheduler updates.
    assert req.step_index == STEPS
    assert req.denoise_completed

    # post_decode released exactly the non-dict cache entries, like the loop.
    assert rec_step.cleaned == [cond_step, uncond_step]


def test_step_scheduler_update_matches_reference_euler_step():
    pipe, sn, p = _make_setup()
    req = _build_step_request(sn, p)
    tok = (H // (PATCH * MERGE)) * (W // (PATCH * MERGE))
    ch = 3 * (PATCH * MERGE) ** 2
    v_pred = torch.full((BATCH, tok, ch), 0.5)

    latents_before = req.latents.clone()
    t, t_next = sn.timesteps[0], sn.timesteps[1]
    expected_z = pipe_mod._patchify(latents_before, PATCH * MERGE) + (t_next - t) * v_pred
    expected = pipe_mod._unpatchify(expected_z, PATCH * MERGE, H, W)

    pipe.step_scheduler(req, v_pred)

    assert req.step_index == 1
    assert torch.equal(req.latents, expected)
    # The original noise tensor is never mutated in place.
    assert req.latents is not latents_before
    assert torch.equal(sn.image_prediction, latents_before)


def test_step_scheduler_ignores_none_noise_pred():
    pipe, sn, p = _make_setup()
    req = _build_step_request(sn, p)
    latents_before = req.latents.clone()

    pipe.step_scheduler(req, None)

    assert req.step_index == 0
    assert torch.equal(req.latents, latents_before)


def test_denoise_step_concatenates_and_slices_per_request():
    """Two requests at different steps: rows concatenate, and runner-style
    per-request slices drive each scheduler update with its own t."""
    pipe, sn_a, p_a = _make_setup()
    _, sn_b, p_b = _make_setup()
    recorder = _install_recorder(pipe)

    req_a = _build_step_request(sn_a, p_a, is_it2i=False)
    req_b = _build_step_request(sn_b, p_b, is_it2i=True)
    req_b.step_index = 1  # mid-flight request joins request a at step 0

    v_pred = pipe.denoise_step(None, states=[req_a, req_b])

    total_rows = req_a.latents.shape[0] + req_b.latents.shape[0]
    assert v_pred.shape[0] == total_rows
    assert [call["step_i"] for call in recorder.denoise_calls] == [0, 1]
    assert [call["is_it2i"] for call in recorder.denoise_calls] == [False, True]

    # Runner contract: slice noise_pred by each request's latents rows.
    offset = 0
    for req in (req_a, req_b):
        rows = req.latents.shape[0]
        pipe.step_scheduler(req, v_pred[offset : offset + rows].clone())
        offset += rows
    assert offset == v_pred.shape[0]
    assert req_a.step_index == 1
    assert req_b.step_index == 2


def _make_prepare_setup():
    pipe, sn_state, _ = _make_setup()
    calls = {"t2i": 0, "it2i": 0}
    pipe._prepare_t2i = lambda p: (calls.__setitem__("t2i", calls["t2i"] + 1) or (sn_state, "thoughts"))
    pipe._prepare_it2i = lambda p, imgs: (calls.__setitem__("it2i", calls["it2i"] + 1) or (sn_state, "edited"))
    return pipe, sn_state, calls


def _prepare_sampling():
    return types.SimpleNamespace(height=None, width=None, num_inference_steps=STEPS, seed=7, extra_args={})


def test_prepare_encode_populates_step_request_state():
    pipe, sn_state, calls = _make_prepare_setup()
    pipe._extract_input_images = lambda prompt: None

    req = StepRequestState(request_id="req-1", sampling=_prepare_sampling(), prompt="a cat")
    returned = pipe.prepare_encode(req)

    assert returned is req
    assert calls == {"t2i": 1, "it2i": 0}
    assert req.latents is sn_state.image_prediction
    # The runner must see exactly num_steps timesteps; the final schedule
    # entry stays on the request-local state for the t_next lookup.
    assert torch.equal(req.timesteps, sn_state.timesteps[:-1])
    assert req.total_steps == STEPS
    assert req.step_index == 0
    assert torch.equal(req.current_timestep, sn_state.timesteps[0])
    assert req.extra[_STEP_DENOISE_STATE] is sn_state
    assert req.extra[_STEP_PARAMS].num_steps == STEPS
    assert req.extra[_STEP_THINK_TEXT] == "thoughts"
    assert req.extra[_STEP_IS_IT2I] is False


def test_prepare_encode_routes_image_requests_to_it2i():
    pipe, sn_state, calls = _make_prepare_setup()
    pipe._extract_input_images = lambda prompt: [object()]

    prompt = {"prompt": "edit this", "multi_modal_data": {"image": ["img.png"]}}
    req = StepRequestState(request_id="req-1", sampling=_prepare_sampling(), prompt=prompt)
    pipe.prepare_encode(req)

    assert calls == {"t2i": 0, "it2i": 1}
    assert req.extra[_STEP_IS_IT2I] is True
    assert req.extra[_STEP_THINK_TEXT] == "edited"


def test_prepare_encode_rejects_text_only_requests():
    pipe, sn_state, _ = _make_prepare_setup()

    prompt = {"prompt": "just chat", "modalities": ["text"]}
    req = StepRequestState(request_id="req-1", sampling=_prepare_sampling(), prompt=prompt)
    with pytest.raises(ValueError, match="text-only"):
        pipe.prepare_encode(req)


def test_post_decode_releases_caches_even_on_decode_failure(monkeypatch):
    pipe, sn, p = _make_setup()
    released = []
    monkeypatch.setattr(pipe_mod, "clear_flash_kv_cache", lambda cache: released.append(cache))
    req = _build_step_request(sn, p, think_text="reasoning...")
    cond, uncond = sn.caches["cond"], sn.caches["uncond"]

    out = pipe.post_decode(req)
    assert out.output["metadata"]["text"] == {"think_text": "reasoning..."}
    assert released == [cond, uncond]
    # The release empties the cache dict, so a repeat release is a no-op.
    assert not sn.caches
    pipe.release_step_state(req)
    assert released == [cond, uncond]

    def exploding_decode(image_prediction, think_text=""):
        raise RuntimeError("decode exploded")

    pipe._build_diffusion_output = exploding_decode
    # A fresh request proves the finally-path releases even when decode raises.
    _, sn2, _ = _make_setup()
    cond2, uncond2 = sn2.caches["cond"], sn2.caches["uncond"]
    req2 = _build_step_request(sn2, p)
    with pytest.raises(RuntimeError, match="decode exploded"):
        pipe.post_decode(req2)
    assert released == [cond, uncond, cond2, uncond2]


def test_pre_process_func_routes_text_requests_to_full_forward():
    from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import (
        get_sensenova_u1_pre_process_func,
    )

    fn = get_sensenova_u1_pre_process_func(types.SimpleNamespace(step_execution=True))

    text_req = types.SimpleNamespace(prompt={"prompt": "hi", "modalities": ["text"]}, use_step_execution=True)
    assert fn(text_req) is text_req
    assert text_req.use_step_execution is False

    img_req = types.SimpleNamespace(prompt={"prompt": "hi", "modalities": ["image"]}, use_step_execution=True)
    fn(img_req)
    assert img_req.use_step_execution is True

    plain_req = types.SimpleNamespace(prompt="hi", use_step_execution=True)
    fn(plain_req)
    assert plain_req.use_step_execution is True

    fn_off = get_sensenova_u1_pre_process_func(types.SimpleNamespace(step_execution=False))
    text_req_off = types.SimpleNamespace(prompt={"modalities": ["text"]}, use_step_execution=True)
    fn_off(text_req_off)
    assert text_req_off.use_step_execution is True


def test_pre_process_func_registered_for_pipeline():
    from vllm_omni.diffusion.registry import _DIFFUSION_PRE_PROCESS_FUNCS

    assert _DIFFUSION_PRE_PROCESS_FUNCS["SenseNovaU1Pipeline"] == "get_sensenova_u1_pre_process_func"
