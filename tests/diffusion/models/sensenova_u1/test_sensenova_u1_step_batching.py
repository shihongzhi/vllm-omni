# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step-wise batching tests for the SenseNova-U1.5 pipeline.

A step-mode wave carries several in-flight requests through one
``denoise_step`` call. The pipeline forwards each request on its own state
(flash KV caches, geometry, CFG branches) and concatenates the velocity rows,
so the runner's per-request row slices keep every request evolving exactly as
it would alone. These tests pin that multi-request contract:

- a batched pair reproduces each request's solo denoise call sequence and
  final pixels bit-for-bit;
- heterogeneous waves stay safe: mixed t2i/it2i, unequal step counts, and a
  mid-flight admission that must not disturb the request already running;
- cancelling one request releases its caches and leaves its wave peer intact.

The driver below mirrors ``DiffusionModelRunner._execute_stepwise_core``: one
``denoise_step`` per wave over the still-active requests, then a
``step_scheduler`` update per request on its own ``latents.shape[0]`` row
slice, with ``post_decode`` firing per request as soon as its own schedule is
exhausted.
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
    """Records every model call and answers deterministically.

    Both stubs are pure functions of their inputs: a request must see the
    same response for the same image state no matter how waves interleave it
    with peers, which is exactly what the solo-vs-batched comparison pins.
    (Random replay would couple calls through one generator and cannot be
    aligned across different wave interleavings.)
    """

    def __init__(self):
        self.extract_calls: list[torch.Tensor] = []
        self.denoise_calls: list[dict] = []
        self.cleaned: list[object] = []

    def extract_feature(self, image_input, gen_model, grid_hw):
        self.extract_calls.append(image_input.clone())
        pooled = image_input.float().mean(dim=-1, keepdim=True)
        embeds = pooled.expand(*image_input.shape[:-1], DIM)
        return (embeds * 7.0).to(image_input.dtype)

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
        return torch.tanh(z) * (1.0 - t) + 0.25


def _make_setup(steps: int = STEPS, seed: int = 1234):
    """Build an isolated pipeline plus one request-local denoising state."""
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

    g = torch.Generator().manual_seed(seed)
    state = SenseNovaDenoiseState(
        image_prediction=torch.randn(BATCH, 3, H, W, generator=g),
        timesteps=1.0 - torch.arange(steps + 1, dtype=torch.float32) / steps,
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
    p = types.SimpleNamespace(batch_size=BATCH, num_steps=steps, image_size=[H, W])
    return pipe, state, p


def _install_recorder(pipe):
    recorder = _Recorder()
    pipe._extract_feature = recorder.extract_feature
    pipe._denoise = recorder.denoise
    return recorder


def _track_cleanup(monkeypatch, recorder):
    monkeypatch.setattr(pipe_mod, "clear_flash_kv_cache", lambda cache: recorder.cleaned.append(cache))


def _build_step_request(sn_state, p, request_id="req-1", think_text="", is_it2i=False):
    """Assemble the StepRequestState exactly as prepare_encode lays it out."""
    req = StepRequestState(request_id=request_id, sampling=types.SimpleNamespace(), prompt="a cat")
    req.latents = sn_state.image_prediction
    req.timesteps = sn_state.timesteps[:-1]
    req.extra[_STEP_DENOISE_STATE] = sn_state
    req.extra[_STEP_PARAMS] = p
    req.extra[_STEP_THINK_TEXT] = think_text
    req.extra[_STEP_IS_IT2I] = is_it2i
    return req


def _drive_wave(pipe, reqs):
    """Advance one scheduler wave the way the runner does.

    A single ``denoise_step`` covers every still-active request; each request
    then consumes its own ``latents.shape[0]`` row slice of the concatenated
    velocity, and the slices must cover it exactly.
    """
    v_pred = pipe.denoise_step(None, states=list(reqs))
    offset = 0
    for req in reqs:
        rows = req.latents.shape[0]
        pipe.step_scheduler(req, v_pred[offset : offset + rows])
        offset += rows
    assert offset == v_pred.shape[0]


def _run_to_completion(pipe, reqs):
    """Drive requests to completion, returning outputs and post_decode order.

    Mirrors the runner loop: each wave covers the still-active requests only,
    and a request is decoded as soon as its own schedule is exhausted, while
    its peers keep stepping.
    """
    outputs = {}
    decode_order = []
    while True:
        active = [req for req in reqs if not req.denoise_completed]
        if not active:
            break
        _drive_wave(pipe, active)
        for req in active:
            if req.denoise_completed:
                outputs[req.request_id] = pipe.post_decode(req)
                decode_order.append(req.request_id)
    return outputs, decode_order


def _assert_matching_calls(calls, expected):
    assert len(calls) == len(expected)
    for got, want in zip(calls, expected):
        assert got["step_i"] == want["step_i"]
        assert got["is_it2i"] == want["is_it2i"]
        assert torch.equal(got["image_prediction"], want["image_prediction"])
        assert torch.equal(got["t"], want["t"])
        assert torch.equal(got["z"], want["z"])
        assert torch.equal(got["image_embeds"], want["image_embeds"])


def _solo_run(monkeypatch, seed, request_id):
    """Run one request alone on its own pipeline; return calls, output, caches."""
    pipe, sn, p = _make_setup(seed=seed)
    recorder = _install_recorder(pipe)
    _track_cleanup(monkeypatch, recorder)
    req = _build_step_request(sn, p, request_id=request_id)
    outputs, _ = _run_to_completion(pipe, [req])
    return recorder.denoise_calls, outputs[request_id], sn.caches


def test_step_batch_matches_sequential_execution(monkeypatch):
    """Batching two requests into shared waves must change nothing per request.

    Every denoise forward, and the final pixels, must match a run where each
    request drove the step methods alone. The batch recorder sees the wave
    order (req-a then req-b per wave), so its call log is the interleaving of
    the two solo logs.
    """
    solo_a_calls, solo_a_out, _ = _solo_run(monkeypatch, seed=1234, request_id="req-a")
    solo_b_calls, solo_b_out, _ = _solo_run(monkeypatch, seed=5678, request_id="req-b")

    pipe, sn_a, p_a = _make_setup(seed=1234)
    _, sn_b, p_b = _make_setup(seed=5678)
    recorder = _install_recorder(pipe)
    _track_cleanup(monkeypatch, recorder)
    req_a = _build_step_request(sn_a, p_a, request_id="req-a")
    req_b = _build_step_request(sn_b, p_b, request_id="req-b")

    outputs, decode_order = _run_to_completion(pipe, [req_a, req_b])

    assert decode_order == ["req-a", "req-b"]
    interleaved = [call for pair in zip(solo_a_calls, solo_b_calls) for call in pair]
    _assert_matching_calls(recorder.denoise_calls, interleaved)

    for request_id, solo_out in (("req-a", solo_a_out), ("req-b", solo_b_out)):
        solo_img = solo_out.output["payload"]["image"]
        batch_img = outputs[request_id].output["payload"]["image"]
        assert np.array_equal(np.asarray(batch_img), np.asarray(solo_img))

    # Each request's caches were released exactly once, in decode order.
    assert recorder.cleaned == [
        sn_a.caches["cond"], sn_a.caches["uncond"],
        sn_b.caches["cond"], sn_b.caches["uncond"],
    ]


def test_step_batch_mixed_t2i_and_it2i_requests(monkeypatch):
    """A wave may mix t2i and it2i requests: each forward uses its own branch
    layout and CFG structure, and the concatenated rows stay per-request."""
    pipe, sn_t2i, p = _make_setup()
    _, sn_it2i, _ = _make_setup(seed=5678)
    recorder = _install_recorder(pipe)
    _track_cleanup(monkeypatch, recorder)
    req_t2i = _build_step_request(sn_t2i, p, request_id="req-t2i", is_it2i=False)
    req_it2i = _build_step_request(sn_it2i, p, request_id="req-it2i", is_it2i=True)

    outputs, decode_order = _run_to_completion(pipe, [req_t2i, req_it2i])

    assert decode_order == ["req-t2i", "req-it2i"]
    # One forward per request per wave, in states order.
    assert [call["is_it2i"] for call in recorder.denoise_calls] == [False, True] * STEPS
    assert req_t2i.step_index == req_it2i.step_index == STEPS
    assert outputs["req-t2i"].output["payload"]["image"] is not None
    assert outputs["req-it2i"].output["payload"]["image"] is not None
    # Both requests released their non-dict caches (img_cond dict is skipped).
    assert recorder.cleaned.count(sn_t2i.caches["cond"]) == 1
    assert recorder.cleaned.count(sn_it2i.caches["cond"]) == 1


def test_step_batch_unequal_step_counts(monkeypatch):
    """A shorter request retires mid-wave while its longer peer keeps going."""
    pipe, sn_long, p_long = _make_setup(steps=3, seed=1234)
    _, sn_short, p_short = _make_setup(steps=2, seed=5678)
    recorder = _install_recorder(pipe)
    _track_cleanup(monkeypatch, recorder)
    req_long = _build_step_request(sn_long, p_long, request_id="req-long")
    req_short = _build_step_request(sn_short, p_short, request_id="req-short")

    outputs, decode_order = _run_to_completion(pipe, [req_long, req_short])

    # The 2-step request finishes first; the final wave carries req-long alone.
    assert decode_order == ["req-short", "req-long"]
    assert [call["step_i"] for call in recorder.denoise_calls] == [0, 0, 1, 1, 2]
    assert req_short.step_index == 2
    assert req_long.step_index == 3
    assert outputs["req-short"].output["payload"]["image"] is not None
    assert outputs["req-long"].output["payload"]["image"] is not None


def test_mid_flight_admission_continues_older_request(monkeypatch):
    """A request admitted mid-denoise joins the wave without disturbing it.

    req-a steps alone first; req-b is then prepared the way the runner
    prepares a newly admitted request (prepare_encode on a fresh state) and
    joins from its own step 0. req-a's forwards must stay bit-identical to a
    solo run -- including the step it already took.
    """
    solo_calls, solo_out, _ = _solo_run(monkeypatch, seed=1234, request_id="req-a")

    pipe, sn_a, p_a = _make_setup(seed=1234)
    recorder = _install_recorder(pipe)
    _track_cleanup(monkeypatch, recorder)
    req_a = _build_step_request(sn_a, p_a, request_id="req-a")
    _drive_wave(pipe, [req_a])
    assert req_a.step_index == 1

    # Admission: the runner builds the new state and calls prepare_encode,
    # which parks the request-local denoising state on it.
    _, sn_b, _ = _make_setup(seed=5678)
    pipe._extract_input_images = lambda prompt: None
    pipe._prepare_t2i = lambda p: (sn_b, "")
    req_b = StepRequestState(
        request_id="req-b",
        sampling=types.SimpleNamespace(
            height=H, width=W, num_inference_steps=STEPS, seed=9, extra_args={}
        ),
        prompt="a boat",
    )
    returned = pipe.prepare_encode(req_b)
    assert returned is req_b
    assert req_b.extra[_STEP_DENOISE_STATE] is sn_b

    outputs, decode_order = _run_to_completion(pipe, [req_a, req_b])

    assert decode_order == ["req-a", "req-b"]
    # Wave log: [a0] then [a1, b0], [a2, b1], [b2]. req-a owns indices 0, 1, 3.
    _assert_matching_calls(
        [recorder.denoise_calls[i] for i in (0, 1, 3)],
        solo_calls,
    )
    solo_img = solo_out.output["payload"]["image"]
    assert np.array_equal(np.asarray(outputs["req-a"].output["payload"]["image"]), np.asarray(solo_img))
    # The admitted request completed its own schedule.
    assert req_b.step_index == STEPS
    assert outputs["req-b"].output["payload"]["image"] is not None


def test_aborted_request_peer_unaffected(monkeypatch):
    """Aborting one request mid-denoise releases its caches and spares peers.

    The abort takes effect at a wave boundary (the runner pops the state when
    the next wave assembles); the cancelled request's caches are released via
    the caller-side cleanup the request-local state supports, and the wave
    peer keeps evolving exactly as a solo run.
    """
    solo_calls, solo_out, _ = _solo_run(monkeypatch, seed=1234, request_id="req-a")

    pipe, sn_a, p_a = _make_setup(seed=1234)
    _, sn_b, p_b = _make_setup(seed=5678)
    recorder = _install_recorder(pipe)
    _track_cleanup(monkeypatch, recorder)
    req_a = _build_step_request(sn_a, p_a, request_id="req-a")
    req_b = _build_step_request(sn_b, p_b, request_id="req-b")

    _drive_wave(pipe, [req_a, req_b])

    # Cancel req-b between waves: the runner drops its state; the request-local
    # caches are released from the caller side (idempotent with post_decode's
    # finally path, which this request never reaches).
    pipe._cleanup_denoise_caches(sn_b.caches)
    assert recorder.cleaned == [sn_b.caches["cond"], sn_b.caches["uncond"]]

    while not req_a.denoise_completed:
        _drive_wave(pipe, [req_a])
    out_a = pipe.post_decode(req_a)

    # Wave log: [a0, b0] then [a1], [a2]. req-a owns indices 0, 2, 3.
    _assert_matching_calls(
        [recorder.denoise_calls[i] for i in (0, 2, 3)],
        solo_calls,
    )
    solo_img = solo_out.output["payload"]["image"]
    assert np.array_equal(np.asarray(out_a.output["payload"]["image"]), np.asarray(solo_img))
    # Peer caches released once; the aborted request's release is unchanged.
    assert recorder.cleaned.count(sn_a.caches["cond"]) == 1
    assert recorder.cleaned.count(sn_b.caches["cond"]) == 1


def test_step_batch_noise_is_per_request():
    """Denoising noise is seeded per request, so concurrent requests stay
    independent: same seed reproduces, different seeds differ."""
    _, sn_a, _ = _make_setup(seed=1234)
    _, sn_a_repeat, _ = _make_setup(seed=1234)
    _, sn_b, _ = _make_setup(seed=5678)

    assert torch.equal(sn_a.image_prediction, sn_a_repeat.image_prediction)
    assert not torch.equal(sn_a.image_prediction, sn_b.image_prediction)
