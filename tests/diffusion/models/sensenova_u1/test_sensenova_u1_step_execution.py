# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SenseNova-U1 step-execution lifecycle tests.

Drives the step contract (``prepare_encode`` / ``denoise_step`` /
``step_scheduler`` / ``post_decode``) on CPU against stand-in encoders and a
deterministic noise predictor, and checks parity with the request-level
denoising loop that ``forward()`` runs. The stand-ins keep the real CFG
branching, the real timestep / noise-scale embeddings, and the real Euler
scheduler math in play, so any divergence between the two paths shows up as
a numeric difference.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_PATCH = 16
_MERGE = 2
_HIDDEN = 32

# Fixed projection so the stand-in vision encoder is deterministic.
_IN_DIM = _PATCH * _PATCH * 3
_PROJ = torch.randn(_IN_DIM, _HIDDEN, generator=torch.Generator().manual_seed(123))
_EMBED_PROJ = torch.randn(_HIDDEN, 3 * (_PATCH * _MERGE) ** 2, generator=torch.Generator().manual_seed(7))

_kv_serial = itertools.count()


class _FakeKV:
    """Stand-in prefix KV cache with a per-object serial.

    The serial gives the stand-in noise predictor a branch-dependent offset,
    so CFG combination math (and cfg_zero_star's step dependence) is actually
    exercised instead of cancelling between identical branches.
    """

    def __init__(self):
        self.serial = next(_kv_serial)
        self.layers = []


class _FakeLM:
    """Stand-in language model for embedding lookups and prefix forwards."""

    def __call__(self, *, input_ids=None, inputs_embeds=None, embed_only=False, **_kwargs):
        if embed_only:
            seq_len = int(input_ids.shape[1])
            return SimpleNamespace(inputs_embeds=torch.zeros(1, seq_len, _HIDDEN))
        return SimpleNamespace(
            inputs_embeds=None,
            past_key_values=_FakeKV(),
            hidden_states=None,
            logits=torch.zeros(1, 1, 8),
        )


class _FakeTokenizer:
    """Deterministic stand-in: one token per whitespace-separated word."""

    def __call__(self, query, return_tensors="pt", add_special_tokens=True):
        num_tokens = max(1, len(query.split())) + (1 if add_special_tokens else 0)
        input_ids = torch.arange(1, num_tokens + 1, dtype=torch.long).unsqueeze(0)
        return {"input_ids": input_ids}

    def convert_tokens_to_ids(self, token):
        return {"<img>": 11, "</img>": 12, "<IMG_CONTEXT>": 13, "<|im_end|>": 14, "</think>": 15}.get(token, 0)


def _fake_extract_feature(pixel_values, gen_model=False, grid_hw=None):
    """Stand-in ViT: pool merge^2 patch rows into token rows, then project."""
    token_rows = pixel_values.view(-1, _MERGE * _MERGE, pixel_values.shape[-1]).mean(dim=1)
    return token_rows.float() @ _PROJ


def _stand_in_predict_noise(**kwargs):
    """Deterministic x0-style predictor over the real branch kwargs."""
    z = kwargs["z"]
    t = float(kwargs["t"])
    embeds = kwargs["input_embeds"]
    branch_offset = 0.25 * (kwargs["past_key_values"].serial % 3)
    x_pred = z * (1.0 - 0.5 * t) + (embeds @ _EMBED_PROJ) * 0.5 + branch_offset
    return (x_pred - z) / (1.0 - kwargs["t"]).clamp_min(kwargs["t_eps"])


def _make_pipeline(
    monkeypatch,
    *,
    add_noise_scale_embedding: bool = False,
) -> tuple:
    """A bare pipeline carrying only what the denoise path touches."""
    from vllm_omni.diffusion.models import sensenova_u1 as pkg
    from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import (
        SenseNovaU1Pipeline,
        TimestepEmbedder,
    )

    pipeline = object.__new__(SenseNovaU1Pipeline)
    pipeline.device = "cpu"
    pipeline.patch_size = _PATCH
    pipeline.merge_size = _MERGE
    pipeline.downsample_ratio = 1.0 / _MERGE
    pipeline.od_config = SimpleNamespace(dtype=torch.float32)
    pipeline.model_cfg = SimpleNamespace(
        noise_scale=1.0,
        noise_scale_mode="constant",
        noise_scale_base_image_seq_len=256,
        noise_scale_max_value=8.0,
        add_noise_scale_embedding=add_noise_scale_embedding,
        use_pixel_head=False,
    )
    pipeline.tokenizer = _FakeTokenizer()
    pipeline.img_context_token_id = 13
    pipeline.img_start_token_id = 11
    pipeline.language_model = _FakeLM()
    pipeline._interrupt = False

    torch.manual_seed(0)
    timestep_embedder = TimestepEmbedder(_HIDDEN)
    for param in timestep_embedder.parameters():
        torch.nn.init.normal_(param)
    fm_modules = {"vision_model_mot_gen": torch.nn.Identity(), "timestep_embedder": timestep_embedder}
    if add_noise_scale_embedding:
        noise_scale_embedder = TimestepEmbedder(_HIDDEN)
        for param in noise_scale_embedder.parameters():
            torch.nn.init.normal_(param)
        fm_modules["noise_scale_embedder"] = noise_scale_embedder
    pipeline.fm_modules = torch.nn.ModuleDict(fm_modules)

    pipeline._extract_feature = _fake_extract_feature
    pipeline.predict_noise = _stand_in_predict_noise

    cleared: list = []
    monkeypatch.setattr(pkg.pipeline_sensenova_u1, "prepare_flash_kv_cache", lambda kv, current_len, batch_size: None)
    monkeypatch.setattr(pkg.pipeline_sensenova_u1, "clear_flash_kv_cache", cleared.append)
    monkeypatch.setattr(pkg.pipeline_sensenova_u1, "_to_pil", lambda batch: [batch])
    return pipeline, cleared


def _sampling(**extra_args):
    return SimpleNamespace(
        height=64,
        width=96,
        num_inference_steps=6,
        seed=11,
        extra_args=dict(extra_args),
    )


def _t2i_prompt(**overrides):
    prompt = {"prompt": "a cat in a hat", "modalities": ["image"]}
    prompt.update(overrides)
    return prompt


def _run_request_level(pipeline, sampling, prompt):
    req = SimpleNamespace(prompts=[prompt], sampling_params=sampling)
    p = pipeline._parse_request(req)
    return pipeline._forward_t2i(p)


def _run_step_lifecycle(pipeline, sampling, prompt, request_id="req-step"):
    from vllm_omni.diffusion.worker.input_batch import InputBatch
    from vllm_omni.diffusion.worker.utils import StepRequestState

    state = StepRequestState(request_id=request_id, sampling=sampling, prompt=prompt)
    pipeline.prepare_encode(state)
    cached_batch = None
    while not state.denoise_completed:
        input_batch = InputBatch.make_batch([state], cached_batch=cached_batch)
        cached_batch = input_batch
        noise_pred = pipeline.denoise_step(input_batch, states=[state])
        pipeline.step_scheduler(state, noise_pred)
    return pipeline.post_decode(state), state


def test_pipeline_satisfies_step_execution_protocol():
    from vllm_omni.diffusion.models.interface import supports_step_execution
    from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import SenseNovaU1Pipeline

    assert supports_step_execution(SenseNovaU1Pipeline)


def test_step_execution_matches_request_level_t2i_loop(monkeypatch):
    """Stepping the lifecycle must reproduce forward()'s denoising loop.

    ``cfg_norm="cfg_zero_star"`` makes the CFG combine depend on ``step_i``,
    so a mis-threaded step index (or timestep) diverges from the reference.
    """
    from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import _patchify

    pipeline, cleared = _make_pipeline(monkeypatch, add_noise_scale_embedding=True)
    sampling = _sampling(cfg_scale=4.0, cfg_norm="cfg_zero_star", timestep_shift=3.0)
    prompt = _t2i_prompt()

    reference = _run_request_level(pipeline, sampling, prompt)
    reference_cleared = len(cleared)
    step_output, state = _run_step_lifecycle(pipeline, sampling, prompt)

    reference_image = reference.output["payload"]["image"]
    assert isinstance(reference_image, torch.Tensor)
    torch.testing.assert_close(step_output.output["payload"]["image"], reference_image)
    torch.testing.assert_close(state.latents, _patchify(reference_image, _PATCH * _MERGE))
    assert step_output.output["metadata"] == reference.output["metadata"] == {}

    # The request-level finalize and post_decode each release the branch KV.
    assert len(cleared) - reference_cleared == 2  # cond + uncond


def test_step_execution_matches_request_level_it2i_loop(monkeypatch):
    """The dual-CFG editing path reaches the same pixels through both paths."""
    pipeline, cleared = _make_pipeline(monkeypatch)
    sampling = _sampling(cfg_scale=2.5, img_cfg_scale=1.5)
    prompt = _t2i_prompt(multi_modal_data={"image": [Image.new("RGB", (512, 512))]})

    req = SimpleNamespace(prompts=[prompt], sampling_params=sampling)
    p = pipeline._parse_request(req)
    reference = pipeline._forward_it2i(p, [Image.new("RGB", (512, 512))])
    reference_cleared = len(cleared)
    step_output, state = _run_step_lifecycle(pipeline, sampling, prompt)

    reference_image = reference.output["payload"]["image"]
    torch.testing.assert_close(step_output.output["payload"]["image"], reference_image)
    assert len(cleared) - reference_cleared == 3  # cond + img_cond + uncond


def test_step_execution_matches_request_level_think_loop(monkeypatch):
    """Think requests keep their think text and latents identical."""
    pipeline, _ = _make_pipeline(monkeypatch)
    pipeline._generate_think = lambda outputs, past_kv, t_idx, max_think_tokens=1024: (
        past_kv,
        t_idx + 3,
        "thinking about the cat",
    )
    sampling = _sampling(cfg_scale=1.0, think=True)
    prompt = _t2i_prompt()

    reference = _run_request_level(pipeline, sampling, prompt)
    step_output, _ = _run_step_lifecycle(pipeline, sampling, prompt)

    expected_metadata = {"text": {"think_text": "thinking about the cat"}}
    assert reference.output["metadata"] == expected_metadata
    assert step_output.output["metadata"] == expected_metadata
    torch.testing.assert_close(
        step_output.output["payload"]["image"],
        reference.output["payload"]["image"],
    )


def test_prepare_encode_seeds_runner_visible_state(monkeypatch):
    from vllm_omni.diffusion.models.sensenova_u1 import pipeline_sensenova_u1 as mod
    from vllm_omni.diffusion.worker.utils import StepRequestState

    pipeline, _ = _make_pipeline(monkeypatch)
    sampling = _sampling(cfg_scale=4.0)
    state = StepRequestState(request_id="req-0", sampling=sampling, prompt=_t2i_prompt())
    pipeline.prepare_encode(state)

    num_steps = sampling.num_inference_steps
    token_h = sampling.height // (_PATCH * _MERGE)
    token_w = sampling.width // (_PATCH * _MERGE)
    # The runner slices the batched velocity by this row count.
    assert state.latents.shape == (1, token_h * token_w, 3 * (_PATCH * _MERGE) ** 2)
    assert state.total_steps == num_steps
    assert state.step_index == 0
    assert state.do_true_cfg is False

    timesteps = torch.linspace(0.0, 1.0, num_steps + 1)
    expected_t0 = 1.0 - (3.0 * (1.0 - timesteps[0]) / (1.0 + 2.0 * (1.0 - timesteps[0])))
    torch.testing.assert_close(state.current_timestep, expected_t0.reshape(()))

    for key in (mod._STEP_PARAMS, mod._STEP_NS, mod._STEP_CACHES, mod._STEP_THINK_TEXT, mod._STEP_IS_IT2I):
        assert key in state.extra


def test_prepare_encode_rejects_text_modality_requests(monkeypatch):
    from vllm_omni.diffusion.worker.utils import StepRequestState

    pipeline, _ = _make_pipeline(monkeypatch)
    state = StepRequestState(request_id="req-text", sampling=_sampling(), prompt=_t2i_prompt(modalities=["text"]))
    with pytest.raises(ValueError, match="request-level"):
        pipeline.prepare_encode(state)


def test_interrupt_cancels_pending_steps_at_boundary(monkeypatch):
    """denoise_step() reports cancellation with None; state freezes."""
    from vllm_omni.diffusion.worker.utils import StepRequestState

    pipeline, _ = _make_pipeline(monkeypatch)
    state = StepRequestState(request_id="req-0", sampling=_sampling(), prompt=_t2i_prompt())
    pipeline.prepare_encode(state)
    for _ in range(2):
        noise_pred = pipeline.denoise_step(SimpleNamespace(states=(state,)), states=[state])
        pipeline.step_scheduler(state, noise_pred)

    frozen_step = state.step_index
    frozen_latents = state.latents.clone()
    pipeline._interrupt = True

    assert pipeline.denoise_step(SimpleNamespace(states=(state,)), states=[state]) is None
    # Defensive no-op if a caller still invokes the scheduler hook.
    pipeline.step_scheduler(state, torch.zeros_like(state.latents))
    assert state.step_index == frozen_step
    torch.testing.assert_close(state.latents, frozen_latents)
    assert not state.denoise_completed

    # The next request resets the flag, exactly like forward() does.
    fresh = StepRequestState(request_id="req-1", sampling=_sampling(), prompt=_t2i_prompt())
    pipeline.prepare_encode(fresh)
    assert pipeline.interrupt is False
    assert pipeline.denoise_step(SimpleNamespace(states=(fresh,)), states=[fresh]) is not None


def test_batched_step_execution_matches_independent_requests(monkeypatch):
    """Co-batched requests land where they would have landed alone.

    Uses the real InputBatch gather/scatter and the runner's row slicing, with
    requests that differ in seed, cfg scale, step count, and latent rows.
    """
    from vllm_omni.diffusion.worker.input_batch import InputBatch
    from vllm_omni.diffusion.worker.utils import StepRequestState

    pipeline, _ = _make_pipeline(monkeypatch)
    specs = [
        ("req-0", _sampling(seed=11, cfg_scale=4.0), 6, 1),
        ("req-1", _sampling(seed=23, cfg_scale=1.0), 3, 2),  # batch_size=2 latent rows
    ]

    alone: dict[str, torch.Tensor] = {}
    for request_id, sampling, _, _ in specs:
        _, state = _run_step_lifecycle(pipeline, sampling, _t2i_prompt(), request_id=request_id)
        alone[request_id] = state.latents.clone()

    states = []
    for request_id, sampling, num_steps, batch_size in specs:
        sampling.num_inference_steps = num_steps
        sampling.extra_args["batch_size"] = batch_size
        states.append(StepRequestState(request_id=request_id, sampling=sampling, prompt=_t2i_prompt()))
        pipeline.prepare_encode(states[-1])

    active = list(states)
    cached_batch = None
    while active:
        input_batch = InputBatch.make_batch(active, cached_batch=cached_batch)
        cached_batch = input_batch
        noise_pred = pipeline.denoise_step(input_batch, states=active)
        offset = 0
        for state in active:
            rows = state.latents.shape[0]
            pipeline.step_scheduler(state, noise_pred[offset : offset + rows])
            offset += rows
        assert offset == noise_pred.shape[0]
        # Finished requests leave the batch, exactly like the runner drops them.
        active = [state for state in active if not state.denoise_completed]

    for state in states:
        torch.testing.assert_close(state.latents, alone[state.request_id])


def test_pre_process_routes_text_requests_to_request_level_path():
    from vllm_omni.diffusion.models.sensenova_u1 import get_sensenova_u1_pre_process_func

    step_on = get_sensenova_u1_pre_process_func(SimpleNamespace(step_execution=True))
    text_request = SimpleNamespace(prompt=_t2i_prompt(modalities=["text"]), use_step_execution=True)
    assert step_on(text_request).use_step_execution is False

    image_request = SimpleNamespace(prompt=_t2i_prompt(), use_step_execution=True)
    assert step_on(image_request).use_step_execution is True

    # Without step execution nothing is rerouted.
    step_off = get_sensenova_u1_pre_process_func(SimpleNamespace(step_execution=False))
    plain = SimpleNamespace(prompt=_t2i_prompt(modalities=["text"]), use_step_execution=True)
    assert step_off(plain).use_step_execution is True
