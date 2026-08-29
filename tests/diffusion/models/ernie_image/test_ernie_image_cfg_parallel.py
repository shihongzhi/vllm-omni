# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

import vllm_omni.diffusion.distributed.cfg_parallel as cfg_parallel_module
import vllm_omni.diffusion.cache.cachedit.model_specific as model_specific_module
from vllm_omni.diffusion.models.ernie_image.pipeline_ernie_image import ErnieImagePipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

LATENT_SHAPE = (1, 4, 8, 8)


class _StubTransformer:
    dtype = torch.float32

    def __init__(self):
        self.config = SimpleNamespace(in_channels=LATENT_SHAPE[1], text_in_dim=8)
        self.calls = []

    def __call__(self, *, hidden_states, timestep, text_bth, text_lens, return_dict):
        self.calls.append(
            {
                "batch": hidden_states.shape[0],
                "text_sum": text_bth.sum(dim=(1, 2)).view(-1, 1, 1, 1),
            }
        )
        # Prediction depends only on the text branch: cond -> 4.0, uncond -> 0.0.
        pred = torch.ones_like(hidden_states) * text_bth.sum(dim=(1, 2)).view(-1, 1, 1, 1)
        return (pred,)


class _StubScheduler:
    def __init__(self, num_steps=2):
        self.timesteps = [torch.tensor(1000 - i) for i in range(num_steps)]
        self.captured_preds = []

    def set_timesteps(self, sigmas, device=None):
        pass

    def step(self, pred, t, latents, return_dict=False):
        self.captured_preds.append(pred.clone())
        return (latents,)


def _make_pipe(is_distilled, transformer, scheduler):
    pipe = ErnieImagePipeline.__new__(ErnieImagePipeline)
    pipe._execution_device = torch.device("cpu")
    pipe.transformer = transformer
    pipe.scheduler = scheduler
    pipe.is_distilled = is_distilled
    pipe.vae_scale_factor = 16
    pipe.check_inputs = lambda **_kwargs: None
    # Cond prompt encodes with apply_pe=True, uncond with apply_pe=False.
    pipe.encode_prompt = lambda _prompts, _device, _n, width=None, height=None, apply_pe=True: [
        torch.ones((1, 4)) if apply_pe else torch.zeros((1, 4))
    ]
    return pipe


def _make_req(guidance_scale=4.0, num_inference_steps=2):
    return SimpleNamespace(
        prompts=["cat"],
        sampling_params=SimpleNamespace(
            height=None,
            width=None,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=None,
            num_outputs_per_prompt=1,
            extra_args={},
        ),
    )


def _run_forward(pipe):
    latents = torch.zeros(LATENT_SHAPE)
    return pipe.forward(
        _make_req(),
        prompt_embeds=None,
        negative_prompt_embeds=None,
        latents=latents,
        output_type="latent",
    )


def test_sequential_cfg_issues_two_single_batch_forwards_per_step():
    transformer, scheduler = _StubTransformer(), _StubScheduler()
    pipe = _make_pipe(is_distilled=False, transformer=transformer, scheduler=scheduler)

    _run_forward(pipe)

    assert len(transformer.calls) == 4  # 2 steps x (cond + uncond)
    assert all(call["batch"] == 1 for call in transformer.calls)
    # Branch order per step: cond (text_sum=4) then uncond (text_sum=0).
    assert [call["text_sum"].item() for call in transformer.calls] == pytest.approx([4.0, 0.0, 4.0, 0.0])
    # Plain CFG formula: uncond + scale * (cond - uncond) = 0 + 4 * 4 = 16.
    for pred in scheduler.captured_preds:
        assert torch.allclose(pred, torch.full(LATENT_SHAPE, 16.0))


def test_distilled_runs_single_branch():
    transformer, scheduler = _StubTransformer(), _StubScheduler()
    pipe = _make_pipe(is_distilled=True, transformer=transformer, scheduler=scheduler)

    _run_forward(pipe)

    assert len(transformer.calls) == 2  # 2 steps x cond only
    assert all(call["batch"] == 1 for call in transformer.calls)
    for pred in scheduler.captured_preds:
        assert torch.allclose(pred, torch.full(LATENT_SHAPE, 4.0))


def test_cfg_parallel_rank0_matches_sequential_result(monkeypatch):
    monkeypatch.setattr(cfg_parallel_module, "get_classifier_free_guidance_world_size", lambda: 2)
    monkeypatch.setattr(cfg_parallel_module, "get_classifier_free_guidance_rank", lambda: 0)

    class _FakeGroup:
        def all_gather(self, tensor, separate_tensors=True):
            # Rank 1 computes the uncond branch; for this stub both branches
            # are derived from text_bth, so the gathered pair mirrors rank 0.
            return [tensor, tensor * 0.0]

    monkeypatch.setattr(cfg_parallel_module, "get_cfg_group", lambda: _FakeGroup())

    transformer, scheduler = _StubTransformer(), _StubScheduler()
    pipe = _make_pipe(is_distilled=False, transformer=transformer, scheduler=scheduler)

    _run_forward(pipe)

    # Rank 0 only computes the positive branch: 1 forward per step.
    assert len(transformer.calls) == 2
    assert all(call["batch"] == 1 for call in transformer.calls)
    for pred in scheduler.captured_preds:
        assert torch.allclose(pred, torch.full(LATENT_SHAPE, 16.0))


def test_predict_noise_forwards_to_transformer():
    transformer, _ = _StubTransformer(), _StubScheduler()
    pipe = _make_pipe(is_distilled=False, transformer=transformer, scheduler=_StubScheduler())

    hidden = torch.zeros(LATENT_SHAPE)
    text = torch.ones((1, 4, 8))
    out = pipe.predict_noise(
        hidden_states=hidden,
        timestep=torch.zeros(1),
        text_bth=text,
        text_lens=torch.tensor([4]),
    )

    assert transformer.calls and transformer.calls[0]["batch"] == 1
    assert out.shape == LATENT_SHAPE


def test_ernie_cache_enabler_registered():
    model_specific_module.register_custom_dit_enablers()
    assert (
        model_specific_module.CUSTOM_DIT_ENABLERS.get("ErnieImagePipeline")
        is model_specific_module.enable_cache_for_ernie_image
    )


@pytest.mark.parametrize("is_distilled, expected_separate_cfg", [(False, True), (True, False)])
def test_enabler_has_separate_cfg_follows_checkpoint(is_distilled, expected_separate_cfg, monkeypatch):
    captured = {}
    transformer = SimpleNamespace(layers=["layer"])

    def fake_get_transformer(pipeline):
        return pipeline.transformer

    def fake_enable_cache_for_dit(pipeline, cache_config, block_adapter):
        captured["block_adapter"] = block_adapter

    monkeypatch.setattr(model_specific_module, "_default_get_pipeline_transformer", fake_get_transformer)
    monkeypatch.setattr(model_specific_module, "enable_cache_for_dit", fake_enable_cache_for_dit)

    class _FakeBlockAdapter:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(model_specific_module, "BlockAdapter", _FakeBlockAdapter)

    pipeline = SimpleNamespace(is_distilled=is_distilled, transformer=transformer)
    model_specific_module.enable_cache_for_ernie_image(pipeline, cache_config=None)

    assert captured["has_separate_cfg"] is expected_separate_cfg
    assert captured["blocks"] == [transformer.layers]
