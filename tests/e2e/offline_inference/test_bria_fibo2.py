# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""End-to-end text-to-image test for Bria Fibo-2.0 Dev through the full engine.

Needs the gated checkpoint in the local HF cache (``HF_HOME`` pointing at the
baseline cache works), so it runs locally rather than in CI. When the parity
baseline reference data is also available (``FIBO2_REF_DATA``), the decoded
image is additionally compared against the baseline capture.
"""

import os

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OfflineOmniClient
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = os.environ.get("FIBO2_MODEL", "briaai/Fibo-2.0-dev")
TURBO_MODEL = os.environ.get("FIBO2_TURBO_MODEL", "briaai/Fibo-2.0-edit-turbo")
REF_DATA = os.environ.get("FIBO2_REF_DATA", "/root/autodl-fs/fibo-baseline/ref-data/run1")

CAPTION = (
    '{"short_description":"A red bicycle leaning against a white brick wall on a sunny morning.",'
    '"objects":[{"description":"a red city bicycle with a wicker basket","location":"center"}],'
    '"background_setting":"a white brick wall with a green door",'
    '"lighting":{"conditions":"bright morning sun","direction":"from the right"},'
    '"style_medium":"photograph"}'
)


@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards={"cuda": 1})
@pytest.mark.parametrize("omni_runner", [(MODEL, None)], indirect=True)
def test_bria_fibo2_t2i(offline_client: OfflineOmniClient):
    sampling = OmniDiffusionSamplingParams(
        height=1024,
        width=1024,
        num_inference_steps=30,
        guidance_scale=5.0,
        seed=42,
    )
    request_config = {
        "model": MODEL,
        "prompt": CAPTION,
        "sampling_params": sampling,
    }
    # OfflineOmniClient.send_diffusion_request returns a single response
    resp = offline_client.send_diffusion_request(request_config)
    assert resp.success, f"request failed: {getattr(resp, 'error_message', None)}"
    assert resp.images, f"no images in response: text={resp.text_content}"
    img = resp.images[0]

    baseline = os.path.join(REF_DATA, "image.png")
    if not os.path.exists(baseline):
        pytest.skip("baseline reference image not available; e2e smoke only")

    import math

    import numpy as np
    from PIL import Image

    got = np.asarray(img, dtype=np.float32)
    assert got.shape == (1024, 1024, 3), f"unexpected output shape {got.shape}"
    ref = np.asarray(Image.open(baseline), dtype=np.float32)
    af, bf = got.flatten(), ref.flatten()
    cos = float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-8))
    mse = float(np.mean((af / 255 - bf / 255) ** 2))
    psnr = 10 * math.log10(1.0 / mse) if mse > 0 else 99.0
    print(f"e2e vs baseline: cosine={cos:.4f} psnr={psnr:.2f}dB")
    # 30 denoising steps of bf16 kernel noise vs the diffusers baseline land
    # around cosine 0.998 / 29 dB; a broken component collapses far below.
    assert cos >= 0.99, f"e2e image diverged from baseline: cosine={cos:.4f}"


@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards={"cuda": 1})
@pytest.mark.parametrize("omni_runner", [(MODEL, None)], indirect=True)
def test_bria_fibo2_edit(offline_client: OfflineOmniClient):
    """Edit path: instruction + image to edit (the baseline T2I capture),
    compared against the diffusers edit baseline (FIBO2_EDIT_REF)."""
    import json as _json

    edit_ref = os.environ.get("FIBO2_EDIT_REF", "/root/autodl-fs/fibo-baseline/ref-data/edit1")
    source = os.path.join(os.environ.get("FIBO2_REF_DATA", "/root/autodl-fs/fibo-baseline/ref-data/run1"), "image.png")
    edited = os.path.join(edit_ref, "edited_image.png")
    meta_path = os.path.join(edit_ref, "meta.json")
    if not (os.path.exists(edited) and os.path.exists(meta_path)):
        pytest.skip("edit baseline reference not available; e2e smoke only")

    meta = _json.load(open(meta_path))
    instruction = meta["stats"]["instruction"]

    sampling = OmniDiffusionSamplingParams(
        num_inference_steps=30,
        guidance_scale=5.0,
        seed=42,
    )
    request_config = {
        "model": MODEL,
        "prompt": instruction,
        "images": [source],
        "sampling_params": sampling,
    }
    resp = offline_client.send_diffusion_request(request_config)
    assert resp.success, f"edit request failed: {getattr(resp, 'error_message', None)}"
    assert resp.images, f"no images in edit response: text={resp.text_content}"
    img = resp.images[0]

    import math

    import numpy as np
    from PIL import Image

    got = np.asarray(img, dtype=np.float32)
    assert got.ndim == 3 and got.shape[2] == 3, f"bad image shape {got.shape}"
    assert got.shape[0] % 16 == 0 and got.shape[1] % 16 == 0, f"size not multiple of 16: {got.shape}"
    exp_h, exp_w = meta["stats"]["height"], meta["stats"]["width"]
    assert got.shape == (exp_h, exp_w, 3), f"unexpected edit output shape {got.shape}"
    ref = np.asarray(Image.open(edited), dtype=np.float32)
    af, bf = got.flatten(), ref.flatten()
    cos = float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-8))
    mse = float(np.mean((af / 255 - bf / 255) ** 2))
    psnr = 10 * math.log10(1.0 / mse) if mse > 0 else 99.0
    print(f"edit e2e vs baseline: cosine={cos:.4f} psnr={psnr:.2f}dB")
    # The edit path transcribes most tokens from the context image, so it
    # tracks the baseline much tighter than generation (pipeline-level smoke
    # lands at cosine 0.99995 / 49 dB).
    assert cos >= 0.995, f"e2e edit diverged from baseline: cosine={cos:.4f}"


@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards={"cuda": 1})
@pytest.mark.parametrize("omni_runner", [(TURBO_MODEL, None)], indirect=True)
def test_bria_fibo2_edit_turbo(offline_client: OfflineOmniClient):
    """Edit Turbo: 4 steps, guidance 1.0 (no CFG, single forward per step),
    compared against the diffusers edit-turbo baseline (FIBO2_TURBO_REF)."""
    import json as _json

    turbo_ref = os.environ.get("FIBO2_TURBO_REF", "/root/autodl-fs/fibo-baseline/ref-data/edit-turbo1")
    source = os.path.join(os.environ.get("FIBO2_REF_DATA", "/root/autodl-fs/fibo-baseline/ref-data/run1"), "image.png")
    edited = os.path.join(turbo_ref, "edited_image.png")
    meta_path = os.path.join(turbo_ref, "meta.json")
    if not (os.path.exists(edited) and os.path.exists(meta_path)):
        pytest.skip("edit-turbo baseline reference not available; e2e smoke only")

    meta = _json.load(open(meta_path))
    instruction = meta["stats"]["instruction"]

    sampling = OmniDiffusionSamplingParams(
        num_inference_steps=4,
        guidance_scale=1.0,
        seed=42,
    )
    request_config = {
        "model": TURBO_MODEL,
        "prompt": instruction,
        "images": [source],
        "sampling_params": sampling,
    }
    resp = offline_client.send_diffusion_request(request_config)
    assert resp.success, f"turbo edit request failed: {getattr(resp, 'error_message', None)}"
    assert resp.images, f"no images in turbo response: text={resp.text_content}"
    img = resp.images[0]

    import math

    import numpy as np
    from PIL import Image

    got = np.asarray(img, dtype=np.float32)
    assert got.ndim == 3 and got.shape[2] == 3, f"bad image shape {got.shape}"
    assert got.shape[0] % 16 == 0 and got.shape[1] % 16 == 0, f"size not multiple of 16: {got.shape}"
    exp_h, exp_w = meta["stats"]["height"], meta["stats"]["width"]
    assert got.shape == (exp_h, exp_w, 3), f"unexpected turbo output shape {got.shape}"
    ref = np.asarray(Image.open(edited), dtype=np.float32)
    af, bf = got.flatten(), ref.flatten()
    cos = float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-8))
    mse = float(np.mean((af / 255 - bf / 255) ** 2))
    psnr = 10 * math.log10(1.0 / mse) if mse > 0 else 99.0
    print(f"turbo edit e2e vs baseline: cosine={cos:.4f} psnr={psnr:.2f}dB")
    assert cos >= 0.995, f"e2e turbo edit diverged from baseline: cosine={cos:.4f}"
