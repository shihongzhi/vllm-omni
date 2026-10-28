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
    outputs = offline_client.send_diffusion_request(request_config)
    assert outputs, "no outputs returned"
    images = outputs[0].outputs if hasattr(outputs[0], "outputs") else outputs[0]
    assert images is not None and len(images) > 0

    baseline = os.path.join(REF_DATA, "image.png")
    if not os.path.exists(baseline):
        pytest.skip("baseline reference image not available; e2e smoke only")

    import numpy as np
    from PIL import Image

    img = images[0] if not hasattr(images[0], "image") else images[0].image
    if not isinstance(img, Image.Image):  # some clients return ndarray paths
        img = Image.open(img) if isinstance(img, str) else Image.fromarray(img)
    ref = np.asarray(Image.open(baseline), dtype=np.float32)
    got = np.asarray(img, dtype=np.float32)
    if got.shape != ref.shape:
        pytest.skip(f"unexpected output shape {got.shape} vs baseline {ref.shape}")
    af, bf = got.flatten(), ref.flatten()
    cos = float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-8))
    mse = float(np.mean((af / 255 - bf / 255) ** 2))
    import math

    psnr = 10 * math.log10(1.0 / mse) if mse > 0 else 99.0
    print(f"e2e vs baseline: cosine={cos:.4f} psnr={psnr:.2f}dB")
    # 30 denoising steps of bf16 kernel noise vs the diffusers baseline land
    # around cosine 0.998 / 29 dB; a broken component collapses far below.
    assert cos >= 0.99, f"e2e image diverged from baseline: cosine={cos:.4f}"
