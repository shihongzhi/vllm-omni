# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.models.minimax_h3.encoder import MiniMaxH3Qwen3VLEncoder

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _make_stub_encoder() -> MiniMaxH3Qwen3VLEncoder:
    enc = MiniMaxH3Qwen3VLEncoder(
        model_path="",
        device=torch.device("cpu"),
        load_model=False,
    )
    # Registering a submodule makes ``is_loaded`` true and provides a CPU
    # parameter for the device guard inside ``encode_ids``.
    enc.text_model = nn.Linear(2, 2)
    return enc


def test_encode_ids_restores_cudnn_sdp_flag():
    enc = _make_stub_encoder()
    enc._encode = lambda *a, **k: torch.zeros(3, 4)
    original = torch.backends.cuda.cudnn_sdp_enabled()
    try:
        for initial in (False, True):
            torch.backends.cuda.enable_cudnn_sdp(initial)
            out = enc.encode_ids(torch.tensor([1, 2, 3]))
            assert out.shape == (3, 4)
            assert torch.backends.cuda.cudnn_sdp_enabled() is initial
    finally:
        torch.backends.cuda.enable_cudnn_sdp(original)


def test_encode_ids_restores_cudnn_sdp_flag_on_error():
    enc = _make_stub_encoder()

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    enc._encode = _boom
    original = torch.backends.cuda.cudnn_sdp_enabled()
    try:
        torch.backends.cuda.enable_cudnn_sdp(False)
        with pytest.raises(RuntimeError, match="boom"):
            enc.encode_ids(torch.tensor([1, 2, 3]))
        assert torch.backends.cuda.cudnn_sdp_enabled() is False
    finally:
        torch.backends.cuda.enable_cudnn_sdp(original)
