"""Dump reference tensors from the HF ArkttsModel for vLLM-port validation.

MUST run in the *baseline* environment (transformers>=4.57,<5 -- transformers
5.x produces all-zero codes for this checkpoint):

    /path/to/audio8-baseline/bin/python dump_reference_tensors.py \
        --output-dir /home/featurize/work/audio8_baseline

For each variant (no-reference, voice-clone) it re-runs the exact golden
greedy generation while hooking ``_prepare_prompt`` / ``_embed`` /
``_slow_step`` / ``_sample_semantic`` / ``_generate_codebooks``, and saves
``ref_tensors_<variant>.pt`` containing:

  prompt_ids / prompt_mask / position_ids   -- prefill driver
  prefill_embeds                            -- _embed output at prefill
  step_* lists                              -- per-iteration decode driver
      (input_ids / cache_position / position_ids / attention_mask /
       logits / slow_hidden / embeds)
  semantic_ids / ras_previous / codebooks   -- per-iteration decisions
  meta                                      -- text, params, config snapshot

The script asserts the hooked run still reproduces the golden ``.npy`` codes,
proving the hooks did not perturb the computation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

MODEL_ID = "AutoArk-AI/Audio8-TTS-Preview-0.6b"
GOLDEN_TEXT = "Hello, this is a golden baseline test for the Audio8 text to speech model integration."
CLONE_TEXT = "Voice cloning baseline with a reference audio sample."
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.8
TOP_P = 0.95
TOP_K = 50
SEED = 42


def install_hooks(model) -> dict:
    state: dict[str, list] = {
        "prepare": [],
        "embeds": [],
        "slow_inputs": [],
        "slow_outputs": [],
        "semantics": [],
        "ras_previous": [],
        "codebooks": [],
        "fast_inputs": [],
    }

    def wrap(name: str, capture) -> None:
        original = getattr(model, name)

        def hooked(*args, **kwargs):
            out = original(*args, **kwargs)
            capture(args, kwargs, out)
            return out

        setattr(model, name, hooked)

    def on_prepare(args, kwargs, out):
        state["prepare"].append({"prompt": out[0].cpu(), "prompt_mask": out[1].cpu()})

    def on_embed(args, kwargs, out):
        state["embeds"].append(out.detach().to("cpu", torch.bfloat16))

    def on_slow_step(args, kwargs, out):
        state["slow_inputs"].append(
            {
                "input_ids": args[0].cpu(),
                "cache_position": args[1].cpu(),
                "position_ids": args[2].cpu(),
                "attention_mask": args[3].cpu(),
            }
        )
        state["slow_outputs"].append(
            {
                "logits": out[0].detach().to("cpu", torch.bfloat16),
                "slow_hidden": out[1].detach().to("cpu", torch.bfloat16),
            }
        )

    def on_sample_semantic(args, kwargs, out):
        # positional: (history, logits, processors, top_k, top_p,
        #              temperature, previous, do_sample, generator)
        previous = args[6]
        state["semantics"].append(int(out.reshape(-1)[0]))
        state["ras_previous"].append(None if previous is None else previous.cpu())

    def on_generate_codebooks(args, kwargs, out):
        state["fast_inputs"].append(
            {
                "slow_hidden": args[0].detach().to("cpu", torch.bfloat16),
                "semantic": args[1].cpu(),
            }
        )
        state["codebooks"].append(out.cpu())

    wrap("_prepare_prompt", on_prepare)
    wrap("_embed", on_embed)
    wrap("_slow_step", on_slow_step)
    wrap("_sample_semantic", on_sample_semantic)
    wrap("_generate_codebooks", on_generate_codebooks)
    return state


def run_variant(model, processor, device, *, text, reference_audio, reference_text, meta):
    generator = torch.Generator(device=device).manual_seed(SEED)
    state = install_hooks(model)

    processor_kwargs: dict = {"text": [text], "return_tensors": "pt"}
    if reference_audio is not None:
        processor_kwargs.update(reference_audio=[reference_audio], reference_text=[reference_text])
    inputs = processor(**processor_kwargs)
    inputs = {name: value.to(device) for name, value in inputs.items()}
    output = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        do_sample=False,  # greedy, matches the golden runs
        generator=generator,
        return_dict_in_generate=True,
    )

    assert bool(output.finished[0]), "generation did not finish with EOS"
    cfg = model.config
    eos = int(cfg.eos_token_id)
    semantic_ids = state["semantics"]
    valid_frames = sum(1 for s in semantic_ids if s != eos)
    assert valid_frames == int(output.code_lengths[0])

    prompt = state["prepare"][0]["prompt"]
    prompt_mask = state["prepare"][0]["prompt_mask"]
    position_ids = prompt_mask.long().cumsum(-1).sub(1).clamp_min(0)

    dump = {
        "meta": meta,
        "prompt_ids": prompt,
        "prompt_mask": prompt_mask,
        "position_ids": position_ids,
        "prefill_embeds": state["embeds"][0][0],  # [T, H]
        "prefill_logits": state["slow_outputs"][0]["logits"],  # [1, vocab]
        "prefill_slow_hidden": state["slow_outputs"][0]["slow_hidden"],  # [1, 1, H]
        "step_embeds": [e[0, 0] for e in state["embeds"][1:]],  # list of [H]
        "step_inputs": state["slow_inputs"][1:],
        "step_logits": [o["logits"] for o in state["slow_outputs"][1:]],
        "step_slow_hidden": [o["slow_hidden"] for o in state["slow_outputs"][1:]],
        "semantic_ids": semantic_ids,
        "ras_previous": state["ras_previous"],
        "codebooks": state["codebooks"],  # list of [1, 10] per iteration
        "fast_inputs": state["fast_inputs"],
        "config_snapshot": {
            k: getattr(cfg, k)
            for k in (
                "dim",
                "n_layer",
                "n_head",
                "n_local_heads",
                "head_dim",
                "rope_base",
                "max_seq_len",
                "norm_eps",
                "intermediate_size",
                "vocab_size",
                "codebook_size",
                "num_codebooks",
                "semantic_begin_id",
                "semantic_end_id",
                "eos_token_id",
                "pad_token_id",
                "n_fast_layer",
                "fast_dim",
                "fast_head_dim",
                "fast_n_head",
                "fast_n_local_heads",
                "fast_intermediate_size",
                "norm_fastlayer_input",
                "ras_window_size",
                "ras_temperature",
                "ras_top_p",
                "codec_sample_rate",
            )
        },
    }

    # Hooks must not perturb the run: reproduce the golden codes exactly.
    stacked = torch.cat(dump["codebooks"], dim=0)[:valid_frames]  # [T, 10]
    if meta["golden_npy"] is not None:
        golden = np.load(meta["golden_npy"])
        assert stacked.numpy().T.shape == golden.shape, (
            f"frame count mismatch: hooked={stacked.shape[0]}, golden={golden.shape[1]}"
        )
        assert np.array_equal(stacked.numpy().T, golden), "hooked run diverged from golden codes"
    print(
        f"[dump] {meta['variant']}: prompt_width={prompt.shape[-1]}, "
        f"iterations={len(semantic_ids)}, frames={valid_frames}, "
        f"golden_match={'yes' if meta['golden_npy'] else 'n/a'}"
    )
    return dump


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("/home/featurize/work/audio8_baseline"))
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args()

    from transformers import AutoModel, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=dtype).eval().to(device)

    out_dir: Path = args.output_dir
    no_ref = run_variant(
        model,
        processor,
        device,
        text=GOLDEN_TEXT,
        reference_audio=None,
        reference_text=None,
        meta={"variant": "no_ref", "text": GOLDEN_TEXT, "golden_npy": str(out_dir / "golden_no_ref.npy")},
    )
    torch.save(no_ref, out_dir / "ref_tensors_no_ref.pt")

    clone = run_variant(
        model,
        processor,
        device,
        text=CLONE_TEXT,
        reference_audio=str(out_dir / "golden_no_ref.wav"),
        reference_text=GOLDEN_TEXT,
        meta={"variant": "clone", "text": CLONE_TEXT, "golden_npy": str(out_dir / "golden_clone.npy")},
    )
    torch.save(clone, out_dir / "ref_tensors_clone.pt")
    print(f"[dump] saved to {out_dir}/ref_tensors_no_ref.pt and ref_tensors_clone.pt")


if __name__ == "__main__":
    main()
