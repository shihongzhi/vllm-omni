#!/usr/bin/env python
"""Fibo-2.0 Dev Diffusers 基线：固定输入生成 + 对照数据捕获。

对照 BriaFibo2Pipeline.__call__ 的 T2I 路径手工复刻一遍（Phase A），
再用官方 pipe.__call__ 以相同的 prompt_embeds/latents 重跑（Phase B），
verify 模式下作为独立进程再跑一遍 capture 并逐项比对（Phase C），
证明「同一份初始 latents 重复生成一致结果」。

用法:
  python run_baseline.py --mode capture   # 首次运行，保存全部对照数据到 <out>/run1
  python run_baseline.py --mode verify    # 独立进程重跑，与 <out>/run1 逐项比对
"""

# ruff: noqa
# ^ Verbatim snapshot of the baseline generator script; it runs in the external
#   diffusers environment, not under this repo's lint policy.

import argparse
import hashlib
import json
import os
import subprocess
import time

import numpy as np
import torch
from diffusers import BriaFibo2Pipeline
from diffusers.pipelines.bria_fibo2.pipeline_bria_fibo2 import (
    DEFAULT_NEGATIVE_PROMPT,
    calculate_shift,
    retrieve_timesteps,
)
from PIL import Image
from safetensors.torch import load_file, save_file

# 固定输入：diffusers 文档示例 caption（bria_fibo2 pipeline docstring），无 text_render
CAPTION = {
    "short_description": "A red bicycle leaning against a white brick wall on a sunny morning.",
    "objects": [{"description": "a red city bicycle with a wicker basket", "location": "center"}],
    "background_setting": "a white brick wall with a green door",
    "lighting": {"conditions": "bright morning sun", "direction": "from the right"},
    "style_medium": "photograph",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sync():
    torch.cuda.synchronize()


def load_pipeline(model_id):
    try:
        pipe = BriaFibo2Pipeline.from_pretrained(model_id, dtype=torch.bfloat16)
    except TypeError:  # 兼容旧的 torch_dtype 参数名
        pipe = BriaFibo2Pipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def build_prompt():
    raw = json.dumps(CAPTION, separators=(",", ":"), ensure_ascii=False)
    caption, _ = BriaFibo2Pipeline._parse_prompt(raw)
    prompt = caption if isinstance(caption, str) else json.dumps(caption, separators=(",", ":"), ensure_ascii=False)
    return raw, prompt


def vae_decode(pipe, latents):
    """复刻 __call__ 第 7 步：反 BatchNorm 归一化后 VAE 解码。"""
    lat = pipe._patchify_latents(latents)
    bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(lat.device, lat.dtype)
    bn_scale = (1 / torch.sqrt(pipe.vae.bn.running_var + pipe.vae.config.batch_norm_eps)).view(1, -1, 1, 1)
    lat = lat / bn_scale.to(lat.device, lat.dtype) + bn_mean
    lat = pipe._unpatchify_latents(lat)
    image = pipe.vae.decode(lat.to(pipe.vae.dtype), return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


def capture_run(pipe, args):
    """Phase A：逐行复刻 __call__ 的 T2I 去噪路径，收集全部中间量。"""
    device = pipe._execution_device
    stats = {
        "steps": args.steps,
        "guidance": args.guidance,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
    }

    # 3. 文本编码（正/负）。负 prompt 用 Fibo 训练用的固定 null caption
    _, prompt = build_prompt()
    t0 = time.perf_counter()
    prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(prompt, None, device, 1)
    sync()
    stats["encode_cond_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    neg_embeds, neg_embeds_mask = pipe.encode_prompt(DEFAULT_NEGATIVE_PROMPT, None, device, 1)
    sync()
    stats["encode_neg_s"] = time.perf_counter() - t0

    # 4. 初始 noise latents（fp32，cuda 生成器 seed=42）
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    latents = pipe.prepare_latents(1, pipe.transformer.config.in_channels, args.height, args.width, device, gen)
    initial_latents = latents.clone()

    # 5. timesteps：sigmas 1 -> 1/steps，按 1024 分辨率 image_seq_len 做 shift
    patch_size = pipe.transformer.config.patch_size
    image_seq_len = (latents.shape[2] // patch_size) * (latents.shape[3] // patch_size)
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.base_image_seq_len,
        pipe.scheduler.config.max_image_seq_len,
        pipe.scheduler.config.base_shift,
        pipe.scheduler.config.max_shift,
    )
    sigmas = np.linspace(1.0, 1 / args.steps, args.steps)
    timesteps, num_steps = retrieve_timesteps(pipe.scheduler, args.steps, device, sigmas=sigmas, mu=mu)
    stats["sigmas"] = sigmas.tolist()
    stats["timesteps"] = timesteps.tolist()
    stats["mu"] = float(mu)
    stats["image_seq_len"] = int(image_seq_len)

    # 6. 去噪循环：cond / uncond 两次独立前向 + CFG 合并 + scheduler.step
    # 额外采样中间/尾步（0 为既有 first_step；15/29 供 vllm parity 复查整段去噪区间）
    capture_at = sorted(set([0] + [s for s in (args.steps // 2, args.steps - 1) if 0 < s < args.steps]))
    captured_steps: dict[int, dict] = {}
    first = {}
    trajectory = {}
    torch.cuda.reset_peak_memory_stats()
    step_times = []
    for i, t in enumerate(timesteps):
        t0 = time.perf_counter()
        timestep = t.expand(latents.shape[0]) / 1000
        latent_model_input = latents.to(pipe.transformer.dtype)
        noise_pred = pipe.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_embeds_mask,
            context_latents=None,
            return_dict=False,
        )[0].float()
        noise_pred_uncond = pipe.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=neg_embeds,
            encoder_attention_mask=neg_embeds_mask,
            context_latents=None,
            return_dict=False,
        )[0].float()
        if i in capture_at:
            captured_steps[i] = {
                "latent_model_input": latent_model_input.clone(),
                "timestep": timestep.clone(),
                "noise_pred_cond": noise_pred.clone(),
                "noise_pred_uncond": noise_pred_uncond.clone(),
            }
        guided = noise_pred_uncond + args.guidance * (noise_pred - noise_pred_uncond)
        if i == 0:
            first = {
                "noise_pred_cond": noise_pred.clone(),
                "noise_pred_uncond": noise_pred_uncond.clone(),
                "noise_pred_guided": guided.clone(),
                "latent_model_input": latent_model_input.clone(),
            }
        latents = pipe.scheduler.step(guided, t, latents, return_dict=False)[0]
        sync()
        step_times.append(time.perf_counter() - t0)
        trajectory[f"latents_step_{i:03d}"] = latents.clone()
    stats["per_step_s"] = step_times
    stats["loop_s"] = float(sum(step_times))
    stats["peak_mem_loop_gb"] = torch.cuda.max_memory_allocated() / 2**30
    final_latents = latents.clone()

    # 7. VAE 解码
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    image = vae_decode(pipe, final_latents)
    sync()
    stats["vae_decode_s"] = time.perf_counter() - t0
    stats["peak_mem_decode_gb"] = torch.cuda.max_memory_allocated() / 2**30

    return {
        "prompt": prompt,
        "prompt_embeds": prompt_embeds,
        "prompt_embeds_mask": prompt_embeds_mask,
        "neg_embeds": neg_embeds,
        "neg_embeds_mask": neg_embeds_mask,
        "initial_latents": initial_latents,
        "first": first,
        "captured_steps": captured_steps,
        "trajectory": trajectory,
        "final_latents": latents.clone(),
        "image": image,
        "stats": stats,
    }


def official_run(pipe, cap, args):
    """Phase B：官方 __call__ 以 capture 得到的 embeds/latents 重跑，返回最终 latents。"""
    out = pipe(
        prompt_embeds=cap["prompt_embeds"].clone(),
        prompt_embeds_mask=cap["prompt_embeds_mask"].clone() if cap["prompt_embeds_mask"] is not None else None,
        negative_prompt_embeds=cap["neg_embeds"].clone(),
        negative_prompt_embeds_mask=cap["neg_embeds_mask"].clone() if cap["neg_embeds_mask"] is not None else None,
        latents=cap["initial_latents"].clone(),
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        output_type="latent",
        return_dict=False,
    )
    return out[0]


def save_artifacts(cap, official_latents, out_dir, pipe, args, load_s):
    tensors_dir = os.path.join(out_dir, "tensors")
    os.makedirs(tensors_dir, exist_ok=True)
    save_file(
        {"initial_latents": cap["initial_latents"].contiguous()},
        os.path.join(tensors_dir, "initial_latents.safetensors"),
    )
    em = {"prompt_embeds": cap["prompt_embeds"].contiguous()}
    if cap["prompt_embeds_mask"] is not None:
        em["prompt_embeds_mask"] = cap["prompt_embeds_mask"].contiguous()
    save_file(em, os.path.join(tensors_dir, "prompt_embeds.safetensors"))
    nm = {"negative_prompt_embeds": cap["neg_embeds"].contiguous()}
    if cap["neg_embeds_mask"] is not None:
        nm["negative_prompt_embeds_mask"] = cap["neg_embeds_mask"].contiguous()
    save_file(nm, os.path.join(tensors_dir, "negative_prompt_embeds.safetensors"))
    save_file({k: v.contiguous() for k, v in cap["first"].items()}, os.path.join(tensors_dir, "first_step.safetensors"))
    sel = {}
    for i, d in cap["captured_steps"].items():
        for kk, v in d.items():
            sel[f"step_{i:03d}_{kk}"] = v.contiguous()
    save_file(sel, os.path.join(tensors_dir, "selected_steps.safetensors"))
    save_file(
        {k: v.contiguous() for k, v in cap["trajectory"].items()}, os.path.join(tensors_dir, "trajectory.safetensors")
    )
    save_file(
        {"final_latents": cap["final_latents"].contiguous(), "official_final_latents": official_latents.contiguous()},
        os.path.join(tensors_dir, "final_latents.safetensors"),
    )
    cap["image"].save(os.path.join(out_dir, "image.png"))

    with open(os.path.join(out_dir, "prompt.txt"), "w") as f:
        f.write(cap["prompt"] + "\n")
    with open(os.path.join(out_dir, "caption_raw.json"), "w") as f:
        json.dump(CAPTION, f, indent=2, ensure_ascii=False)

    same = torch.equal(cap["final_latents"], official_latents)
    max_diff = (cap["final_latents"] - official_latents).abs().max().item()
    # 权重快照 sha（HF 下载后的 snapshot 目录名）
    model_sha = "unknown"
    snap_root = os.path.join(os.environ.get("HF_HOME", ""), "hub", "models--briaai--Fibo-2.0-dev", "snapshots")
    if os.path.isdir(snap_root):
        shas = os.listdir(snap_root)
        model_sha = shas[0] if shas else "unknown"
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except Exception:
        driver = "n/a"
    import accelerate
    import diffusers
    import PIL
    import transformers

    meta = {
        "model_id": args.model,
        "model_sha": model_sha,
        "run_config": {
            "mode": args.mode,
            "steps": args.steps,
            "guidance": args.guidance,
            "seed": args.seed,
            "height": args.height,
            "width": args.width,
            "dtype": "bfloat16",
            "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
            "caption_raw": CAPTION,
            "prompt_cleaned": cap["prompt"],
        },
        "versions": {
            "python": subprocess.run(["python", "-V"], capture_output=True, text=True).stdout.strip(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "diffusers": diffusers.__version__,
            "transformers": transformers.__version__,
            "accelerate": accelerate.__version__,
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "gpu_driver": driver,
            "gpu": torch.cuda.get_device_name(0),
        },
        "tensor_shapes": {
            "prompt_embeds": list(cap["prompt_embeds"].shape),
            "prompt_embeds_mask": None if cap["prompt_embeds_mask"] is None else list(cap["prompt_embeds_mask"].shape),
            "initial_latents": list(cap["initial_latents"].shape),
            "first_step_noise_pred_cond": list(cap["first"]["noise_pred_cond"].shape),
            "final_latents": list(cap["final_latents"].shape),
        },
        "timing_s": cap["stats"],
        "peak_mem": {
            "after_load_gb": cap["stats"].get("peak_after_load_gb"),
            "loop_gb": cap["stats"]["peak_mem_loop_gb"],
            "decode_gb": cap["stats"]["peak_mem_decode_gb"],
            "load_s": load_s,
        },
        "manual_vs_official": {"final_latents_bitwise_equal": bool(same), "max_abs_diff": max_diff},
        "artifacts_sha256": {},
    }
    for fn in sorted(os.listdir(tensors_dir)):
        meta["artifacts_sha256"][fn] = sha256_file(os.path.join(tensors_dir, fn))
    meta["artifacts_sha256"]["image.png"] = sha256_file(os.path.join(out_dir, "image.png"))
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def compare(a, b, name, report):
    a, b = a.cpu(), b.cpu()  # safetensors 读回在 CPU，统一设备后再比对
    if a.shape != b.shape or a.dtype != b.dtype:
        report[name] = {"pass": False, "reason": f"shape/dtype mismatch {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}"}
        return
    equal = torch.equal(a, b)
    max_diff = (a.float() - b.float()).abs().max().item()
    report[name] = {
        "pass": bool(equal),
        "bitwise_equal": bool(equal),
        "max_abs_diff": max_diff,
        "shape": list(a.shape),
        "dtype": str(a.dtype),
    }


def verify(args):
    """Phase C：独立进程重跑 capture，与 run1 逐项比对。"""
    ref_dir = os.path.join(args.out, "run1")
    t0 = time.perf_counter()
    pipe = load_pipeline(args.model)
    load_s = time.perf_counter() - t0
    with torch.no_grad():  # 与官方 @torch.no_grad() 行为一致
        cap = capture_run(pipe, args)
    report = {}
    tdir = os.path.join(ref_dir, "tensors")
    compare(
        cap["initial_latents"],
        load_file(os.path.join(tdir, "initial_latents.safetensors"))["initial_latents"],
        "initial_latents",
        report,
    )
    em = load_file(os.path.join(tdir, "prompt_embeds.safetensors"))
    compare(cap["prompt_embeds"], em["prompt_embeds"], "prompt_embeds", report)
    if "prompt_embeds_mask" in em:
        compare(cap["prompt_embeds_mask"], em["prompt_embeds_mask"], "prompt_embeds_mask", report)
    else:
        report["prompt_embeds_mask"] = {"pass": cap["prompt_embeds_mask"] is None, "note": "both None expected"}
    fs = load_file(os.path.join(tdir, "first_step.safetensors"))
    for k in ("noise_pred_cond", "noise_pred_uncond", "noise_pred_guided"):
        compare(cap["first"][k], fs[k], f"first_step.{k}", report)
    if "selected_steps.safetensors" in os.listdir(tdir):
        sel_ref = load_file(os.path.join(tdir, "selected_steps.safetensors"))
        sel_now = {f"step_{i:03d}_{kk}": v for i, d in cap["captured_steps"].items() for kk, v in d.items()}
        sel_ok = all(torch.equal(sel_now[k].cpu(), sel_ref[k].cpu()) for k in sel_ref)
        sel_max = max((sel_now[k].cpu().float() - sel_ref[k].float()).abs().max().item() for k in sel_ref)
        report["selected_steps_bitwise"] = {
            "pass": bool(sel_ok),
            "max_abs_diff": sel_max,
            "keys": sorted(sel_ref.keys()),
        }
    traj = load_file(os.path.join(tdir, "trajectory.safetensors"))
    step_ok = all(torch.equal(cap["trajectory"][k].cpu(), traj[k].cpu()) for k in traj)
    max_step_diff = max((cap["trajectory"][k].cpu().float() - traj[k].float()).abs().max().item() for k in traj)
    report["trajectory_all_30_steps"] = {
        "pass": bool(step_ok),
        "bitwise_equal": bool(step_ok),
        "max_abs_diff": max_step_diff,
    }
    compare(
        cap["final_latents"],
        load_file(os.path.join(tdir, "final_latents.safetensors"))["final_latents"],
        "final_latents",
        report,
    )
    ref = Image.open(os.path.join(ref_dir, "image.png"))
    img_equal = bool(np.array_equal(np.asarray(cap["image"]), np.asarray(ref)))
    report["image_pixels"] = {"pass": img_equal}
    all_pass = all(v.get("pass", False) for v in report.values())
    out = {
        "verified_against": ref_dir,
        "all_pass": all_pass,
        "load_s": load_s,
        "timing_s": cap["stats"],
        "comparisons": report,
    }
    vdir = os.path.join(args.out, "verify")
    os.makedirs(vdir, exist_ok=True)
    with open(os.path.join(vdir, "report.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v.get("pass") for k, v in report.items()}, indent=2))
    print(f"ALL_PASS={all_pass}")
    return 0 if all_pass else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="briaai/Fibo-2.0-dev")
    p.add_argument("--out", default="/root/autodl-fs/fibo-baseline/ref-data")
    p.add_argument("--mode", choices=["capture", "verify"], default="capture")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    args = p.parse_args()

    if args.mode == "verify":
        raise SystemExit(verify(args))

    t0 = time.perf_counter()
    pipe = load_pipeline(args.model)
    load_s = time.perf_counter() - t0
    # 官方 __call__ 带 @torch.no_grad()；手工复刻必须同样关闭 autograd，
    # 否则 44 层中间激活 + attention 矩阵全部保留（~54GB），首个 forward 即 OOM
    with torch.no_grad():
        cap = capture_run(pipe, args)
    cap["stats"]["peak_after_load_gb"] = torch.cuda.max_memory_allocated() / 2**30
    with torch.no_grad():
        official = official_run(pipe, cap, args)
    out_dir = os.path.join(args.out, "run1")
    meta = save_artifacts(cap, official, out_dir, pipe, args, load_s)
    print(
        json.dumps(
            {k: meta[k] for k in ("tensor_shapes", "timing_s", "peak_mem", "manual_vs_official")},
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
