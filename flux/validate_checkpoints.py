import argparse, glob, json, os, re, sys, time

P = argparse.ArgumentParser()
P.add_argument("--base", default="/workspace/models/FLUX.2-klein-base-4B")
P.add_argument("--runs_dir", default="/workspace/runs/klein-base-4b-overfit20")
P.add_argument("--dataset", default="/workspace/datasets/exteriors-v5-overfit20")
P.add_argument("--val_split_ratio", type=float, default=0.2)
P.add_argument("--val_split_seed", type=int, default=42)
P.add_argument("--max_samples", type=int, default=None, help="cap on held-out images per checkpoint")
P.add_argument("--height", type=int, default=1360)
P.add_argument("--width", type=int, default=2048)
P.add_argument("--steps", type=int, default=28)
P.add_argument("--guidance_scale", type=float, default=4.0)
P.add_argument("--image_guidance_scale", type=float, default=1.5)
P.add_argument("--max_sequence_length", type=int, default=512)
P.add_argument("--out", default="/workspace/runs/ckpt_validation")
P.add_argument("--wandb_project", default="dreambooth-flux2-image2img-lora")
P.add_argument("--run_name", default="ckpt-validation")
P.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
P.add_argument("--threads", type=int, default=None, help="CPU threads for torch (cpu device only)")
P.add_argument("--watch", action="store_true", help="keep polling runs_dir for new checkpoints")
P.add_argument("--poll_seconds", type=int, default=300)
P.add_argument("--settle_seconds", type=int, default=120, help="checkpoint file must be unchanged this long before use")
P.add_argument("--min_free_gb", type=float, default=80.0, help="cgroup memory headroom required before loading a checkpoint")
a = P.parse_args()

if a.device == "cpu":
    # must happen before torch import; never let this job see the GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if a.threads:
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[k] = str(a.threads)

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.transforms import functional as TF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from channel_concat_denoise import denoise_channel_concat
from ckpt_table import build_and_log as log_results_table
from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline, Flux2Transformer2DModel, FlowMatchEulerDiscreteScheduler

if a.device == "cpu" and a.threads:
    torch.set_num_threads(a.threads)
dev, dt = a.device, torch.bfloat16
os.makedirs(a.out, exist_ok=True)
done_path = os.path.join(a.out, "done.json")
done = json.load(open(done_path)) if os.path.exists(done_path) else {}
print(f"device={dev} dtype={dt} threads={torch.get_num_threads()} already done: {sorted(int(k) for k in done if k.isdigit())}", flush=True)


def free_cuda():
    if dev == "cuda":
        torch.cuda.empty_cache()


def cgroup_free_gb():
    try:
        cur = int(open("/sys/fs/cgroup/memory.current").read())
        mx = open("/sys/fs/cgroup/memory.max").read().strip()
        return (int(mx) - cur) / 2**30 if mx != "max" else float("inf")
    except OSError:
        return float("inf")


def wait_for_memory():
    while (free := cgroup_free_gb()) < a.min_free_gb:
        print(f"  only {free:.0f} GB free in cgroup, waiting for {a.min_free_gb:.0f} GB", flush=True)
        time.sleep(60)


def ckpt_step(path):
    return int(re.search(r"checkpoint-(\d+)", path).group(1))


def ready_checkpoints():
    out = []
    for cdir in glob.glob(os.path.join(a.runs_dir, "checkpoint-*", "transformer")):
        w = os.path.join(cdir, "diffusion_pytorch_model.safetensors")
        if not (os.path.exists(w) and os.path.exists(os.path.join(cdir, "config.json"))):
            continue
        if time.time() - os.path.getmtime(w) < a.settle_seconds:
            continue
        if str(ckpt_step(cdir)) not in done:
            out.append(cdir)
    return sorted(out, key=ckpt_step)


# same held-out split the training run used
ds = load_from_disk(a.dataset)
if isinstance(ds, dict):
    ds = ds["train"]
val = ds.train_test_split(test_size=a.val_split_ratio, seed=a.val_split_seed)["test"]
if a.max_samples:
    val = val.select(range(min(a.max_samples, len(val))))
print(f"val images: {len(val)}", flush=True)


def prep(img, H, W):
    img = img.convert("RGB")
    w, h = img.size
    s = max(W / w, H / h)
    img = img.resize((round(w * s), round(h * s)), Image.BICUBIC)
    left, top = (img.width - W) // 2, (img.height - H) // 2
    img = img.crop((left, top, left + W, top + H))
    return TF.to_tensor(img) * 2 - 1  # [-1,1], matches training


import wandb

run = wandb.init(
    project=a.wandb_project,
    name=a.run_name,
    id=re.sub(r"[^A-Za-z0-9_-]", "-", a.run_name),
    resume="allow",
    notes="Post-hoc validation of every surviving checkpoint on the held-out split",
    config={k: v for k, v in vars(a).items()},
)
# log against the checkpoint step so restarts never violate wandb's monotonic-step rule
run.define_metric("ckpt_step")
run.define_metric("*", step_metric="ckpt_step")
t0 = time.time()
pipe = Flux2KleinPipeline.from_pretrained(a.base, transformer=None, vae=None, torch_dtype=dt).to(dev)
samples, originals = [], []
with torch.no_grad():
    neg_pe, _ = pipe.encode_prompt(
        prompt=[""], device=dev, num_images_per_prompt=1, max_sequence_length=a.max_sequence_length
    )
    neg_pe = neg_pe.cpu()
    for r in val:
        pe, tids = pipe.encode_prompt(
            prompt=[r["caption"]], device=dev, num_images_per_prompt=1, max_sequence_length=a.max_sequence_length
        )
        if not done.get("_originals_logged"):
            originals.append(wandb.Image(r["merged_img"].convert("RGB"), caption=f"sample {len(samples)} merged (4K)"))
            originals.append(wandb.Image(r["edited_img"].convert("RGB"), caption=f"sample {len(samples)} edited (4K)"))
        samples.append(
            {
                "caption": r["caption"],
                "src": prep(r["merged_img"], a.height, a.width),
                "tgt": prep(r["edited_img"], a.height, a.width),
                "pe": pe.cpu(),
                "tids": tids.cpu(),
            }
        )
del pipe
free_cuda()
if originals:
    run.log({"eval/originals_4k": originals, "ckpt_step": 0})
    done["_originals_logged"] = True
    json.dump(done, open(done_path, "w"), indent=1)
    del originals
print(f"prompts encoded in {time.time() - t0:.0f}s", flush=True)

vae = AutoencoderKLFlux2.from_pretrained(a.base, subfolder="vae", torch_dtype=dt).to(dev)
bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(dev)
bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(dev)
for s in samples:
    with torch.no_grad():
        s["cond_lat"] = vae.encode(s["src"][None].to(dev, dt)).latent_dist.mode().cpu()
print(f"cond latents ready at {time.time() - t0:.0f}s", flush=True)

sched = FlowMatchEulerDiscreteScheduler.from_pretrained(a.base, subfolder="scheduler")


def evaluate(cdir):
    step = ckpt_step(cdir)
    print(f"\n=== checkpoint-{step}", flush=True)
    wait_for_memory()
    t_load = time.time()
    tr = Flux2Transformer2DModel.from_pretrained(cdir, torch_dtype=dt).to(dev)
    tr.eval()
    print(f"  loaded in {time.time() - t_load:.0f}s", flush=True)
    psnr = PeakSignalNoiseRatio(data_range=1.0, dim=(1, 2, 3), reduction="elementwise_mean").to(dev)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(dev)
    mses, strips, preds_full = [], [], []
    for i, s in enumerate(samples):
        t_s = time.time()
        dec = denoise_channel_concat(
            transformer=tr, vae=vae, scheduler=sched, cond_latents=s["cond_lat"].to(dev, torch.float32),
            prompt_embeds=s["pe"], text_ids=s["tids"], latents_bn_mean=bn_mean, latents_bn_std=bn_std,
            num_inference_steps=a.steps, guidance_scale=a.guidance_scale, image_guidance_scale=a.image_guidance_scale,
            negative_prompt_embeds=neg_pe,
            generator=torch.Generator("cpu").manual_seed(i), device=dev, dtype=dt,
        )
        pred = (dec * 0.5 + 0.5).clamp(0, 1).float()
        tgt = (s["tgt"][None].to(dev) * 0.5 + 0.5).clamp(0, 1).float()
        if pred.shape[-2:] != tgt.shape[-2:]:
            pred = F.interpolate(pred, size=tgt.shape[-2:], mode="bilinear", align_corners=False)
        mses.append(F.mse_loss(pred, tgt).item())
        psnr.update(pred, tgt)
        lpips.update(pred, tgt)
        trip = [TF.to_pil_image(x[0].cpu()) for x in ((s["src"][None] * 0.5 + 0.5).clamp(0, 1), tgt, pred)]
        full = os.path.join(a.out, "full")
        os.makedirs(full, exist_ok=True)
        trip[2].save(os.path.join(full, f"ckpt{step}_sample{i}_pred.png"))
        preds_full.append(wandb.Image(trip[2], caption=f"ckpt-{step} sample {i} pred (2K) mse={mses[-1]:.4f}"))
        for name, im in (("src", trip[0]), ("tgt", trip[1])):
            fp = os.path.join(full, f"sample{i}_{name}.png")
            if not os.path.exists(fp):
                im.save(fp)
        W = 560
        H = int(W * trip[0].height / trip[0].width)
        strip = Image.new("RGB", (W * 3, H), "white")
        for k, im in enumerate(trip):
            strip.paste(im.resize((W, H)), (k * W, 0))
        strip.save(os.path.join(a.out, f"ckpt{step}_sample{i}.png"))
        strips.append(wandb.Image(strip, caption=f"ckpt-{step} sample {i} mse={mses[-1]:.4f}"))
        print(f"  sample {i}: mse={mses[-1]:.4f} ({time.time() - t_s:.0f}s)", flush=True)
    m = {"val_mse": sum(mses) / len(mses), "val_psnr": psnr.compute().item(), "val_lpips": lpips.compute().item()}
    xw = tr.x_embedder.weight
    half = xw.shape[1] // 2
    m["cond_img_ratio"] = (xw[:, half:].abs().mean() / xw[:, :half].abs().mean()).item()
    print(f"  => step {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in m.items()), flush=True)
    run.log({**m, "ckpt_step": step, "eval/compare": strips, "eval/pred_2k": preds_full})
    done[str(step)] = m
    json.dump(done, open(done_path, "w"), indent=1)
    n_rows = log_results_table(
        run, a.out, [s["caption"] for s in samples], done, guidance_scale=a.guidance_scale, num_steps=a.steps
    )
    print(f"  results table: {n_rows} rows", flush=True)
    del tr, psnr, lpips
    free_cuda()
    return m


while True:
    ckpts = ready_checkpoints()
    if ckpts:
        print("checkpoints:", [ckpt_step(c) for c in ckpts], flush=True)
    for cdir in ckpts:
        if not os.path.exists(cdir):
            continue  # rotated away by checkpoints_total_limit before we got to it
        try:
            evaluate(cdir)
        except Exception as e:
            # checkpoint rotated away mid-eval or half-written; skip it, keep watching
            print(f"  !! checkpoint-{ckpt_step(cdir)} failed: {type(e).__name__}: {e}", flush=True)
            free_cuda()
    if not a.watch:
        break
    time.sleep(a.poll_seconds)

print("\n==== SUMMARY ====")
for step in sorted((k for k in done if k.isdigit()), key=int):
    print(f"checkpoint-{step}: " + ", ".join(f"{k}={v:.4f}" for k, v in done[step].items()))
print("wandb:", run.url)
run.finish()
