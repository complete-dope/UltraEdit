import argparse, glob, os, re, sys
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.transforms import functional as TF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from channel_concat_denoise import denoise_channel_concat
from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline, Flux2Transformer2DModel, FlowMatchEulerDiscreteScheduler

P = argparse.ArgumentParser()
P.add_argument("--base", default="/workspace/models/FLUX.2-klein-base-4B")
P.add_argument("--runs_dir", default="/workspace/runs/klein-base-4b-overfit20")
P.add_argument("--dataset", default="/workspace/datasets/exteriors-v5-overfit20")
P.add_argument("--val_split_ratio", type=float, default=0.2)
P.add_argument("--val_split_seed", type=int, default=42)
P.add_argument("--height", type=int, default=1360)
P.add_argument("--width", type=int, default=2048)
P.add_argument("--steps", type=int, default=28)
P.add_argument("--guidance_scale", type=float, default=4.0)
P.add_argument("--image_guidance_scale", type=float, default=1.5)
P.add_argument("--out", default="/workspace/runs/ckpt_validation")
P.add_argument("--wandb_project", default="dreambooth-flux2-image2img-lora")
a = P.parse_args()

dev, dt = "cuda", torch.bfloat16
os.makedirs(a.out, exist_ok=True)

# same held-out split the training run used
ds = load_from_disk(a.dataset)
val = ds.train_test_split(test_size=a.val_split_ratio, seed=a.val_split_seed)["test"]
print(f"val images: {len(val)}")


def prep(img, H, W):
    img = img.convert("RGB")
    w, h = img.size
    s = max(W / w, H / h)
    img = img.resize((round(w * s), round(h * s)), Image.BICUBIC)
    left, top = (img.width - W) // 2, (img.height - H) // 2
    img = img.crop((left, top, left + W, top + H))
    return TF.to_tensor(img) * 2 - 1  # [-1,1], matches training


pipe = Flux2KleinPipeline.from_pretrained(a.base, transformer=None, vae=None, torch_dtype=dt).to(dev)
neg_pe, _ = pipe.encode_prompt(prompt=[""], device=dev, num_images_per_prompt=1)
neg_pe = neg_pe.cpu()
samples = []
for r in val:
    pe, tids = pipe.encode_prompt(prompt=[r["caption"]], device=dev, num_images_per_prompt=1)
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
torch.cuda.empty_cache()

vae = AutoencoderKLFlux2.from_pretrained(a.base, subfolder="vae", torch_dtype=dt).to(dev)
bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(dev)
bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(dev)
for s in samples:
    with torch.no_grad():
        s["cond_lat"] = vae.encode(s["src"][None].to(dev, dt)).latent_dist.mode().cpu()

ckpts = sorted(
    glob.glob(os.path.join(a.runs_dir, "checkpoint-*", "transformer")),
    key=lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1)),
)
print("checkpoints:", [re.search(r"checkpoint-(\d+)", c).group(1) for c in ckpts])

import wandb

run = wandb.init(project=a.wandb_project, name="ckpt-validation", reinit=True,
                 notes="Post-hoc validation of every surviving checkpoint on the held-out split")
sched = FlowMatchEulerDiscreteScheduler.from_pretrained(a.base, subfolder="scheduler")
results = []

for cdir in ckpts:
    step = int(re.search(r"checkpoint-(\d+)", cdir).group(1))
    print(f"\n=== checkpoint-{step}")
    tr = Flux2Transformer2DModel.from_pretrained(cdir, torch_dtype=dt).to(dev)
    tr.eval()
    psnr = PeakSignalNoiseRatio(data_range=1.0, dim=(1, 2, 3), reduction="elementwise_mean").to(dev)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(dev)
    mses, strips = [], []
    for i, s in enumerate(samples):
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
        W = 560
        H = int(W * trip[0].height / trip[0].width)
        strip = Image.new("RGB", (W * 3, H), "white")
        for k, im in enumerate(trip):
            strip.paste(im.resize((W, H)), (k * W, 0))
        strip.save(os.path.join(a.out, f"ckpt{step}_sample{i}.png"))
        strips.append(wandb.Image(strip, caption=f"ckpt-{step} sample {i} mse={mses[-1]:.4f}"))
        print(f"  sample {i}: mse={mses[-1]:.4f}")
    m = {"val_mse": sum(mses) / len(mses), "val_psnr": psnr.compute().item(), "val_lpips": lpips.compute().item()}
    xw = tr.x_embedder.weight
    half = xw.shape[1] // 2
    m["cond_img_ratio"] = (xw[:, half:].abs().mean() / xw[:, :half].abs().mean()).item()
    print(f"  => step {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in m.items()))
    run.log({**m, "eval/compare": strips}, step=step)
    results.append((step, m))
    del tr, psnr, lpips
    torch.cuda.empty_cache()

print("\n==== SUMMARY ====")
for step, m in results:
    print(f"checkpoint-{step}: " + ", ".join(f"{k}={v:.4f}" for k, v in m.items()))
print("wandb:", run.url)
run.finish()
