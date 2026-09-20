import os, sys, json, io, time
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import functional as TF

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the flux/ dir, for channel_concat_denoise
from channel_concat_denoise import denoise_channel_concat  # noqa: F401
from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline, Flux2Transformer2DModel, FlowMatchEulerDiscreteScheduler

BASE = os.environ.get("ABL_BASE", "/workspace/models/FLUX.2-klein-base-4B")
CKPT = os.environ.get("ABL_CKPT", "/workspace/checkpoints/checkpoint-17000/transformer")
DATA = os.environ.get("ABL_DATA", "/workspace/flux-ablation/data/exteriors-v5/data/test-00000-of-00001.parquet")
RESULTS = os.environ.get("ABL_RESULTS", "/workspace/flux-ablation/results")
PROMPT = "good sky_color, good greenary, good construction, good clouds"
W, H = 2048, 1360
DEV = "cuda"
DT = torch.bfloat16


def log(msg, file=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if file:
        with open(file, "a") as f:
            f.write(line + "\n")


def prep(img, height=H, width=W):
    img = img.convert("RGB")
    w, h = img.size
    s = max(width / w, height / h)
    img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
    left, top = (img.width - width) // 2, (img.height - height) // 2
    img = img.crop((left, top, left + width, top + height))
    return TF.to_tensor(img) * 2 - 1


def to_pil(t):
    t = (t.detach().float().cpu().clamp(-1, 1) + 1) / 2
    return Image.fromarray((t.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))


def load_pairs(n=4, indices=None):
    import pyarrow.parquet as pq
    tbl = pq.read_table(DATA).to_pandas()
    rows = []
    idx = indices if indices is not None else list(range(len(tbl)))
    for i in idx:
        r = tbl.iloc[i]
        if r["is_vertical"]:
            continue
        rows.append(
            dict(
                idx=int(i),
                id=r["id"],
                caption=r["caption"],
                src=Image.open(io.BytesIO(r["merged_img"]["bytes"])),
                tgt=Image.open(io.BytesIO(r["edited_img"]["bytes"])),
                meta={k: str(r[k]) for k in ["shot_type", "construction", "greenary", "sky_replacement", "sky_color", "cloud_type"]},
            )
        )
        if len(rows) >= n:
            break
    return rows


def load_vae():
    vae = AutoencoderKLFlux2.from_pretrained(BASE, subfolder="vae", torch_dtype=DT).to(DEV).eval()
    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(DEV)
    bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(DEV)
    return vae, bn_mean, bn_std


@torch.no_grad()
def encode_img(vae, pil):
    x = prep(pil).unsqueeze(0).to(DEV, DT)
    return vae.encode(x).latent_dist.mode()  # (1, 32, H/8, W/8) raw


@torch.no_grad()
def decode_lat(vae, lat):
    return vae.decode(lat.to(vae.dtype)).sample[0]


def load_transformer(path=CKPT, dtype=DT):
    return Flux2Transformer2DModel.from_pretrained(path, torch_dtype=dtype).to(DEV).eval()


def load_scheduler():
    return FlowMatchEulerDiscreteScheduler.from_pretrained(BASE, subfolder="scheduler")


class TextEncoder:
    def __init__(self):
        self.pipe = Flux2KleinPipeline.from_pretrained(BASE, transformer=None, vae=None, torch_dtype=DT).to(DEV)

    @torch.no_grad()
    def __call__(self, prompt, max_len=512):
        pe, tids = self.pipe.encode_prompt(prompt=[prompt], device=DEV, num_images_per_prompt=1, max_sequence_length=max_len)
        return pe, tids

    def token_positions(self, prompt, max_len=512):
        tok = self.pipe.tokenizer
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = tok(text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len)
        ids = enc["input_ids"][0].tolist()
        toks = tok.convert_ids_to_tokens(ids)
        return toks, enc["attention_mask"][0].tolist()

    def free(self):
        self.pipe.to("cpu")
        del self.pipe
        torch.cuda.empty_cache()


def norm_pack(lat, bn_mean, bn_std):
    p = Flux2KleinPipeline._patchify_latents(lat)
    p = (p - bn_mean) / bn_std
    return p  # (1, 128, h, w) normalized, patchified


def pack(p):
    return Flux2KleinPipeline._pack_latents(p)


def latent_ids(p):
    return Flux2KleinPipeline._prepare_latent_ids(p).to(DEV)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
