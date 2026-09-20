import os, sys, logging, torch, numpy as np
logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib, channel_concat_denoise; importlib.reload(channel_concat_denoise)
from channel_concat_denoise import denoise_channel_concat
from ablation_common import *
OUT = os.path.join(RESULTS, "eval_fix_check"); os.makedirs(OUT, exist_ok=True)
p = load_pairs(n=1, indices=[2])[0]
te = TextEncoder(); pe, tids = te(PROMPT); neg, _ = te(""); te.free()
vae, bn_mean, bn_std = load_vae()
cond_lat = encode_img(vae, p["src"]).float()
tr = load_transformer(); sched = load_scheduler()
def run(name, **kw):
    dec = denoise_channel_decoded = denoise_channel_concat(transformer=tr, vae=vae, scheduler=sched, cond_latents=cond_lat, prompt_embeds=pe, text_ids=tids,
        latents_bn_mean=bn_mean, latents_bn_std=bn_std, num_inference_steps=28, generator=torch.Generator("cpu").manual_seed(0), device=DEV, dtype=DT, **kw)
    img = to_pil(dec[0]).resize((1024, 680)); img.save(os.path.join(OUT, name + ".jpg"), quality=92); return np.asarray(img).astype(np.float32)
a = run("fixed_unguided", guidance_scale=1.0)
b = run("fixed_txt4_img1.5", guidance_scale=4.0, image_guidance_scale=1.5, negative_prompt_embeds=neg)
c = run("fixed_txt4_img1", guidance_scale=4.0, image_guidance_scale=1.0, negative_prompt_embeds=neg)
d = run("fixed_txt4_no_negative_warns", guidance_scale=4.0, image_guidance_scale=1.5)  # must warn and fall back
from PIL import Image
ref_n = np.asarray(Image.open(os.path.join(RESULTS, "t1_swap/pair2_normal.jpg"))).astype(np.float32)
ref_c = np.asarray(Image.open(os.path.join(RESULTS, "t1_swap/pair2_cfg_txt4_img1.5.jpg"))).astype(np.float32)
ref_t = np.asarray(Image.open(os.path.join(RESULTS, "t1_swap/pair2_cfg_txt4.jpg"))).astype(np.float32)
mad = lambda x, y: float(np.abs(x - y).mean())
print(f"fixed unguided       vs sampling.py normal            : mean |diff| = {mad(a, ref_n):.2f} /255")
print(f"fixed txt4 img1.5    vs sampling.py cfg_txt4_img1.5   : mean |diff| = {mad(b, ref_c):.2f} /255")
print(f"fixed txt4 img1.0    vs sampling.py cfg_txt4          : mean |diff| = {mad(c, ref_t):.2f} /255")
print(f"fixed no-negative    vs fixed unguided (fallback)     : mean |diff| = {mad(d, a):.2f} /255")
print(f"guided vs unguided (should differ)                    : mean |diff| = {mad(b, a):.2f} /255")
