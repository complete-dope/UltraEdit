# Tells how much is each file affecting here

# Test 1c: full sampling. normal / swapped cond / null cond / identity cond / prompt variants / CFG variants.
import os, torch, json, itertools
import numpy as np
from PIL import Image, ImageDraw
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ablation_common import *
from cfg_sampling import sample, unnorm_unpatch

PAIR_IDX = [int(x) for x in os.environ.get("PAIR_IDX", "").split(",") if x]
OUT = os.path.join(RESULTS, os.environ.get("OUT_NAME", "t1_swap")); os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "log.txt"); open(LOG, "w").close()
STEPS = 28

PROMPTS = {
    "default": PROMPT,
    "null": "",
    "caption": None,  # per pair
    "sunset": "dramatic orange sunset sky with clouds",
    "bluesky": "clear deep blue sky, no clouds",
    "overcast": "overcast grey cloudy sky",
    "night": "night time, dark sky with stars, lights on in the house",
    "snow": "snow covering the ground and roof, winter",
    "instruct": "edit the exterior",
}

pairs = load_pairs(n=len(PAIR_IDX) or 3, indices=PAIR_IDX or None)
te = TextEncoder()
emb = {k: te(v) for k, v in PROMPTS.items() if v is not None}
emb.update({f"caption{p['idx']}": te(p["caption"]) for p in pairs})
for p in pairs:
    log(f"pair {p['idx']} caption: {p['caption']!r} meta={p['meta']}", LOG)
te.free()
vae, bn_mean, bn_std = load_vae()
lat = {p["idx"]: dict(src=norm_pack(encode_img(vae, p["src"]), bn_mean, bn_std), tgt=norm_pack(encode_img(vae, p["tgt"]), bn_mean, bn_std)) for p in pairs}
pix = {p["idx"]: dict(src=prep(p["src"]), tgt=prep(p["tgt"])) for p in pairs}
for k, v in pix.items():
    to_pil(v["src"]).resize((1024, 680)).save(os.path.join(OUT, f"pair{k}_src.jpg"), quality=92)
    to_pil(v["tgt"]).resize((1024, 680)).save(os.path.join(OUT, f"pair{k}_tgt.jpg"), quality=92)
tr = load_transformer(); sched = load_scheduler()

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(DEV)


def metrics(img, ref):
    a = ((img.float().clamp(-1, 1) + 1) / 2).unsqueeze(0).to(DEV); b = ((ref.float().clamp(-1, 1) + 1) / 2).unsqueeze(0).to(DEV)
    mse = ((a - b) ** 2).mean().item()
    with torch.no_grad():
        l = lpips(torch.nn.functional.interpolate(a, size=(680, 1024), mode="area"), torch.nn.functional.interpolate(b, size=(680, 1024), mode="area")).item()
    return dict(lpips=l, psnr=10 * np.log10(1 / max(mse, 1e-10)), mse=mse)


def sky_rgb(img):
    return ((img[:, : img.shape[1] // 4].float().mean(dim=(1, 2)) + 1) / 2 * 255).round().tolist()


results = []


def run(name, pair, cond_key, cond_pair=None, prompt="default", s_txt=1.0, s_img=1.0, seed=0, drop_cond=False):
    cond_pair = pair if cond_pair is None else cond_pair
    pe, tids = emb[prompt if prompt != "caption" else f"caption{pair}"]
    t0 = time.time()
    out = sample(tr, sched, lat[cond_pair][cond_key], pe, tids, steps=STEPS, seed=seed, pe_null=emb["null"][0], s_txt=s_txt, s_img=s_img, drop_cond=drop_cond)
    img = decode_lat(vae, unnorm_unpatch(out, bn_mean, bn_std))
    fn = f"pair{pair}_{name}.jpg"
    to_pil(img).resize((1024, 680)).save(os.path.join(OUT, fn), quality=92)
    r = dict(name=name, pair=pair, file=fn, cond=f"{cond_pair}.{cond_key}" if not drop_cond else "zeros", prompt=prompt, s_txt=s_txt, s_img=s_img, seed=seed, secs=round(time.time() - t0, 1))
    for ref_pair in sorted(pix):
        for k in ["src", "tgt"]:
            m = metrics(img, pix[ref_pair][k])
            r[f"lpips_vs_{ref_pair}.{k}"] = round(m["lpips"], 4); r[f"psnr_vs_{ref_pair}.{k}"] = round(m["psnr"], 2)
    r["sky_rgb"] = sky_rgb(img); r["sky_rgb_src"] = sky_rgb(pix[pair]["src"]); r["sky_rgb_tgt"] = sky_rgb(pix[pair]["tgt"])
    results.append(r)
    log(json.dumps(r), LOG)
    save_json(results, os.path.join(OUT, "results.json"))
    return img


ids_ = [p["idx"] for p in pairs]
A = ids_[0]
for pair, other in list(zip(ids_, ids_[1:] + ids_[:1])):
    run("normal", pair, "src")
    if os.environ.get("FULL", "1") == "1": run("normal_seed1", pair, "src", seed=1)
    run("swap_cond_from_other", pair, "src", cond_pair=other)
    run("cond_zeros", pair, "src", drop_cond=True)
    run("cond_is_target", pair, "tgt")
    run("prompt_null", pair, "src", prompt="null")
    run("prompt_caption", pair, "src", prompt="caption")
    run("prompt_instruct", pair, "src", prompt="instruct")
    run("cfg_txt4", pair, "src", s_txt=4.0, s_img=1.0)
    run("cfg_txt4_img1.5", pair, "src", s_txt=4.0, s_img=1.5)
    if os.environ.get("FULL", "1") == "1": run("cfg_txt7.5_img1.5", pair, "src", s_txt=7.5, s_img=1.5)
    for pr in (["sunset", "bluesky", "overcast", "night", "snow"] if os.environ.get("FULL", "1") == "1" else ["sunset", "overcast", "night"]):
        run(f"prompt_{pr}", pair, "src", prompt=pr)
        run(f"prompt_{pr}_cfg4", pair, "src", prompt=pr, s_txt=4.0)
    # unconditional both: what does the model draw with no image and no text?
    run("uncond_all", pair, "src", prompt="null", drop_cond=True)

# contact sheets
names = [r["name"] for r in results if r["pair"] == A]
for pair in ids_:
    rs = {r["name"]: r for r in results if r["pair"] == pair}
    cols = 5; rows = int(np.ceil((len(names) + 2) / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.5))
    ax = ax.flatten()
    ax[0].imshow(Image.open(os.path.join(OUT, f"pair{pair}_src.jpg"))); ax[0].set_title("INPUT (cond)")
    ax[1].imshow(Image.open(os.path.join(OUT, f"pair{pair}_tgt.jpg"))); ax[1].set_title("TARGET (edited)")
    for i, n in enumerate(names):
        r = rs[n]; ax[i + 2].imshow(Image.open(os.path.join(OUT, r["file"])))
        ax[i + 2].set_title(f"{n}\nlpips src={r[f'lpips_vs_{pair}.src']:.3f} tgt={r[f'lpips_vs_{pair}.tgt']:.3f}", fontsize=9)
    for a in ax: a.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, f"sheet_pair{pair}.jpg"), dpi=70); plt.close()
log("DONE t1_swap", LOG)
