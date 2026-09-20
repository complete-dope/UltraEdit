# Compares the latent of the edited and source and then we measure model error seperately on each kind of area with and wo the input image, 

# Does the cond help where src==tgt (structure) and hurt where src!=tgt (the edit)? Per-region flow-matching loss.
import os
import json
import torch
import numpy as np
import matplotliba
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ablation_common import *

OUT = os.path.join(RESULTS, "t1_region_loss")
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "log.txt")
open(LOG, "w").close()

SIGMAS = [0.95, 0.8, 0.7, 0.5, 0.3, 0.1]
NSEED = 3
CHANGED_QUANTILE = 0.7
UNCHANGED_QUANTILE = 0.3
VARIANTS = ["cond_src", "cond_zero", "cond_tgt", "pred_is_src_velocity"]
REGIONS = ["all", "changed", "unchanged"]

torch.manual_seed(0)
pairs = load_pairs(n=4, indices=list(range(24)))

te = TextEncoder()
prompt_embeds, txt_ids = te(PROMPT)
te.free()

vae, bn_mean, bn_std = load_vae()
lat = [dict(src=norm_pack(encode_img(vae, p["src"]), bn_mean, bn_std),
            tgt=norm_pack(encode_img(vae, p["tgt"]), bn_mean, bn_std)) for p in pairs]
del vae
torch.cuda.empty_cache()

tr = load_transformer()
img_ids = latent_ids(lat[0]["tgt"])
h, w = lat[0]["tgt"].shape[-2:]


def split_regions(tgt, src):
    diff = (tgt - src).norm(dim=-1)[0]
    changed = diff > diff.quantile(CHANGED_QUANTILE)
    unchanged = diff <= diff.quantile(UNCHANGED_QUANTILE)
    return diff, changed, unchanged


def make_noise(shape, seed):
    return torch.randn(shape, generator=torch.Generator("cpu").manual_seed(seed)).to(DEV)


@torch.no_grad()
def predict(tgt, cond, sigma, noise):
    noisy = (1 - sigma) * tgt + sigma * noise
    inp = torch.cat([noisy, cond], -1).to(DT)
    return tr(hidden_states=inp, timestep=torch.tensor([sigma], device=DEV, dtype=DT), guidance=None,
              encoder_hidden_states=prompt_embeds.to(DEV, DT), txt_ids=txt_ids.to(DEV), img_ids=img_ids,
              return_dict=False)[0].float()


def predict_variants(tgt, src, sigma, noise):
    return {
        "cond_src": predict(tgt, src, sigma, noise),
        "cond_zero": predict(tgt, torch.zeros_like(src), sigma, noise),
        "cond_tgt": predict(tgt, tgt, sigma, noise),
        "pred_is_src_velocity": noise - src,  # what a "copy the source" model would output
    }


def token_errors(pred, target, changed, unchanged):
    err = ((pred - target) ** 2).mean(-1)[0]
    return dict(all=err.mean().item(), changed=err[changed].mean().item(), unchanged=err[unchanged].mean().item())


rows = []
for pair, l in zip(pairs, lat):
    tgt = pack(l["tgt"]).float()
    src = pack(l["src"]).float()
    diff, changed, unchanged = split_regions(tgt, src)
    print("Shape of diff is : ", diff.shape) # diff is a tensor here 
    np.save(os.path.join(OUT, f"diff_{pair['idx']}.npy"), diff.view(h, w).cpu().numpy())

    for sigma in SIGMAS:
        acc = {v: {r: [] for r in REGIONS} for v in VARIANTS}
        first_seed_preds = None
        for seed in range(NSEED):
            noise = make_noise(tgt.shape, seed)
            target = noise - tgt
            preds = predict_variants(tgt, src, sigma, noise)
            if seed == 0:
                first_seed_preds = preds
            for name, p in preds.items():
                for region, value in token_errors(p, target, changed, unchanged).items():
                    acc[name][region].append(value)

        # velocity target is noise - tgt, so copying src means adding (tgt - src) to it
        cond_effect = first_seed_preds["cond_src"] - first_seed_preds["cond_zero"]
        toward_src = torch.nn.functional.cosine_similarity(cond_effect.flatten(), (tgt - src).flatten(), dim=0).item()

        row = dict(pair=pair["idx"], sigma=sigma)
        for name, regions in acc.items():
            for region, values in regions.items():
                row[f"{name}_{region}"] = float(np.mean(values))
        row["cos_conddelta_vs_copysrc"] = toward_src
        rows.append(row)
        log(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), LOG)

save_json(rows, os.path.join(OUT, "results.json"))

fig, ax = plt.subplots(len(pairs), 3, figsize=(16, 3.6 * len(pairs)))
for i, pair in enumerate(pairs):
    pair_rows = [r for r in rows if r["pair"] == pair["idx"]]
    for j, region in enumerate(["all", "unchanged", "changed"]):
        for name, marker in zip(VARIANTS, ["o-", "x--", "s:", "^-."]):
            ax[i, j].plot(SIGMAS, [r[f"{name}_{region}"] for r in pair_rows], marker, label=name)
        ax[i, j].set_title(f"pair {pair['idx']} loss on {region} tokens")
        ax[i, j].set_xlabel("sigma")
        ax[i, j].legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "region_loss.png"), dpi=100)
plt.close()

fig, ax = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 3.5))
for i, pair in enumerate(pairs):
    ax[i].imshow(np.load(os.path.join(OUT, f"diff_{pair['idx']}.npy")), cmap="magma")
    ax[i].set_title(f"pair {pair['idx']} |tgt - src| latent")
    ax[i].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "change_maps.png"), dpi=100)
plt.close()

log("DONE t1_region_loss", LOG)
