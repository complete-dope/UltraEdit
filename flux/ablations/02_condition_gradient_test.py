# GRADIENTS FOR THE LOSS W.R.T THE CONDITION LATENT , vs W.R.T THE NOISY LATENT
# This tells how grads are respecting the inputs that we have here, and how values are changing at different sigma (noise) values in this 
 
# Test 1b: gradient of the flow-matching loss w.r.t. the condition latent, vs w.r.t. the noisy latent.
# Plus loss deltas when the condition is zeroed / swapped / replaced with noise.
import os, torch, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ablation_common import *

OUT = os.path.join(RESULTS, "t1_grad"); os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "log.txt"); open(LOG, "w").close()
torch.manual_seed(0)

pairs = load_pairs(n=3)
log(f"pairs: {[p['id'] for p in pairs]}", LOG)
te = TextEncoder() # defined where ? 
pe, tids = te(PROMPT)
pe_null, _ = te("")
pe_cap, _ = te(pairs[0]["caption"])
te.free()
vae, bn_mean, bn_std = load_vae()

lat = [dict(src=norm_pack(encode_img(vae, p["src"]), bn_mean, bn_std), tgt=norm_pack(encode_img(vae, p["tgt"]), bn_mean, bn_std)) for p in pairs] # pairs here are 

for p, l in zip(pairs, lat):
    to_pil(prep(p["src"])).resize((512, 340)).save(os.path.join(OUT, f"{p['idx']}_src.jpg")) # source image file 
    to_pil(prep(p["tgt"])).resize((512, 340)).save(os.path.join(OUT, f"{p['idx']}_tgt.jpg")) # target image file 

del vae; torch.cuda.empty_cache()

tr = load_transformer()
tr.enable_gradient_checkpointing()
for p_ in tr.parameters():
    p_.requires_grad_(False)
log(f"transformer loaded, {sum(p.numel() for p in tr.parameters())/1e9:.2f}B params", LOG)

ids = latent_ids(lat[0]["tgt"])

def fm_loss(tgt, cond, sigma, noise, prompt_embeds, need_grad=("cond", "noisy")):
    x = pack(tgt); c = pack(cond)
    noisy = (1 - sigma) * x + sigma * noise # ((sigma * noise) + (1-sigma) * X) 
    noisy = noisy.detach().requires_grad_("noisy" in need_grad)
    c = c.detach().requires_grad_("cond" in need_grad)
    inp = torch.cat([noisy, c], dim=-1).to(DT)
    pred = tr(hidden_states=inp, timestep=torch.tensor([sigma], device=DEV, dtype=DT), guidance=None,
              encoder_hidden_states=prompt_embeds.to(DEV, DT), txt_ids=tids.to(DEV), img_ids=ids, return_dict=False)[0].float()
    target = noise - x
    loss = ((pred - target) ** 2).mean()
    return loss, noisy, c, pred


results = []
sigmas = [0.95, 0.7, 0.5, 0.3, 0.1]
for pi, (p, l) in enumerate(zip(pairs, lat)):
    x = pack(l["tgt"]).float(); noise = torch.randn_like(x)
    other = lat[(pi + 1) % len(lat)]
    for sigma in sigmas: # traversing from 
        loss, noisy, c, pred = fm_loss(l["tgt"].float(), l["src"].float(), sigma, noise, pe)
        loss.backward()
        g_c, g_n = c.grad.float(), noisy.grad.float()
        r = dict(pair=p["idx"], sigma=sigma, loss=loss.item(), grad_cond_norm=g_c.norm().item(), grad_noisy_norm=g_n.norm().item(),grad_cond_absmean=g_c.abs().mean().item(), grad_noisy_absmean=g_n.abs().mean().item())
        
        r["grad_ratio_cond_over_noisy"] = r["grad_cond_norm"] / r["grad_noisy_norm"]
        # spatial map of |grad| over tokens -> latent grid
        h, w = l["tgt"].shape[-2:]
        gmap = g_c.norm(dim=-1).view(h, w).cpu().numpy()
        np.save(os.path.join(OUT, f"gradmap_{p['idx']}_s{sigma}.npy"), gmap)
        tr.zero_grad(set_to_none=True)
        with torch.no_grad():
            # loss counterfactuals: how much does the prediction depend on the cond?
            pred_ref = pred.detach()
            l0, *_ , pred0 = fm_loss(l["tgt"].float(), torch.zeros_like(l["src"]).float(), sigma, noise, pe, need_grad=())
            ls, *_ , preds = fm_loss(l["tgt"].float(), other["src"].float(), sigma, noise, pe, need_grad=())
            ln, *_ , predn = fm_loss(l["tgt"].float(), torch.randn_like(l["src"]).float(), sigma, noise, pe, need_grad=())
            lt, *_ , predt = fm_loss(l["tgt"].float(), l["tgt"].float(), sigma, noise, pe, need_grad=())
            lnp, *_ , prednp = fm_loss(l["tgt"].float(), l["src"].float(), sigma, noise, pe_null, need_grad=())
            lcap, *_ , predcap = fm_loss(l["tgt"].float(), l["src"].float(), sigma, noise, pe_cap, need_grad=())
        r.update(loss_cond_zero=l0.item(), loss_cond_swapped=ls.item(), loss_cond_noise=ln.item(), loss_cond_is_target=lt.item(),
                 loss_null_prompt=lnp.item(), loss_caption_prompt=lcap.item(),
                 pred_delta_cond_zero=((pred0 - pred_ref).norm() / pred_ref.norm()).item(),
                 pred_delta_cond_swapped=((preds - pred_ref).norm() / pred_ref.norm()).item(),
                 pred_delta_cond_noise=((predn - pred_ref).norm() / pred_ref.norm()).item(),
                 pred_delta_null_prompt=((prednp - pred_ref).norm() / pred_ref.norm()).item(),
                 pred_delta_caption_prompt=((predcap - pred_ref).norm() / pred_ref.norm()).item())
        results.append(r)
        log(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}), LOG)
        torch.cuda.empty_cache()

save_json(results, os.path.join(OUT, "results.json"))

# plots
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
for p in pairs:
    rs = [r for r in results if r["pair"] == p["idx"]]
    ax[0].plot([r["sigma"] for r in rs], [r["grad_ratio_cond_over_noisy"] for r in rs], "o-", label=f"pair {p['idx']}")
    ax[1].plot([r["sigma"] for r in rs], [r["loss"] for r in rs], "o-", label=f"{p['idx']} cond=src")
    ax[1].plot([r["sigma"] for r in rs], [r["loss_cond_zero"] for r in rs], "x--", label=f"{p['idx']} cond=0")
    ax[1].plot([r["sigma"] for r in rs], [r["loss_cond_swapped"] for r in rs], "s:", label=f"{p['idx']} cond=other img")
    for k, m in [("pred_delta_cond_zero", "x--"), ("pred_delta_cond_swapped", "s:"), ("pred_delta_null_prompt", "^-."), ("pred_delta_caption_prompt", "d-")]:
        ax[2].plot([r["sigma"] for r in rs], [r[k] for r in rs], m, label=f"{p['idx']} {k[11:]}")
ax[0].set_title("||dL/d cond|| / ||dL/d noisy||"); ax[0].set_xlabel("sigma"); ax[0].legend(); ax[0].set_yscale("log")
ax[1].set_title("flow-matching loss"); ax[1].set_xlabel("sigma"); ax[1].legend(fontsize=7)
ax[2].set_title("relative change of prediction when input is replaced"); ax[2].set_xlabel("sigma"); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "grad_and_loss.png"), dpi=110); plt.close()

fig, ax = plt.subplots(len(pairs), len(sigmas) + 1, figsize=(4 * (len(sigmas) + 1), 2.8 * len(pairs)))
for i, p in enumerate(pairs):
    ax[i, 0].imshow(prep(p["src"]).permute(1, 2, 0) * .5 + .5); ax[i, 0].set_title(f"cond {p['idx']}"); ax[i, 0].axis("off")
    for j, s in enumerate(sigmas):
        g = np.load(os.path.join(OUT, f"gradmap_{p['idx']}_s{s}.npy"))
        ax[i, j + 1].imshow(g, cmap="magma"); ax[i, j + 1].set_title(f"|dL/dcond| sigma={s}"); ax[i, j + 1].axis("off")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "grad_maps.png"), dpi=100); plt.close()
log("DONE t1_grad", LOG)
