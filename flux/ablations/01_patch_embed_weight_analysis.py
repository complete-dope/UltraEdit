# HOW BETTER THE MDOEL HAS LEARNED THE NEW WEIGHTS HERE 

# Test 1a: is the condition half of x_embedder non-trivial after training?
import os, torch, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from safetensors.torch import load_file
from safetensors import safe_open
from ablation_common import CKPT, BASE, RESULTS, log, save_json

OUT = os.path.join(RESULTS, "t1_weights"); os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "log.txt"); open(LOG, "w").close()

# Diffusion model weights
with safe_open(os.path.join(CKPT, "diffusion_pytorch_model.safetensors"), "pt") as f:
    W = f.get_tensor("x_embedder.weight").float()          # [3072, 256]
    proj_out = f.get_tensor("proj_out.weight").float()

# Base model weights
with safe_open(os.path.join(BASE, "transformer", "diffusion_pytorch_model.safetensors"), "pt") as f:
    W0 = f.get_tensor("x_embedder.weight").float()         # [3072, 128]
    proj_out0 = f.get_tensor("proj_out.weight").float()

log(f"ckpt x_embedder {tuple(W.shape)}  base x_embedder {tuple(W0.shape)}", LOG)
half = W.shape[1] // 2

Wn, Wc = W[:, :half], W[:, half:] # Noise weights , conditioned channels
res = {}
def stats(name, t):
    d = dict(fro=t.norm().item(), abs_mean=t.abs().mean().item(), abs_max=t.abs().max().item(),
             std=t.std().item(), frac_exact_zero=(t == 0).float().mean().item())
    res[name] = d
    log(f"{name:28s} fro={d['fro']:.4f} abs_mean={d['abs_mean']:.6f} abs_max={d['abs_max']:.4f} std={d['std']:.6f} zeros={d['frac_exact_zero']:.4f}", LOG)

stats("noise_half (trained)", Wn)
stats("cond_half (trained)", Wc)
stats("noise_half (base)", W0)
stats("noise_half drift (trn-base)", Wn - W0)
res["ratio_cond_over_noise_absmean"] = Wc.abs().mean().item() / Wn.abs().mean().item()
res["ratio_cond_over_noise_fro"] = (Wc.norm() / Wn.norm()).item()
res["noise_half_rel_drift_fro"] = ((Wn - W0).norm() / W0.norm()).item()
res["proj_out_rel_drift_fro"] = ((proj_out - proj_out0).norm() / proj_out0.norm()).item()
log(f"cond/noise abs-mean ratio = {res['ratio_cond_over_noise_absmean']:.4f}   (trainer logs this as cond_img_ratio)", LOG)
log(f"cond/noise frobenius ratio = {res['ratio_cond_over_noise_fro']:.4f}", LOG)
log(f"noise half relative drift from base = {res['noise_half_rel_drift_fro']:.4f}", LOG)
log(f"proj_out relative drift from base   = {res['proj_out_rel_drift_fro']:.4f}", LOG)

# per input column norm. Column layout after _patchify_latents: c*4 + (py*2+px), c in 0..31
col_n = Wn.norm(dim=0).numpy(); col_c = Wc.norm(dim=0).numpy(); col_0 = W0.norm(dim=0).numpy()
res["col_norm_noise_mean/min/max"] = [float(col_n.mean()), float(col_n.min()), float(col_n.max())]
res["col_norm_cond_mean/min/max"] = [float(col_c.mean()), float(col_c.min()), float(col_c.max())]
per_ch_c = col_c.reshape(32, 4).mean(1); per_ch_n = col_n.reshape(32, 4).mean(1)
res["per_latent_channel_cond_norm"] = per_ch_c.tolist()
res["per_latent_channel_noise_norm"] = per_ch_n.tolist()
log("per latent channel cond col-norm: " + " ".join(f"{v:.2f}" for v in per_ch_c), LOG)
log("per latent channel noise col-norm: " + " ".join(f"{v:.2f}" for v in per_ch_n), LOG)

# How aligned are the two halves? If cond half ~ k * noise half, the model just averages the two images.
cos_cols = torch.nn.functional.cosine_similarity(Wn, Wc, dim=0)   # per input channel, across 3072 outputs
res["colwise_cos(noise,cond)_mean"] = cos_cols.mean().item()
res["colwise_cos(noise,cond)_min/max"] = [cos_cols.min().item(), cos_cols.max().item()]
log(f"column-wise cosine(noise col i, cond col i): mean={cos_cols.mean():.4f} min={cos_cols.min():.4f} max={cos_cols.max():.4f}", LOG)
# least squares: Wc ≈ Wn @ M ; residual tells how much of cond half is NOT a linear re-use of noise half
M = torch.linalg.lstsq(Wn, Wc).solution
resid = (Wc - Wn @ M).norm() / Wc.norm()
res["cond_half_unexplained_by_noise_half"] = resid.item()
log(f"fraction of cond half NOT in column space of noise half = {resid:.4f} (0 = pure re-use of the noise pathway, 1 = orthogonal)", LOG)
# singular value spectra
sn = torch.linalg.svdvals(Wn).numpy(); sc = torch.linalg.svdvals(Wc).numpy()
res["top5_sv_noise"] = sn[:5].tolist(); res["top5_sv_cond"] = sc[:5].tolist()
res["effective_rank_noise"] = float(np.exp(-(p := sn**2 / (sn**2).sum()) @ np.log(p + 1e-12)))
res["effective_rank_cond"] = float(np.exp(-(p := sc**2 / (sc**2).sum()) @ np.log(p + 1e-12)))
log(f"effective rank noise={res['effective_rank_noise']:.1f} cond={res['effective_rank_cond']:.1f} (of 128)", LOG)

fig, ax = plt.subplots(2, 2, figsize=(14, 8))
ax[0, 0].plot(col_0, label="base noise half", alpha=.6); ax[0, 0].plot(col_n, label="trained noise half"); ax[0, 0].plot(col_c, label="trained cond half")
ax[0, 0].set_title("x_embedder column L2 norm per input channel"); ax[0, 0].set_xlabel("input channel (0-127 noise | 128-255 cond shown overlaid)"); ax[0, 0].legend()
ax[0, 1].bar(np.arange(32) - .2, per_ch_n, .4, label="noise"); ax[0, 1].bar(np.arange(32) + .2, per_ch_c, .4, label="cond")
ax[0, 1].set_title("mean col norm per VAE latent channel (32)"); ax[0, 1].legend()
ax[1, 0].semilogy(sn, label="noise half"); ax[1, 0].semilogy(sc, label="cond half"); ax[1, 0].set_title("singular values"); ax[1, 0].legend()
ax[1, 1].hist(Wn.flatten().numpy(), bins=200, alpha=.5, label="noise", density=True); ax[1, 1].hist(Wc.flatten().numpy(), bins=200, alpha=.5, label="cond", density=True)
ax[1, 1].set_yscale("log"); ax[1, 1].set_title("weight value distribution"); ax[1, 1].legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT, "x_embedder_analysis.png"), dpi=110); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(16, 5))
im = ax[0].imshow(Wn[:256].numpy(), aspect="auto", cmap="RdBu", vmin=-.05, vmax=.05); ax[0].set_title("noise half (first 256 output rows)")
ax[1].imshow(Wc[:256].numpy(), aspect="auto", cmap="RdBu", vmin=-.05, vmax=.05); ax[1].set_title("cond half (first 256 output rows)")
plt.colorbar(im, ax=ax); plt.savefig(os.path.join(OUT, "x_embedder_heatmap.png"), dpi=110); plt.close()

# whole-model drift from base
drift = {}
with safe_open(os.path.join(CKPT, "diffusion_pytorch_model.safetensors"), "pt") as f, safe_open(os.path.join(BASE, "transformer", "diffusion_pytorch_model.safetensors"), "pt") as g:
    keys = [k for k in f.keys() if k in set(g.keys()) and k != "x_embedder.weight"]
    for k in keys:
        a, b = f.get_tensor(k).float(), g.get_tensor(k).float()
        drift[k] = ((a - b).norm() / (b.norm() + 1e-12)).item()
grp = {}
for k, v in drift.items():
    g_ = ".".join(k.split(".")[:2]) if k.startswith(("transformer_blocks", "single_transformer_blocks")) else k.split(".")[0]
    grp.setdefault(g_, []).append(v)
grp = {k: float(np.mean(v)) for k, v in grp.items()}
res["relative_drift_by_module_group"] = grp
log("relative weight drift from base by group: " + json.dumps({k: round(v, 4) for k, v in grp.items()}), LOG)
plt.figure(figsize=(14, 4)); ks = list(grp); plt.bar(range(len(ks)), [grp[k] for k in ks]); plt.xticks(range(len(ks)), ks, rotation=90, fontsize=7)
plt.ylabel("||W_trained - W_base|| / ||W_base||"); plt.title("full-finetune drift per block"); plt.tight_layout(); plt.savefig(os.path.join(OUT, "drift_by_block.png"), dpi=110); plt.close()

save_json(res, os.path.join(OUT, "results.json"))
log("DONE t1_weights", LOG)
