# Test 2: attention maps of joint (double) and single blocks during sampling.
import os, sys, torch, json, math
import numpy as np
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from diffusers.models.transformers import transformer_flux2 as tf2
from diffusers.models.transformers.transformer_flux2 import _get_qkv_projections, apply_rotary_emb, dispatch_attention_fn
from ablation_common import *
from cfg_sampling import sample, unnorm_unpatch

OUT = os.path.join(RESULTS, "t2_attn_pair" + os.environ.get("ATTN_PAIR", "19")); os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "log.txt"); open(LOG, "w").close()
STEPS = 28
TARGET_SIGMAS = [0.9, 0.5, 0.2]
COL_LAYERS = ["double0", "double2", "double4", "single0", "single5", "single10", "single15", "single19"]

CAP = dict(active=False, tag=None, layer=None, q_idx=None, txt_len=None, word_tok=None, col_layers=set(), store={})


def capture(query, key, layer):
    """query/key: (1, N, heads, d) after norm+rope, text first."""
    if not CAP["active"]:
        return
    q = query[0].permute(1, 0, 2).float(); k = key[0].permute(1, 0, 2).float()   # (heads, N, d)
    scale = 1 / math.sqrt(q.shape[-1])
    T = CAP["txt_len"]; N = q.shape[1]
    st = CAP["store"].setdefault(CAP["tag"], {}).setdefault(layer, {})
    # rows for selected image queries + edit-word text queries
    qi = torch.tensor(CAP["q_idx"] + CAP["word_tok"], device=q.device)
    logits = torch.einsum("hqd,hkd->hqk", q[:, qi], k) * scale
    attn = logits.softmax(-1)                                                    # (heads, nq, N)
    st["rows_headmean"] = attn.mean(0).cpu().numpy()
    ent = -(attn * (attn + 1e-12).log()).sum(-1)                                 # (heads, nq)
    st["rows_entropy_per_head"] = ent.cpu().numpy()
    if layer in CAP["col_layers"]:
        # every image query: mass on text (real / pad) and on the edit-word tokens; per-head mean
        real = CAP["txt_real_mask"].to(q.device)                                 # (T,) bool
        wt = torch.zeros(N, dtype=torch.bool, device=q.device); wt[CAP["word_tok"]] = True
        tot_real = torch.zeros(N - T, device=q.device); tot_pad = torch.zeros(N - T, device=q.device); tot_w = torch.zeros(N - T, device=q.device)
        selfm = torch.zeros(N - T, device=q.device)
        for h in range(q.shape[0]):
            a = (q[h, T:] @ k[h].T * scale).softmax(-1)                          # (N_img, N)
            tot_real += a[:, :T][:, real].sum(-1); tot_pad += a[:, :T][:, ~real].sum(-1); tot_w += a[:, wt].sum(-1)
            selfm += a[torch.arange(N - T, device=q.device), T + torch.arange(N - T, device=q.device)]
        H = q.shape[0]
        st["img2txt_real"] = (tot_real / H).cpu().numpy(); st["img2txt_pad"] = (tot_pad / H).cpu().numpy()
        st["img2word"] = (tot_w / H).cpu().numpy(); st["img_self"] = (selfm / H).cpu().numpy()


class DoubleProc(tf2.Flux2AttnProcessor):
    def __init__(self, name): super().__init__(); self.name = name
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
        query, key, value, eq, ek, ev = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        query = attn.norm_q(query.unflatten(-1, (-1, attn.head_dim))); key = attn.norm_k(key.unflatten(-1, (-1, attn.head_dim))); value = value.unflatten(-1, (-1, attn.head_dim))
        eq = attn.norm_added_q(eq.unflatten(-1, (-1, attn.head_dim))); ek = attn.norm_added_k(ek.unflatten(-1, (-1, attn.head_dim))); ev = ev.unflatten(-1, (-1, attn.head_dim))
        query = torch.cat([eq, query], 1); key = torch.cat([ek, key], 1); value = torch.cat([ev, value], 1)
        query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1); key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        capture(query, key, self.name)
        hs = dispatch_attention_fn(query, key, value, attn_mask=attention_mask, backend=self._attention_backend, parallel_config=self._parallel_config)
        hs = hs.flatten(2, 3).to(query.dtype)
        ehs, hs = hs.split_with_sizes([encoder_hidden_states.shape[1], hs.shape[1] - encoder_hidden_states.shape[1]], dim=1)
        return attn.to_out[1](attn.to_out[0](hs)), attn.to_add_out(ehs)


class SingleProc(tf2.Flux2ParallelSelfAttnProcessor):
    def __init__(self, name): super().__init__(); self.name = name
    def __call__(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None):
        hidden_states = attn.to_qkv_mlp_proj(hidden_states)
        qkv_dim = 3 * attn.inner_dim; mlp_dim = attn.mlp_hidden_dim * attn.mlp_mult_factor
        local_qkv = hidden_states.shape[-1] * qkv_dim // (qkv_dim + mlp_dim)
        qkv, mlp = torch.split(hidden_states, [local_qkv, hidden_states.shape[-1] - local_qkv], dim=-1)
        query, key, value = qkv.chunk(3, dim=-1)
        query = attn.norm_q(query.unflatten(-1, (-1, attn.head_dim))); key = attn.norm_k(key.unflatten(-1, (-1, attn.head_dim))); value = value.unflatten(-1, (-1, attn.head_dim))
        query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1); key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        capture(query, key, self.name)
        hs = dispatch_attention_fn(query, key, value, attn_mask=attention_mask, backend=self._attention_backend, parallel_config=self._parallel_config)
        hs = hs.flatten(2, 3).to(query.dtype)
        return attn.to_out(torch.cat([hs, attn.mlp_act_fn(mlp)], -1))


def install(tr):
    for i, b in enumerate(tr.transformer_blocks): b.attn.set_processor(DoubleProc(f"double{i}"))
    for i, b in enumerate(tr.single_transformer_blocks): b.attn.set_processor(SingleProc(f"single{i}"))


def pick_queries(src_pix, h, w):
    g = src_pix.mean(0, keepdim=True).unsqueeze(0)
    g = F.interpolate(g, size=(h, w), mode="area")[0, 0]
    sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=g.dtype); sy = sx.T
    e = (F.conv2d(g[None, None], sx[None, None], padding=1) ** 2 + F.conv2d(g[None, None], sy[None, None], padding=1) ** 2)[0, 0].sqrt()
    qs = {}
    band = e.clone(); band[: int(.25 * h)] = 0; band[int(.75 * h):] = 0; band[:, : int(.15 * w)] = 0; band[:, int(.85 * w):] = 0
    r, c = divmod(int(band.argmax()), w); qs["detail_max_edge"] = (r, c)
    band2 = band.clone(); band2[max(0, r - 12): r + 12, max(0, c - 12): c + 12] = 0
    r2, c2 = divmod(int(band2.argmax()), w); qs["detail_2nd_edge"] = (r2, c2)
    qs["sky"] = (int(.08 * h), int(.5 * w)); qs["ground"] = (int(.92 * h), int(.3 * w)); qs["center"] = (h // 2, w // 2)
    return qs


def locality(row_img, r, c, h, w, radii=(2, 4, 8)):
    m = row_img.reshape(h, w); tot = m.sum()
    out = dict(img_mass=float(tot))
    p = m / max(tot, 1e-12)
    out["entropy_norm"] = float(-(p * np.log(p + 1e-12)).sum() / np.log(h * w))
    for rad in radii:
        out[f"mass_r{rad}"] = float(p[max(0, r - rad): r + rad + 1, max(0, c - rad): c + rad + 1].sum())
    out["self_mass"] = float(p[r, c])
    # what fraction a uniform map would give inside r4, for reference
    out["uniform_r4"] = float(min(9, h) * min(9, w) / (h * w))
    return out


def run_model(model_name, tr, cond_p, pe, tids, toks, mask, words, src_pix, drop_cond=False, prompt_name="default"):
    h, w = cond_p.shape[-2:]
    T = pe.shape[1]
    qs = pick_queries(src_pix, h, w)
    q_idx = [T + r * w + c for r, c in qs.values()]
    word_tok = {wd: [i for i, t in enumerate(toks) if mask[i] and wd in t.lower()] for wd in words}
    log(f"[{model_name}/{prompt_name}] queries={qs} word tokens={word_tok} real text tokens={sum(mask)}/{T} pad_side={'left' if mask[0]==0 else 'right'}", LOG)
    wt_flat = [i for v in word_tok.values() for i in v]
    CAP.update(q_idx=q_idx, txt_len=T, word_tok=wt_flat, txt_real_mask=torch.tensor(mask, dtype=torch.bool), col_layers=set(COL_LAYERS), store={}, active=False)
    # the shifted schedule never reaches low sigma (last step ~0.3 at this token count): capture the step
    # whose sigma is nearest each target, using the exact schedule sample() will build
    from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu
    _sched = load_scheduler(); _sig = np.linspace(1.0, 1 / STEPS, STEPS)
    _sched.set_timesteps(sigmas=_sig, device="cpu", mu=compute_empirical_mu(image_seq_len=h * w, num_steps=STEPS))
    sched_sigmas = (_sched.timesteps.float() / 1000).tolist()
    step_for = {ts: int(np.argmin([abs(sg - ts) for sg in sched_sigmas])) for ts in TARGET_SIGMAS}
    log(f"  schedule sigmas: {[round(x, 3) for x in sched_sigmas]}", LOG)
    log(f"  capture steps: {step_for} (actual sigmas {[round(sched_sigmas[i], 3) for i in step_for.values()]})", LOG)
    sig_at = {}

    def hook(i, sigma):
        CAP["active"] = False
        for ts, si in step_for.items():
            if i == si:
                sig_at[ts] = (i, sigma); CAP["active"] = True; CAP["tag"] = f"sigma{ts}"
                log(f"  capturing step {i} sigma={sigma:.3f} as {CAP['tag']}", LOG)

    out = sample(tr, sched, cond_p, pe, tids, steps=STEPS, seed=0, step_hook=hook, drop_cond=drop_cond)
    CAP["active"] = False
    img = decode_lat(vae, unnorm_unpatch(out, bn_mean, bn_std))
    to_pil(img).resize((1024, 680)).save(os.path.join(OUT, f"{model_name}_{prompt_name}_sample.jpg"), quality=90)
    store = CAP["store"]
    layers = [f"double{i}" for i in range(len(tr.transformer_blocks))] + [f"single{i}" for i in range(len(tr.single_transformer_blocks))]
    stats = {}
    nq = len(qs); qnames = list(qs)
    # ---- locality stats for every layer / timestep / query
    for tag, ls in store.items():
        for L in layers:
            rows = ls[L]["rows_headmean"]; ent = ls[L]["rows_entropy_per_head"]
            for qi, qn in enumerate(qnames):
                r, c = qs[qn]; row = rows[qi]
                s = locality(row[T:], r, c, h, w)
                s["txt_mass_real"] = float(row[:T][np.array(mask, bool)].sum()); s["txt_mass_pad"] = float(row[:T][~np.array(mask, bool)].sum())
                s["entropy_heads_min/mean/max"] = [float(ent[:, qi].min()), float(ent[:, qi].mean()), float(ent[:, qi].max())]
                stats[f"{tag}/{L}/{qn}"] = s
            for wd, idxs in word_tok.items():
                if not idxs: continue
                rr = rows[nq:][[wt_flat.index(i) for i in idxs]].mean(0)
                s = dict(img_mass=float(rr[T:].sum()), txt_mass=float(rr[:T].sum()))
                p = rr[T:] / max(rr[T:].sum(), 1e-12); s["entropy_norm_over_img"] = float(-(p * np.log(p + 1e-12)).sum() / np.log(h * w))
                stats[f"{tag}/{L}/word:{wd}"] = s
            if "img2word" in ls[L]:
                stats[f"{tag}/{L}/img2txt"] = dict(real_mean=float(ls[L]["img2txt_real"].mean()), pad_mean=float(ls[L]["img2txt_pad"].mean()),
                                                   word_mean=float(ls[L]["img2word"].mean()), word_max=float(ls[L]["img2word"].max()), self_mean=float(ls[L]["img_self"].mean()))
    save_json(dict(queries=qs, word_tokens=word_tok, sig_at=sig_at, stats=stats), os.path.join(OUT, f"{model_name}_{prompt_name}_stats.json"))
    np.savez_compressed(os.path.join(OUT, f"{model_name}_{prompt_name}_raw.npz"),
                        **{f"{tag}|{L}|{k}": v for tag, ls in store.items() for L, d in ls.items() for k, v in d.items()})
    for k, v in stats.items():
        if any(x in k for x in COL_LAYERS) and ("detail_max_edge" in k or "img2txt" in k or "word:" in k):
            log(f"  {k}: " + json.dumps({a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()}), LOG)

    # ---- figures
    bg = ((src_pix.permute(1, 2, 0).numpy() + 1) / 2)
    bg_small = np.array(Image.fromarray((bg * 255).astype(np.uint8)).resize((w * 4, h * 4)))
    def heat(a, row2d, r=None, c=None, cmap="inferno", crop=None):
        m = row2d / max(row2d.max(), 1e-12)
        m = m ** 0.3                                        # gamma so the tail is visible; pure white = the peak
        if crop is not None:
            r0, r1, c0, c1 = crop; m = m[r0:r1, c0:c1]; bgc = bg_small[r0 * 4:r1 * 4, c0 * 4:c1 * 4]
        else:
            bgc = bg_small
        a.imshow(bgc, alpha=.35); a.imshow(np.kron(m, np.ones((4, 4))), cmap=cmap, alpha=.75, vmin=0, vmax=1)
        if r is not None:
            rr, cc = (r - crop[0], c - crop[2]) if crop is not None else (r, c)
            a.plot(cc * 4 + 2, rr * 4 + 2, "c+", ms=9, mew=1.5)
        a.axis("off")

    for qn in qnames:
        r, c = qs[qn]; qi = qnames.index(qn)
        crop = (max(0, r - 14), min(h, r + 15), max(0, c - 22), min(w, c + 23))
        fig, ax = plt.subplots(2 * len(TARGET_SIGMAS), len(COL_LAYERS), figsize=(3.2 * len(COL_LAYERS), 2.4 * 2 * len(TARGET_SIGMAS)))
        for i, ts in enumerate(TARGET_SIGMAS):
            tag = f"sigma{ts}"
            for j, L in enumerate(COL_LAYERS):
                row = store[tag][L]["rows_headmean"][qi][T:].reshape(h, w)
                s = stats[f"{tag}/{L}/{qn}"]
                heat(ax[2 * i, j], row, r, c)
                ax[2 * i, j].set_title(f"{L} s={sig_at[ts][1]:.2f}  full frame\nr4={s['mass_r4']:.2f} r8={s['mass_r8']:.2f} H={s['entropy_norm']:.2f} txt={s['txt_mass_real']+s['txt_mass_pad']:.2f}", fontsize=8)
                heat(ax[2 * i + 1, j], row, r, c, crop=crop)
                ax[2 * i + 1, j].set_title("zoom +-14 rows / +-22 cols", fontsize=7)
        plt.suptitle(f"{model_name} / prompt={prompt_name} / query={qn} at (row {r}, col {c}) of {h}x{w}  [image-key part of the row, head-mean, gamma 0.3]")
        plt.tight_layout(); plt.savefig(os.path.join(OUT, f"{model_name}_{prompt_name}_row_{qn}.jpg"), dpi=80); plt.close()
    # text -> image rows for words
    for wd, idxs in word_tok.items():
        if not idxs: continue
        fig, ax = plt.subplots(len(TARGET_SIGMAS), len(COL_LAYERS), figsize=(3.2 * len(COL_LAYERS), 2.4 * len(TARGET_SIGMAS)))
        for i, ts in enumerate(TARGET_SIGMAS):
            tag = f"sigma{ts}"
            for j, L in enumerate(COL_LAYERS):
                rows = store[tag][L]["rows_headmean"][nq:]
                row = rows[[wt_flat.index(k) for k in idxs]].mean(0)[T:].reshape(h, w)
                a = ax[i, j]; heat(a, row)
                s = stats[f"{tag}/{L}/word:{wd}"]
                a.set_title(f"{L} s={sig_at[ts][1]:.2f}\nimg_mass={s['img_mass']:.2f} H={s['entropy_norm_over_img']:.2f}", fontsize=8)
        plt.suptitle(f"{model_name} / prompt={prompt_name} / TEXT token '{wd}' -> image keys")
        plt.tight_layout(); plt.savefig(os.path.join(OUT, f"{model_name}_{prompt_name}_txt2img_{wd}.jpg"), dpi=80); plt.close()
        # image -> word column maps
        fig, ax = plt.subplots(len(TARGET_SIGMAS), len(COL_LAYERS), figsize=(3.2 * len(COL_LAYERS), 2.4 * len(TARGET_SIGMAS)))
        for i, ts in enumerate(TARGET_SIGMAS):
            tag = f"sigma{ts}"
            for j, L in enumerate(COL_LAYERS):
                if "img2word" not in store[tag][L]: continue
                col = store[tag][L]["img2word"].reshape(h, w)
                a = ax[i, j]; heat(a, col, cmap="viridis")
                a.set_title(f"{L} s={sig_at[ts][1]:.2f}\nmean={col.mean():.4f} max={col.max():.4f}", fontsize=8)
        plt.suptitle(f"{model_name} / prompt={prompt_name} / how much each IMAGE token attends to ALL edit-word tokens {list(word_tok)} (head-mean)")
        plt.tight_layout(); plt.savefig(os.path.join(OUT, f"{model_name}_{prompt_name}_img2txt_words.jpg"), dpi=80); plt.close()
        break  # one img2txt figure (it sums all words)
    # image -> text total (real vs pad) maps
    fig, ax = plt.subplots(2 * len(TARGET_SIGMAS), len(COL_LAYERS), figsize=(3.2 * len(COL_LAYERS), 2.4 * 2 * len(TARGET_SIGMAS)))
    for i, ts in enumerate(TARGET_SIGMAS):
        tag = f"sigma{ts}"
        for j, L in enumerate(COL_LAYERS):
            for k, key in enumerate(["img2txt_real", "img2txt_pad"]):
                col = store[tag][L][key].reshape(h, w); a = ax[2 * i + k, j]
                a.imshow(col, cmap="viridis", vmin=0, vmax=max(col.max(), 1e-6)); a.axis("off"); a.set_title(f"{L} {tag} {key}\nmean={col.mean():.3f}", fontsize=8)
    plt.suptitle(f"{model_name} / prompt={prompt_name} / image-token attention mass on real text tokens vs pad tokens")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, f"{model_name}_{prompt_name}_img2txt_total.jpg"), dpi=80); plt.close()
    # locality curve across all layers
    fig, ax = plt.subplots(1, 3, figsize=(18, 4))
    for ts in TARGET_SIGMAS:
        tag = f"sigma{ts}"
        ax[0].plot([stats[f"{tag}/{L}/detail_max_edge"]["mass_r4"] for L in layers], "o-", label=f"sigma={sig_at[ts][1]:.2f}")
        ax[1].plot([stats[f"{tag}/{L}/detail_max_edge"]["entropy_norm"] for L in layers], "o-", label=tag)
        ax[2].plot([stats[f"{tag}/{L}/detail_max_edge"]["txt_mass_real"] + stats[f"{tag}/{L}/detail_max_edge"]["txt_mass_pad"] for L in layers], "o-", label=tag)
    for a, t in zip(ax, ["mass within radius 4 (detail query)", "normalized entropy over image keys", "mass on text tokens"]):
        a.set_title(t); a.set_xticks(range(len(layers))); a.set_xticklabels(layers, rotation=90, fontsize=7); a.legend()
    ax[0].axhline(stats[f"sigma0.5/double0/detail_max_edge"]["uniform_r4"], color="k", ls="--", label="uniform")
    plt.suptitle(f"{model_name} / prompt={prompt_name}"); plt.tight_layout(); plt.savefig(os.path.join(OUT, f"{model_name}_{prompt_name}_locality_curve.jpg"), dpi=90); plt.close()
    # query marker image
    fig = plt.figure(figsize=(10, 6.6)); plt.imshow(bg)
    for qn, (r, c) in qs.items(): plt.plot(c * 16 + 8, r * 16 + 8, "o", ms=10, mfc="none", mew=2); plt.text(c * 16 + 12, r * 16, qn, color="w", fontsize=9, bbox=dict(fc="k", alpha=.5))
    plt.axis("off"); plt.savefig(os.path.join(OUT, f"{model_name}_{prompt_name}_queries.jpg"), dpi=80, bbox_inches="tight"); plt.close()
    return stats


pairs = load_pairs(n=1, indices=[int(os.environ.get("ATTN_PAIR", "19"))])
p = pairs[0]
log(f"attention test on pair {p['idx']}", LOG)
te = TextEncoder()
pe_def, tids = te(PROMPT); toks_def, mask_def = te.token_positions(PROMPT)
SUNSET = "dramatic orange sunset sky with clouds"
pe_sun, _ = te(SUNSET); toks_sun, mask_sun = te.token_positions(SUNSET)
log("default prompt real tokens: " + str([t for t, m in zip(toks_def, mask_def) if m]), LOG)
te.free()
vae, bn_mean, bn_std = load_vae()
cond_p = norm_pack(encode_img(vae, p["src"]), bn_mean, bn_std)
src_pix = prep(p["src"])
sched = load_scheduler()

tr = load_transformer(); install(tr)
run_model("trained", tr, cond_p, pe_def, tids, toks_def, mask_def, ["sky", "cloud", "green", "construction"], src_pix)
run_model("trained", tr, cond_p, pe_sun, tids, toks_sun, mask_sun, ["sunset", "orange", "sky", "cloud"], src_pix, prompt_name="sunset")
run_model("trained_nocond", tr, cond_p, pe_def, tids, toks_def, mask_def, ["sky", "cloud", "green", "construction"], src_pix, drop_cond=True)
del tr; torch.cuda.empty_cache()

# base model reference: same loop but the base x_embedder only takes the 128 noise channels
base = load_transformer(os.path.join(BASE, "transformer")); install(base)
_orig_fwd = base.forward
def fwd_base(hidden_states, **kw):
    return _orig_fwd(hidden_states=hidden_states[..., :128], **kw)
base.forward = fwd_base
run_model("base", base, cond_p, pe_def, tids, toks_def, mask_def, ["sky", "cloud", "green", "construction"], src_pix)
run_model("base", base, cond_p, pe_sun, tids, toks_sun, mask_sun, ["sunset", "orange", "sky", "cloud"], src_pix, prompt_name="sunset")
log("DONE t2_attn", LOG)
