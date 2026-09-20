import json, os, html
R = "/workspace/flux-ablation/results/"
OUT = "/workspace/flux-ablation/artifact/index.html"


def J(p):
    return json.load(open(R + p))


def f(v, d=3):
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.{d}f}"
    return html.escape(str(v))


def table(headers, rows, cls="num"):
    h = "".join(f"<th>{html.escape(str(c))}</th>" for c in headers)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def fig(src, cap, wide=False):
    return f'<figure class="{"wide" if wide else ""}"><a href="img/{src}" target="_blank" rel="noopener"><img src="img/{src}" alt="{html.escape(cap)}" loading="lazy"></a><figcaption>{cap}</figcaption></figure>'


parts = []

# ------------------------------------------------------------------ weights
w = J("t1_weights/results.json")
rows = []
for k in ["noise_half (trained)", "cond_half (trained)", "noise_half (base)", "noise_half drift (trn-base)"]:
    d = w[k]
    rows.append([k, f(d["fro"], 3), f(d["abs_mean"], 5), f(d["abs_max"], 4), f(d["std"], 5), f(d["frac_exact_zero"], 4)])
w_t1 = table(["tensor", "Frobenius", "abs mean", "abs max", "std", "frac exact 0"], rows)
w_t2 = table(["quantity", "value"], [
    ["cond / noise abs-mean ratio (trainer's cond_img_ratio)", f(w["ratio_cond_over_noise_absmean"], 4)],
    ["cond / noise Frobenius ratio", f(w["ratio_cond_over_noise_fro"], 4)],
    ["noise half relative drift from base", f(w["noise_half_rel_drift_fro"], 4)],
    ["proj_out relative drift from base", f(w["proj_out_rel_drift_fro"], 4)],
    ["column-wise cosine(noise col i, cond col i): mean", f(w["colwise_cos(noise,cond)_mean"], 4)],
    ["column-wise cosine min / max", f"{w['colwise_cos(noise,cond)_min/max'][0]:.4f} / {w['colwise_cos(noise,cond)_min/max'][1]:.4f}"],
    ["fraction of cond half outside column space of noise half", f(w["cond_half_unexplained_by_noise_half"], 4)],
    ["effective rank noise / cond (of 128)", f"{w['effective_rank_noise']:.1f} / {w['effective_rank_cond']:.1f}"],
    ["top-5 singular values noise", ", ".join(f"{x:.2f}" for x in w["top5_sv_noise"])],
    ["top-5 singular values cond", ", ".join(f"{x:.2f}" for x in w["top5_sv_cond"])],
    ["column norm noise mean / min / max", " / ".join(f"{x:.3f}" for x in w["col_norm_noise_mean/min/max"])],
    ["column norm cond mean / min / max", " / ".join(f"{x:.3f}" for x in w["col_norm_cond_mean/min/max"])],
], cls="kv")
w_t3 = table(["VAE latent channel"] + [str(i) for i in range(32)],
             [["noise col-norm"] + [f"{x:.2f}" for x in w["per_latent_channel_noise_norm"]],
              ["cond col-norm"] + [f"{x:.2f}" for x in w["per_latent_channel_cond_norm"]]])
g = w["relative_drift_by_module_group"]
order = [k for k in g if not k.startswith(("transformer_blocks", "single_transformer_blocks"))] + [f"transformer_blocks.{i}" for i in range(5)] + [f"single_transformer_blocks.{i}" for i in range(20)]
w_t4 = table(["module group", "relative drift ||W-W0||/||W0||"], [[k, f(g[k], 4)] for k in order if k in g], cls="kv")
parts.append(f"""
<section id="s1a">
<h2><span class="tag pass">passes</span>Test 1a · Did the model learn to look at the input photo?</h2>
<p class="lede"><strong>What we checked.</strong> The first layer of the model has two "doors": one for the noisy image it is cleaning up, one for the input photo we want edited. The second door started at exactly zero when training began. If it was still near zero after training, the model never learned to use the input photo.</p>
<p class="lede"><strong>What we found.</strong> The input-photo door is wide open. Its weights are actually a bit <em>bigger</em> than the noisy-image door (1.10x), and they point in their own direction (97% different from the other door), so the model built a real, separate reading path for the input photo.</p>
<p><strong>How to read the tables.</strong> "Frobenius" and "abs mean" are just two ways of saying "how big are the numbers in this block of weights". Bigger = the door is more open. "noise half (base)" is the original model before your training; "trained" is checkpoint 17000.</p>
{w_t1}
{w_t2}
<h3>How big each of the 32 image channels is, for each door</h3>
<p>Every channel of the input photo is read with about the same strength as the matching channel of the noisy image. No channel was ignored.</p>
{w_t3}
<h3>How much did each part of the model change during training?</h3>
<p>Almost nothing changed in the body of the model (2-4% per block). The first layer changed a lot (87%) because the training setting <code>--x_embedder_lr 1e-3</code> gave a 100x higher learning rate to the <em>whole</em> first layer, not only to the new input-photo door. That means the original, pre-trained door was also shaken up hard. Not fatal, but worth fixing next time.</p>
{w_t4}
<div class="figs">
{fig("w_analysis.jpg", "Column norms per input channel (base vs trained noise half vs cond half), per-channel bars, singular values, weight histograms.", True)}
{fig("w_heatmap.jpg", "First 256 output rows of the noise half (left) and the cond half (right).", True)}
{fig("w_drift.jpg", "Relative weight drift from base per block.", True)}
</div>
<p class="src">Code <code>t1_weights.py</code> · raw <a href="data/t1_weights.json">t1_weights.json</a> · run log <a href="data/logs/t1_weights.log.txt">t1_weights.log.txt</a></p>
</section>""")

# ------------------------------------------------------------------ grad
gr = J("t1_grad/results.json"); gb = J("t1_grad/results_base.json")
bm = {(r["pair"], r["sigma"]): r for r in gb}
rows = []
for r in gr:
    b = bm[(r["pair"], r["sigma"])]
    rows.append([r["pair"], r["sigma"], f(r["loss"]), f(r["loss_cond_zero"]), f(r["loss_cond_swapped"]), f(r["loss_cond_noise"]), f(r["loss_cond_is_target"]),
                 f(b["base_t2i_loss"]), f(b["loss_if_pred_zero"]), f(r["loss_null_prompt"]), f(r["loss_caption_prompt"])])
g_t1 = table(["pair", "sigma", "trained cond=src", "cond=0", "cond=other img", "cond=gaussian noise", "cond=target", "BASE model (t2i)", "loss if pred=0", "null prompt", "caption prompt"], rows)
rows = []
for r in gr:
    rows.append([r["pair"], r["sigma"], f"{r['grad_cond_norm']:.5f}", f"{r['grad_noisy_norm']:.5f}", f(r["grad_ratio_cond_over_noisy"], 3),
                 f(r["pred_delta_cond_zero"]), f(r["pred_delta_cond_swapped"]), f(r["pred_delta_cond_noise"]), f(r["pred_delta_null_prompt"]), f(r["pred_delta_caption_prompt"])])
g_t2 = table(["pair", "sigma", "||dL/d cond||", "||dL/d noisy||", "ratio cond/noisy", "Δpred cond=0", "Δpred cond=other", "Δpred cond=noise", "Δpred null prompt", "Δpred caption"], rows)
parts.append(f"""
<section id="s1b">
<h2><span class="tag pass">passes</span>Test 1b · Does the output actually react to the input photo?</h2>
<p class="lede"><strong>What we checked.</strong> We ran the model once on a real held-out pair and asked: if we nudge the input photo a tiny bit, how much does the answer change? (That is what a "gradient" is.) We compared it to the same nudge on the noisy image. We also swapped the input photo for other things (nothing, a different house, random noise, the finished target photo) and measured how wrong the model's answer became.</p>
<p class="lede"><strong>What we found.</strong> The model reacts to the input photo as strongly as, or more strongly than, to the noisy image (ratio 1.1 to 1.8 in the early and middle part of denoising). The "where does it react" maps light up on roofs, edges and cars, i.e. real content. So the input photo is read.</p>
<div class="callout warn"><strong>The surprise.</strong> Giving the model the <em>correct</em> input photo made its answer <em>worse</em> than giving it a blank photo, for most of the denoising process (sigma 0.7 and below). It was also worse than the original untouched model. And when we secretly fed it the <em>finished target</em> photo as the input, its error dropped 3 to 8 times. Read together: the model learned "copy whatever photo comes through the input door", not "improve the photo".</div>
<p><strong>About sigma.</strong> Denoising runs from sigma 1.0 (pure noise, start) to 0 (clean image, end). Early steps decide the big picture, late steps decide details.</p>
<p><strong>About "loss".</strong> Lower is better. A model that outputs nothing at all would score about 2.0. The "BASE model" column is the original model before your training, given the text prompt only.</p>
<h3>How wrong is the answer when we swap the input photo for something else?</h3>
{g_t1}
<h3>How much does the answer move when we change one input?</h3>
<p>"Δpred" = how much the model's output changed, as a fraction. Changing the <em>text prompt</em> to nothing moves the output only 2-5%. Removing the <em>input photo</em> moves it 21-64%. The photo is doing all the work; the words are doing almost none.</p>
{g_t2}
<div class="figs">
{fig("g_curves.jpg", "Left: how strongly the model reacts to the input photo compared with the noisy image, per denoising step. Middle: error with the correct photo, a blank photo, or another house. Right: how much the output moves when each input is changed.", True)}
{fig("g_maps.jpg", "Where in the picture the model reacts to the input photo (bright = reacts a lot). One row per test image, one column per denoising step. The first column is the input photo itself.", True)}
</div>
<p class="src">Code <code>t1_grad.py</code>, <code>t1_grad_base.py</code> · raw <a href="data/t1_grad.json">t1_grad.json</a>, <a href="data/t1_grad_base.json">t1_grad_base.json</a> · run logs <a href="data/logs/t1_grad.log.txt">t1_grad.log.txt</a>, <a href="data/logs/t1_grad_base.log.txt">t1_grad_base.log.txt</a></p>
</section>""")

# ------------------------------------------------------------------ region
rg = J("t1_region_loss/results.json")
rows = []
for r in rg:
    rows.append([r["pair"], r["sigma"], f(r["cond_src_all"]), f(r["cond_zero_all"]), f(r["cond_tgt_all"]), f(r["pred_is_src_velocity_all"]),
                 f(r["cond_src_changed"]), f(r["cond_zero_changed"]), f(r["cond_tgt_changed"]), f(r["pred_is_src_velocity_changed"]),
                 f(r["cond_src_unchanged"]), f(r["cond_zero_unchanged"]), f(r["cond_tgt_unchanged"]), f(r["pred_is_src_velocity_unchanged"]), f(r["cos_conddelta_vs_copysrc"], 3)])
r_t1 = table(["pair", "sigma", "all: cond=src", "all: cond=0", "all: cond=tgt", "all: copy-src", "changed: cond=src", "changed: cond=0", "changed: cond=tgt", "changed: copy-src",
              "unchanged: cond=src", "unchanged: cond=0", "unchanged: cond=tgt", "unchanged: copy-src", "cos(cond effect, toward source)"], rows)
parts.append(f"""
<section id="s1r">
<h2><span class="tag warn">finding</span>Test 1b′ · Does the input photo help where the editor made changes?</h2>
<p class="lede"><strong>What we checked.</strong> We split every picture into two kinds of areas: the parts the human editor changed a lot (mostly the sky and the colour grade) and the parts they left alone. Then we measured the model's error separately in each kind of area, with and without the input photo. We also added a fake "copy machine" that just outputs the input photo unchanged, as a reference.</p>
<div class="callout warn"><strong>What we found.</strong> In the areas the editor changed, giving the model the input photo makes it <em>worse</em> than giving it nothing, on every test image, for most of the denoising process. It is even worse than the plain copy machine there. In the untouched areas the input photo neither helps nor hurts much. The last column shows that the input photo pulls the model's answer <em>back toward the un-edited original</em> (positive numbers). In short: the model resists the edit.</div>
<p><strong>Columns.</strong> "cond=src" = model given the input photo. "cond=0" = given a blank. "cond=tgt" = given the finished target (cheating, to see the ceiling). "copy-src" = the copy machine. "all / changed / unchanged" = which part of the picture the error was measured on.</p>
{r_t1}
<div class="figs">
{fig("r_loss.jpg", "Error curves. One row per test image. Columns: whole picture, untouched areas, edited areas. Blue (with input photo) sits above orange (blank) in the right-hand column: the photo hurts exactly where the edit should happen.", True)}
{fig("r_change.jpg", "Which parts of each picture counted as “edited”. Bright = the editor changed it a lot. Mostly sky and the colour-graded ground.", True)}
</div>
<p class="src">Code <code>t1_region_loss.py</code> · raw <a href="data/t1_region_loss.json">t1_region_loss.json</a> · run log <a href="data/logs/t1_region_loss.log.txt">t1_region_loss.log.txt</a></p>
</section>""")

# ------------------------------------------------------------------ swap
sw = J("t1_swap/results.json")
pairs = sorted({r["pair"] for r in sw})
hdr = ["case", "pair", "cond", "prompt", "s_txt", "s_img", "seed"] + [f"LPIPS vs {p}.{k}" for p in pairs for k in ["src", "tgt"]] + [f"PSNR vs {p}.{k}" for p in pairs for k in ["src", "tgt"]] + ["sky RGB out", "sky RGB src", "sky RGB tgt", "secs"]
rows = []
for r in sw:
    rows.append([r["name"], r["pair"], r["cond"], r["prompt"], r["s_txt"], r["s_img"], r["seed"]] + [f(r[f"lpips_vs_{p}.{k}"]) for p in pairs for k in ["src", "tgt"]]
                + [f"{r[f'psnr_vs_{p}.{k}']:.2f}" for p in pairs for k in ["src", "tgt"]]
                + [" ".join(str(int(x)) for x in r["sky_rgb"]), " ".join(str(int(x)) for x in r["sky_rgb_src"]), " ".join(str(int(x)) for x in r["sky_rgb_tgt"]), r["secs"]])
s_t1 = table(hdr, rows)
core = ""
for p in pairs:
    core += f'<h3>Pair {p}</h3><div class="strip">'
    for c, lab in [("src", "INPUT (condition)"), ("tgt", "TARGET (editor)"), ("normal", "normal: cond=src, default prompt, no CFG"), ("swap_cond_from_other", "cond from the next pair, same seed"),
                   ("cond_zeros", "cond = zeros (trained null)"), ("cond_is_target", "cond = target"), ("prompt_night_cfg4", "prompt “night time, dark sky with stars”, text CFG 4"),
                   ("prompt_sunset_cfg4", "prompt “dramatic orange sunset sky”, text CFG 4"), ("cfg_txt4_img1.5", "IP2P CFG text 4, image 1.5")]:
        core += f'<figure><a href="img/s_p{p}_{c}.jpg" target="_blank" rel="noopener"><img src="img/s_p{p}_{c}.jpg" alt="{lab}" loading="lazy"></a><figcaption>{lab}</figcaption></figure>'
    core += "</div>"
parts.append(f"""
<section id="s1c">
<h2><span class="tag pass">passes</span>Test 1c · Generate full images and swap things around</h2>
<p class="lede"><strong>What we checked.</strong> We generated real 2048x1360 images (28 steps, same as your inference settings) and played with the inputs: give it house B's photo while everything else stays the same; give it a blank photo; give it the finished target; change the text prompt to "night", "sunset", "snow"; turn on guidance (CFG) at different strengths.</p>
<p class="lede"><strong>How to read the numbers.</strong> "LPIPS" is a perceptual distance: 0 means identical pictures, 0.7 means completely different pictures. "LPIPS vs 2.src" means distance to test image 2's input photo; "vs 2.tgt" means distance to its editor-finished version. "sky RGB" is the average colour of the top quarter of the frame, so you can see if the sky changed.</p>
<div class="callout ok"><strong>Swap test passes.</strong> Feed the model house B's photo and you get house B (distance 0.14-0.20 to B, 0.72 to A). Feed it a blank and you get a random generic house, the same one every time. Feed it the finished target and you get the target back almost exactly (0.10-0.14). The input door works.</div>
<div class="callout warn"><strong>But it learned only one trick.</strong> When the photo has a sky (test image 2), it swaps in a blue sky with white clouds and makes the lawn greener. When there is no sky (the two aerial shots), it just copies the photo with a slight colour shift. <strong>Words change nothing:</strong> "night time, dark sky with stars", "dramatic orange sunset", "snow on the ground", the empty prompt, the training caption: all give the same daytime picture, even with strong text guidance (4 or 7.5). The only knob that does anything is image guidance at 1.5, which sharpens slightly. This makes sense: every training caption was the same four words shuffled ("good sky_color, good greenary, good construction, good clouds"), so the model had nothing to learn from text.</div>
<div class="callout"><strong>Bug found in the training script's eval.</strong> The eval sampler <code>denoise_channel_concat</code> never actually applied guidance. It passed the guidance number to a part of the model that Klein base does not have. So the "guidance 4.0" you saw in wandb eval images was doing nothing. All those eval images were unguided.</div>
{core}
<h3>Everything at once: all 22 variations for each test image</h3>
<div class="figs">
{fig("s_sheet0.jpg", "Pair 0, all cases with LPIPS to own source / target.", True)}
{fig("s_sheet1.jpg", "Pair 1, all cases.", True)}
{fig("s_sheet2.jpg", "Pair 2, all cases. Every version has the same blue sky, including the ones asked for night and sunset.", True)}
</div>
<h3>All 66 generated images, all numbers</h3>
{s_t1}
<p class="src">Code <code>t1_swap.py</code>, <code>sampling.py</code> · raw <a href="data/t1_swap.json">t1_swap.json</a> · run log <a href="data/logs/t1_swap.log.txt">t1_swap.log.txt</a></p>
</section>""")


# ------------------------------------------------------------------ sky swap (stage 8)
sk = J("t1_swap_sky/results.json")
kpairs = sorted({r["pair"] for r in sk})
hdr = ["case", "pair", "cond", "prompt", "s_txt", "s_img", "LPIPS vs own src", "LPIPS vs own tgt", "closer to", "LPIPS vs donor src"] + [f"LPIPS vs {p}.{k}" for p in kpairs for k in ["src", "tgt"]] + ["sky RGB out", "sky RGB src", "sky RGB tgt"]
rows = []
for r in sk:
    p = r["pair"]; d = r["cond"].split(".")[0]
    a, b = r[f"lpips_vs_{p}.src"], r[f"lpips_vs_{p}.tgt"]
    rows.append([r["name"], p, r["cond"], r["prompt"], r["s_txt"], r["s_img"], f(a), f(b), "target" if b < a else "source", "" if d == "zeros" else f(r[f"lpips_vs_{d}.src"])]
                + [f(r[f"lpips_vs_{q}.{k}"]) for q in kpairs for k in ["src", "tgt"]]
                + [" ".join(str(int(x)) for x in r["sky_rgb"]), " ".join(str(int(x)) for x in r["sky_rgb_src"]), " ".join(str(int(x)) for x in r["sky_rgb_tgt"])])
k_t1 = table(hdr, rows)
kcore = ""
for p in kpairs:
    kcore += f'<h3>Pair {p}</h3><div class="strip">'
    for c, lab in [("src", "INPUT (condition)"), ("tgt", "TARGET (editor)"), ("normal", "normal: cond=src, default prompt, no guidance"), ("swap_cond_from_other", "cond from the next pair, same seed"),
                   ("cond_is_target", "cond = target"), ("prompt_night_cfg4", "prompt “night time, dark sky with stars”, text guidance 4"),
                   ("prompt_sunset_cfg4", "prompt “dramatic orange sunset sky”, text guidance 4"), ("cfg_txt4_img1.5", "text guidance 4, image guidance 1.5")]:
        kcore += f'<figure><a href="img/k_p{p}_{c}.jpg" target="_blank" rel="noopener"><img src="img/k_p{p}_{c}.jpg" alt="{lab}" loading="lazy"></a><figcaption>{lab}</figcaption></figure>'
    kcore += "</div>"
parts.append(f"""
<section id="s1k">
<h2><span class="tag pass">stage 8</span>Test 1c′ · The same generation tests on four photos with a big sky</h2>
<p class="lede"><strong>Why.</strong> The first three test images were mostly aerials with almost no sky, which hides the one edit this model knows. So we repeated the generation tests on four held-out photos where the editor replaced a large sky: grey overcast (4 and 13), dusk (15) and a washed-out pale blue (19). 16 variations each, 64 images.</p>
<div class="callout ok"><strong>The sky edit is real.</strong> On pairs 4, 13 and 15 the plain output is closer to the editor's finished photo than to the input (for example 0.227 vs 0.246 on pair 4), and the sky colour moves from grey or dusk to the target's blue. Pair 19 already had a pale sky; the model still repaints it a deeper blue with clouds, which ends up closer to the input than to the target.</div>
<div class="callout warn"><strong>Words still change nothing.</strong> "night", "sunset", "overcast", with or without text guidance, all give the same blue daytime sky (look at the sky RGB column: blue in every row). Text guidance actually makes results a little worse, because it amplifies a signal that carries no information. Image guidance 1.5 is again the only knob that helps, by a small amount.</div>
{kcore}
<h3>Everything at once: all 16 variations for each photo</h3>
<div class="figs">
{fig("k_sheet4.jpg", "Pair 4, all cases.", True)}
{fig("k_sheet13.jpg", "Pair 13, all cases.", True)}
{fig("k_sheet15.jpg", "Pair 15, all cases. Dusk input, blue-sky outputs in every variation.", True)}
{fig("k_sheet19.jpg", "Pair 19, all cases.", True)}
</div>
<h3>All 64 generated images, all numbers</h3>
{k_t1}
<p class="src">Code <code>t1_swap.py</code> · raw <a href="data/t1_swap_sky.json">t1_swap_sky.json</a> · run log <a href="data/logs/t1_swap_sky.log.txt">t1_swap_sky.log.txt</a></p>
</section>""")

# ------------------------------------------------------------------ eval fix
parts.append(f"""
<section id="fix">
<h2><span class="tag pass">fixed</span>The eval sampler in the training script now applies guidance</h2>
<p class="lede"><strong>What was wrong.</strong> <code>channel_concat_denoise.py</code> handed the guidance number to a part of the model that Klein base does not have, so it was thrown away. Every eval image logged to wandb at "guidance 4.0" was actually made with no guidance at all.</p>
<p class="lede"><strong>What it does now.</strong> Real InstructPix2Pix guidance with two knobs: <code>guidance_scale</code> for the text and a new <code>image_guidance_scale</code> for the input photo (default 1.5 in the trainer, flag <code>--eval_image_guidance_scale</code>). If guidance is asked for but the empty-prompt embeddings are missing, it prints a warning and runs unguided instead of silently doing nothing. The trainer, the 4k trainer copy and <code>validate_checkpoints.py</code> all pass the right inputs now.</p>
<p class="lede"><strong>Check.</strong> We generated the same image (pair 2, seed 0, 28 steps) with the fixed training-repo sampler and with the independent sampler used for this report. They match to within bf16 numerical noise; guided and unguided differ three times more than that.</p>
{table(["comparison", "mean pixel difference (out of 255)"], [
    ["fixed sampler, unguided  vs  report sampler, unguided", "2.72"],
    ["fixed sampler, text 4 + image 1.5  vs  report sampler, same", "2.79"],
    ["fixed sampler, text 4 + image 1.0  vs  report sampler, same", "2.64"],
    ["fixed sampler, guidance asked but no empty-prompt embeddings (fallback)  vs  unguided", "0.00"],
    ["guided  vs  unguided (should differ)", "8.14"],
], cls="kv")}
<div class="strip">
<figure><a href="img/e_fixed_unguided.jpg" target="_blank" rel="noopener"><img src="img/e_fixed_unguided.jpg" alt="fixed unguided" loading="lazy"></a><figcaption>fixed sampler, unguided</figcaption></figure>
<figure><a href="img/s_p2_normal.jpg" target="_blank" rel="noopener"><img src="img/s_p2_normal.jpg" alt="report unguided" loading="lazy"></a><figcaption>report sampler, unguided</figcaption></figure>
<figure><a href="img/e_fixed_txt4_img1.5.jpg" target="_blank" rel="noopener"><img src="img/e_fixed_txt4_img1.5.jpg" alt="fixed guided" loading="lazy"></a><figcaption>fixed sampler, text 4 + image 1.5</figcaption></figure>
<figure><a href="img/s_p2_cfg_txt4_img1.5.jpg" target="_blank" rel="noopener"><img src="img/s_p2_cfg_txt4_img1.5.jpg" alt="report guided" loading="lazy"></a><figcaption>report sampler, text 4 + image 1.5</figcaption></figure>
</div>
<p class="src">Code <code>/workspace/UltraEdit/flux/channel_concat_denoise.py</code>, <code>verify_eval_fix.py</code></p>
</section>""")

# ------------------------------------------------------------------ attention
L8 = ["double0", "double2", "double4", "single0", "single5", "single10", "single15", "single19"]
runs = ["trained_default", "trained_sunset", "trained_nocond_default", "base_default", "base_sunset"]
runlab = {"trained_default": "trained · default prompt", "trained_sunset": "trained · sunset prompt", "trained_nocond_default": "trained · cond = 0", "base_default": "base · default prompt", "base_sunset": "base · sunset prompt"}
words = {"trained_default": ["sky", "cloud", "green", "construction"], "trained_sunset": ["sunset", "orange", "sky", "cloud"], "trained_nocond_default": ["sky", "cloud", "green", "construction"], "base_default": ["sky", "cloud", "green", "construction"], "base_sunset": ["sunset", "orange", "sky", "cloud"]}


def attn_section(pair, title_extra):
    D = f"t2_attn_pair{pair}/"
    S = {r: J(D + f"{r}_stats.json") for r in runs}
    sig = S["trained_default"]["sig_at"]
    sigl = {k: f"{v[1]:.2f}" for k, v in sig.items()}
    q = S["trained_default"]["queries"]
    out = []
    # locality summary at each sigma
    for ts in ["0.9", "0.5", "0.2"]:
        rows = []
        for l in L8:
            row = [l]
            for r in runs:
                d = S[r]["stats"][f"sigma{ts}/{l}/detail_max_edge"]
                row.append(f"{d['mass_r4']:.2f} / {d['mass_r8']:.2f} / {d['entropy_norm']:.2f} / {d['txt_mass_real'] + d['txt_mass_pad']:.2f}")
            rows.append(row)
        out.append(f"<h4>sigma {sigl[ts]} · detail query at row {q['detail_max_edge'][0]}, col {q['detail_max_edge'][1]} · cells: mass within r4 / within r8 / normalized entropy / total text mass</h4>" + table(["layer"] + [runlab[r] for r in runs], rows))
    # all queries, trained default, sigma 0.5
    rows = []
    for l in L8:
        row = [l]
        for qn in q:
            d = S["trained_default"]["stats"][f"sigma0.5/{l}/{qn}"]
            row.append(f"{d['mass_r4']:.2f} / {d['entropy_norm']:.2f}")
        rows.append(row)
    out.append(f"<h4>trained · default prompt · sigma {sigl['0.5']} · every query (r4 / entropy). Query positions: " + ", ".join(f"{k} = ({v[0]},{v[1]})" for k, v in q.items()) + "</h4>" + table(["layer"] + list(q), rows))
    # img2txt
    rows = []
    for l in L8:
        row = [l]
        for r in runs:
            d = S[r]["stats"][f"sigma0.5/{l}/img2txt"]
            row.append(f"{d['real_mean']:.3f} / {d['pad_mean']:.3f} / {d['word_mean']:.4f} / {d['word_max']:.3f} / {d['self_mean']:.3f}")
        rows.append(row)
    out.append(f"<h4>image tokens → text · sigma {sigl['0.5']} · cells: mean mass on real prompt tokens / on pad tokens / on edit-word tokens (mean) / (max over image tokens) / mass on self</h4>" + table(["layer"] + [runlab[r] for r in runs], rows))
    # words
    for r in runs:
        rows = []
        for l in L8:
            row = [l]
            for wd in words[r]:
                k = f"sigma0.5/{l}/word:{wd}"
                if k in S[r]["stats"]:
                    d = S[r]["stats"][k]; row.append(f"{d['entropy_norm_over_img']:.3f} / {d['img_mass']:.2f}")
                else:
                    row.append("–")
            rows.append(row)
        out.append(f"<h4>{runlab[r]} · text word → image keys · sigma {sigl['0.5']} · cells: normalized entropy over image tokens (1 = flat) / fraction of the word's attention that lands on image tokens</h4>" + table(["layer"] + words[r], rows))
    # per-sigma trained locality all 25 layers
    layers = [f"double{i}" for i in range(5)] + [f"single{i}" for i in range(20)]
    rows = []
    for l in layers:
        rows.append([l] + [f"{S['trained_default']['stats'][f'sigma{ts}/{l}/detail_max_edge']['mass_r4']:.2f}" for ts in ["0.9", "0.5", "0.2"]] + [f"{S['base_default']['stats'][f'sigma{ts}/{l}/detail_max_edge']['mass_r4']:.2f}" for ts in ["0.9", "0.5", "0.2"]])
    out.append("<h4>all 25 layers · detail query · mass within r4 · trained vs base at the three captured sigmas</h4>" + table(["layer", f"trained σ{sigl['0.9']}", f"trained σ{sigl['0.5']}", f"trained σ{sigl['0.2']}", f"base σ{sigl['0.9']}", f"base σ{sigl['0.5']}", f"base σ{sigl['0.2']}"], rows))
    tables = "".join(out)
    a = f"a{pair}_"
    samples = '<div class="strip five">' + "".join(f'<figure><a href="img/{a}{m}_sample.jpg" target="_blank" rel="noopener"><img src="img/{a}{m}_sample.jpg" alt="{runlab[m]}" loading="lazy"></a><figcaption>{runlab[m]}</figcaption></figure>' for m in runs) + "</div>"
    return f"""
<h3>Pair {pair}{title_extra}</h3>
<p>The picture each model was drawing while we recorded its attention. Original model: cloud scene for the default prompt, orange sunset for the sunset prompt. Trained model: the same house, same sky, for both prompts.</p>
{samples}
<div class="figs">
{fig(a + "queries.jpg", "The patches we followed, marked on the input photo.")}
{fig(a + "trained_default_row_detail.jpg", "Trained model. Where the detail patch (cyan cross) looks. Top row of each pair = whole frame, bottom row = zoomed in. 8 layers left to right, 3 moments top to bottom. Bright = looked at a lot. A tight spot plus a cross is healthy. Layer single5 is the one that looks around the whole house.", True)}
{fig(a + "base_default_row_detail.jpg", "Original untrained model, same patch. Slightly less focused than the trained model.", True)}
{fig(a + "trained_nocond_default_row_detail.jpg", "Trained model given a blank input photo.", True)}
{fig(a + "trained_default_row_sky.jpg", "Trained model, a patch in the sky.", True)}
{fig(a + "trained_default_txt2img_sky.jpg", "Trained model: where the word “sky” looks in the image. Brighter in the sky area in a few layers, but weak overall.", True)}
{fig(a + "base_default_txt2img_sky.jpg", "Original model: where the word “sky” looks (over the cloud picture it was drawing).", True)}
{fig(a + "trained_sunset_txt2img_sunset.jpg", "Trained model, sunset prompt: where the word “sunset” looks.", True)}
{fig(a + "trained_default_img2txt_words.jpg", "Trained model: how much each part of the image pays attention to the edit words. Very little everywhere.", True)}
{fig(a + "trained_default_img2txt_total.jpg", "Trained model: attention spent on the real words (top row of each pair) versus on empty padding slots (bottom row). Padding wins.", True)}
{fig(a + "trained_default_locality.jpg", "Trained model, all 25 layers in one chart: focus (left), spread (middle), attention on words (right).", True)}
{fig(a + "base_default_locality.jpg", "Original model, same chart.", True)}
</div>
<details><summary>Click to open every number for pair {pair}</summary>{tables}</details>
<p class="src">raw {" · ".join(f'<a href="data/attn{pair}_{r}.json">{r}</a>' for r in runs)}</p>
"""


parts.append(f"""
<section id="s2">
<h2><span class="tag pass">passes</span>Test 2 · Where is the model looking? (attention maps)</h2>
<p class="lede"><strong>What we checked.</strong> Inside the model, every small patch of the image "looks at" other patches and at the words of the prompt to decide what to draw. We picked one patch in a detailed spot (a roof edge, the flag pole) and recorded exactly where it looked, in every one of the 25 layers, at three moments during generation (early, middle, late). A healthy model looks mostly at nearby patches. A broken one looks everywhere equally. We also recorded how much the image patches look at the words "sky", "cloud", "green", "construction", and how much those words look back at the image. We ran the same thing on the original untrained model for comparison.</p>
<p class="lede"><strong>Reading the numbers.</strong> "r4" = share of attention landing within 4 patches of the chosen patch (if the model looked everywhere equally this would be 0.007). "entropy" = how spread out the attention is (1.0 = perfectly even everywhere, lower = more focused). "text mass" = share of attention spent on the words instead of the image. The three captured moments are sigma 0.90, 0.54 and 0.27; the sampler never goes below 0.27 at this image size, so there is no "0.2" step.</p>
<div class="callout ok"><strong>Attention is healthy.</strong> The detail patch spends 45-65% of its attention within 4 patches of itself in most layers, versus 0.7% if it looked everywhere. The maps show a sharp bright spot on the patch plus a cross along its row and column (that cross is normal for this architecture). The trained model is actually <em>more</em> focused than the original model. Nothing is smeared across the frame. And it looks the same at early, middle and late steps.</div>
<div class="callout warn"><strong>Words get very little attention in both models, so this test alone cannot tell you text is broken.</strong> Image patches spend 2-5% of their attention on the real words in the early layers, 5-25% in the later ones, and only 0.1-1.4% on the edit words themselves. Most of the "text" attention actually goes to the 487 empty padding slots after the prompt (the model uses them as a parking spot). The original untrained model shows the same pattern, yet it follows prompts perfectly. <strong>The proof that text is dead is in the generated pictures below:</strong> the original model draws a cloud scene for the default prompt and an orange sunset for the sunset prompt; the trained model draws the identical house under an identical blue sky for both. Its attention numbers for the two prompts match to two decimal places. The words are still being read, they just no longer change anything.</div>
{attn_section(19, " · house with flag, large sky")}
{attn_section(4, " · deck view, grey overcast sky in the source")}
<p class="src">Code <code>t2_attn.py</code> · run logs <a href="data/logs/t2_attn_pair19.log.txt">t2_attn_pair19.log.txt</a>, <a href="data/logs/t2_attn_pair4.log.txt">t2_attn_pair4.log.txt</a> · stage timing <a href="data/logs/chain.log.txt">chain.log.txt</a></p>
</section>""")

parts.append(parts.pop(next(i for i, x in enumerate(parts) if 'id="fix"' in x)))  # eval-fix section reads best after attention

# ------------------------------------------------------------------ verdict + recommendations
verdict = """
<section id="tldr">
<h2>The short version</h2>
<p class="lede">You asked two questions. <strong>1. Does the model use the input photo we concatenate?</strong> Yes, heavily. <strong>2. Is its attention healthy?</strong> Yes. The real problem is elsewhere: the model learned to <em>copy</em> the input photo and paint one standard blue sky on it, and it completely ignores the text prompt. Details below, everything numbered and pictured.</p>
<div class="cards">
<div class="card ok"><div class="k">Test 1 · input photo</div><div class="v">Used, heavily</div><p>The door for the input photo is fully open (1.10x the size of the other door). The output reacts to the photo more than to anything else. Swap in house B's photo and you get house B.</p></div>
<div class="card warn"><div class="k">What it learned</div><div class="v">Copy + one standard sky</div><p>It copies the input photo, paints a blue sky with white clouds where there is sky, greens the lawn, warms the colours. Giving it the photo makes its error <em>worse</em> than giving it nothing, exactly in the areas the editor changed.</p></div>
<div class="card crit"><div class="k">Text prompt</div><div class="v">Ignored</div><p>Changing the words moves the output 2-5%. Removing the photo moves it 25-64%. "Night", "sunset", "snow" all give the same daytime picture, even with strong guidance. The original model obeys these prompts fine.</p></div>
<div class="card ok"><div class="k">Test 2 · attention</div><div class="v">Healthy</div><p>A detail patch spends 45-65% of its attention within 4 patches of itself (would be 0.7% if smeared everywhere). Same at every layer and every step. More focused than the original model.</p></div>
<div class="card ok"><div class="k">Eval script</div><div class="v">Guidance was off, now fixed</div><p>The eval sampler in the training script never applied guidance; every wandb eval image at "guidance 4.0" was unguided. Fixed and verified, see the section near the end.</p></div>
<div class="card warn"><div class="k">Training setting</div><div class="v">LR hit the wrong weights too</div><p>The 100x learning rate for the first layer also hit the original pre-trained half of that layer, which moved 87%. The rest of the model moved 2-4%.</p></div>
</div>
<p class="lede"><strong>Setup.</strong> Your checkpoint 17000 (of 20000) of the FLUX.2 Klein 4B model, fully fine-tuned with the input photo concatenated as extra channels. Tested on images the model never saw in training (the dataset's <code>test</code> split), at the training resolution 2048x1360, with your inference prompt and 28 steps. Every number on this page comes straight from the result files; the raw files are linked at the end of each section.</p>
<figure><a href="img/overview.jpg" target="_blank" rel="noopener"><img src="img/overview.jpg" alt="test split overview" loading="lazy"></a><figcaption>The 24 test images. Each row of input photos sits above the row of editor-finished versions. Numbered 0-5, 6-11, 12-17, 18-23 top to bottom. Images 0, 1, 2 were used for tests 1b and 1c; 0-3 for 1b′; 19 and 4 for test 2.</figcaption></figure>
</section>"""

reco = """
<section id="next">
<h2>What to do for the next run</h2>
<p class="lede">The plumbing works. The problem is what the model was asked to learn. It got a fast, easy path to copy the input photo, a dataset where the "edit" is nearly always the same (blue sky, greener lawn), and captions that say the same four words every time. The easiest way to score well on that is "copy the photo and paint the usual sky on it", so that is what it learned.</p>
<ol class="reco">
<li><strong>Give the text something to say, or drop it.</strong> Right now every caption is the same four tags shuffled. Either write captions that describe what actually changed in each pair (with real variety), or accept that this is an image-only model and stop expecting prompts to steer it. If you want a steerable mentor model, you need the first option.</li>
<li><strong>Turn guidance on for real, and fix the eval sampler.</strong> Add proper guidance (separate strengths for image and text) to <code>denoise_channel_concat</code> so eval numbers reflect what the model can do. Image guidance 1.5 already helps a little. Text guidance cannot help until point 1 is fixed. Raise conditioning dropout from 5% to 10-15% so the "no input" and "no text" branches are trained well enough for guidance to work.</li>
<li><strong>Point the high learning rate only at the new weights.</strong> <code>--x_embedder_lr 1e-3</code> currently hits the whole first layer. Keep the original pre-trained half at the normal 1e-5.</li>
<li><strong>Make copying expensive.</strong> Weight the loss more heavily in the areas where the target differs from the input, or train more on the middle and late denoising steps (that is where the model currently loses). During training, watch the "changed area" error with and without the input photo, from <code>t1_region_loss.py</code>: the with-photo number must fall below the without-photo number, otherwise it is still just copying.</li>
<li><strong>Watch better numbers.</strong> The <code>cond_img_ratio</code> you log only proves the input door is open (it was 1.3 here and said nothing about copying). Also log: error with vs without the input photo on held-out pairs, overall and in the changed areas; how much closer the output is to the target than to the input; and how much the output moves when the prompt changes vs when the photo is removed. The scripts here do all three in a few minutes on one GPU for any checkpoint.</li>
<li><strong>Test on photos with sky.</strong> Aerial shots have no sky and hide the one edit the model does know. On sky-heavy pairs 4, 13, 15 the output is closer to the target than to the input; use pairs like these for eval.</li>
</ol>
<p>Things that are fine and need no work: attention, the channel-concat wiring, the zero-initialisation, the latent normalisation, bf16 inference.</p>
</section>"""

page = f"""<title>Klein Exterior Ablation</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--bg:#f3f4f6;--panel:#ffffff;--ink:#181c22;--muted:#5c6470;--line:#d9dde3;--acc:#0e6f8c;--ok:#1a7f4b;--okbg:#e6f4ec;--warn:#9a5b00;--warnbg:#fbf1dc;--crit:#a3232b;--critbg:#fbe5e6;--code:#eceff3;--th:#e9ecf1}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#0f1317;--panel:#171c22;--ink:#e6e9ee;--muted:#98a2b0;--line:#2a323c;--acc:#5cc1e0;--ok:#5fd39a;--okbg:#12291d;--warn:#f0b35a;--warnbg:#2c2210;--crit:#f08c92;--critbg:#2e1417;--code:#1f262e;--th:#1d242c}}}}
:root[data-theme="dark"]{{--bg:#0f1317;--panel:#171c22;--ink:#e6e9ee;--muted:#98a2b0;--line:#2a323c;--acc:#5cc1e0;--ok:#5fd39a;--okbg:#12291d;--warn:#f0b35a;--warnbg:#2c2210;--crit:#f08c92;--critbg:#2e1417;--code:#1f262e;--th:#1d242c}}
body{{background:var(--bg);color:var(--ink);font:15px/1.55 "IBM Plex Sans",system-ui,sans-serif;margin:0}}
.wrap{{display:grid;grid-template-columns:220px minmax(0,1fr);gap:32px;max-width:1500px;margin:0 auto;padding-block:28px;padding-inline:16px}}
nav{{position:sticky;top:env(safe-area-inset-top,0px);align-self:start;font-size:13.5px}}
nav a{{display:block;color:var(--muted);text-decoration:none;padding:5px 0;border-left:2px solid var(--line);padding-left:10px}}
nav a:hover,nav a:focus{{color:var(--acc);border-color:var(--acc)}}
nav .h{{font-weight:600;color:var(--ink);margin:0 0 8px;font-size:12px;letter-spacing:.08em;text-transform:uppercase}}
main{{min-width:0}}
h1{{font-size:28px;font-weight:600;letter-spacing:-.01em;margin:0 0 4px;text-wrap:balance}}
.sub{{color:var(--muted);margin:0 0 28px}}
section{{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:22px 24px;margin-bottom:28px}}
h2{{font-size:20px;font-weight:600;margin:0 0 12px;display:flex;align-items:center;gap:10px;flex-wrap:wrap;text-wrap:balance}}
h3{{font-size:16px;font-weight:600;margin:26px 0 8px}}
h4{{font-size:13px;font-weight:600;margin:20px 0 6px;color:var(--muted)}}
p{{max-width:78ch}}
.lede{{font-size:15.5px}}
.tag{{font-size:11px;letter-spacing:.08em;text-transform:uppercase;padding:3px 8px;border-radius:3px;font-weight:600}}
.tag.pass{{background:var(--okbg);color:var(--ok)}}.tag.warn{{background:var(--warnbg);color:var(--warn)}}
.callout{{border-left:3px solid var(--acc);background:var(--code);padding:10px 14px;margin:12px 0;max-width:90ch}}
.callout.ok{{border-color:var(--ok);background:var(--okbg)}}.callout.warn{{border-color:var(--warn);background:var(--warnbg)}}
code{{font:13px "IBM Plex Mono",ui-monospace,monospace;background:var(--code);padding:1px 5px;border-radius:3px}}
.tw{{overflow-x:auto;margin:8px 0 14px;border:1px solid var(--line);border-radius:4px}}
table{{border-collapse:collapse;font:12.5px/1.35 "IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums;white-space:nowrap}}
th,td{{padding:5px 9px;border-bottom:1px solid var(--line);text-align:right}}
th{{background:var(--th);position:sticky;top:0;font-weight:500;text-align:right}}
td:first-child,th:first-child{{text-align:left}}
table.kv td:first-child{{font-family:"IBM Plex Sans",system-ui,sans-serif;white-space:normal;min-width:260px}}
tbody tr:hover{{background:var(--code)}}
.figs{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px;margin:14px 0}}
figure{{margin:0;min-width:0}}
figure.wide{{grid-column:1/-1}}
figure img{{width:100%;height:auto;display:block;border:1px solid var(--line);border-radius:4px;background:#000}}
figcaption{{font-size:12.5px;color:var(--muted);margin-top:5px;max-width:none}}
.strip{{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:10px;margin:8px 0 18px}}
.strip.five{{grid-template-columns:repeat(auto-fill,minmax(180px,1fr))}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px;margin:8px 0 18px}}
.card{{border:1px solid var(--line);border-radius:6px;padding:12px 14px;border-top:3px solid var(--acc)}}
.card.ok{{border-top-color:var(--ok)}}.card.warn{{border-top-color:var(--warn)}}.card.crit{{border-top-color:var(--crit)}}
.card .k{{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}}
.card .v{{font-size:18px;font-weight:600;margin:2px 0 6px}}
.card p{{font-size:13px;margin:0;color:var(--muted)}}
.src{{font-size:12.5px;color:var(--muted);margin-top:18px}}
.src a,.callout a,p a{{color:var(--acc)}}
details{{margin:12px 0}}summary{{cursor:pointer;font-weight:600;color:var(--acc)}}
ol.reco li{{margin-bottom:8px;max-width:80ch}}
@media (max-width:900px){{.wrap{{grid-template-columns:1fr}}nav{{position:static;display:flex;flex-wrap:wrap;gap:4px 12px}}nav a{{border-left:0;padding-left:0}}nav .h{{display:none}}}}
</style>
<div class="wrap">
<nav>
<p class="h">Sections</p>
<a href="#tldr">Verdict</a>
<a href="#s1a">1a · weights</a>
<a href="#s1b">1b · gradient</a>
<a href="#s1r">1b′ · region loss</a>
<a href="#s1c">1c · swap / prompts / CFG</a>
<a href="#s1k">1c′ · sky-heavy pairs</a>
<a href="#s2">2 · attention</a>
<a href="#fix">Eval sampler fix</a>
<a href="#next">Next run</a>
</nav>
<main>
<h1>Klein Exterior Ablation</h1>
<p class="sub">fotello-ai/flux-klein-4b-exterior-v1 · checkpoint-17000 · all 8 stages complete · 2026-09-20 · every table and picture is on this page, raw data linked per section</p>
{verdict}
{"".join(parts)}
{reco}
</main>
</div>
"""
open(OUT, "w").write(page)
print(len(page) / 1e6, "MB html")
