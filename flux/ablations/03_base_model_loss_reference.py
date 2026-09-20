# reference: the same flow-matching loss on the BASE model (text-to-image, no cond) so t1_grad losses have a baseline
import os, torch, json
from ablation_common import *
OUT = os.path.join(RESULTS, "t1_grad"); LOG = os.path.join(OUT, "log_base.txt"); open(LOG, "w").close()
torch.manual_seed(0)
pairs = load_pairs(n=3)
te = TextEncoder(); pe, tids = te(PROMPT); te.free()
vae, bn_mean, bn_std = load_vae()
lat = [norm_pack(encode_img(vae, p["tgt"]), bn_mean, bn_std) for p in pairs]
del vae; torch.cuda.empty_cache()
tr = load_transformer(os.path.join(BASE, "transformer"))
ids = latent_ids(lat[0]); res = []
with torch.no_grad():
    for pi, l in enumerate(lat):
        x = pack(l).float(); noise = torch.randn_like(x)
        for sigma in [0.95, 0.7, 0.5, 0.3, 0.1]:
            noisy = ((1 - sigma) * x + sigma * noise).to(DT)
            pred = tr(hidden_states=noisy, timestep=torch.tensor([sigma], device=DEV, dtype=DT), guidance=None, encoder_hidden_states=pe.to(DEV, DT), txt_ids=tids.to(DEV), img_ids=ids, return_dict=False)[0].float()
            loss = ((pred - (noise - x)) ** 2).mean().item()
            zero_loss = ((noise - x) ** 2).mean().item()
            r = dict(pair=pairs[pi]["idx"], sigma=sigma, base_t2i_loss=loss, loss_if_pred_zero=zero_loss); res.append(r); log(json.dumps(r), LOG)
save_json(res, os.path.join(OUT, "results_base.json")); log("DONE", LOG)
