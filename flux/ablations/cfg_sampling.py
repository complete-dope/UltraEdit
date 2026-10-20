import numpy as np, torch
from diffusers import Flux2KleinPipeline
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu
from ablation_common import DEV, DT, pack, latent_ids


@torch.no_grad()
def sample(transformer, scheduler, cond_p, pe, tids, steps=28, seed=0, pe_null=None, s_txt=1.0, s_img=1.0,
           step_hook=None, latents=None, drop_cond=False):
    """cond_p: normalized patchified cond (1,128,h,w). IP2P-style CFG when s_txt>1 or s_img>1.
    e = e(0,0) + s_img*(e(cI,0)-e(0,0)) + s_txt*(e(cI,cT)-e(cI,0)); s_img=1,s_txt=1 -> plain conditional."""
    packed_cond = pack(cond_p).to(DEV, DT)
    if drop_cond:
        packed_cond = torch.zeros_like(packed_cond)
    if latents is None:
        latents = torch.randn(cond_p.shape, generator=torch.Generator("cpu").manual_seed(seed)).to(DEV, DT)
    x = pack(latents)
    ids = latent_ids(cond_p)
    sigmas = np.linspace(1.0, 1 / steps, steps)
    mu = compute_empirical_mu(image_seq_len=x.shape[1], num_steps=steps)
    scheduler.set_timesteps(sigmas=sigmas, device=DEV, mu=mu)
    use_cfg = (s_txt != 1.0) or (s_img != 1.0)

    def fwd(xx, cc, p, t):
        return transformer(hidden_states=torch.cat([xx, cc], dim=-1), timestep=(t.expand(1) / 1000).to(DT), guidance=None,
                           encoder_hidden_states=p.to(DEV, DT), txt_ids=tids.to(DEV), img_ids=ids, return_dict=False)[0]

    for i, t in enumerate(scheduler.timesteps):
        if step_hook is not None:
            step_hook(i, float(t) / 1000)
        e_full = fwd(x, packed_cond, pe, t)
        if use_cfg:
            e_img = fwd(x, packed_cond, pe_null, t)
            e_none = fwd(x, torch.zeros_like(packed_cond), pe_null, t)
            pred = e_none + s_img * (e_img - e_none) + s_txt * (e_full - e_img)
        else:
            pred = e_full
        x = scheduler.step(pred.float(), t, x.float(), return_dict=False)[0].to(DT)
    out = Flux2KleinPipeline._unpack_latents_with_ids(x, ids)
    return out  # normalized patchified (1,128,h,w)


def unnorm_unpatch(out, bn_mean, bn_std):
    out = out.float() * bn_std + bn_mean
    return Flux2KleinPipeline._unpatchify_latents(out)
