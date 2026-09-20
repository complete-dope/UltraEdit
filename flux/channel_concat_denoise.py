import logging

import numpy as np
import torch
from diffusers import Flux2KleinPipeline
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu

logger = logging.getLogger(__name__)
_warned_no_negative = False


@torch.no_grad()
def denoise_channel_concat(
    transformer,
    vae,
    scheduler,
    cond_latents,
    prompt_embeds,
    text_ids,
    latents_bn_mean,
    latents_bn_std,
    num_inference_steps=28,
    guidance_scale=None,
    image_guidance_scale=1.0,
    negative_prompt_embeds=None,
    generator=None,
    device=None,
    dtype=None,
):
    """Sampling loop for --channel_concat_cond models.

    The stock Flux2KleinPipeline appends cond latents as extra tokens (dim=1). A channel-concat
    model has a widened x_embedder and instead expects them on the channel dim (-1), so it needs
    its own loop. Mirrors flow_matching_loss's input prep exactly.

    Guidance. Klein *base* has no guidance embedding (guidance_embeds=False), so `guidance_scale`
    is applied as real classifier-free guidance, InstructPix2Pix style, using the two null
    branches the trainer learns through --conditioning_dropout_prob:

        e = e(0, 0) + image_guidance_scale * (e(cI, 0) - e(0, 0))
                    + guidance_scale       * (e(cI, cT) - e(cI, 0))

    image-null = zero latents (as in training), text-null = `negative_prompt_embeds` (the empty
    prompt). With image_guidance_scale == 1 the e(0,0) branch cancels and only two forwards run.
    For guidance-distilled variants (guidance_embeds=True) the scale is fed to the embedding as before.
    """
    global _warned_no_negative
    device = device or transformer.device
    dtype = dtype or torch.bfloat16

    cond = Flux2KleinPipeline._patchify_latents(cond_latents)
    cond = (cond - latents_bn_mean) / latents_bn_std
    packed_cond = Flux2KleinPipeline._pack_latents(cond).to(device=device, dtype=dtype)

    latents = torch.randn(cond.shape, generator=generator, device="cpu").to(device=device, dtype=dtype)
    packed = Flux2KleinPipeline._pack_latents(latents)
    img_ids = Flux2KleinPipeline._prepare_latent_ids(cond).to(device=device)

    # Flux2's scheduler uses dynamic shifting, so it needs mu derived from the image sequence length
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    if getattr(scheduler.config, "use_flow_sigmas", False):
        sigmas = None
    mu = compute_empirical_mu(image_seq_len=packed.shape[1], num_steps=num_inference_steps)
    scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)

    has_guidance_embed = getattr(transformer.config, "guidance_embeds", False)
    guidance = None
    if has_guidance_embed and guidance_scale is not None:
        guidance = torch.full([packed.shape[0]], guidance_scale, device=device)

    s_txt = float(guidance_scale) if (guidance_scale is not None and not has_guidance_embed) else 1.0
    s_img = float(image_guidance_scale) if image_guidance_scale is not None else 1.0
    use_cfg = (s_txt != 1.0) or (s_img != 1.0)
    if use_cfg and negative_prompt_embeds is None:
        if not _warned_no_negative:
            logger.warning(
                "denoise_channel_concat: guidance_scale=%s / image_guidance_scale=%s requested but no "
                "negative_prompt_embeds given; sampling WITHOUT guidance.", guidance_scale, image_guidance_scale
            )
            _warned_no_negative = True
        use_cfg = False

    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    text_ids = text_ids.to(device=device)
    if use_cfg:
        negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)
        if negative_prompt_embeds.shape[0] != prompt_embeds.shape[0]:
            negative_prompt_embeds = negative_prompt_embeds.expand(prompt_embeds.shape[0], -1, -1)
    zero_cond = torch.zeros_like(packed_cond)

    def fwd(x, c, pe, t):
        return transformer(
            hidden_states=torch.cat([x, c], dim=-1),
            timestep=t.expand(x.shape[0]).to(x.dtype) / 1000,
            guidance=guidance,
            encoder_hidden_states=pe,
            txt_ids=text_ids,
            img_ids=img_ids,
            return_dict=False,
        )[0][:, : x.shape[1], :]

    for t in scheduler.timesteps:
        e_full = fwd(packed, packed_cond, prompt_embeds, t)
        if use_cfg:
            e_img = fwd(packed, packed_cond, negative_prompt_embeds, t)
            if s_img != 1.0:
                e_none = fwd(packed, zero_cond, negative_prompt_embeds, t)
                pred = e_none + s_img * (e_img - e_none) + s_txt * (e_full - e_img)
            else:
                pred = e_img + s_txt * (e_full - e_img)
        else:
            pred = e_full
        packed = scheduler.step(pred.to(torch.float32), t, packed.to(torch.float32), return_dict=False)[0].to(dtype)

    out = Flux2KleinPipeline._unpack_latents_with_ids(packed, img_ids)
    out = out * latents_bn_std + latents_bn_mean
    out = Flux2KleinPipeline._unpatchify_latents(out)
    return vae.decode(out.to(vae.dtype)).sample
