import numpy as np
import torch
from diffusers import Flux2KleinPipeline
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu


# Its used to infer the concatenated channels  
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
    generator=None,
    device=None,
    dtype=None,
):
    """Sampling loop for --channel_concat_cond models.

    The stock Flux2KleinPipeline appends cond latents as extra tokens (dim=1). A channel-concat
    model has a widened x_embedder and instead expects them on the channel dim (-1), so it needs
    its own loop. Mirrors flow_matching_loss's input prep exactly.
    """
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
    guidance = None
    if getattr(transformer.config, "guidance_embeds", False) and guidance_scale is not None:
        guidance = torch.full([packed.shape[0]], guidance_scale, device=device)

    for t in scheduler.timesteps:
        model_in = torch.cat([packed, packed_cond], dim=-1)
        timestep = t.expand(packed.shape[0]).to(packed.dtype)
        pred = transformer(
            hidden_states=model_in,
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds.to(device=device, dtype=dtype),
            txt_ids=text_ids.to(device=device),
            img_ids=img_ids,
            return_dict=False,
        )[0]
        pred = pred[:, : packed.shape[1], :]
        packed = scheduler.step(pred.to(torch.float32), t, packed.to(torch.float32), return_dict=False)[0].to(dtype)

    out = Flux2KleinPipeline._unpack_latents_with_ids(packed, img_ids)
    out = out * latents_bn_std + latents_bn_mean
    out = Flux2KleinPipeline._unpatchify_latents(out)
    return vae.decode(out.to(vae.dtype)).sample
