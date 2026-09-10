import argparse, os, torch
from PIL import Image
from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline, Flux2Transformer2DModel

P = argparse.ArgumentParser()
P.add_argument("--base", default="/workspace/models/FLUX.2-klein-base-4B")
P.add_argument("--transformer", required=True, help="checkpoint-N/transformer from the run")
P.add_argument("--image", required=True)
P.add_argument("--prompt", default="good sky_color, good greenary, good construction, good clouds")
P.add_argument("--height", type=int, default=1360)
P.add_argument("--width", type=int, default=2048)
P.add_argument("--steps", type=int, default=28)
P.add_argument("--guidance_scale", type=float, default=4.0)
P.add_argument("--seed", type=int, default=42)
P.add_argument("--out", default="/workspace/runs/infer_out.png")
a = P.parse_args()

transformer = Flux2Transformer2DModel.from_pretrained(a.transformer, torch_dtype=torch.bfloat16)
pipe = Flux2KleinPipeline.from_pretrained(a.base, transformer=transformer, torch_dtype=torch.bfloat16)
pipe.to("cuda")

src = Image.open(a.image).convert("RGB").resize((a.width, a.height), Image.LANCZOS)
img = pipe(
    image=src,
    prompt=a.prompt,
    height=a.height,
    width=a.width,
    num_inference_steps=a.steps,
    guidance_scale=a.guidance_scale,
    generator=torch.Generator("cuda").manual_seed(a.seed),
).images[0]
os.makedirs(os.path.dirname(a.out), exist_ok=True)
img.save(a.out)
print("saved", a.out, img.size)
