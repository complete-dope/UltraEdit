import torch
from PIL import Image
from torchvision.transforms import functional as TF

try:
    import wandb
except ImportError:
    wandb = None


class InferenceTable:
    COLUMNS = [
        "step",
        "epoch",
        "sample",
        "prompt",
        "seed",
        "guidance_scale",
        "num_inference_steps",
        "source",
        "target",
        "prediction",
        "mse",
        "psnr",
        "lpips",
    ]

    def __init__(self, accelerator):
        self.tracker = None
        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                self.tracker = tracker
        self.rows = []
        self.strips = []

    @property
    def enabled(self):
        return self.tracker is not None and wandb is not None

    @staticmethod
    def _to_pil(img, value_range):
        if img is None:
            return None
        if isinstance(img, torch.Tensor):
            img = img.detach().float().cpu()
            if img.ndim == 4:
                img = img[0]
            if value_range == (-1, 1):
                img = img * 0.5 + 0.5
            return TF.to_pil_image(img.clamp(0, 1))
        return img

    @staticmethod
    def _strip(*pils, width=640):
        pils = [p for p in pils if p is not None]
        if not pils:
            return None
        h = max(1, int(width * pils[0].height / pils[0].width))
        out = Image.new("RGB", (width * len(pils), h), "white")
        for i, p in enumerate(pils):
            out.paste(p.resize((width, h)), (i * width, 0))
        return out

    @staticmethod
    def _to_image(img, value_range):
        if img is None:
            return None
        if isinstance(img, torch.Tensor):
            img = img.detach().float().cpu()
            if img.ndim == 4:
                img = img[0]
            if value_range == (-1, 1):
                img = img * 0.5 + 0.5
            img = TF.to_pil_image(img.clamp(0, 1))
        return wandb.Image(img)

    def add(
        self,
        *,
        step,
        epoch,
        sample,
        prompt,
        seed,
        guidance_scale,
        num_inference_steps,
        source,
        prediction,
        target=None,
        mse=None,
        psnr=None,
        lpips=None,
        source_range=(-1, 1),
        target_range=(-1, 1),
        prediction_range=(0, 1),
    ):
        if not self.enabled:
            return
        strip = self._strip(
            self._to_pil(source, source_range),
            self._to_pil(target, target_range),
            self._to_pil(prediction, prediction_range),
        )
        if strip is not None:
            cap = f"s{sample}"
            if psnr is not None:
                cap += f" psnr={psnr:.2f}"
            if lpips is not None:
                cap += f" lpips={lpips:.3f}"
            self.strips.append(wandb.Image(strip, caption=cap))
        self.rows.append(
            [
                step,
                epoch,
                sample,
                prompt,
                seed,
                guidance_scale,
                num_inference_steps,
                self._to_image(source, source_range),
                self._to_image(target, target_range),
                self._to_image(prediction, prediction_range),
                mse,
                psnr,
                lpips,
            ]
        )

    def log(self, key, step):
        if not self.enabled or not self.rows:
            return
        table = wandb.Table(columns=self.COLUMNS)
        for row in self.rows:
            table.add_data(*row)
        # wandb 0.30 drops media nested in a Table, so log the images as plain panels too
        payload = {key: table}
        for label, idx in (("source", 7), ("target", 8), ("prediction", 9)):
            imgs = [r[idx] for r in self.rows if r[idx] is not None]
            if imgs:
                payload[f"{key}/{label}"] = imgs
        if self.strips:
            # one panel with source | target | prediction side by side, easiest to eyeball across steps
            payload[f"{key}/compare"] = self.strips
        self.tracker.log(payload, step=step)
        self.rows = []
        self.strips = []
