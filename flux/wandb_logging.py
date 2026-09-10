import torch
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

    @property
    def enabled(self):
        return self.tracker is not None and wandb is not None

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
        self.tracker.log({key: table}, step=step)
        self.rows = []
