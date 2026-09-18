#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "diffusers @ git+https://github.com/huggingface/diffusers.git",
#     "torch>=2.0.0",
#     "accelerate>=0.31.0",
#     "transformers>=4.41.2",
#     "ftfy",
#     "tensorboard",
#     "Jinja2",
#     "peft>=0.11.1",
#     "sentencepiece",
#     "torchvision",
#     "datasets",
#     "bitsandbytes",
#     "prodigyopt",
# ]
# ///

import argparse
import copy
import itertools
import json
import logging
import hashlib
import math
import multiprocessing as mp
import os
import time
import random
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import HfApi, create_repo, upload_folder
from huggingface_hub.errors import EntryNotFoundError
from peft import LoraConfig, prepare_model_for_kbit_training, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

import diffusers
from diffusers import (
    AutoencoderKLFlux2,
    BitsAndBytesConfig,
    FlowMatchEulerDiscreteScheduler,
    Flux2KleinPipeline,
    Flux2Transformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
from diffusers.training_utils import (
    _collate_lora_metadata,
    _to_cpu_contiguous,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    find_nearest_bucket,
    free_memory,
    generate_aspect_ratio_buckets,
    get_fsdp_kwargs_from_accelerator,
    offload_models,
    parse_buckets_string,
    wrap_with_fsdp,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
    load_image,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import is_compiled_module


if getattr(torch, "distributed", None) is not None:
    import torch.distributed as dist

if is_wandb_available():
    import wandb

# Change it based on the codebase
_FLUX_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
if _FLUX_DIR not in sys.path:
    sys.path.insert(0, _FLUX_DIR)

from channel_concat_denoise import denoise_channel_concat
from wandb_logging import InferenceTable

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.41.0.dev0")

logger = get_logger(__name__)


def save_model_card(
    repo_id: str,
    images=None,
    base_model: str = None,
    instance_prompt=None,
    validation_prompt=None,
    repo_folder=None,
    fp8_training=False,
):
    widget_dict = []
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            widget_dict.append(
                {"text": validation_prompt if validation_prompt else " ", "output": {"url": f"image_{i}.png"}}
            )

    model_description = f"""
# Flux.2 [Klein] DreamBooth LoRA - {repo_id}

<Gallery />

## Model description

These are {repo_id} DreamBooth LoRA weights for {base_model}.

The weights were trained using [DreamBooth](https://dreambooth.github.io/) with the [Flux2 diffusers trainer](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_flux2.md).

FP8 training? {fp8_training}

## Trigger words

You should use `{instance_prompt}` to trigger the image generation.

## Download model

[Download the *.safetensors LoRA]({repo_id}/tree/main) in the Files & versions tab.

## Use it with the [🧨 diffusers library](https://github.com/huggingface/diffusers)

```py
from diffusers import AutoPipelineForText2Image
import torch
pipeline = AutoPipelineForText2Image.from_pretrained("black-forest-labs/FLUX.2", torch_dtype=torch.bfloat16).to('cuda')
pipeline.load_lora_weights('{repo_id}', weight_name='pytorch_lora_weights.safetensors')
image = pipeline('{validation_prompt if validation_prompt else instance_prompt}').images[0]
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation on loading LoRAs in diffusers](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters)

## License

Please adhere to the licensing terms as described [here](https://huggingface.co/black-forest-labs/FLUX.2/blob/main/LICENSE.md).
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="other",
        base_model=base_model,
        prompt=instance_prompt,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "text-to-image",
        "diffusers-training",
        "diffusers",
        "lora",
        "flux2",
        "flux2-diffusers",
        "template:sd-lora",
    ]

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


def log_validation(
    pipeline,
    args,
    accelerator,
    pipeline_args,
    epoch,
    torch_dtype,
    is_final_validation=False,
    global_step=None,
):
    args.num_validation_images = args.num_validation_images if args.num_validation_images else 1
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline = pipeline.to(dtype=torch_dtype)
    pipeline.enable_model_cpu_offload()
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None
    autocast_ctx = torch.autocast(accelerator.device.type) if not is_final_validation else nullcontext()

    images = []
    for _ in range(args.num_validation_images):
        with autocast_ctx:
            image = pipeline(
                image=pipeline_args["image"],
                prompt_embeds=pipeline_args["prompt_embeds"],
                negative_prompt_embeds=pipeline_args["negative_prompt_embeds"],
                generator=generator,
            ).images[0]
            images.append(image)

    phase_name = "test" if is_final_validation else "validation"
    table = InferenceTable(accelerator)
    for i, image in enumerate(images):
        table.add(
            step=global_step,
            epoch=epoch,
            sample=i,
            prompt=args.validation_prompt,
            seed=args.seed,
            guidance_scale=None,
            num_inference_steps=None,
            source=pipeline_args["image"],
            prediction=image,
        )
    table.log(f"{phase_name}/samples", step=global_step)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    free_memory()

    return images


def module_filter_fn(mod: torch.nn.Module, fqn: str):
    # don't convert the output module
    if fqn == "proj_out":
        return False
    # don't convert linear modules with weight dimensions not divisible by 16
    if isinstance(mod, torch.nn.Linear):
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
    return True


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--bnb_quantization_config_path",
        type=str,
        default=None,
        help="Quantization config in a JSON file that will be used to define the bitsandbytes quant config of the DiT.",
    )
    parser.add_argument(
        "--do_fp8_training",
        action="store_true",
        help="if we are doing FP8 training.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help=("A folder containing the training data. "),
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'image'.",
    )
    parser.add_argument(
        "--cond_image_column",
        type=str,
        default=None,
        help="Column in the dataset containing the condition image. Must be specified when performing I2I fine-tuning",
    )
    parser.add_argument(
        "--channel_concat_cond",
        action="store_true",
        help="InstructPix2Pix-style conditioning: widen x_embedder and concat the cond latents along the channel dim "
        "(pixel-aligned) instead of appending them as extra tokens.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="The column of the dataset containing the instance prompt for each image",
    )

    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")

    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=False,
        help="The prompt with identifier specifying the instance, e.g. 'photo of a TOK dog', 'in the style of TOK'",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        help="path to an image that is used during validation as the condition image to verify that the model is learning.",
    )
    parser.add_argument(
        "--skip_final_inference",
        default=False,
        action="store_true",
        help="Whether to skip the final inference step with loaded lora weights upon training completion. This will run intermediate validation inference if `validation_prompt` is provided. Specify to reduce memory.",
    )
    parser.add_argument(
        "--final_validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during a final validation to verify that the model is learning. Ignored if `--validation_prompt` is provided.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--local_dataset_path",
        type=str,
        default=None,
        help="Local dataset: a `save_to_disk` folder or an imagefolder with metadata.jsonl. Alternative to --dataset_name.",
    )
    parser.add_argument(
        "--original_image_height",
        type=int,
        default=None,
        help="Canonical source height. Images not already this size are resized to it before cropping.",
    )
    parser.add_argument("--original_image_width", type=int, default=None, help="Canonical source width.")
    parser.add_argument(
        "--train_height",
        type=int,
        default=None,
        help="Height of the window cropped (no resize) from each training pair. Overrides --resolution / bucketing.",
    )
    parser.add_argument("--train_width", type=int, default=None, help="Width of the training crop window.")
    parser.add_argument(
        "--validation_height",
        type=int,
        default=None,
        help="Height of the center crop used for the validation split. Defaults to --train_height.",
    )
    parser.add_argument("--validation_width", type=int, default=None, help="Width of the validation center crop.")
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=None,
        help="InstructPix2Pix CFG dropout. With p: drop text only for p, image only for p, both for p. "
        "Needed for image_guidance_scale at inference. 0.05 in the paper. Off when unset.",
    )
    parser.add_argument(
        "--val_split_ratio",
        type=float,
        default=0.0,
        help="Fraction of --dataset_name held out (seeded) for validation loss and image metrics. 0 disables eval.",
    )
    parser.add_argument("--val_split_seed", type=int, default=42, help="Seed used to carve out the validation split.")
    parser.add_argument(
        "--eval_steps", type=int, default=250, help="Run validation loss + image metrics every X steps."
    )
    parser.add_argument(
        "--num_eval_samples",
        type=int,
        default=16,
        help="Number of validation pairs to fully sample for MSE / PSNR / LPIPS. Val loss uses the whole split.",
    )
    parser.add_argument("--eval_inference_steps", type=int, default=28, help="Sampling steps for image metrics.")
    parser.add_argument("--eval_guidance_scale", type=float, default=4.0, help="Guidance scale for image metrics.")
    parser.add_argument(
        "--training_mode",
        type=str,
        default="full",
        choices=["lora", "full"],
        help="`lora` trains LoRA adapters on the transformer; `full` finetunes all transformer weights.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=4,
        help="LoRA alpha to be used for additional scaling.",
    )
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="Dropout probability for LoRA layers")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="flux-dreambooth-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--aspect_ratio_buckets",
        type=str,
        default=None,
        help=(
            "Aspect ratio buckets to use for training. Define as a string of 'h1,w1;h2,w2;...'. "
            "e.g. '1024,1024;768,1360;1360,768;880,1168;1168,880;1248,832;832,1248'. "
            "Requires --use_aspect_ratio_buckets. Images are resized to cover and cropped to the nearest "
            "listed bucket (smaller images are upscaled). When set, --resolution is ignored."
        ),
    )
    # sd3-pix2pix flag names, accepted as aliases for the flux column args
    parser.add_argument("--edit_prompt_column", type=str, default='caption', help="Alias for --caption_column.")
    parser.add_argument("--original_image_column", type=str, default='merged_img', help="Alias for --cond_image_column.")
    parser.add_argument("--edited_image_column", type=str, default='edited_img', help="Alias for --image_column.")
    parser.add_argument(
        "--resolution_height",
        type=int,
        default=1360,
        help="Target height. With --resolution_width, images are resized to cover and cropped to this exact "
        "size (one fixed bucket). Overrides --resolution.",
    )
    parser.add_argument("--resolution_width", type=int, default=2048, help="Target width, see --resolution_height.")
    parser.add_argument(
        "--input_height",
        type=int,
        default=None,
        help="Canvas height. With --input_width, every training pair is first resized to this canvas; random "
        "crops of --resolution_height x --resolution_width are then taken from it (see --random_crop_ratio).",
    )
    parser.add_argument("--input_width", type=int, default=None, help="Canvas width, see --input_height.")
    parser.add_argument(
        "--random_crop_ratio",
        type=float,
        default=0.0,
        help="Per-step probability of training on a random resolution-sized window of the canvas instead of "
        "the whole image resized to the resolution. Requires --input_height/--input_width.",
    )
    parser.add_argument(
        "--x_embedder_lr",
        type=float,
        default=None,
        help="Separate LR for x_embedder. With --channel_concat_cond its cond half is zero-init, so at the "
        "body LR it grows too slowly to ever use the cond image. Defaults to --learning_rate.",
    )
    parser.add_argument(
        "--transformer_dtype",
        type=str,
        default='fp32',
        choices=["fp32", "bf16", "fp16"],
        help="dtype the transformer weights are held in. Defaults to the --mixed_precision dtype. Use fp32 to "
        "keep master weights in fp32 while autocast still computes in bf16.",
    )
    parser.add_argument(
        "--use_aspect_ratio_buckets",
        action="store_true",
        help=(
            "Enable aspect-ratio bucketing. Without --aspect_ratio_buckets, the buckets are computed on the "
            "fly from --resolution and capped to each image's own resolution, so smaller images are assigned "
            "to a smaller bucket instead of being upscaled. Provide --aspect_ratio_buckets to use an explicit list."
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="the FLUX.1 dev variant is a guidance distilled model",
    )

    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )

    parser.add_argument(
        "--lora_layers",
        type=str,
        default=None,
        help=(
            'The transformer modules to apply LoRA training on. Please specify the layers in a comma separated. E.g. - "to_k,to_q,to_v,to_out.0" will result in lora training of attention layers only'
        ),
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--push_checkpoints_to_hub",
        action="store_true",
        help="Upload every `checkpoint-N` folder to `checkpoints/checkpoint-N` in --hub_model_id right after saving.",
    )
    parser.add_argument(
        "--hub_checkpoints_limit",
        type=int,
        default=5,
        help="Max checkpoints kept on the Hub. The oldest is deleted before the next one is uploaded.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--upcast_before_saving",
        action="store_true",
        default=False,
        help=(
            "Whether to upcast the trained transformer layers to float32 before saving (at the end of training). "
            "Defaults to precision dtype used for training to save memory"
        ),
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Whether to offload the VAE and the text encoder to CPU when they are not used.",
    )

    parser.add_argument(
        "--latent_dir",
        type=str,
        default=None,
        help="Directory with index.jsonl, encode_config.json and <split>/*.safetensors of pre-encoded, patchified, "
        "bn-normalized latents (keys `edited`, `merged`). When set the VAE is never loaded.",
    )
    parser.add_argument("--latent_split", type=str, default="train", help="Split of --latent_dir used for training.")
    parser.add_argument(
        "--latent_val_split",
        type=str,
        default="test",
        help="Split of --latent_dir used for validation; empty string disables eval.",
    )
    parser.add_argument("--max_train_samples", type=int, default=None, help="Use only the first N training latents.")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Use only the first N validation latents.")
    parser.add_argument(
        "--preload_latents", action="store_true", help="Read every training latent into RAM once instead of per step."
    )
    parser.add_argument(
        "--attention_backend",
        type=str,
        default=None,
        help="diffusers attention backend for the transformer, e.g. `xformers`, `flash`, `native`.",
    )
    parser.add_argument(
        "--compile_transformer", action="store_true", help="torch.compile every transformer block forward."
    )
    parser.add_argument("--compile_mode", type=str, default="default", help="torch.compile mode for the blocks.")
    parser.add_argument(
        "--tracker_project_name", type=str, default="dreambooth-flux2-image2img-lora", help="wandb project name."
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--enable_npu_flash_attention", action="store_true", help="Enabla Flash Attention for NPU")
    parser.add_argument("--fsdp_text_encoder", action="store_true", help="Use FSDP for text encoder")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    for alias, canonical in (
        ("edit_prompt_column", "caption_column"),
        ("original_image_column", "cond_image_column"),
        ("edited_image_column", "image_column"),
    ):
        if getattr(args, alias) is not None:
            setattr(args, canonical, getattr(args, alias))

    if args.local_dataset_path == "":
        args.local_dataset_path = None
    if args.dataset_name is not None and args.local_dataset_path is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--local_dataset_path`")
    if args.local_dataset_path is not None:
        # downstream code treats a local dataset exactly like a Hub one
        args.dataset_name = args.local_dataset_path

    if args.val_split_ratio > 0 and args.dataset_name is None:
        raise ValueError("--val_split_ratio requires --dataset_name or --local_dataset_path.")
    if not 0 <= args.val_split_ratio < 1:
        raise ValueError("--val_split_ratio must be in [0, 1).")

    for pair in (
        ("original_image_height", "original_image_width"),
        ("train_height", "train_width"),
        ("validation_height", "validation_width"),
    ):
        vals = [getattr(args, name) for name in pair]
        if (vals[0] is None) != (vals[1] is None):
            raise ValueError(f"--{pair[0]} and --{pair[1]} must be set together.")
    if args.validation_height is not None and args.train_height is None:
        raise ValueError("--validation_height/--validation_width require --train_height/--train_width.")
    for name in ("train_height", "train_width", "validation_height", "validation_width"):
        val = getattr(args, name)
        if val is not None and val % 16 != 0:
            raise ValueError(f"--{name} must be a multiple of 16, got {val}.")

    if (args.input_height is None) != (args.input_width is None):
        raise ValueError("--input_height and --input_width must be given together.")
    if not 0 <= args.random_crop_ratio <= 1:
        raise ValueError("--random_crop_ratio must be in [0, 1].")
    if args.random_crop_ratio > 0:
        if args.input_height is None or args.resolution_height is None:
            raise ValueError("--random_crop_ratio requires --input_height/--input_width and --resolution_height/--resolution_width.")
        if args.input_height < args.resolution_height or args.input_width < args.resolution_width:
            raise ValueError("--input_height/--input_width must be at least --resolution_height/--resolution_width.")
        if args.input_height % 8 or args.input_width % 8 or args.resolution_height % 16 or args.resolution_width % 16:
            raise ValueError("--input_* must be multiples of 8 and --resolution_* multiples of 16 for latent-space crops.")

    if args.channel_concat_cond and (args.validation_prompt or args.final_validation_prompt):
        raise ValueError(
            "--channel_concat_cond changes the transformer input layout; Flux2KleinPipeline validation still "
            "appends cond tokens along the sequence and would fail. Disable validation prompts for now."
        )
    if args.cond_image_column is None:
        raise ValueError(
            "you must provide --cond_image_column for image-to-image training. Otherwise please see Flux2 text-to-image training example."
        )
    else:
        assert args.image_column is not None
        assert args.caption_column is not None

    if args.latent_dir is not None:
        if args.instance_prompt is not None:
            raise ValueError("--instance_prompt is unused with --latent_dir; captions come from index.jsonl.")
        if args.dataset_name is not None or args.instance_data_dir is not None:
            raise ValueError("--latent_dir replaces --dataset_name/--local_dataset_path/--instance_data_dir.")
        if args.random_crop_ratio > 0:
            raise ValueError("--random_crop_ratio is a pixel-space crop and is not supported with --latent_dir.")
        if args.validation_prompt or args.final_validation_prompt:
            raise ValueError("Pixel-space validation prompts need the VAE; disable them with --latent_dir.")
        if args.latent_val_split == "":
            args.latent_val_split = None
    elif args.dataset_name is None and args.instance_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--instance_data_dir`")

    if args.dataset_name is not None and args.instance_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--instance_data_dir`")

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def sample_crop_params():
    """(canvas_w, canvas_h, top, left) of a random resolution-sized window on the canvas, or None to use the whole image.
    Offsets are 16px-aligned so the window maps onto whole 2x2 latent patches."""
    if args.random_crop_ratio > 0 and random.random() < args.random_crop_ratio:
        max_top = args.input_height - args.resolution_height
        max_left = args.input_width - args.resolution_width
        top = random.randrange(0, max_top + 1, 16)
        left = random.randrange(0, max_left + 1, 16)
        return (args.input_width, args.input_height, top, left)
    return None


def crop_window(x, crop):
    """Cut the crop window out of a (..., H, W) pixel tensor."""
    _, _, top, left = crop
    h, w = args.resolution_height, args.resolution_width
    return x[..., top : top + h, left : left + w].contiguous()


def to_pixels(x, device=None, dtype=None):
    if x.dtype == torch.uint8:
        return x.to(device=device, dtype=dtype or torch.float32).div_(127.5).sub_(1.0)
    return x.to(device=device, dtype=dtype)


_PAIR_POOL = {}


def _u8_chw(arr):
    return torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)


def _preprocess_pair(job):
    # Forked worker: decode, resize, crop, and write uint8 pixels into the shared memmaps.
    j, i = job
    p = _PAIR_POOL
    row = p["dataset"][i]
    image = exif_transpose(row[p["image_column"]]).convert("RGB")
    dest = exif_transpose(row[p["cond_column"]]).convert("RGB") if p["cond_column"] else None
    canonical, canvas_wh = p["canonical"], p["canvas_wh"]
    if canonical is not None and image.size != canonical:
        image = image.resize(canonical, Image.LANCZOS)
    if canonical is not None and dest is not None and dest.size != canonical:
        dest = dest.resize(canonical, Image.LANCZOS)
    if p["canvas"] is not None:
        p["canvas"][j] = np.asarray(image if image.size == canvas_wh else image.resize(canvas_wh, Image.LANCZOS))
        if dest is not None:
            p["canvas_cond"][j] = np.asarray(dest if dest.size == canvas_wh else dest.resize(canvas_wh, Image.LANCZOS))
    image, dest = DreamBoothDataset.paired_transform(
        None,
        image,
        dest_image=dest,
        size=p["target"],
        center_crop=p["center_crop"],
        random_flip=p["random_flip"],
        resize=p["resize"],
        to_tensor=False,
    )
    p["crops"][j] = np.asarray(image)
    if dest is not None:
        p["crops_cond"][j] = np.asarray(dest)
    return j


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        size=1024,
        repeats=1,
        center_crop=False,
        buckets=None,
        use_aspect_ratio_buckets=False,
        bucket_divisibility=16,
        bucket_base_resolutions=None,
        split="train",
        val_split_ratio=0.0,
        split_seed=42,
        random_flip=None,
        original_size=None,
        crop_size=None,
        input_size=None,
    ):
        self.size = size
        # (height, width) canvas for random crops; canvas tensors are only built while emit_canvas is set
        self.input_size = input_size
        self.emit_canvas = False
        self.resolution = size
        self.center_crop = center_crop
        self.random_flip = args.random_flip if random_flip is None else random_flip
        # (height, width). When crop_size is set the pair is cropped to it with no resize.
        self.original_size = original_size
        self.crop_size = crop_size
        # Fixed canvas + crop geometry: decode and resize in a process pool instead of one image at a time.
        self._fast = False

        self.instance_prompt = instance_prompt
        self.custom_instance_prompts = None

        # Explicit user-provided bucket list (or None). The concrete list of buckets actually used is
        # built from the data in `self.buckets` during preprocessing below.
        self._explicit_buckets = buckets
        self.use_aspect_ratio_buckets = use_aspect_ratio_buckets
        self.bucket_divisibility = bucket_divisibility
        self.bucket_base_resolutions = bucket_base_resolutions

        # if --dataset_name is provided or a metadata jsonl file is provided in the local --instance_data directory,
        # we load the training data using load_dataset
        if args.dataset_name is not None:
            try:
                from datasets import load_dataset, load_from_disk
            except ImportError:
                raise ImportError(
                    "You are trying to load your data using the datasets library. If you wish to train using custom "
                    "captions please install the datasets library: `pip install datasets`. If you wish to load a "
                    "local folder containing images only, specify --instance_data_dir instead."
                )
            # Downloading and loading a dataset from the hub.
            # See more about loading custom images at
            # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script
            if os.path.isfile(os.path.join(args.dataset_name, "dataset_info.json")) or os.path.isfile(
                os.path.join(args.dataset_name, "dataset_dict.json")
            ):
                dataset = load_from_disk(args.dataset_name)
                if not isinstance(dataset, dict):
                    dataset = {"train": dataset}
            else:
                dataset = load_dataset(
                    args.dataset_name,
                    args.dataset_config_name,
                    cache_dir=args.cache_dir,
                )
            if val_split_ratio > 0:
                splits = dataset["train"].train_test_split(test_size=val_split_ratio, seed=split_seed)
                dataset["train"] = splits["train" if split == "train" else "test"]
            # Preprocessing the datasets.
            column_names = dataset["train"].column_names

            # 6. Get the column names for input/target.
            if args.cond_image_column is not None and args.cond_image_column not in column_names:
                raise ValueError(
                    f"`--cond_image_column` value '{args.cond_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                )
            if args.image_column is None:
                image_column = column_names[0]
                logger.info(f"image column defaulting to {image_column}")
            else:
                image_column = args.image_column
                if image_column not in column_names:
                    raise ValueError(
                        f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                    )
            cond_image_column = args.cond_image_column
            fixed_bucket = self._explicit_buckets is not None and len(self._explicit_buckets) == 1
            self._fast = self.crop_size is not None or (fixed_bucket and not self.use_aspect_ratio_buckets)
            instance_images, cond_images = [], None
            if self._fast:
                self._preprocess_parallel(dataset["train"], image_column, cond_image_column, repeats)
            else:
                instance_images = dataset["train"][image_column]
                if cond_image_column is not None:
                    cond_images = [dataset["train"][i][cond_image_column] for i in range(len(dataset["train"]))]
                    assert len(instance_images) == len(cond_images)

            if args.caption_column is None:
                logger.info(
                    "No caption column provided, defaulting to instance_prompt for all images. If your dataset "
                    "contains captions/prompts for the images, make sure to specify the "
                    "column as --caption_column"
                )
                self.custom_instance_prompts = None
            else:
                if args.caption_column not in column_names:
                    raise ValueError(
                        f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                    )
                custom_instance_prompts = dataset["train"][args.caption_column]
                # create final list of captions according to --repeats
                self.custom_instance_prompts = []
                for caption in custom_instance_prompts:
                    self.custom_instance_prompts.extend(itertools.repeat(caption, repeats))
        else:
            self.instance_data_root = Path(instance_data_root)
            if not self.instance_data_root.exists():
                raise ValueError("Instance images root doesn't exists.")

            instance_images = [Image.open(path) for path in list(Path(instance_data_root).iterdir())]
            self.custom_instance_prompts = None

        self.instance_images = []
        self.cond_images = []
        for i, img in enumerate(instance_images):
            self.instance_images.extend(itertools.repeat(img, repeats))
            if args.dataset_name is not None and cond_images is not None:
                self.cond_images.extend(itertools.repeat(cond_images[i], repeats))

        self.pixel_values = []
        self.cond_pixel_values = []
        self.buckets = []
        bucket_to_idx = {}
        for i, image in enumerate(self.instance_images):
            image = exif_transpose(image)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            dest_image = None
            if self.cond_images and self.crop_size is not None:
                dest_image = exif_transpose(self.cond_images[i]).convert("RGB")
            elif self.cond_images:  # todo: take care of max area for buckets
                dest_image = self.cond_images[i]
                image_width, image_height = dest_image.size
                if image_width * image_height > 1024 * 1024:
                    dest_image = Flux2ImageProcessor._resize_to_target_area(dest_image, 1024 * 1024)
                    image_width, image_height = dest_image.size

                multiple_of = 2 ** (4 - 1)  # 2 ** (len(vae.config.block_out_channels) - 1), temp!
                image_width = (image_width // multiple_of) * multiple_of
                image_height = (image_height // multiple_of) * multiple_of
                image_processor = Flux2ImageProcessor()
                dest_image = image_processor.preprocess(
                    dest_image, height=image_height, width=image_width, resize_mode="crop"
                )
                # Convert back to PIL
                dest_image = dest_image.squeeze(0)
                if dest_image.min() < 0:
                    dest_image = (dest_image + 1) / 2
                dest_image = (torch.clamp(dest_image, 0, 1) * 255).byte().cpu()

                if dest_image.shape[0] == 1:
                    # Gray scale image
                    dest_image = Image.fromarray(dest_image.squeeze().numpy(), mode="L")
                else:
                    # RGB scale image: (C, H, W) -> (H, W, C)
                    dest_image = TF.to_pil_image(dest_image)

                dest_image = exif_transpose(dest_image)
                if not dest_image.mode == "RGB":
                    dest_image = dest_image.convert("RGB")

            if self.original_size is not None:
                canonical = (self.original_size[1], self.original_size[0])
                if image.size != canonical:
                    image = image.resize(canonical, Image.LANCZOS)
                if dest_image is not None and dest_image.size != canonical:
                    dest_image = dest_image.resize(canonical, Image.LANCZOS)

            width, height = image.size

            # Assign the image to a bucket.
            target = self.crop_size if self.crop_size is not None else self._bucket_for_image(height, width)
            if target not in bucket_to_idx:
                bucket_to_idx[target] = len(self.buckets)
                self.buckets.append(target)
            bucket_idx = bucket_to_idx[target]

            # based on the bucket assignment, define the transformations
            image, dest_image = self.paired_transform(
                image,
                dest_image=dest_image,
                size=target,
                center_crop=self.center_crop,
                random_flip=self.random_flip,
                resize=self.crop_size is None,
            )
            self.pixel_values.append((image, bucket_idx))
            if dest_image is not None:
                self.cond_pixel_values.append((dest_image, bucket_idx))

        if self._fast:
            target, n, has_cond = self._fast_layout
            self.buckets = [target]
            self.pixel_values = [(j, 0) for j in range(n)]
            self.cond_pixel_values = [(j, 0) for j in range(n)] if has_cond else []
            self.num_instance_images = n
        else:
            self.num_instance_images = len(self.instance_images)
        self._length = self.num_instance_images

    def __len__(self):
        return self._length

    def _preprocess_parallel(self, dataset, image_column, cond_column, repeats):
        # Pixels live as uint8 memmaps in shared memory: built once by local rank 0, mapped read-only by every
        # rank and DataLoader worker, and reused by later runs with the same dataset fingerprint and geometry.
        src = [i for i in range(len(dataset)) for _ in range(repeats)]
        n = len(src)
        target = self.crop_size if self.crop_size is not None else tuple(self._explicit_buckets[0])
        ch, cw = target
        canonical = None if self.original_size is None else (self.original_size[1], self.original_size[0])
        canvas_wh = None if self.input_size is None else (self.input_size[1], self.input_size[0])
        shapes = {"crops": (n, ch, cw, 3)}
        if cond_column:
            shapes["crops_cond"] = (n, ch, cw, 3)
        if canvas_wh is not None:
            shapes["canvas"] = (n, canvas_wh[1], canvas_wh[0], 3)
            if cond_column:
                shapes["canvas_cond"] = (n, canvas_wh[1], canvas_wh[0], 3)
        spec = [dataset._fingerprint, image_column, cond_column, repeats, target, canvas_wh, canonical,
                self.center_crop, self.random_flip, sorted(shapes.items())]
        key = hashlib.sha1(json.dumps(spec, default=str).encode()).hexdigest()[:16]
        cache_dir = Path(os.environ.get("PIXEL_CACHE_DIR", "/dev/shm/klein_pixel_cache")) / key
        done = cache_dir / "done"
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank == 0 and not done.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
            cache_dir.mkdir(parents=True)
            arrays = {
                k: np.lib.format.open_memmap(cache_dir / f"{k}.npy", mode="w+", dtype=np.uint8, shape=shp)
                for k, shp in shapes.items()
            }
            _PAIR_POOL.update(
                dataset=dataset,
                image_column=image_column,
                cond_column=cond_column,
                canonical=canonical,
                canvas_wh=canvas_wh,
                target=target,
                resize=self.crop_size is None,
                center_crop=self.center_crop,
                random_flip=self.random_flip,
                crops=arrays["crops"],
                crops_cond=arrays.get("crops_cond"),
                canvas=arrays.get("canvas"),
                canvas_cond=arrays.get("canvas_cond"),
            )
            workers = max(1, min(64, (os.cpu_count() or 1) - 8, n))
            with mp.get_context("fork").Pool(workers) as pool:
                for _ in tqdm(
                    pool.imap_unordered(_preprocess_pair, list(enumerate(src)), chunksize=2),
                    total=n,
                    desc=f"Building pixel cache {cache_dir} ({workers} procs)",
                ):
                    pass
            _PAIR_POOL.clear()
            for a in arrays.values():
                a.flush()
            del arrays
            done.touch()
        else:
            waited = 0
            while not done.exists():
                time.sleep(2)
                waited += 2
                if waited > 7200:
                    raise RuntimeError(f"Timed out waiting for pixel cache {cache_dir}")
        arrays = {k: np.load(cache_dir / f"{k}.npy", mmap_mode="r") for k in shapes}
        for k, shp in shapes.items():
            if arrays[k].shape != shp:
                raise RuntimeError(f"Pixel cache {cache_dir}/{k}.npy has shape {arrays[k].shape}, expected {shp}")
        self._crops = arrays["crops"]
        self._crops_cond = arrays.get("crops_cond")
        self._canvas = arrays.get("canvas")
        self._canvas_cond = arrays.get("canvas_cond")
        gb = sum(a.nbytes for a in arrays.values()) / 1024**3
        if local_rank == 0:
            print(f"Pixel cache ready: {cache_dir} ({n} pairs, {gb:.1f} GB shared)")
        self._fast_layout = (target, n, bool(cond_column))

    def _canvas_tensor(self, image):
        image = exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        canvas = (self.input_size[1], self.input_size[0])
        if image.size != canvas:
            image = image.resize(canvas, Image.LANCZOS)
        return TF.normalize(TF.to_tensor(image), [0.5], [0.5])

    def __getitem__(self, index):
        example = {}
        idx = index % self.num_instance_images
        example["index"] = idx
        if self._fast:
            example["instance_images"] = _u8_chw(self._crops[idx])
            example["bucket_idx"] = 0
            if self._crops_cond is not None:
                example["cond_images"] = _u8_chw(self._crops_cond[idx])
            if self.emit_canvas and self._canvas is not None:
                example["canvas_images"] = _u8_chw(self._canvas[idx])
                if self._canvas_cond is not None:
                    example["canvas_cond_images"] = _u8_chw(self._canvas_cond[idx])
        else:
            instance_image, bucket_idx = self.pixel_values[idx]
            example["instance_images"] = instance_image
            example["bucket_idx"] = bucket_idx
            if self.cond_pixel_values:
                dest_image, _ = self.cond_pixel_values[idx]
                example["cond_images"] = dest_image
            if self.emit_canvas and self.input_size is not None:
                example["canvas_images"] = self._canvas_tensor(self.instance_images[idx])
                if self.cond_images:
                    example["canvas_cond_images"] = self._canvas_tensor(self.cond_images[idx])

        if self.custom_instance_prompts:
            caption = self.custom_instance_prompts[index % self.num_instance_images]
            if caption:
                example["instance_prompt"] = caption
            else:
                example["instance_prompt"] = self.instance_prompt

        else:  # custom prompts were provided, but length does not match size of image dataset
            example["instance_prompt"] = self.instance_prompt

        return example

    def _bucket_for_image(self, height, width):
        # An explicit bucket list takes priority: pick the nearest, upscaling smaller images to cover it.
        if self._explicit_buckets is not None:
            return self._explicit_buckets[find_nearest_bucket(height, width, self._explicit_buckets)]
        # On-the-fly bucketing: cap the ladder to the image's own resolution so smaller images are
        # assigned to a smaller bucket rather than being upscaled (mirrors ostris' bucketing).
        if self.use_aspect_ratio_buckets:
            resolution = min(self.resolution, round((height * width) ** 0.5))
            ladder = generate_aspect_ratio_buckets(
                resolution,
                divisibility=self.bucket_divisibility,
                base_resolutions=self.bucket_base_resolutions,
            )
            return ladder[find_nearest_bucket(height, width, ladder)]
        # No bucketing: a single square bucket reproduces the fixed-size resize + crop.
        return (self.resolution, self.resolution)

    def paired_transform(
        self, image, dest_image=None, size=(224, 224), center_crop=False, random_flip=False, resize=True, to_tensor=True
    ):
        # Resize preserving aspect ratio so the image covers the bucket, then crop to the bucket size.
        # The same geometry is applied to the conditioning image so the pair stays aligned.
        target_height, target_width = size
        width, height = image.size
        if resize:
            scale = max(target_height / height, target_width / width)
            new_size = [round(height * scale), round(width * scale)]
            image = TF.resize(image, new_size, interpolation=transforms.InterpolationMode.BILINEAR)
            if dest_image is not None:
                dest_image = TF.resize(dest_image, new_size, interpolation=transforms.InterpolationMode.BILINEAR)
        elif height < target_height or width < target_width:
            raise ValueError(
                f"Image {width}x{height} is smaller than the crop {target_width}x{target_height}; "
                "set --original_image_height/--original_image_width or lower the crop size."
            )
        if dest_image is not None and dest_image.size != image.size:
            raise ValueError(f"Condition image {dest_image.size} does not match target image {image.size}.")
        if center_crop:
            image = TF.center_crop(image, size)
            if dest_image is not None:
                dest_image = TF.center_crop(dest_image, size)
        else:
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=size)
            image = TF.crop(image, i, j, h, w)
            if dest_image is not None:
                dest_image = TF.crop(dest_image, i, j, h, w)
        if random_flip and random.random() < 0.5:
            image = TF.hflip(image)
            if dest_image is not None:
                dest_image = TF.hflip(dest_image)
        if not to_tensor:
            return image, dest_image
        image = TF.normalize(TF.to_tensor(image), [0.5], [0.5])
        if dest_image is not None:
            dest_image = TF.normalize(TF.to_tensor(dest_image), [0.5], [0.5])
        return (image, dest_image) if dest_image is not None else (image, None)


def _stack_pixels(examples, key):
    # uint8 stays uint8 across the DataLoader hop (4x less shared memory); to_pixels() converts on device.
    stacked = torch.stack([example[key] for example in examples]).contiguous()
    return stacked if stacked.dtype == torch.uint8 else stacked.float()


def _base_batch(examples):
    return {
        "prompts": [example["instance_prompt"] for example in examples],
        "indices": [example["index"] for example in examples],
    }


def latent_collate_fn(examples):
    batch = _base_batch(examples)
    batch["latents"] = torch.stack([example["latents"] for example in examples]).contiguous()
    batch["cond_latents"] = torch.stack([example["cond_latents"] for example in examples]).contiguous()
    batch["keys"] = [example["key"] for example in examples]
    return batch


def pixel_collate_fn(examples):
    batch = _base_batch(examples)
    batch["pixel_values"] = _stack_pixels(examples, "instance_images")
    for key, out in (
        ("cond_images", "cond_pixel_values"),
        ("canvas_images", "canvas_pixel_values"),
        ("canvas_cond_images", "canvas_cond_pixel_values"),
    ):
        if key in examples[0]:
            batch[out] = _stack_pixels(examples, key)
    return batch


class BucketBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset: DreamBoothDataset,
        batch_size: int,
        drop_last: bool = False,
        shuffle_batches_each_epoch: bool = True,
    ):
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got drop_last={}".format(drop_last))

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle_batches_each_epoch = shuffle_batches_each_epoch

        # Group indices by bucket
        self.bucket_indices = [[] for _ in range(len(self.dataset.buckets))]
        for idx, (_, bucket_idx) in enumerate(self.dataset.pixel_values):
            self.bucket_indices[bucket_idx].append(idx)

        self.sampler_len = 0
        self.batches = []

        # Pre-generate batches for each bucket
        for indices_in_bucket in self.bucket_indices:
            # Shuffle indices within the bucket
            random.shuffle(indices_in_bucket)
            # Create batches
            for i in range(0, len(indices_in_bucket), self.batch_size):
                batch = indices_in_bucket[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue  # Skip partial batch if drop_last is True
                self.batches.append(batch)
                self.sampler_len += 1  # Count the number of batches

        if not self.shuffle_batches_each_epoch:
            # Shuffle the precomputed batches once to mix buckets while keeping
            # the order stable across epochs for step-indexed caches.
            random.shuffle(self.batches)

    def __iter__(self):
        if self.shuffle_batches_each_epoch:
            random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return self.sampler_len


class LatentDataset(Dataset):
    """Pre-encoded pairs from --latent_dir: `edited` (target) and `merged` (source), each already patchified to
    (C*4, H/16, W/16) and bn-normalized, exactly what flow_matching_loss feeds the transformer."""

    def __init__(self, latent_dir, split, max_samples=None, repeats=1, preload=False):
        self.root = Path(latent_dir)
        with open(self.root / "index.jsonl") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        rows = [r for r in rows if r["split"] == split]
        rows.sort(key=lambda r: r["key"])
        missing = [r["key"] for r in rows if not (self.root / r["file"]).exists()]
        if missing:
            logger.warning(f"{len(missing)} {split} latents listed in index.jsonl are not on disk; skipping them")
            rows = [r for r in rows if (self.root / r["file"]).exists()]
        if max_samples is not None:
            rows = rows[:max_samples]
        if not rows:
            raise ValueError(f"No `{split}` latents found under {latent_dir}")
        shapes = {tuple(r["edited_latent_shape"]) for r in rows} | {tuple(r["merged_latent_shape"]) for r in rows}
        if len(shapes) != 1:
            raise ValueError(f"All latents must share one shape for a single bucket, got {sorted(shapes)}")
        self.rows = rows
        self.latent_shape = shapes.pop()
        self.num_instance_images = len(rows)
        self._length = len(rows) * repeats
        uncaptioned = [r["key"] for r in rows if not r.get("caption")]
        if uncaptioned:
            raise ValueError(
                f"{len(uncaptioned)} `{split}` rows in index.jsonl have no `caption` "
                f"(e.g. {uncaptioned[:3]}); re-encode the latents with captions. "
                "--instance_prompt is not a fallback here."
            )
        captions = [r["caption"] for r in rows]
        self.custom_instance_prompts = [captions[i % len(rows)] for i in range(self._length)]
        # single bucket; BucketBatchSampler reads (payload, bucket_idx) pairs from pixel_values
        self.buckets = [tuple(self.latent_shape[1:])]
        self.pixel_values = [(i, 0) for i in range(self._length)]
        self.input_size = None
        self.emit_canvas = False
        self._cache = None
        if preload:
            with ThreadPoolExecutor(max_workers=16) as pool:
                self._cache = list(
                    tqdm(
                        pool.map(self._load, range(len(rows))),
                        total=len(rows),
                        desc=f"Preloading {split} latents",
                        disable=int(os.environ.get("LOCAL_RANK", 0)) != 0,
                    )
                )

    def _load(self, idx):
        tensors = load_file(str(self.root / self.rows[idx]["file"]))
        return tensors["edited"], tensors["merged"]

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        idx = index % self.num_instance_images
        edited, merged = self._cache[idx] if self._cache is not None else self._load(idx)
        return {
            "index": idx,
            "bucket_idx": 0,
            "latents": edited,
            "cond_latents": merged,
            "key": self.rows[idx]["key"],
            "instance_prompt": self.custom_instance_prompts[index],
        }


def latent_preview(latents, size=(512, 342)):
    """Cheap VAE-free visual of a normalized patchified latent: unpatchify and map the first 3 channels to RGB."""
    x = latents.detach().float().cpu()
    if x.ndim == 3:
        x = x.unsqueeze(0)
    x = Flux2KleinPipeline._unpatchify_latents(x)[0, :3]
    lo, hi = x.flatten(1).quantile(0.01, dim=1), x.flatten(1).quantile(0.99, dim=1)
    x = ((x - lo[:, None, None]) / (hi - lo).clamp_min(1e-6)[:, None, None]).clamp(0, 1)
    return TF.to_pil_image(x).resize(size, Image.BILINEAR)


class LatentEvalTable:
    COLUMNS = [
        "step",
        "epoch",
        "sample",
        "key",
        "prompt",
        "seed",
        "guidance_scale",
        "num_inference_steps",
        "source",
        "target",
        "prediction",
        "latent_mse",
        "latent_psnr",
        "latent_cosine",
        "val_loss",
    ]
    _tables = {}

    def __init__(self, accelerator):
        self.tracker = next((t for t in accelerator.trackers if t.name == "wandb"), None)
        self.rows, self.strips = [], []

    @property
    def enabled(self):
        return self.tracker is not None and is_wandb_available()

    def add(self, *, source, target, prediction, sample, latent_psnr, latent_mse, **cols):
        if not self.enabled:
            return
        pils = [latent_preview(source), latent_preview(target), latent_preview(prediction)]
        strip = Image.new("RGB", (sum(p.width for p in pils), pils[0].height), "white")
        for i, p in enumerate(pils):
            strip.paste(p, (i * p.width, 0))
        self.strips.append(wandb.Image(strip, caption=f"s{sample} mse={latent_mse:.4f} psnr={latent_psnr:.2f}"))
        row = dict(cols, sample=sample, latent_mse=latent_mse, latent_psnr=latent_psnr)
        row.update(source=wandb.Image(pils[0]), target=wandb.Image(pils[1]), prediction=wandb.Image(pils[2]))
        self.rows.append([row.get(c) for c in self.COLUMNS])

    def log(self, key, step):
        if not self.enabled or not self.rows:
            return
        table = self._tables.get(key)
        if table is None:
            table = wandb.Table(columns=self.COLUMNS, log_mode="INCREMENTAL")
            self._tables[key] = table
        for row in self.rows:
            table.add_data(*row)
        self.tracker.log({key: table, f"{key}_compare": self.strips}, step=step)
        self.rows, self.strips = [], []


def _block_forward(block, *args, **kwargs):
    return type(block).forward(block, *args, **kwargs)


def _block_forward_ckpt(block, *args, **kwargs):
    if not torch.is_grad_enabled():
        return type(block).forward(block, *args, **kwargs)
    return torch.utils.checkpoint.checkpoint(type(block).forward, block, *args, use_reentrant=False, **kwargs)


def compile_transformer_blocks(model, mode="default", checkpoint=False):
    # One compiled function shared by every block: dynamo guards on module structure, not identity, so
    # same-class blocks reuse a graph instead of recompiling 25 times. Activation checkpointing sits
    # INSIDE the compiled region (torchtitan order); checkpoint wrapped around a compiled forward breaks
    # recompute metadata checks. FSDP hooks stay outside, and block class names stay intact for FSDP's
    # transformer_layer_cls_to_wrap policy.
    import functools

    torch._dynamo.config.recompile_limit = max(getattr(torch._dynamo.config, "recompile_limit", 8), 64)
    compiled = torch.compile(_block_forward_ckpt if checkpoint else _block_forward, mode=mode, dynamic=False)
    n = 0
    for name in ("transformer_blocks", "single_transformer_blocks"):
        for block in getattr(model, name, []):
            block.forward = functools.partial(compiled, block)
            n += 1
    logger.info(f"torch.compile applied to {n} transformer blocks (mode={mode}, checkpoint_inside={checkpoint})")


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `hf auth login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )
    if args.do_fp8_training:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub or args.push_checkpoints_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    def remote_checkpoints(api):
        try:
            names = [
                Path(entry.path).name
                for entry in api.list_repo_tree(repo_id, path_in_repo="checkpoints")
                if Path(entry.path).name.startswith("checkpoint-")
            ]
        except EntryNotFoundError:
            names = []
        return sorted(names, key=lambda x: int(x.split("-")[1]))

    def strip_bin_states(api, name):
        """Only the newest Hub checkpoint keeps the .bin resume state; older ones keep safetensors."""
        stale = [
            entry.path
            for entry in api.list_repo_tree(repo_id, path_in_repo=f"checkpoints/{name}")
            if entry.path.endswith(".bin")
        ]
        if stale:
            logger.info(f"Dropping resume state from {repo_id}/checkpoints/{name}: {len(stale)} .bin files")
            api.delete_files(repo_id=repo_id, delete_patterns=stale, commit_message=f"Strip .bin state from {name}")

    def push_checkpoint_to_hub(save_path):
        api = HfApi()
        remote = remote_checkpoints(api)
        while args.hub_checkpoints_limit > 0 and len(remote) >= args.hub_checkpoints_limit:
            oldest = remote.pop(0)
            logger.info(f"Hub checkpoint limit reached, deleting {oldest} from {repo_id}")
            api.delete_folder(path_in_repo=f"checkpoints/{oldest}", repo_id=repo_id)
        name = Path(save_path).name
        logger.info(f"Uploading {name} to {repo_id}/checkpoints/{name}")
        upload_folder(
            repo_id=repo_id,
            folder_path=save_path,
            path_in_repo=f"checkpoints/{name}",
            commit_message=f"Training checkpoint {name}",
        )
        for older in remote:
            if older != name:
                strip_bin_states(api, older)

    # Load the tokenizers
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # held separately: fp32 master weights + bf16 autocast is the stable setup for a full finetune
    transformer_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}.get(
        args.transformer_dtype, weight_dtype
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        revision=args.revision,
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    use_latents = args.latent_dir is not None
    if use_latents:
        # Latents were encoded offline; the bn statistics come from the encoder's config, not the VAE weights.
        vae = None
        with open(os.path.join(args.latent_dir, "encode_config.json")) as f:
            encode_config = json.load(f)
        latents_bn_mean = torch.tensor(encode_config["bn_running_mean"]).view(1, -1, 1, 1).to(accelerator.device)
        latents_bn_std = torch.sqrt(
            torch.tensor(encode_config["bn_running_var"]).view(1, -1, 1, 1) + encode_config["batch_norm_eps"]
        ).to(accelerator.device)
        logger.info(
            f"Using pre-encoded latents from {args.latent_dir} (vae={encode_config.get('vae')}, "
            f"size={encode_config.get('size')}, normalization={encode_config.get('normalization')}); VAE not loaded."
        )
    else:
        print("---x--- FOUND NO LATENTS FOR THE IMAGES SO LOADING IN VAE MODEL AND ENCODING IT USING THAT VAE MODEL HERE")
        vae = AutoencoderKLFlux2.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=args.revision,
            variant=args.variant,
        )
        latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(accelerator.device)
        latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
            accelerator.device
        )

    quantization_config = None
    if args.bnb_quantization_config_path is not None:
        with open(args.bnb_quantization_config_path, "r") as f:
            config_kwargs = json.load(f)
            if "load_in_4bit" in config_kwargs and config_kwargs["load_in_4bit"]:
                config_kwargs["bnb_4bit_compute_dtype"] = weight_dtype
        quantization_config = BitsAndBytesConfig(**config_kwargs)

    transformer = Flux2Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
        quantization_config=quantization_config,
        torch_dtype=transformer_dtype,
    )
    if args.bnb_quantization_config_path is not None:
        transformer = prepare_model_for_kbit_training(transformer, use_gradient_checkpointing=False)

    def widen_x_embedder(model):
        # Zero-init the new cond half so step 0 behaves exactly like the pretrained model.
        old_proj = model.x_embedder
        new_in = 2 * old_proj.in_features  # 128 -> 256
        new_proj = torch.nn.Linear(new_in, old_proj.out_features, bias=False).to(
            device=old_proj.weight.device, dtype=old_proj.weight.dtype
        )
        with torch.no_grad():
            new_proj.weight.zero_()
            new_proj.weight[:, : old_proj.in_features].copy_(old_proj.weight)
        model.x_embedder = new_proj
        # out_channels defaults to in_channels, but proj_out is NOT widened -- record it
        # explicitly or the checkpoint cannot be reloaded with from_pretrained
        model.register_to_config(in_channels=new_in, out_channels=old_proj.in_features)
        logger.info(f"Widened x_embedder in_features {old_proj.in_features} -> {new_in} for channel concat") # register_to-config only writes metadata into the model's config dict, touches nothing in weight

    if args.channel_concat_cond:
        widen_x_embedder(transformer)

    text_encoder = Qwen3ForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder.requires_grad_(False)

    is_full_finetune = args.training_mode == "full"
    if is_full_finetune and args.bnb_quantization_config_path is not None:
        raise ValueError("Full finetuning is not supported with a bitsandbytes-quantized transformer.")

    # LoRA: only the adapter layers are trained. Full: every transformer weight is trained.
    transformer.requires_grad_(is_full_finetune)
    if vae is not None:
        vae.requires_grad_(False)  # Always FALSE, training / finetuning a VAE is out of scope here

    if args.enable_npu_flash_attention:
        if is_torch_npu_available():
            logger.info("npu flash attention enabled.")
            transformer.set_attention_backend("_native_npu")
        else:
            raise ValueError("npu flash attention requires torch_npu extensions and is supported only on npu device ")

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    to_kwargs = {"dtype": weight_dtype, "device": accelerator.device} if not args.offload else {"dtype": weight_dtype}
    # flux vae is stable in bf16 so load it in weight_dtype to reduce memory
    if vae is not None:
        vae.to(**to_kwargs)
    # we never offload the transformer to CPU, so we can just use the accelerator device
    transformer_to_kwargs = (
        {"device": accelerator.device}
        if args.bnb_quantization_config_path is not None
        else {"device": accelerator.device, "dtype": transformer_dtype}
    )

    is_fsdp = getattr(accelerator.state, "fsdp_plugin", None) is not None
    if not is_fsdp:
        transformer.to(**transformer_to_kwargs)

    if args.do_fp8_training:
        convert_to_float8_training(
            transformer, module_filter_fn=module_filter_fn, config=Float8LinearConfig(pad_inner_dim=True)
        )

    text_encoder.to(**to_kwargs)
    # Initialize a text encoding pipeline and keep it to CPU for now.
    text_encoding_pipeline = Flux2KleinPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=None,
        transformer=None,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=None,
        revision=args.revision,
    )

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    transformer_lora_config = None
    if not is_full_finetune:
        if args.lora_layers is not None:
            target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
        else:
            # target_modules = ["to_k", "to_q", "to_v", "to_out.0"] # just train transformer_blocks

            # train transformer_blocks and single_transformer_blocks
            target_modules = ["to_k", "to_q", "to_v", "to_out.0"] + [
                "to_qkv_mlp_proj",
                *[f"single_transformer_blocks.{i}.attn.to_out" for i in range(24)],
            ]

        # now we will add new LoRA weights the transformer layers
        transformer_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=target_modules,
            # the widened x_embedder has no pretrained weights for the cond half, train it fully
            modules_to_save=["x_embedder"] if args.channel_concat_cond else None,
        )
        transformer.add_adapter(transformer_lora_config)

    if args.attention_backend is not None:
        transformer.set_attention_backend(args.attention_backend)
        logger.info(f"transformer attention backend: {args.attention_backend}")
    if args.compile_transformer:
        if args.gradient_checkpointing:
            transformer.disable_gradient_checkpointing()
        compile_transformer_blocks(transformer, args.compile_mode, checkpoint=args.gradient_checkpointing)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_full_transformer(model, save_directory, state_dict=None):
        # FSDP hands us an already-gathered state dict, so bypass save_pretrained and write it directly
        if state_dict is None:
            model.save_pretrained(save_directory)
            return
        os.makedirs(save_directory, exist_ok=True)
        model.save_config(save_directory)
        save_file(_to_cpu_contiguous(state_dict), os.path.join(save_directory, "diffusion_pytorch_model.safetensors"))

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        transformer_cls = type(unwrap_model(transformer))

        # 1) Validate and pick the transformer model
        modules_to_save: dict[str, Any] = {}
        transformer_model = None

        for model in models:
            if isinstance(unwrap_model(model), transformer_cls):
                transformer_model = model
                modules_to_save["transformer"] = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        if transformer_model is None:
            raise ValueError("No transformer model found in 'models'")

        # 2) Optionally gather FSDP state dict once
        state_dict = accelerator.get_state_dict(model) if is_fsdp else None

        if is_full_finetune:
            if accelerator.is_main_process:
                if weights:
                    weights.pop()
                save_full_transformer(
                    unwrap_model(transformer_model), os.path.join(output_dir, "transformer"), state_dict=state_dict
                )
            return

        # 3) Only main process materializes the LoRA state dict
        transformer_lora_layers_to_save = None
        if accelerator.is_main_process:
            peft_kwargs = {}
            if is_fsdp:
                peft_kwargs["state_dict"] = state_dict

            transformer_lora_layers_to_save = get_peft_model_state_dict(
                unwrap_model(transformer_model) if is_fsdp else transformer_model,
                **peft_kwargs,
            )

            if is_fsdp:
                transformer_lora_layers_to_save = _to_cpu_contiguous(transformer_lora_layers_to_save)

            # make sure to pop weight so that corresponding model is not saved again
            if weights:
                weights.pop()

            Flux2KleinPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                **_collate_lora_metadata(modules_to_save),
            )

    def load_model_hook(models, input_dir):
        transformer_ = None

        if not is_fsdp:
            while len(models) > 0:
                model = models.pop()

                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    transformer_ = unwrap_model(model)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")
        else:
            transformer_ = Flux2Transformer2DModel.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="transformer",
            )
            # checkpoint was saved with the widened layer, rebuild it before loading weights
            if args.channel_concat_cond:
                widen_x_embedder(transformer_)
            if not is_full_finetune:
                transformer_.add_adapter(transformer_lora_config)

        if is_full_finetune:
            loaded = Flux2Transformer2DModel.from_pretrained(input_dir, subfolder="transformer")
            transformer_.load_state_dict(loaded.state_dict())
            del loaded
            free_memory()
            return

        lora_state_dict = Flux2KleinPipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [transformer_]
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # Optimization parameters
    if args.x_embedder_lr is not None:
        xe_params = [p for k, p in transformer.named_parameters() if p.requires_grad and "x_embedder" in k]
        body_params = [p for k, p in transformer.named_parameters() if p.requires_grad and "x_embedder" not in k]
        params_to_optimize = [
            {"params": body_params, "lr": args.learning_rate},
            {"params": xe_params, "lr": args.x_embedder_lr},
        ]
        logger.info(
            f"param groups: body={len(body_params)} tensors @ {args.learning_rate}, "
            f"x_embedder={len(xe_params)} tensors @ {args.x_embedder_lr}"
        )
    else:
        params_to_optimize = [{"params": transformer_trainable_parameters, "lr": args.learning_rate}]

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    # Resolve the bucketing mode. Bucketing must be enabled explicitly with --use_aspect_ratio_buckets;
    # a bucket list without that flag is an error. With the flag, an explicit --aspect_ratio_buckets list
    # drives assignment, otherwise buckets are computed on the fly inside the dataset. Without the flag a
    # single square bucket reproduces the fixed-size resize + crop.
    if args.aspect_ratio_buckets is not None and not args.use_aspect_ratio_buckets:
        raise ValueError("--aspect_ratio_buckets requires --use_aspect_ratio_buckets to be set.")
    if (args.resolution_height is None) != (args.resolution_width is None):
        raise ValueError("--resolution_height and --resolution_width must be given together.")
    if args.resolution_height is not None and args.aspect_ratio_buckets is not None:
        raise ValueError("Specify only one of --resolution_height/--resolution_width or --aspect_ratio_buckets.")
    if args.resolution_height is not None:
        buckets = [(args.resolution_height, args.resolution_width)]
        use_aspect_ratio_buckets = False
        logger.info(f"Using fixed resolution bucket: {buckets}")
    elif args.aspect_ratio_buckets is not None:
        buckets = parse_buckets_string(args.aspect_ratio_buckets)
        use_aspect_ratio_buckets = False
        logger.info(f"Using explicit aspect ratio buckets: {buckets}")
    elif args.use_aspect_ratio_buckets:
        buckets = None
        use_aspect_ratio_buckets = True
        logger.info(
            "No --aspect_ratio_buckets provided; auto-computing aspect ratio buckets on the fly from --resolution."
        )
    else:
        buckets = [(args.resolution, args.resolution)]
        use_aspect_ratio_buckets = False

    original_size = None
    if args.original_image_height is not None:
        original_size = (args.original_image_height, args.original_image_width)
    train_crop_size = None
    val_crop_size = None
    if args.train_height is not None:
        train_crop_size = (args.train_height, args.train_width)
        val_crop_size = train_crop_size
        if args.validation_height is not None:
            val_crop_size = (args.validation_height, args.validation_width)

    # Dataset and DataLoaders creation:
    val_dataset = None
    if use_latents:
        train_dataset = LatentDataset(
            args.latent_dir,
            args.latent_split,
            max_samples=args.max_train_samples,
            repeats=args.repeats,
            preload=args.preload_latents,
        )
        logger.info(
            f"Training on {train_dataset.num_instance_images} `{args.latent_split}` latents of shape "
            f"{train_dataset.latent_shape} ({train_dataset.latent_shape[1] * train_dataset.latent_shape[2]} tokens)."
        )
        if args.latent_val_split:
            val_dataset = LatentDataset(
                args.latent_dir,
                args.latent_val_split,
                max_samples=args.max_val_samples,
                preload=True,
            )
            logger.info(f"Validation on {len(val_dataset)} `{args.latent_val_split}` latents.")
    else:
        train_dataset = DreamBoothDataset(
            instance_data_root=args.instance_data_dir,
            instance_prompt=args.instance_prompt,
            size=args.resolution,
            repeats=args.repeats,
            center_crop=args.center_crop,
            buckets=buckets,
            use_aspect_ratio_buckets=use_aspect_ratio_buckets,
            split="train",
            val_split_ratio=args.val_split_ratio,
            split_seed=args.val_split_seed,
            original_size=original_size,
            crop_size=train_crop_size,
            input_size=(args.input_height, args.input_width) if args.random_crop_ratio > 0 else None,
        )
    if train_dataset.input_size is not None and len(train_dataset.buckets) != 1:
        raise ValueError("--random_crop_ratio needs a single fixed resolution bucket.")
    if args.val_split_ratio > 0 and not use_latents:
        val_dataset = DreamBoothDataset(
            instance_data_root=args.instance_data_dir,
            instance_prompt=args.instance_prompt,
            size=args.resolution,
            repeats=1,
            center_crop=True,
            buckets=buckets,
            use_aspect_ratio_buckets=use_aspect_ratio_buckets,
            split="val",
            val_split_ratio=args.val_split_ratio,
            split_seed=args.val_split_seed,
            random_flip=False,
            original_size=original_size,
            crop_size=val_crop_size,
        )
        logger.info(f"Validation split: {len(val_dataset)} pairs held out from training.")
    # Caches are keyed by dataset index (carried in batch["indices"]), so they survive dataloader
    # sharding across ranks and per-epoch reshuffling.
    precompute_prompts = bool(train_dataset.custom_instance_prompts)
    collate = latent_collate_fn if use_latents else pixel_collate_fn
    batch_sampler = BucketBatchSampler(
        train_dataset,
        batch_size=args.train_batch_size,
        drop_last=True,
        shuffle_batches_each_epoch=True,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate,
        num_workers=args.dataloader_num_workers,
    )

    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_sampler=BucketBatchSampler(
                val_dataset, batch_size=args.train_batch_size, drop_last=False, shuffle_batches_each_epoch=False
            ),
            collate_fn=collate,
            num_workers=args.dataloader_num_workers,
        )

    def compute_text_embeddings(prompt, text_encoding_pipeline):
        with torch.no_grad():
            prompt_embeds, text_ids = text_encoding_pipeline.encode_prompt(
                prompt=prompt, max_sequence_length=args.max_sequence_length
            )
        return prompt_embeds, text_ids

    # If no type of tuning is done on the text_encoder and custom instance prompts are NOT
    # provided (i.e. the --instance_prompt is used for all images), we encode the instance prompt once to avoid
    # the redundant encoding.
    if not train_dataset.custom_instance_prompts:
        with offload_models(text_encoding_pipeline, device=accelerator.device, offload=args.offload):
            instance_prompt_hidden_states, instance_text_ids = compute_text_embeddings(
                args.instance_prompt, text_encoding_pipeline
            )

    null_prompt_embeds = None
    if args.conditioning_dropout_prob is not None:
        with offload_models(text_encoding_pipeline, device=accelerator.device, offload=args.offload):
            null_prompt_embeds, _ = compute_text_embeddings("", text_encoding_pipeline)
        null_prompt_embeds = null_prompt_embeds.cpu()

    if args.validation_prompt is not None:
        validation_image = load_image(args.validation_image).convert("RGB")
        validation_kwargs = {"image": validation_image}
        with offload_models(text_encoding_pipeline, device=accelerator.device, offload=args.offload):
            validation_kwargs["prompt_embeds"], _text_ids = compute_text_embeddings(
                args.validation_prompt, text_encoding_pipeline
            )
            validation_kwargs["negative_prompt_embeds"], _text_ids = compute_text_embeddings(
                "", text_encoding_pipeline
            )

    # Init FSDP for text encoder
    if args.fsdp_text_encoder:
        fsdp_kwargs = get_fsdp_kwargs_from_accelerator(accelerator)
        text_encoder_fsdp = wrap_with_fsdp(
            model=text_encoding_pipeline.text_encoder,
            device=accelerator.device,
            offload=args.offload,
            limit_all_gathers=True,
            use_orig_params=True,
            fsdp_kwargs=fsdp_kwargs,
        )

        text_encoding_pipeline.text_encoder = text_encoder_fsdp
        dist.barrier()

    # If custom instance prompts are NOT provided (i.e. the instance prompt is used for all images),
    # pack the statically computed variables appropriately here. This is so that we don't
    # have to pass them to the dataloader.
    if not train_dataset.custom_instance_prompts:
        prompt_embeds = instance_prompt_hidden_states
        text_ids = instance_text_ids

    # Only text embeddings are cached. Image latents are never cached: every step VAE-encodes either the
    # full resized pair or a fresh pixel-space crop of the canvas.
    if precompute_prompts:
        prompt_embeds_cache = {}
        text_ids_cache = {}
        train_dataset.emit_canvas = False

        # Prompts pad to max_sequence_length, so encode each unique caption once instead of decoding
        # every image through the dataloader.
        all_prompts = [p or args.instance_prompt for p in train_dataset.custom_instance_prompts[: len(train_dataset)]]
        unique_prompts = sorted(set(all_prompts))
        unique_cache = {}
        with offload_models(
            text_encoding_pipeline, device=accelerator.device, offload=args.offload and not args.fsdp_text_encoder
        ):
            for i in tqdm(range(0, len(unique_prompts), args.sample_batch_size), desc="Caching prompt embeddings"):
                chunk = unique_prompts[i : i + args.sample_batch_size]
                with torch.no_grad():
                    prompt_embeds, text_ids = compute_text_embeddings(chunk, text_encoding_pipeline)
                prompt_embeds, text_ids = prompt_embeds.cpu(), text_ids.cpu()
                if text_ids.shape[0] != prompt_embeds.shape[0]:
                    text_ids = text_ids.unsqueeze(0).expand(prompt_embeds.shape[0], *text_ids.shape)
                for prompt, e, t in zip(chunk, prompt_embeds, text_ids):
                    unique_cache[prompt] = (e, t)
        for idx, prompt in enumerate(all_prompts):
            prompt_embeds_cache[idx], text_ids_cache[idx] = unique_cache[prompt]
        logger.info(f"Cached {len(unique_prompts)} unique prompts for {len(all_prompts)} images.")

        def gather(cache, indices):
            return torch.stack([cache[i] for i in indices]).to(accelerator.device)

    # The random crop is cut from canvas pixels each step, so emit them during training.
    train_dataset.emit_canvas = train_dataset.input_size is not None

    # Validation pairs are always cached (latents, embeddings, pixels) so eval never needs the VAE or text encoder.
    val_cache = []
    val_negative_prompt_embeds = None
    if val_dataloader is not None:
        with torch.no_grad():
            with offload_models(text_encoding_pipeline, device=accelerator.device, offload=args.offload):
                val_negative_prompt_embeds, _ = compute_text_embeddings("", text_encoding_pipeline)
                val_negative_prompt_embeds = val_negative_prompt_embeds.cpu()
            for batch in tqdm(val_dataloader, desc="Caching validation set"):
                if use_latents:
                    item = {
                        "latents": batch["latents"],
                        "cond_latents": batch["cond_latents"],
                        "prompts": batch["prompts"],
                        "keys": batch["keys"],
                    }
                else:
                    item = {
                        "pixel_values": to_pixels(batch["pixel_values"]),
                        "cond_pixel_values": to_pixels(batch["cond_pixel_values"]),
                        "prompts": batch["prompts"],
                    }
                    with offload_models(vae, device=accelerator.device, offload=args.offload):
                        item["latents"] = (
                            vae.encode(to_pixels(batch["pixel_values"], accelerator.device, vae.dtype))
                            .latent_dist.mode()
                            .cpu()
                        )
                        item["cond_latents"] = (
                            vae.encode(to_pixels(batch["cond_pixel_values"], accelerator.device, vae.dtype))
                            .latent_dist.mode()
                            .cpu()
                        )
                if train_dataset.custom_instance_prompts:
                    with offload_models(text_encoding_pipeline, device=accelerator.device, offload=args.offload):
                        embeds, ids = compute_text_embeddings(batch["prompts"], text_encoding_pipeline)
                else:
                    embeds = instance_prompt_hidden_states.repeat(len(batch["prompts"]), 1, 1)
                    ids = instance_text_ids.repeat(len(batch["prompts"]), 1, 1)
                item["prompt_embeds"], item["text_ids"] = embeds.cpu(), ids.cpu()
                val_cache.append(item)

    # move back to cpu before deleting to ensure memory is freed see: https://github.com/huggingface/diffusers/issues/11376#issue-3008144624
    text_encoding_pipeline = text_encoding_pipeline.to("cpu")
    del text_encoder, tokenizer
    free_memory()

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * accelerator.num_processes * num_update_steps_per_epoch
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )
    logger.info(
        "post-prepare optimizer groups: "
        + ", ".join(f"[{k}] lr={g['lr']} n={len(g['params'])}" for k, g in enumerate(optimizer.param_groups))
    )
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            # a run killed mid-save leaves a stub directory behind; it is not resumable
            dirs = [
                d
                for d in dirs
                if os.path.exists(os.path.join(args.output_dir, d, "pytorch_model_fsdp.bin"))
                or os.path.exists(
                    os.path.join(args.output_dir, d, "transformer", "diffusion_pytorch_model.safetensors")
                )
            ]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            ckpt_dir = os.path.join(args.output_dir, path)
            global_step = int(path.split("-")[1])
            has_opt = any(
                os.path.exists(os.path.join(ckpt_dir, f)) for f in ("optimizer.bin", "optimizer_0", "optimizer_0.bin")
            )
            if has_opt:
                accelerator.load_state(ckpt_dir)
            elif is_fsdp:
                # optimizer state was stripped to stay inside the disk quota: take the weights,
                # rebuild the moments from scratch and fast-forward the LR schedule
                logger.warning(f"{path} has no optimizer state; resuming weights-only")
                from accelerate.utils import load_fsdp_model

                load_fsdp_model(accelerator.state.fsdp_plugin, accelerator, transformer, ckpt_dir, 0)
                for _ in range(global_step):
                    lr_scheduler.step()
            else:
                accelerator.load_state(ckpt_dir)

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def flow_matching_loss(
        model_input, cond_model_input, prompt_embeds, text_ids, generator=None, conditioning_dropout_prob=None
    ):
        if not use_latents:
            model_input = Flux2KleinPipeline._patchify_latents(model_input)
            model_input = (model_input - latents_bn_mean) / latents_bn_std

            cond_model_input = Flux2KleinPipeline._patchify_latents(cond_model_input)
            cond_model_input = (cond_model_input - latents_bn_mean) / latents_bn_std

        if conditioning_dropout_prob is not None:
            # InstructPix2Pix schedule on one draw: text dropped for p < 2q, image dropped for q <= p < 3q.
            # Image null = zeros in normalized latent space; the inference pipeline must use the same null.
            bsz = model_input.shape[0]
            random_p = torch.rand(bsz, device=model_input.device)
            prompt_mask = (random_p < 2 * conditioning_dropout_prob).reshape(bsz, 1, 1)
            null_embeds = null_prompt_embeds.to(prompt_embeds.device, dtype=prompt_embeds.dtype).expand_as(prompt_embeds)
            prompt_embeds = torch.where(prompt_mask, null_embeds, prompt_embeds)
            image_keep = (random_p < conditioning_dropout_prob) | (random_p >= 3 * conditioning_dropout_prob)
            cond_model_input = cond_model_input * image_keep.to(cond_model_input.dtype).reshape(bsz, 1, 1, 1)

        model_input_ids = Flux2KleinPipeline._prepare_latent_ids(model_input).to(device=model_input.device)
        # Each batch element is an independent training sample with a single
        # conditional image. Generate temporal IDs for one sample and expand
        # across the batch, avoiding incorrect cross-sample temporal offsets.
        cond_model_input_ids = Flux2KleinPipeline._prepare_image_ids([cond_model_input[0:1]]).to(
            device=cond_model_input.device
        )
        cond_model_input_ids = cond_model_input_ids.expand(cond_model_input.shape[0], -1, -1)

        # Sample noise that we'll add to the latents
        noise_device = model_input.device if generator is None else generator.device
        noise = torch.randn(model_input.shape, generator=generator, device=noise_device, dtype=model_input.dtype)
        noise = noise.to(model_input.device)
        bsz = model_input.shape[0]

        # Sample a random timestep for each image
        # for weighting schemes where we sample timesteps non-uniformly
        u = compute_density_for_timestep_sampling(
            weighting_scheme=args.weighting_scheme,
            batch_size=bsz,
            logit_mean=args.logit_mean,
            logit_std=args.logit_std,
            mode_scale=args.mode_scale,
            generator=generator,
        )
        indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
        timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

        # Add noise according to flow matching.
        # zt = (1 - texp) * x + texp * z1
        sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

        # [B, C, H, W] -> [B, H*W, C]
        # concatenate the model inputs with the cond inputs
        packed_noisy_model_input = Flux2KleinPipeline._pack_latents(noisy_model_input)
        packed_cond_model_input = Flux2KleinPipeline._pack_latents(cond_model_input)
        orig_input_shape = packed_noisy_model_input.shape
        orig_input_ids_shape = model_input_ids.shape

        # concatenate the model inputs with the cond inputs
        if args.channel_concat_cond:
            assert packed_noisy_model_input.shape[1] == packed_cond_model_input.shape[1], (
                "channel concat needs cond and target latents of identical H x W"
            )
            packed_noisy_model_input = torch.cat([packed_noisy_model_input, packed_cond_model_input], dim=-1)
        else:
            packed_noisy_model_input = torch.cat([packed_noisy_model_input, packed_cond_model_input], dim=1)
            model_input_ids = torch.cat([model_input_ids, cond_model_input_ids], dim=1)

        # handle guidance
        if unwrap_model(transformer).config.guidance_embeds:
            guidance = torch.full([1], args.guidance_scale, device=accelerator.device)
            guidance = guidance.expand(model_input.shape[0])
        else:
            guidance = None

        # Predict the noise residual
        model_pred = transformer(
            hidden_states=packed_noisy_model_input,  # (B, image_seq_len, C)
            timestep=timesteps / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,  # B, text_seq_len, 4
            img_ids=model_input_ids,  # B, image_seq_len, 4
            return_dict=False,
        )[0]
        # pruning the condition information
        model_pred = model_pred[:, : orig_input_shape[1], :]
        model_input_ids = model_input_ids[:, : orig_input_ids_shape[1], :]

        model_pred = Flux2KleinPipeline._unpack_latents_with_ids(model_pred, model_input_ids)

        # these weighting schemes use a uniform timestep sampling
        # and instead post-weight the loss
        weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

        # flow matching loss
        target = noise - model_input

        # Compute regular loss.
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        loss = loss.mean()
        return loss

    @torch.no_grad()
    def denoise_latents(cond, prompt_embeds, text_ids, num_inference_steps, guidance_scale, generator):
        # Same input prep as flow_matching_loss, returns the normalized patchified latent prediction.
        from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu

        device, dtype = accelerator.device, weight_dtype
        cond = cond.to(device=device, dtype=dtype)
        packed_cond = Flux2KleinPipeline._pack_latents(cond)
        latents = torch.randn(cond.shape, generator=generator, device="cpu").to(device=device, dtype=dtype) # So here we concatenate channels here ( we are adding in channels from noise and adding in channels from the original image so that we can get the nice image out)
        packed = Flux2KleinPipeline._pack_latents(latents)
        img_ids = Flux2KleinPipeline._prepare_latent_ids(cond).to(device=device)
        cond_ids = Flux2KleinPipeline._prepare_image_ids([cond[0:1]]).to(device=device).expand(cond.shape[0], -1, -1)

        scheduler = copy.deepcopy(noise_scheduler)
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        if getattr(scheduler.config, "use_flow_sigmas", False):
            sigmas = None
        mu = compute_empirical_mu(image_seq_len=packed.shape[1], num_steps=num_inference_steps)
        scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
        guidance = None
        if unwrap_model(transformer).config.guidance_embeds and guidance_scale is not None:
            guidance = torch.full([packed.shape[0]], guidance_scale, device=device)

        for t in scheduler.timesteps:
            if args.channel_concat_cond:
                model_in, ids = torch.cat([packed, packed_cond], dim=-1), img_ids
            else:
                model_in, ids = torch.cat([packed, packed_cond], dim=1), torch.cat([img_ids, cond_ids], dim=1)
            pred = transformer(
                hidden_states=model_in,
                timestep=t.expand(packed.shape[0]).to(dtype) / 1000,
                guidance=guidance,
                encoder_hidden_states=prompt_embeds.to(device=device, dtype=dtype),
                txt_ids=text_ids.to(device=device),
                img_ids=ids,
                return_dict=False,
            )[0][:, : packed.shape[1], :]
            packed = scheduler.step(pred.float(), t, packed.float(), return_dict=False)[0].to(dtype)
        return Flux2KleinPipeline._unpack_latents_with_ids(packed, img_ids)

    def run_latent_eval(step, epoch):
        transformer.eval()
        device = accelerator.device
        table = LatentEvalTable(accelerator)
        generator = torch.Generator(device="cpu").manual_seed(args.seed if args.seed is not None else 0)
        autocast_ctx = torch.autocast(device.type) if device.type != "mps" else nullcontext()

        val_losses, per_sample_loss = [], {}
        with torch.no_grad(), autocast_ctx:
            for item in val_cache:
                for i in range(item["latents"].shape[0]):
                    loss = flow_matching_loss(
                        item["latents"][i : i + 1].to(device, dtype=weight_dtype),
                        item["cond_latents"][i : i + 1].to(device, dtype=weight_dtype),
                        item["prompt_embeds"][i : i + 1].to(device),
                        item["text_ids"][i : i + 1].to(device) if item["text_ids"].ndim == 3 else item["text_ids"].to(device),
                        generator=generator,
                    ).item()
                    val_losses.append(loss)
                    per_sample_loss[item["keys"][i]] = loss
        logs = {"val_loss": sum(val_losses) / len(val_losses)}

        mse_sum, psnr_sum, cos_sum, n_done = 0.0, 0.0, 0.0, 0
        # normalized latents dumped for the offline decode script (flux/4k/decode_eval_results.py)
        dump = {} if accelerator.is_main_process else None
        with torch.no_grad(), autocast_ctx:
            for item in val_cache:
                for i in range(item["latents"].shape[0]):
                    if n_done >= args.num_eval_samples:
                        break
                    target = item["latents"][i : i + 1].to(device, dtype=torch.float32)
                    pred = denoise_latents(
                        item["cond_latents"][i : i + 1],
                        item["prompt_embeds"][i : i + 1],
                        item["text_ids"][i : i + 1] if item["text_ids"].ndim == 3 else item["text_ids"],
                        num_inference_steps=args.eval_inference_steps,
                        guidance_scale=args.eval_guidance_scale,
                        generator=torch.Generator(device="cpu").manual_seed(n_done),
                    ).float()
                    mse = F.mse_loss(pred, target).item()
                    # latents are unit-variance normalized, so psnr uses the observed target range
                    peak = (target.max() - target.min()).item()
                    psnr = 10 * math.log10(peak**2 / max(mse, 1e-12))
                    cos = F.cosine_similarity(pred.flatten(), target.flatten(), dim=0).item()
                    mse_sum, psnr_sum, cos_sum = mse_sum + mse, psnr_sum + psnr, cos_sum + cos
                    table.add(
                        step=step,
                        epoch=epoch,
                        sample=n_done,
                        key=item["keys"][i],
                        prompt=item["prompts"][i],
                        seed=n_done,
                        guidance_scale=args.eval_guidance_scale,
                        num_inference_steps=args.eval_inference_steps,
                        source=item["cond_latents"][i],
                        target=item["latents"][i],
                        prediction=pred[0],
                        latent_mse=mse,
                        latent_psnr=psnr,
                        latent_cosine=cos,
                        val_loss=per_sample_loss[item["keys"][i]],
                    )
                    if dump is not None:
                        k = item["keys"][i]
                        dump[f"{k}/source"] = item["cond_latents"][i].to(torch.bfloat16).cpu().contiguous()
                        dump[f"{k}/target"] = item["latents"][i].to(torch.bfloat16).cpu().contiguous()
                        dump[f"{k}/prediction"] = pred[0].to(torch.bfloat16).cpu().contiguous()
                        dump[f"{k}/latent_mse"] = torch.tensor([mse])
                        dump[f"{k}/val_loss"] = torch.tensor([per_sample_loss[k]])
                    n_done += 1
        if dump:
            dump_dir = os.path.join(args.output_dir, "eval_latents")
            os.makedirs(dump_dir, exist_ok=True)
            prompts = {item["keys"][i]: item["prompts"][i] for item in val_cache for i in range(len(item["keys"]))}
            meta = {"step": str(step), "epoch": str(epoch), "prompts": json.dumps(prompts), "split": str(args.latent_val_split)}
            tmp = os.path.join(dump_dir, f"step-{step:06d}.safetensors.tmp")
            save_file(dump, tmp, metadata=meta)
            os.replace(tmp, tmp[: -len(".tmp")])
        if n_done:
            logs.update(
                {"val_latent_mse": mse_sum / n_done, "val_latent_psnr": psnr_sum / n_done, "val_latent_cosine": cos_sum / n_done}
            )

        try:
            xw = unwrap_model(transformer).x_embedder.weight
            xw = xw.to_local() if hasattr(xw, "to_local") else xw
            half = xw.shape[1] // 2
            img_mag = xw[:, :half].abs().mean()
            logs["cond_img_ratio"] = (xw[:, half:].abs().mean() / img_mag.clamp_min(1e-12)).item()
        except Exception as e:
            logger.warning(f"cond_img_ratio unavailable: {e}")

        logger.info(f"step {step} latent eval: " + ", ".join(f"{k}={v:.4f}" for k, v in logs.items()))
        accelerator.log(logs, step=step)
        table.log("eval_table/latent_samples", step=step)
        free_memory()
        transformer.train()

    def run_eval(step, epoch):
        if use_latents:
            return run_latent_eval(step, epoch)
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        transformer.eval()
        device = accelerator.device
        table = InferenceTable(accelerator)
        # fixed noise / timesteps so val_loss is comparable across evals
        generator = torch.Generator(device="cpu").manual_seed(args.seed if args.seed is not None else 0)
        autocast_ctx = torch.autocast(device.type) if device.type != "mps" else nullcontext()

        val_losses = []
        with torch.no_grad(), autocast_ctx:
            for item in val_cache:
                val_losses.append(
                    flow_matching_loss(
                        item["latents"].to(device, dtype=weight_dtype),
                        item["cond_latents"].to(device, dtype=weight_dtype),
                        item["prompt_embeds"].to(device),
                        item["text_ids"].to(device),
                        generator=generator,
                    ).item()
                )
        logs = {"val_loss": sum(val_losses) / len(val_losses)}

        pipeline = Flux2KleinPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=None,
            tokenizer=None,
            transformer=unwrap_model(transformer),
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )
        pipeline.vae.to(device)
        pipeline.set_progress_bar_config(disable=True)

        psnr = PeakSignalNoiseRatio(data_range=1.0, dim=(1, 2, 3), reduction="elementwise_mean").to(device)
        lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(device)
        mse_sum, n_done = 0.0, 0
        with torch.no_grad(), autocast_ctx:
            for item in val_cache:
                for i in range(item["pixel_values"].shape[0]):
                    if n_done >= args.num_eval_samples:
                        break
                    target = (item["pixel_values"][i : i + 1] * 0.5 + 0.5).clamp(0, 1).to(device)
                    cond_image = TF.to_pil_image((item["cond_pixel_values"][i] * 0.5 + 0.5).clamp(0, 1))
                    if args.channel_concat_cond:
                        # stock pipeline appends cond as tokens; this model wants them channel-wise
                        cond_lat = item["cond_latents"][i : i + 1].to(device, dtype=torch.float32)
                        # call the FSDP-wrapped module so params are unsharded; unwrapped weights are flat shards
                        decoded = denoise_channel_concat(
                            transformer=transformer,
                            vae=pipeline.vae,
                            scheduler=pipeline.scheduler,
                            cond_latents=cond_lat,
                            prompt_embeds=item["prompt_embeds"][i : i + 1],
                            text_ids=item["text_ids"][i : i + 1] if item["text_ids"].ndim == 3 else item["text_ids"],
                            latents_bn_mean=latents_bn_mean,
                            latents_bn_std=latents_bn_std,
                            num_inference_steps=args.eval_inference_steps,
                            guidance_scale=args.eval_guidance_scale,
                            generator=torch.Generator(device="cpu").manual_seed(n_done),
                            device=device,
                            dtype=weight_dtype,
                        )
                        pred = (decoded * 0.5 + 0.5).clamp(0, 1).to(device, dtype=torch.float32)
                    else:
                        pred = pipeline(
                            image=cond_image,
                            height=target.shape[-2],
                            width=target.shape[-1],
                            prompt_embeds=item["prompt_embeds"][i : i + 1].to(device),
                            negative_prompt_embeds=val_negative_prompt_embeds.to(device),
                            num_inference_steps=args.eval_inference_steps,
                            guidance_scale=args.eval_guidance_scale,
                            generator=torch.Generator(device=device).manual_seed(n_done),
                            output_type="pt",
                        ).images.to(device, dtype=torch.float32)
                    if pred.shape[-2:] != target.shape[-2:]:
                        logger.warning(f"eval pred {tuple(pred.shape[-2:])} != target {tuple(target.shape[-2:])}, resizing")
                        pred = F.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False)
                    sample_mse = F.mse_loss(pred, target).item()
                    sample_psnr = psnr(pred, target).item()
                    sample_lpips = lpips(pred, target).item()
                    mse_sum += sample_mse
                    table.add(
                        step=step,
                        epoch=epoch,
                        sample=n_done,
                        prompt=item["prompts"][i],
                        seed=n_done,
                        guidance_scale=args.eval_guidance_scale,
                        num_inference_steps=args.eval_inference_steps,
                        source=item["cond_pixel_values"][i],
                        target=item["pixel_values"][i],
                        prediction=pred,
                        mse=sample_mse,
                        psnr=sample_psnr,
                        lpips=sample_lpips,
                    )
                    n_done += 1
        if n_done:
            logs.update(
                {"val_mse": mse_sum / n_done, "val_psnr": psnr.compute().item(), "val_lpips": lpips.compute().item()}
            )

        try:
            xw = unwrap_model(transformer).x_embedder.weight
            xw = xw.to_local() if hasattr(xw, "to_local") else xw
            half = xw.shape[1] // 2
            img_mag = xw[:, :half].abs().mean()
            logs["cond_img_ratio"] = (xw[:, half:].abs().mean() / img_mag.clamp_min(1e-12)).item()
        except Exception as e:
            logger.warning(f"cond_img_ratio unavailable: {e}")

        logger.info(f"step {step} eval: " + ", ".join(f"{k}={v:.4f}" for k, v in logs.items()))
        accelerator.log(logs, step=step)
        table.log("eval_table/samples", step=step)

        del pipeline, psnr, lpips
        free_memory()
        transformer.train()

    epoch = first_epoch  # for the post-loop run_eval when no epoch runs
    step_t0 = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()  # weights are unfreezed here now

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            prompts = batch["prompts"]

            with accelerator.accumulate(*models_to_accumulate):
                if train_dataset.custom_instance_prompts:
                    prompt_embeds = gather(prompt_embeds_cache, batch["indices"])
                    text_ids = gather(text_ids_cache, batch["indices"])
                else:
                    num_repeat_elements = len(prompts)
                    prompt_embeds = prompt_embeds.repeat(num_repeat_elements, 1, 1)
                    text_ids = text_ids.repeat(num_repeat_elements, 1, 1)

                crop = None
                if use_latents:
                    # Pre-encoded, patchified, bn-normalized latents go straight to the transformer
                    model_input = batch["latents"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                    cond_model_input = batch["cond_latents"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                else:
                    # Pick random crop or full image, then VAE-encode the pixels live for this step
                    crop = sample_crop_params()
                    with offload_models(vae, device=accelerator.device, offload=args.offload):
                        if crop is not None:
                            pixel_values = crop_window(batch["canvas_pixel_values"], crop)
                            cond_pixel_values = crop_window(batch["canvas_cond_pixel_values"], crop)
                        else:
                            pixel_values, cond_pixel_values = batch["pixel_values"], batch["cond_pixel_values"]
                        pixel_values = to_pixels(pixel_values, accelerator.device, vae.dtype)  # target / edited
                        cond_pixel_values = to_pixels(cond_pixel_values, accelerator.device, vae.dtype)  # source
                        model_input = vae.encode(pixel_values).latent_dist.mode()
                        cond_model_input = vae.encode(cond_pixel_values).latent_dist.mode()

                loss = flow_matching_loss(
                    model_input,
                    cond_model_input,
                    prompt_embeds,
                    text_ids,
                    conditioning_dropout_prob=args.conditioning_dropout_prob,
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    # Retention has exactly one owner: rank 0. Never let it kill the run.
                    if accelerator.is_main_process and args.checkpoints_total_limit is not None:
                        try:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint, ignore_errors=True)
                        except Exception as e:
                            logger.warning(f"checkpoint retention failed, training continues: {e}")

                    accelerator.wait_for_everyone()

                if (accelerator.is_main_process or is_fsdp) and global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
                    if args.push_checkpoints_to_hub and accelerator.is_main_process:
                        try:
                            push_checkpoint_to_hub(save_path)
                        except Exception as e:
                            # never let a flaky upload kill a training run
                            logger.warning(f"Checkpoint upload failed: {e}")

                if val_cache and (accelerator.is_main_process or is_fsdp) and global_step % args.eval_steps == 0:
                    try:
                        run_eval(global_step, epoch)
                    except Exception as e:
                        logger.error(f"eval at step {global_step} failed, training continues: {e}", exc_info=True)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "random_crop": float(crop is not None)}
            if accelerator.sync_gradients: # sync-gradients 
                now = time.perf_counter()
                step_time = now - step_t0
                step_t0 = now
                samples = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
                tokens = model_input.shape[-2] * model_input.shape[-1] * samples
                logs.update(
                    {
                        "perf/step_time_s": step_time,
                        "perf/samples_per_s": samples / step_time,
                        "perf/img_tokens_per_s": tokens / step_time,
                        "perf/epoch": epoch,
                    }
                )
                if torch.cuda.is_available():
                    logs["perf/gpu_mem_alloc_gb"] = torch.cuda.memory_allocated() / 1024**3
                    logs["perf/gpu_mem_peak_gb"] = torch.cuda.max_memory_allocated() / 1024**3
                    logs["perf/gpu_mem_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
            progress_bar.set_postfix(loss=logs["loss"], lr=logs["lr"], **({"s/it": round(logs["perf/step_time_s"], 2)} if "perf/step_time_s" in logs else {}))
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
                # create pipeline
                pipeline = Flux2KleinPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    text_encoder=None,
                    tokenizer=None,
                    transformer=unwrap_model(transformer),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )
                images = log_validation(
                    pipeline=pipeline,
                    args=args,
                    accelerator=accelerator,
                    pipeline_args=validation_kwargs,
                    epoch=epoch,
                    global_step=global_step,
                    torch_dtype=weight_dtype,
                )

                del pipeline
                free_memory()

    if val_cache and (accelerator.is_main_process or is_fsdp) and global_step % args.eval_steps != 0:
        run_eval(global_step, epoch)

    # Save the trained transformer weights (LoRA adapters or the full model)
    accelerator.wait_for_everyone()

    if is_fsdp:
        transformer = unwrap_model(transformer)
        state_dict = accelerator.get_state_dict(transformer)
    if accelerator.is_main_process and is_full_finetune:
        transformer = unwrap_model(transformer)
        save_dtype = torch.float32 if args.upcast_before_saving else weight_dtype
        if is_fsdp:
            state_dict = {k: v.to(save_dtype) if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()}
        else:
            transformer.to(save_dtype)
            state_dict = None
        save_full_transformer(transformer, os.path.join(args.output_dir, "transformer"), state_dict=state_dict)
    elif accelerator.is_main_process:
        modules_to_save = {}
        if is_fsdp:
            if args.bnb_quantization_config_path is None:
                if args.upcast_before_saving:
                    state_dict = {
                        k: v.to(torch.float32) if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()
                    }
                else:
                    state_dict = {
                        k: v.to(weight_dtype) if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()
                    }

            transformer_lora_layers = get_peft_model_state_dict(
                transformer,
                state_dict=state_dict,
            )
            transformer_lora_layers = {
                k: v.detach().cpu().contiguous() if isinstance(v, torch.Tensor) else v
                for k, v in transformer_lora_layers.items()
            }

        else:
            transformer = unwrap_model(transformer)
            if args.bnb_quantization_config_path is None:
                if args.upcast_before_saving:
                    transformer.to(torch.float32)
                else:
                    transformer = transformer.to(weight_dtype)
            transformer_lora_layers = get_peft_model_state_dict(transformer)

        modules_to_save["transformer"] = transformer

        Flux2KleinPipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
            **_collate_lora_metadata(modules_to_save),
        )

    if accelerator.is_main_process:
        images = []
        run_validation = (args.validation_prompt and args.num_validation_images > 0) or (args.final_validation_prompt)
        should_run_final_inference = not args.skip_final_inference and run_validation
        if should_run_final_inference:
            pipeline_kwargs = {}
            if is_full_finetune:
                pipeline_kwargs["transformer"] = Flux2Transformer2DModel.from_pretrained(
                    args.output_dir, subfolder="transformer", torch_dtype=weight_dtype
                )
            pipeline = Flux2KleinPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
                **pipeline_kwargs,
            )
            if not is_full_finetune:
                # load attention processors
                pipeline.load_lora_weights(args.output_dir)

            # run inference
            images = []
            if args.validation_prompt and args.num_validation_images > 0:
                images = log_validation(
                    pipeline=pipeline,
                    args=args,
                    accelerator=accelerator,
                    pipeline_args=validation_kwargs,
                    epoch=epoch,
                    global_step=global_step,
                    is_final_validation=True,
                    torch_dtype=weight_dtype,
                )
            del pipeline
            free_memory()

        validation_prompt = args.validation_prompt if args.validation_prompt else args.final_validation_prompt
        save_model_card(
            (args.hub_model_id or Path(args.output_dir).name) if not args.push_to_hub else repo_id,
            images=images,
            base_model=args.pretrained_model_name_or_path,
            instance_prompt=args.instance_prompt or train_dataset.custom_instance_prompts[0],
            validation_prompt=validation_prompt,
            repo_folder=args.output_dir,
            fp8_training=args.do_fp8_training,
        )

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                # checkpoint-* already lives under checkpoints/ via --push_checkpoints_to_hub
                ignore_patterns=["step_*", "epoch_*", "checkpoint-*", "logs/*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
