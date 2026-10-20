#!/usr/bin/env bash
# Overfit sanity run: channel-concat (InstructPix2Pix-style) FULL finetune on 20 exterior triplets.
# Goal is not quality -- it is to prove the wiring: train loss must fall toward ~0 over 5000 steps.
# Same layout as train_klein_img2img.sh; only the overfit-relevant values differ.
set -euo pipefail
cd "$(dirname "$0")"

source /workspace/UltraEdit/.venv/bin/activate
export HF_HOME=/workspace/hf_home
export TOKENIZERS_PARALLELISM=false
# no gradient checkpointing means activations fill the card; expandable segments reclaims the
# ~4GB that fragmentation otherwise strands
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="/workspace/models/FLUX.2-klein-base-4B"
DATASET_PATH="/workspace/datasets/exteriors-v5-overfit20"   # save_to_disk folder, 20 rows
OUTPUT_DIR="/workspace/runs/klein-base-4b-overfit20-20k"
RUN_NAME="klein-base-4b-overfit20-20k"

ARGS=(
  # ---------------- model / data ----------------
  --pretrained_model_name_or_path "$MODEL"
  --local_dataset_path "$DATASET_PATH"
  --image_column edited_img                # target / edited image
  --cond_image_column merged_img           # input / original image
  --caption_column caption                 # edit instruction (aka --edit_prompt_column)
  --instance_prompt "edit the exterior"    # fallback when a row has no caption
  --repeats 1
  --max_sequence_length 512
  --channel_concat_cond                    # pix2pix-style aligned channels instead of extra tokens
  --conditioning_dropout_prob 0.0          # 0 for overfit: dropout fights memorisation
# --resolution 1024                        # ignored: resolution_height/width set an exact bucket
  --resolution_height 1360                 # 2K: full image resized to cover, then cropped
  --resolution_width 2048
  --center_crop
  # ---------------- eval ----------------
  --val_split_ratio 0.2                    # 4 of 20 held out for eval, 16 remain for training
  --val_split_seed 42
  --eval_steps 250                         # val_loss + MSE/PSNR/LPIPS + 4 logged images
  --num_eval_samples 4
  --eval_inference_steps 28
  --eval_guidance_scale 4.0
  --skip_final_inference                   # channel_concat_cond hard-errors on validation prompts;
                                           # inference runs separately, see infer_klein_img2img.sh
  # ---------------- optimisation ----------------
  --training_mode full
# --rank 32                                # ignored in full mode
  --output_dir "$OUTPUT_DIR"
  --seed 42
  --train_batch_size 1
  --sample_batch_size 4
  --gradient_accumulation_steps 1          # 1 so 5000 steps == 5000 real updates
  --max_train_steps 20000
  --learning_rate 1e-5                     # full finetune of 4B params: keep the LR low
  --x_embedder_lr 1e-3                     # cond half is zero-init; at 1e-5 it never becomes usable
  --lr_scheduler constant
  --lr_warmup_steps 0
  --optimizer AdamW                        # fp32 Adam states: with bf16 weights, 8bit Adam loses small updates
  --adam_beta1 0.9
  --adam_beta2 0.999
  --adam_weight_decay 1e-4
  --adam_epsilon 1e-8
  --max_grad_norm 1.0
  --guidance_scale 3.5                     # only used if transformer.config.guidance_embeds
  --weighting_scheme none
  --logit_mean 0.0
  --logit_std 1.0
  --mode_scale 1.29
  # ---------------- runtime ----------------
  --mixed_precision bf16                   # autocast compute in bf16...
  --transformer_dtype fp32                 # ...while transformer master weights stay fp32
# --gradient_checkpointing                 # off: trade VRAM for speed, and its recompute pass
                                           # returns fp32 tensors under autocast (CheckpointError)
  --allow_tf32
  # image latents never cached: VAE encodes live each step (random crop or full resize)
  --dataloader_num_workers 2
  --checkpointing_steps 250
  --checkpoints_total_limit 3
  --resume_from_checkpoint latest
  --report_to wandb
  --logging_dir logs
  # no --push_to_hub: checkpoints stay local in $OUTPUT_DIR
)

export WANDB_NAME="$RUN_NAME"
set -a; source "$(git rev-parse --show-toplevel)/.env"; set +a
export WANDB_PROJECT

# FSDP full-shard: a fp32 4B full finetune needs 64GB of weights+grads+Adam per GPU under DDP,
# which OOMs an 80GB A100 at 2K. Sharding over both GPUs brings it to ~32GB each.
accelerate launch --config_file /workspace/hf_home/accelerate/fsdp_config.yaml \
  train_dreambooth_lora_flux2_klein_img2img.py "${ARGS[@]}"
