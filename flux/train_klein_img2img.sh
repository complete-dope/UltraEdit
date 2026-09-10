#!/usr/bin/env bash
# Channel-concat (InstructPix2Pix-style) fine-tune of FLUX.2 klein-base on (source, prompt, edited) triplets.
# Every CLI arg of train_dreambooth_lora_flux2_klein_img2img.py is listed below; commented ones use the script default.
set -euo pipefail
cd "$(dirname "$0")"

MODEL="black-forest-labs/FLUX.2-klein-base-4B"
DATASET_PATH="/path/to/dataset"          # save_to_disk folder or imagefolder with metadata.jsonl
OUTPUT_DIR="./out/klein-base-4b-channel-concat"
RUN_NAME="klein-base-4b-channel-concat"

ARGS=(
  # ---------------- model ----------------
  --pretrained_model_name_or_path "$MODEL"
  --local_dataset_path "$DATASET_PATH"
  --image_column edited                   # target / edited image
  --cond_image_column source              # input / original image
  --caption_column prompt                 # edit instruction
  --instance_prompt "edit the exterior"   # fallback when a row has no caption
  --repeats 1
  --max_sequence_length 512
  --channel_concat_cond                   # pix2pix-style aligned channels instead of extra tokens
  --conditioning_dropout_prob 0.05        # 5% text / 5% image / 5% both, enables image_guidance_scale at inference
  --resolution 1024
  --center_crop
  --val_split_ratio 0.05
  --val_split_seed 42
  --eval_steps 250
  --num_eval_samples 16
  --eval_inference_steps 28
  --eval_guidance_scale 4.0
  --skip_final_inference
  --num_validation_images 4
  --validation_epochs 50
  --training_mode full                    # or lora
  --output_dir "$OUTPUT_DIR"
  --seed 42
  --train_batch_size 1
  --sample_batch_size 4
  --gradient_accumulation_steps 8
  --num_train_epochs 1
  --max_train_steps 5000
  --learning_rate 1e-5
  --lr_scheduler constant_with_warmup
  --lr_warmup_steps 200
  --lr_num_cycles 1
  --lr_power 1.0
  --optimizer AdamW
  --adam_beta1 0.9
  --adam_beta2 0.999
  --adam_weight_decay 1e-4
  --adam_weight_decay_text_encoder 1e-3
  --adam_epsilon 1e-8
  --max_grad_norm 1.0
  --guidance_scale 3.5                    # only used if transformer.config.guidance_embeds
  --weighting_scheme none
  --logit_mean 0.0
  --logit_std 1.0
  --mode_scale 1.29
  --mixed_precision bf16
  --gradient_checkpointing
  --allow_tf32
  --cache_latents
  --offload
  --dataloader_num_workers 4
  --checkpointing_steps 500
  --checkpoints_total_limit 3
  --report_to wandb
  --logging_dir logs
)

export WANDB_NAME="$RUN_NAME"
# export WANDB_PROJECT="fotello-exterior-edit"

accelerate launch train_dreambooth_lora_flux2_klein_img2img.py "${ARGS[@]}"
