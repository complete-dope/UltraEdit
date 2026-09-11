#!/usr/bin/env bash
# Full run: channel-concat (InstructPix2Pix-style) FULL finetune of FLUX.2 klein-base-4B on exteriors-v5
# (1427 (source, caption, edited) triplets, all landscape 4K). Same layout as the overfit20 script that
# proved the wiring (klein-base-4b-overfit20-20k, train loss -> 0.007); only the dataset-scale values differ.
set -euo pipefail
cd "$(dirname "$0")"

source /workspace/UltraEdit/.venv/bin/activate
export HF_HOME=/workspace/hf_home
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="/workspace/models/FLUX.2-klein-base-4B"
DATASET_PATH="/workspace/datasets/exteriors-v5"
OUTPUT_DIR="/workspace/runs/klein-base-4b-exteriors-v5-full"
RUN_NAME="klein-base-4b-exteriors-v5-full"

if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "GPUs busy:"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv; exit 1
fi

ARGS=(
  --pretrained_model_name_or_path "$MODEL"
  --local_dataset_path "$DATASET_PATH"
  --image_column edited_img
  --cond_image_column merged_img
  --caption_column caption
  --instance_prompt "edit the exterior"
  --repeats 1
  --max_sequence_length 512 # token cap for the text prompt
  --channel_concat_cond # decides to concat based on channel on concat it in the tokens
  --conditioning_dropout_prob 0.05
  --input_width 4096
  --input_height 2728
  --resolution_width 2048
  --resolution_height 1360
  --center_crop
  --random_crop_ratio 0.5
  --val_split_ratio 0.02
  --val_split_seed 42
  --eval_steps 500
  --num_eval_samples 8
  --eval_inference_steps 28
  --eval_guidance_scale 4.0
  --skip_final_inference
  --training_mode full
# --rank 32                                # ignored in full mode
  --output_dir "$OUTPUT_DIR"
  --seed 42
  --train_batch_size 1
  --sample_batch_size 4
  --gradient_accumulation_steps 1
  --max_train_steps 10000 # ~14 epochs; at ~3.5 s/it plus evals ~10.5 h
  --learning_rate 1e-5 # full finetune of 4B params: keep the LR low
  --x_embedder_lr 1e-3 # cond half is zero-init; at 1e-5 it never becomes usable
  --lr_scheduler cosine
  --lr_warmup_steps 200
  --lr_num_cycles 1
  --optimizer AdamW
  --adam_beta1 0.9
  --adam_beta2 0.999
  --adam_weight_decay 1e-4
  --adam_epsilon 1e-8
  --max_grad_norm 1.0
  --guidance_scale 3.5
  --weighting_scheme none
  --logit_mean 0.0
  --logit_std 1.0
  --mode_scale 1.29
  --mixed_precision bf16                   # autocast compute in bf16...
  --transformer_dtype fp32                 # ...while transformer master weights stay fp32
# --gradient_checkpointing                 # off: trade VRAM for speed, and its recompute pass
                                           # returns fp32 tensors under autocast (CheckpointError)
  --allow_tf32
  --cache_latents                          # encode all 1427 pairs once (caches live in host RAM), then drop the VAE.
                                           # Side effect: batch order is fixed across epochs (step-indexed caches).
  --dataloader_num_workers 2
  --checkpointing_steps 500                # 58GB each; run ckpt_slim.sh "$OUTPUT_DIR" in tmux to strip middle optimizers
  --checkpoints_total_limit 3
  --resume_from_checkpoint latest
  --report_to wandb                        # project is fixed in the trainer: dreambooth-flux2-image2img-lora
  --logging_dir logs
  # no --push_to_hub: checkpoints stay local in $OUTPUT_DIR
)

export WANDB_NAME="$RUN_NAME"

accelerate launch --config_file /workspace/hf_home/accelerate/fsdp_config.yaml \
  train_dreambooth_lora_flux2_klein_img2img.py "${ARGS[@]}"
