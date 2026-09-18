#!/usr/bin/env bash
# 4K IP2P finetune of FLUX.2 klein 4B on pre-encoded latents, 2 GPUs (FSDP2), xformers + torch.compile.
set -euo pipefail
cd "$(dirname "$0")"

# /workspace is a network FUSE mount and python imports from it take minutes; /opt/flux-venv is a
# local-disk mirror of /workspace/UltraEdit/.venv (cp -a + path fixup). Fall back to the original if absent.
VENV="${VENV:-/opt/flux-venv}"
[[ -x "$VENV/bin/python" ]] || VENV=/workspace/UltraEdit/.venv
source "$VENV/bin/activate"
echo "venv: $VENV"
export HF_HOME=/workspace/hf_home
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=/opt/torchinductor-cache
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

MODEL="/workspace/models/FLUX.2-klein-base-4B"
LATENT_DIR="/workspace/datasets/tmp_exterior_v5_latents/flux2-klein-4b-4096x2736"
RUN_NAME="${RUN_NAME:-klein-4b-4k-latents-500}"
OUTPUT_DIR="/workspace/runs/$RUN_NAME"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
BATCH="${BATCH:-2}"
ACCUM="${ACCUM:-2}"
MAX_STEPS="${MAX_STEPS:-2000}"
EVAL_STEPS="${EVAL_STEPS:-100}"
CKPT_STEPS="${CKPT_STEPS:-250}"
NUM_EVAL="${NUM_EVAL:-8}"
EVAL_INFER_STEPS="${EVAL_INFER_STEPS:-28}"
LR_SCHED="${LR_SCHED:-constant}"
WARMUP="${WARMUP:-0}"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  WANDB_API_KEY="$(grep -oP '^export WANDB_API_KEY=\K\S+' ~/.bashrc || true)"
fi
export WANDB_API_KEY
export WANDB_NAME="$RUN_NAME"
export WANDB_DIR="$OUTPUT_DIR"
# keep every continuation of this run on one wandb run so the curves stay contiguous
export WANDB_RUN_ID="${WANDB_RUN_ID:-q6eyhv0v}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "GPUs busy:"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv; exit 1
fi

ARGS=(
  --pretrained_model_name_or_path "$MODEL"
  --latent_dir "$LATENT_DIR"
  --latent_split train
  --latent_val_split test
  --max_train_samples "$NUM_SAMPLES"
  --preload_latents
  --caption_column caption
  --repeats 1
  --max_sequence_length 512
  --channel_concat_cond
  --conditioning_dropout_prob 0.05
  --eval_steps "$EVAL_STEPS"
  --num_eval_samples "$NUM_EVAL"
  --eval_inference_steps "$EVAL_INFER_STEPS"
  --eval_guidance_scale 4.0
  --skip_final_inference
  --training_mode full
  --output_dir "$OUTPUT_DIR"
  --seed 42
  --train_batch_size "$BATCH"
  --sample_batch_size 8
  --gradient_accumulation_steps "$ACCUM"
  --max_train_steps "$MAX_STEPS"
  --learning_rate 1e-5
  --x_embedder_lr 1e-3
  --lr_scheduler "$LR_SCHED"
  --lr_warmup_steps "$WARMUP"
  --lr_num_cycles 1
  --optimizer AdamW
  --adam_beta1 0.9
  --adam_beta2 0.999
  --adam_weight_decay 1e-4
  --adam_epsilon 1e-8
  --max_grad_norm 1.0
  --guidance_scale 3.5
  --weighting_scheme none
  --mixed_precision bf16
  --transformer_dtype fp32
  --gradient_checkpointing
  --allow_tf32
  --attention_backend xformers
  --compile_transformer
  --dataloader_num_workers 2
  --checkpointing_steps "$CKPT_STEPS"
  --checkpoints_total_limit 2
  --resume_from_checkpoint latest
  --report_to wandb
  --tracker_project_name flux2-klein-4k-latents
  --logging_dir logs
  --push_checkpoints_to_hub
  --hub_model_id fotello-ai/flux-klein-4b-exterior-500-4k-model
  --hub_checkpoints_limit 0
)

mkdir -p "$OUTPUT_DIR"
python -m accelerate.commands.launch --config_file /workspace/hf_home/accelerate/fsdp2_2gpu.yaml \
  train_dreambooth_lora_flux2_klein_img2img.py "${ARGS[@]}" "$@" 2>&1 | tee -a "$OUTPUT_DIR/train.log"
