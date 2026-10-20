#!/bin/bash
# Runs every ablation stage in order on one GPU. Override ABL_CKPT / ABL_BASE / ABL_DATA / ABL_RESULTS as needed.
set -o pipefail
cd "$(dirname "$0")" && source /workspace/venv/bin/activate
export ABL_RESULTS="${ABL_RESULTS:-/workspace/flux-ablation/results}"; mkdir -p "$ABL_RESULTS"
LOG="$ABL_RESULTS/stage_chain.log"
run() {
  echo "=== START $1 $(date)" >> "$LOG"
  env "${@:2}" python "$1" > "$ABL_RESULTS/$(basename "$1" .py).stdout" 2>&1
  echo "=== EXIT $1 code=$? $(date)" >> "$LOG"
}
run 01_patch_embed_weight_analysis.py
run 02_condition_gradient_test.py
run 03_base_model_loss_reference.py
run 04_changed_vs_unchanged_region_loss.py
run 05_condition_swap_and_prompt_sampling.py
run 06_attention_map_analysis.py ATTN_PAIR=19
run 06_attention_map_analysis.py ATTN_PAIR=4
run 05_condition_swap_and_prompt_sampling.py PAIR_IDX=4,13,15,19 OUT_NAME=t1_swap_sky FULL=0
run 07_verify_eval_sampler_fix.py
echo "=== ALL DONE $(date)" >> "$LOG"
