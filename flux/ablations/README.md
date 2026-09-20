# Ablations for the channel-concat Klein img2img model

Scripts used for the 2026-09-20 ablation of `fotello-ai/flux-klein-4b-exterior-v1` checkpoint-17000.
Report and results: `/workspace/flux-ablation/results/REPORT.md`, artifact https://claude.ai/artifact/79ZmwuWoL36CAFxVk2XiK8

Paths are read from env vars with the original run as defaults:
`ABL_BASE` (base model dir), `ABL_CKPT` (checkpoint `transformer/` dir), `ABL_DATA` (test parquet), `ABL_RESULTS` (output dir).
Run from this directory with the venv active (`source /workspace/venv/bin/activate`), or run everything with `./run_all_stages.sh`.

| script | what it does | output |
|---|---|---|
| `01_patch_embed_weight_analysis.py` | Test 1a: x_embedder noise half vs cond half norms, rank, drift of every block from base. CPU only. | `t1_weights/` |
| `02_condition_gradient_test.py` | Test 1b: flow-matching loss and gradient w.r.t. the cond latent vs the noisy latent; loss with cond = 0 / other image / noise / target; prediction change from prompt vs cond. | `t1_grad/` |
| `03_base_model_loss_reference.py` | Same loss on the untouched base model as a reference. | `t1_grad/results_base.json` |
| `04_changed_vs_unchanged_region_loss.py` | Test 1b': loss on tokens the editor changed vs left alone, with/without cond, vs a copy-the-source baseline. | `t1_region_loss/` |
| `05_condition_swap_and_prompt_sampling.py` | Test 1c: full sampling. normal / swapped cond / zero cond / target cond / prompt variants / IP2P CFG. `PAIR_IDX=4,13,15,19 OUT_NAME=t1_swap_sky FULL=0` for the sky-heavy subset. | `t1_swap*/` |
| `06_attention_map_analysis.py` | Test 2: explicit attention rows in all 25 blocks at 3 sigmas for image queries and edit-word text tokens, trained vs zero-cond vs base. `ATTN_PAIR=<test idx>`. | `t2_attn_pair<N>/` |
| `07_verify_eval_sampler_fix.py` | Checks the fixed `../channel_concat_denoise.py` reproduces the ablation sampler's guided/unguided outputs. | `eval_fix_check/` |
| `ablation_common.py` | Shared loading, image prep, text encoding, latent packing. | |
| `cfg_sampling.py` | Channel-concat sampling loop with InstructPix2Pix CFG (text and image scales). | |
| `build_report_page.py` | Builds the HTML report page from the results JSON. | `/workspace/flux-ablation/artifact/` |
| `run_all_stages.sh` | Runs stages 01-07 in order, logs to `$ABL_RESULTS/stage_chain.log`. | |

Metrics worth re-running on any new checkpoint (set `ABL_CKPT`): `02_condition_gradient_test.py` (cond=src vs cond=0 loss,
prompt vs cond prediction change) and `04_changed_vs_unchanged_region_loss.py` (`cond_src_changed` must fall below
`cond_zero_changed`, otherwise the model is still copying the source).
