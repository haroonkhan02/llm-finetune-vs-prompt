#!/usr/bin/env bash
# Fuse the LoRA adapter into the base weights, make a 4-bit copy, and time both on the
# shared 1,000-utterance subset. An unfused re-run comes first as a timing control
# (same session, same thermal state). Resumable: rerun to continue.
set -euo pipefail
cd "$(dirname "$0")/.."
SMALL=mlx-community/Qwen2.5-0.5B-Instruct-bf16
ADAPTER=adapters/qwen05b-lora
FUSED=models/qwen05b-lora-fused
FUSED4=models/qwen05b-lora-fused-4bit

[ -d "$FUSED" ] || python -m mlx_lm fuse --model $SMALL --adapter-path $ADAPTER --save-path $FUSED
[ -d "$FUSED4" ] || python -m mlx_lm convert --hf-path $FUSED --mlx-path $FUSED4 -q --q-bits 4

ftvp-run --system finetuned --backend mlx:$SMALL --adapter $ADAPTER --name qwen05b-lora-rerun
ftvp-run --system finetuned --backend mlx:$FUSED --name qwen05b-lora-fused
ftvp-run --system finetuned --backend mlx:$FUSED4 --name qwen05b-lora-fused-4bit

ftvp-report qwen05b-lora qwen05b-lora-rerun qwen05b-lora-fused qwen05b-lora-fused-4bit \
    --ref qwen05b-lora --out fused --export
