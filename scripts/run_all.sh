#!/usr/bin/env bash
# Run every local system on the shared 1,000-utterance subset, one after another so
# latencies are not distorted by a second model on the GPU. Resumable: rerun to continue.
set -euo pipefail
cd "$(dirname "$0")/.."
SMALL=mlx-community/Qwen2.5-0.5B-Instruct-bf16
LARGE=mlx-community/Qwen2.5-7B-Instruct-4bit
ADAPTER=adapters/qwen05b-lora

ftvp-run --system finetuned --backend mlx:$SMALL --adapter $ADAPTER --name qwen05b-lora
ftvp-run --system zero-shot --backend mlx:$SMALL --name qwen05b-zero
ftvp-run --system few-shot  --backend mlx:$SMALL --name qwen05b-few
ftvp-run --system zero-shot --backend mlx:$LARGE --name qwen7b-zero
ftvp-run --system few-shot  --backend mlx:$LARGE --name qwen7b-few
ftvp-report qwen05b-zero qwen05b-few qwen05b-lora qwen7b-zero qwen7b-few \
    --ref qwen05b-lora --train-log outputs/train.log
