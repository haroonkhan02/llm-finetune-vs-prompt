# Results on the shared MASSIVE en-US test subset (n = 1000)

Quality in %, with 95 % bootstrap intervals over utterances. Latency is wall time per request on an Apple M5 (MLX, batch size 1, greedy decoding).

| Run | System | Model | Intent acc | Slot F1 | Exact match | Valid JSON | Exact schema |
|---|---|---|---:|---:|---:|---:|---:|
| qwen05b-lora | finetuned | mlx:Qwen2.5-0.5B-Instruct-bf16+lora | 83.1 (80.8–85.4) | 70.9 (68.0–73.6) | 60.6 (57.5–63.5) | 100.0 | 100.0 |
| qwen05b-lora-rerun | finetuned | mlx:Qwen2.5-0.5B-Instruct-bf16+lora | 83.1 (80.8–85.4) | 70.9 (68.0–73.6) | 60.6 (57.5–63.5) | 100.0 | 100.0 |
| qwen05b-lora-fused | finetuned | mlx:qwen05b-lora-fused | 83.0 (80.5–85.3) | 70.4 (67.4–73.1) | 60.0 (56.7–62.9) | 100.0 | 100.0 |
| qwen05b-lora-fused-4bit | finetuned | mlx:qwen05b-lora-fused-4bit | 81.3 (78.9–83.7) | 64.2 (61.1–67.1) | 54.8 (51.8–58.0) | 100.0 | 100.0 |

## Speed and tokens per request

| Run | Latency p50 (s) | p95 (s) | First token p50 (s) | Input tokens p50 | Output tokens p50 | Truncated | Seconds per 1k requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen05b-lora | 0.259 | 0.420 | 0.036 | 33 | 20 | 0 | 269 |
| qwen05b-lora-rerun | 0.264 | 0.435 | 0.036 | 33 | 20 | 0 | 277 |
| qwen05b-lora-fused | 0.219 | 0.396 | 0.023 | 33 | 20 | 0 | 237 |
| qwen05b-lora-fused-4bit | 0.079 | 0.123 | 0.016 | 33 | 20 | 0 | 82 |

## Paired difference vs `qwen05b-lora` (other − reference, points)

| Run | Slot F1 Δ (95 % CI) | Exact match Δ (95 % CI) |
|---|---:|---:|
| qwen05b-lora-rerun | 0.0 (0.0 to 0.0) | 0.0 (0.0 to 0.0) |
| qwen05b-lora-fused | -0.5 (-1.3 to 0.2) | -0.6 (-1.3 to 0.1) |
| qwen05b-lora-fused-4bit | -6.7 (-8.9 to -4.3) | -5.8 (-8.3 to -3.4) |

Produced by `ftvp-report` from `outputs/runs/*.jsonl`.
