# Results on the shared MASSIVE en-US test subset (n = 1000)

Quality in %, with 95 % bootstrap intervals over utterances. Latency is wall time per request on an Apple M5 (MLX, batch size 1, greedy decoding).

| Run | System | Model | Intent acc | Slot F1 | Exact match | Valid JSON | Exact schema |
|---|---|---|---:|---:|---:|---:|---:|
| qwen05b-zero | zero-shot | mlx:Qwen2.5-0.5B-Instruct-bf16 | 26.8 (24.3–29.5) | 6.7 (5.3–8.2) | 0.9 (0.3–1.5) | 96.1 | 95.5 |
| qwen05b-few | few-shot | mlx:Qwen2.5-0.5B-Instruct-bf16 | 77.4 (74.8–80.0) | 58.8 (55.3–62.0) | 47.9 (44.7–50.9) | 99.7 | 99.6 |
| qwen05b-lora | finetuned | mlx:Qwen2.5-0.5B-Instruct-bf16+lora | 83.1 (80.8–85.4) | 70.9 (68.0–73.6) | 60.6 (57.5–63.5) | 100.0 | 100.0 |
| qwen7b-zero | zero-shot | mlx:Qwen2.5-7B-Instruct-4bit | 67.4 (64.4–70.4) | 30.6 (27.6–33.4) | 22.6 (20.2–25.3) | 97.3 | 96.8 |
| qwen7b-few | few-shot | mlx:Qwen2.5-7B-Instruct-4bit | 87.5 (85.5–89.4) | 71.4 (68.6–74.2) | 63.8 (60.7–66.9) | 100.0 | 99.8 |

## Speed and tokens per request

| Run | Latency p50 (s) | p95 (s) | First token p50 (s) | Input tokens p50 | Output tokens p50 | Truncated | Seconds per 1k requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen05b-zero | 0.284 | 0.623 | 0.052 | 529 | 26 | 27 | 328 |
| qwen05b-few | 0.240 | 0.408 | 0.074 | 797 | 19 | 2 | 263 |
| qwen05b-lora | 0.259 | 0.420 | 0.036 | 33 | 20 | 0 | 269 |
| qwen7b-zero | 1.342 | 1.792 | 0.626 | 529 | 21 | 0 | 1379 |
| qwen7b-few | 1.565 | 2.170 | 0.902 | 797 | 20 | 0 | 1617 |

## Paired difference vs `qwen05b-lora` (other − reference, points)

| Run | Slot F1 Δ (95 % CI) | Exact match Δ (95 % CI) |
|---|---:|---:|
| qwen05b-zero | -64.2 (-67.2 to -61.3) | -59.7 (-62.7 to -56.6) |
| qwen05b-few | -12.1 (-15.8 to -8.7) | -12.7 (-16.3 to -9.8) |
| qwen7b-zero | -40.3 (-44.0 to -36.6) | -38.0 (-41.7 to -34.8) |
| qwen7b-few | 0.5 (-2.6 to 3.9) | 3.2 (0.1 to 6.2) |

## LoRA training (validation loss on 500 dev utterances)

| Iteration | Validation loss |
|---:|---:|
| 1 | 3.264 |
| 120 | 0.284 |
| 240 | 0.222 |
| 360 | 0.139 |
| 480 | 0.111 |
| 600 | 0.101 |
| 720 | 0.097 |
| 840 | 0.088 |
| 960 | 0.082 |
| 1080 | 0.078 |
| 1200 | 0.077 |
| 1320 | 0.077 |
| 1440 | 0.076 |

Produced by `ftvp-report` from `outputs/runs/*.jsonl`.
