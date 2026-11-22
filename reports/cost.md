# Cost per 1,000 requests

Local columns are measured on an Apple M5 (MLX, one request at a time). The API columns are an estimate: the measured token counts priced at Anthropic list prices (USD per million tokens, input / output: claude-haiku-5-5 0.1 / 0.5, claude-sonnet-5-5 2 / 10, claude-opus-5-5 4 / 20), without prompt caching or batch discounts. Tokens were counted with the Qwen tokenizer, so treat these as orders of magnitude. Fine-tuned runs are not priced: their one-line prompt only works with the tuned weights.

| Run | Input tokens / 1k | Output tokens / 1k | M5 seconds / 1k | Requests / hour (one stream) | claude-haiku-5-5 USD / 1k | claude-sonnet-5-5 USD / 1k | claude-opus-5-5 USD / 1k |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen05b-zero | 529,348 | 31,529 | 328 | 10,964 | 0.069 | 1.374 | 2.748 |
| qwen05b-few | 804,003 | 19,745 | 263 | 13,667 | 0.090 | 1.805 | 3.611 |
| qwen05b-lora | 33,348 | 20,554 | 269 | 13,390 | n/a | n/a | n/a |
| qwen7b-zero | 529,348 | 22,381 | 1379 | 2,611 | 0.064 | 1.283 | 2.565 |
| qwen7b-few | 804,003 | 20,848 | 1617 | 2,227 | 0.091 | 1.816 | 3.633 |

Produced by `ftvp-cost` from `reports/runs/*.jsonl`.
