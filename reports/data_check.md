# MASSIVE en-US: data check

Label space (train): 60 intents, 55 slot types.

| Split | Utterances | Intents | Mean words | Mean slots | No slot | Repeated slot type |
|---|---:|---:|---:|---:|---:|---:|
| train | 11,514 | 60 | 6.9 | 0.99 | 32.6% | 3.1% |
| dev | 2,033 | 59 | 6.9 | 0.99 | 31.8% | 3.3% |
| test | 2,974 | 59 | 6.8 | 0.95 | 33.4% | 2.6% |

Labels in dev/test that never occur in train: none.

Most frequent train intents: calendar_set (810), play_music (639), weather_query (573), calendar_query (566), general_quirky (555).

## Retrieval-only baseline (evaluation subset, n = 1000)

| | Value |
|---|---:|
| Intent of the nearest train utterance (1-NN) | 73.5% |
| Gold intent among the 8 retrieved shots | 91.7% |

## Prompt length per request (median tokens, `mlx-community/Qwen2.5-7B-Instruct-4bit` chat template)

| System | Input | Output (gold answer) |
|---|---:|---:|
| zero-shot | 529 | 19 |
| few-shot (k=8) | 797 | 19 |
| fine-tuned | 33 | 19 |

Produced by `ftvp-check`.
