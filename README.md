<h1 align="center">Fine-tune vs. Prompt a Small LLM</h1>

<p align="center">
  <a href="https://github.com/haroonkhan02/llm-finetune-vs-prompt/actions/workflows/ci.yml"><img src="https://github.com/haroonkhan02/llm-finetune-vs-prompt/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.10--3.12-blue" alt="Python">
  <img src="https://img.shields.io/badge/MLX-LoRA-orange" alt="MLX">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

When is it worth fine-tuning a small model instead of prompting a bigger one? This project
answers that for one narrow task: turning a voice-assistant request into an intent and its
slots, as JSON. I compare a LoRA-tuned 0.5B model on Apple silicon (MLX) with the same
model prompted zero-shot and few-shot, and with a 7B model, on quality, latency and cost
per 1,000 requests.

```text
"wake me up at five am this week"
→ {"intent": "alarm_set", "slots": [["time", "five am"], ["date", "this week"]]}
```

**Status: work in progress.** The LoRA model is trained and all five local systems are
scored below. Error analysis, cost per 1,000 requests and a fused-adapter speed check come
next.

## Task and data

[MASSIVE](https://github.com/alexa/massive) 1.1 (Amazon, CC BY 4.0), English only: 11,514
train, 2,033 dev and 2,974 test utterances, 60 intents and 55 slot types. A third of the
requests have no slot, and about 3 % repeat a slot type ("on monday and on friday"), so
slots are a list of pairs, not a dict. Every system is scored on the same seeded subset of
1,000 test utterances.

From [`reports/data_check.md`](reports/data_check.md):

| Check | Value |
|---|---:|
| Intent of the nearest training utterance (TF-IDF 1-NN, no LLM) | 73.5 % |
| Gold intent among the 8 retrieved few-shot examples | 91.7 % |
| Prompt tokens per request: zero-shot / few-shot (k=8) / fine-tuned | 529 / 797 / 33 |

The 1-NN figure is the floor every model has to beat. The prompt lengths are the main cost
argument for fine-tuning: the tuned model learns the label list and the format, so it does
not pay for them on every request.

## First results

All systems on the same 1,000 test utterances, greedy decoding, one request at a time on an
Apple M5. Full table with 95 % bootstrap intervals: [`reports/results.md`](reports/results.md).

| System | Intent acc | Slot F1 | Exact match | Input tokens | Latency p50 |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B zero-shot | 26.8 % | 6.7 % | 0.9 % | 529 | 0.28 s |
| Qwen2.5-0.5B few-shot (k = 8) | 77.4 % | 58.8 % | 47.9 % | 797 | 0.24 s |
| Qwen2.5-0.5B + LoRA | 83.1 % | 70.9 % | 60.6 % | 33 | 0.26 s |
| Qwen2.5-7B zero-shot (4-bit) | 67.4 % | 30.6 % | 22.6 % | 529 | 1.34 s |
| Qwen2.5-7B few-shot (4-bit) | 87.5 % | 71.4 % | 63.8 % | 797 | 1.57 s |

Fifteen minutes of LoRA training lift the 0.5B model from unusable to roughly the level of
a 7B model prompted with retrieved examples: slot F1 is the same within noise (paired
difference 0.5 points, CI −2.6 to 3.9) and exact match is 3.2 points lower (CI 0.1 to 6.2),
at about a sixth of the latency. Zero-shot, the 0.5B model invents intent names in about a
third of its replies. The tuned model's short prompt does not make it faster yet, because
decoding dominates and the adapter is applied unfused; the 7B model is also 4-bit while the
0.5B runs in bf16, so speed is not a like-for-like hardware comparison.

## Compared systems

```mermaid
flowchart LR
    D[MASSIVE en-US] --> R[TF-IDF retriever over train]
    D --> S[LoRA data: short prompt + JSON answer]
    S --> L[Qwen2.5-0.5B + LoRA, mlx-lm]
    R --> F[few-shot prompts, k = 8]
    D --> Z[zero-shot prompt with all labels]
    Z & F --> Q[Qwen2.5-0.5B and Qwen2.5-7B]
    L & Q --> P[reply parser] --> M[intent acc, slot F1, exact match, format, latency, cost]
```

| System | Model | Prompt |
|---|---|---|
| Retrieval baseline | none | intent of the nearest train utterance |
| Zero-shot | Qwen2.5-0.5B-Instruct, Qwen2.5-7B-Instruct | instructions + all labels |
| Few-shot | same two | + 8 nearest training examples as chat turns |
| Fine-tuned | Qwen2.5-0.5B-Instruct + LoRA (rank 16, 2 epochs) | one-line instruction |
| API model (optional) | Claude | few-shot; requires an Anthropic API key |

Metrics: intent accuracy, slot precision/recall/F1 on (type, value) pairs, exact match,
valid-JSON and exact-schema rates, share of slot values copied from the utterance, median
and p95 latency on an M5, and cost per 1,000 requests. Differences between systems get
paired bootstrap intervals.

## Getting started

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev,mlx]"
ftvp-download      # MASSIVE 1.1 archive (40 MB), keeps only en-US
ftvp-check --tokenizer mlx-community/Qwen2.5-0.5B-Instruct-4bit
ftvp-prepare       # data/sft/{train,valid,test}.jsonl for mlx_lm.lora
python -m mlx_lm lora -c configs/lora.yaml | tee outputs/train.log   # ~15 min on an M5
scripts/run_all.sh # five systems x 1,000 requests, then ftvp-report
pytest -q
```

## Repository structure

```text
llm-finetune-vs-prompt/
├── configs/default.yaml     # dataset URL + checksum, eval subset, k, SFT split sizes
├── configs/lora.yaml        # LoRA hyperparameters for mlx_lm.lora
├── src/ftvp/
│   ├── download.py          # resumable download, checksum, extracts one locale
│   ├── massive.py           # annotated utterance -> intent + (type, value) slots
│   ├── schema.py            # target JSON and a tolerant reply parser
│   ├── metrics.py           # intent / slot / exact-match scores, bootstrap CIs
│   ├── prompts.py           # zero-shot, retrieved few-shot, fine-tuned prompts
│   ├── sft.py               # mlx-lm chat-format training data
│   ├── check.py             # data stats, retrieval baseline, prompt lengths
│   ├── backends.py          # MLX (+ LoRA adapter), Anthropic, fake
│   ├── runner.py            # resumable runs, one JSON record per request
│   └── report.py            # scores, intervals, paired comparison, training curve
├── scripts/run_all.sh       # runs every local system in sequence
├── reports/                 # data check, results, compact per-request runs
└── tests/                   # synthetic fixtures, no data or model needed
```

## Design decisions

| Decision | Why |
|---|---|
| Slots as `[type, value]` pairs | repeated types are common enough to matter |
| Value-level slot F1 | a generative model returns strings, not token spans |
| Parser forgives fences and prose, but records it | format compliance is a result, not a crash |
| Few-shot examples retrieved by similarity | a stronger prompting baseline than fixed examples |
| Same tokenizer family for 0.5B and 7B | prompt costs are directly comparable |
| Loss on the answer only (`mask_prompt`) | the model should learn the labels, not the requests |
| Warmup + cosine learning rate | a constant 1e-4 made the first steps worse than the base model |
| Runs stored per request, re-scored from text | a parser fix never needs another model run |

## License

MIT for the code. MASSIVE is © Amazon, CC BY 4.0.
