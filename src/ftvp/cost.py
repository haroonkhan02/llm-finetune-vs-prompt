"""Cost per 1,000 requests from the measured runs.

Two views, both computed from the per-request records:

- local: seconds of Apple M5 time per 1,000 requests (batch size 1, greedy) and the
  throughput that implies for one stream;
- hosted estimate: what the same prompts would cost if sent to a Claude model at list
  price, from the measured input/output token counts (no prompt caching, no batch
  discount). Token counts come from the Qwen tokenizer, so this is an order-of-magnitude
  estimate, not a bill: Claude's tokenizer counts the same text differently.

    ftvp-cost qwen05b-zero qwen05b-few qwen05b-lora qwen7b-zero qwen7b-few
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Sequence

from ftvp.backends import PRICES
from ftvp.config import load_config, resolve

API_MODELS = ("claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5")


def cost_per_1k(recs: Sequence[dict], models: Sequence[str] = API_MODELS,
                priced: bool = True) -> dict:
    """``priced=False`` for fine-tuned runs: their short prompt only works with the tuned
    weights, so pricing it on a hosted model would be meaningless."""
    if not recs:
        raise ValueError("no records")
    n = len(recs)
    tok_in = sum(r["input_tokens"] for r in recs) / n
    tok_out = sum(r["output_tokens"] for r in recs) / n
    sec = statistics.fmean(r["latency_s"] for r in recs)
    api: dict[str, float | None] = dict.fromkeys(models)
    for m in models if priced else ():
        p_in, p_out = PRICES[m]
        api[m] = 1000 * (tok_in * p_in + tok_out * p_out) / 1e6
    return {"n": n, "input_tokens_per_1k": 1000 * tok_in, "output_tokens_per_1k": 1000 * tok_out,
            "m5_seconds_per_1k": 1000 * sec, "requests_per_hour": 3600 / sec if sec else 0.0,
            "api_usd_per_1k": api}


def render(rows: dict[str, dict], models: Sequence[str] = API_MODELS) -> str:
    lines = ["# Cost per 1,000 requests", "",
             "Local columns are measured on an Apple M5 (MLX, one request at a time). The "
             "API columns are an estimate: the measured token counts priced at Anthropic list "
             "prices (USD per million tokens, input / output: " +
             ", ".join(f"{m} {PRICES[m][0]:g} / {PRICES[m][1]:g}" for m in models) +
             "), without prompt caching or batch discounts. Tokens were counted with the "
             "Qwen tokenizer, so treat these as orders of magnitude. Fine-tuned runs are not "
             "priced: their one-line prompt only works with the tuned weights.", "",
             "| Run | Input tokens / 1k | Output tokens / 1k | M5 seconds / 1k | Requests / hour "
             "(one stream) | " + " | ".join(f"{m} USD / 1k" for m in models) + " |",
             "|---|---:|---:|---:|---:|" + "---:|" * len(models)]
    for name, r in rows.items():
        lines.append(
            f"| {name} | {r['input_tokens_per_1k']:,.0f} | {r['output_tokens_per_1k']:,.0f} | "
            f"{r['m5_seconds_per_1k']:.0f} | {r['requests_per_hour']:,.0f} | " +
            " | ".join("n/a" if r["api_usd_per_1k"][m] is None
                       else f"{r['api_usd_per_1k'][m]:.3f}" for m in models) + " |")
    lines += ["", "Produced by `ftvp-cost` from `reports/runs/*.jsonl`."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    from ftvp.runner import load_run
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+")
    ap.add_argument("--runs-dir", default="reports/runs")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    runs_dir = resolve(args.runs_dir)
    rows = {}
    for name in args.names:
        recs = load_run(runs_dir / f"{name}.jsonl")
        if not recs:
            raise SystemExit(f"no records for run {name!r} in {runs_dir}")
        meta = runs_dir / f"{name}.meta.json"
        system = json.loads(meta.read_text()).get("system") if meta.exists() else None
        rows[name] = cost_per_1k(recs, priced=system != "finetuned")
    out_dir = resolve(cfg.get("reports_dir", "reports"))
    text = render(rows)
    (out_dir / "cost.md").write_text(text, encoding="utf-8")
    (out_dir / "cost.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
