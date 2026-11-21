"""Score finished runs and write ``reports/results.md`` and ``reports/results.json``.

Replies are re-parsed from the stored text, so a parser change re-scores old runs without
calling a model again. Quality gets 95 % bootstrap intervals; every run is also compared
with a reference run on the same utterances (paired bootstrap). Latency is the measured
wall time per request on this machine (batch size 1, greedy).

    ftvp-report qwen05b-zero qwen05b-few qwen05b-lora qwen7b-zero qwen7b-few \
        --ref qwen05b-lora --train-log outputs/train.log
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from ftvp.config import load_config, resolve
from ftvp.massive import Example
from ftvp.metrics import ItemScore, aggregate, bootstrap_ci, paired_bootstrap, score_item
from ftvp.runner import RUNS_DIR, load_run
from ftvp.schema import parse_reply

QUALITY = ("intent_acc", "slot_f1", "exact_match")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_TRAIN = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+[▼▲=]?\s*(\d+)\s")
_VAL = re.compile(r"^\s*(\d+)\s+val\s+([\d.]+)")


def score_records(recs: list[dict]) -> list[ItemScore]:
    out = []
    for r in recs:
        ex = Example(id=r["id"], split="test", utt=r["utt"], intent=r["gold_intent"],
                     slots=tuple(tuple(s) for s in r["gold_slots"]))
        out.append(score_item(ex, parse_reply(r["reply"])))
    return out


def percentile(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def summarise(name: str, recs: list[dict], n_boot: int = 1000) -> dict:
    items = score_records(recs)
    agg = aggregate(items)
    lat = [r["latency_s"] for r in recs]
    return {
        "name": name, **agg,
        "ci": {m: bootstrap_ci(items, m, n_boot=n_boot) for m in QUALITY},
        "latency_p50": statistics.median(lat), "latency_p95": percentile(lat, 0.95),
        "ttft_p50": statistics.median(r.get("ttft_s", 0.0) for r in recs),
        "input_tokens_p50": statistics.median(r["input_tokens"] for r in recs),
        "output_tokens_p50": statistics.median(r["output_tokens"] for r in recs),
        "truncated": sum(r["stop_reason"] == "max_tokens" for r in recs),
        "seconds_per_1k": 1000 * statistics.fmean(lat),
        "api_usd_per_1k": 1000 * statistics.fmean(r.get("cost_usd", 0.0) for r in recs),
    }


def compare(ref: list[dict], other: list[dict], n_boot: int = 1000) -> dict:
    """``other - ref`` on the utterances both runs share."""
    ids = {r["id"] for r in ref} & {r["id"] for r in other}
    a = score_records(sorted((r for r in other if r["id"] in ids), key=lambda r: r["id"]))
    b = score_records(sorted((r for r in ref if r["id"] in ids), key=lambda r: r["id"]))
    return {m: paired_bootstrap(a, b, m, n_boot=n_boot) for m in ("slot_f1", "exact_match")}


def parse_lora_log(text: str) -> dict[str, list[tuple[int, float]]]:
    """Train and validation loss per iteration from ``mlx_lm lora`` console output."""
    train, val = [], []
    for line in _ANSI.sub("", text).splitlines():
        if m := _VAL.match(line):
            val.append((int(m.group(1)), float(m.group(2))))
        elif m := _TRAIN.match(line):
            train.append((int(m.group(1)), float(m.group(2))))
    return {"train": train, "val": val}


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def render(rows: list[dict], metas: dict, ref: str | None, comps: dict, curve: dict | None,
           n: int) -> str:
    lines = [f"# Results on the shared MASSIVE en-US test subset (n = {n})", "",
             "Quality in %, with 95 % bootstrap intervals over utterances. Latency is wall "
             "time per request on an Apple M5 (MLX, batch size 1, greedy decoding).", "",
             "| Run | System | Model | Intent acc | Slot F1 | Exact match | Valid JSON "
             "| Exact schema |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        m = metas.get(r["name"], {})
        ci = r["ci"]
        lines.append(
            f"| {r['name']} | {m.get('system', '?')} | {m.get('backend', '?')} | "
            f"{_pct(r['intent_acc'])} ({_pct(ci['intent_acc'][0])}–{_pct(ci['intent_acc'][1])}) | "
            f"{_pct(r['slot_f1'])} ({_pct(ci['slot_f1'][0])}–{_pct(ci['slot_f1'][1])}) | "
            f"{_pct(r['exact_match'])} ({_pct(ci['exact_match'][0])}–"
            f"{_pct(ci['exact_match'][1])}) | {_pct(r['valid_json'])} | {_pct(r['schema_ok'])} |")
    lines += ["", "## Speed and tokens per request", "",
              "| Run | Latency p50 (s) | p95 (s) | First token p50 (s) | Input tokens p50 "
              "| Output tokens p50 | Truncated | Seconds per 1k requests |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['name']} | {r['latency_p50']:.3f} | {r['latency_p95']:.3f} | "
                     f"{r['ttft_p50']:.3f} | {r['input_tokens_p50']:.0f} | "
                     f"{r['output_tokens_p50']:.0f} | {r['truncated']} | "
                     f"{r['seconds_per_1k']:.0f} |")
    if ref and comps:
        lines += ["", f"## Paired difference vs `{ref}` (other − reference, points)", "",
                  "| Run | Slot F1 Δ (95 % CI) | Exact match Δ (95 % CI) |", "|---|---:|---:|"]
        for name, c in comps.items():
            s, e = c["slot_f1"], c["exact_match"]
            lines.append(f"| {name} | {_pct(s['diff'])} ({_pct(s['lo'])} to {_pct(s['hi'])}) | "
                         f"{_pct(e['diff'])} ({_pct(e['lo'])} to {_pct(e['hi'])}) |")
    if curve and curve["val"]:
        lines += ["", "## LoRA training (validation loss on 500 dev utterances)", "",
                  "| Iteration | Validation loss |", "|---:|---:|"]
        lines += [f"| {it} | {loss:.3f} |" for it, loss in curve["val"]]
    lines += ["", "Produced by `ftvp-report` from `outputs/runs/*.jsonl`."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", help="run names in outputs/runs/")
    ap.add_argument("--ref", default=None, help="run to compare every other run with")
    ap.add_argument("--train-log", default=None, help="mlx_lm lora console log")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    runs_dir = resolve(RUNS_DIR)
    runs, metas = {}, {}
    for name in args.names:
        runs[name] = load_run(runs_dir / f"{name}.jsonl")
        if not runs[name]:
            raise SystemExit(f"no records for run {name!r} in {runs_dir}")
        meta = runs_dir / f"{name}.meta.json"
        metas[name] = json.loads(meta.read_text()) if meta.exists() else {}
    rows = [summarise(n, recs, args.n_boot) for n, recs in runs.items()]
    comps = {}
    if args.ref:
        comps = {n: compare(runs[args.ref], recs, args.n_boot)
                 for n, recs in runs.items() if n != args.ref}
    curve = None
    if args.train_log:
        curve = parse_lora_log(Path(args.train_log).read_text(encoding="utf-8"))
    n = max(len(r) for r in runs.values())
    out_dir = resolve(cfg.get("reports_dir", "reports"))
    out_dir.mkdir(parents=True, exist_ok=True)
    text = render(rows, metas, args.ref, comps, curve, n)
    (out_dir / "results.md").write_text(text, encoding="utf-8")
    (out_dir / "results.json").write_text(json.dumps(
        {"runs": rows, "meta": metas, "ref": args.ref, "paired": comps,
         "training": curve}, indent=2) + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
