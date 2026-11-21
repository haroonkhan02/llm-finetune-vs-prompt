"""Run one system (prompt style + model) over the shared evaluation subset.

Each request is appended to ``outputs/runs/<name>.jsonl`` as soon as it finishes, keyed by
utterance id, so an interrupted run resumes where it stopped. One warm-up request runs
before timing starts (the first MLX call compiles kernels).

    ftvp-run --system zero-shot --backend mlx:mlx-community/Qwen2.5-0.5B-Instruct-bf16 \
             --name qwen05b-zero
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from ftvp.backends import Backend, make_backend
from ftvp.config import load_config, resolve
from ftvp.download import locale_file
from ftvp.massive import Example, by_split, label_space, load_examples
from ftvp.prompts import Retriever, few_shot, finetuned, system_prompt, zero_shot
from ftvp.schema import parse_reply
from ftvp.sft import eval_subset

SYSTEMS = ("zero-shot", "few-shot", "finetuned")
RUNS_DIR = "outputs/runs"


def prompt_builder(system: str, train: list[Example], k: int) -> Callable[[Example], list[dict]]:
    if system == "finetuned":
        return lambda ex: finetuned(ex.utt)
    intents, slot_types = label_space(train)
    sys_text = system_prompt(intents, slot_types)
    if system == "zero-shot":
        return lambda ex: zero_shot(ex.utt, sys_text)
    if system == "few-shot":
        retriever = Retriever(train)
        return lambda ex: few_shot(ex.utt, sys_text, retriever.nearest(ex.utt, k))
    raise ValueError(f"unknown system {system!r}; choose from {SYSTEMS}")


def load_run(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def record(ex: Example, messages: list[dict], comp) -> dict:
    pred = parse_reply(comp.text)
    return {
        "id": ex.id, "utt": ex.utt, "gold_intent": ex.intent,
        "gold_slots": [list(s) for s in ex.slots],
        "reply": comp.text, "pred_intent": pred.intent,
        "pred_slots": [list(s) for s in pred.slots],
        "valid_json": pred.valid_json, "schema_ok": pred.schema_ok, "clean": pred.clean,
        "errors": pred.errors, "n_messages": len(messages),
        **{k: v for k, v in comp.as_dict().items() if k != "text"},
    }


def run(backend: Backend, examples: Iterable[Example], build: Callable[[Example], list[dict]],
        out_path: Path, warmup: bool = True, log_every: int = 100) -> int:
    """Run ``examples`` not yet in ``out_path``; return how many were newly done."""
    done = {r["id"] for r in load_run(out_path)}
    todo = [ex for ex in examples if ex.id not in done]
    if not todo:
        return 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if warmup:
        backend.complete(build(todo[0]))
    t0 = time.perf_counter()
    with open(out_path, "a", encoding="utf-8") as f:
        for i, ex in enumerate(todo, 1):
            msgs = build(ex)
            f.write(json.dumps(record(ex, msgs, backend.complete(msgs)),
                               ensure_ascii=False) + "\n")
            f.flush()
            if log_every and i % log_every == 0:
                rate = i / (time.perf_counter() - t0)
                print(f"{out_path.stem}: {i}/{len(todo)} ({rate:.1f}/s)", flush=True)
    return len(todo)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", choices=SYSTEMS, required=True)
    ap.add_argument("--backend", required=True,
                    help="mlx:<hf repo> | anthropic:<model>[:effort] | fake")
    ap.add_argument("--adapter", default=None, help="LoRA adapter directory (mlx only)")
    ap.add_argument("--name", required=True, help="run name -> outputs/runs/<name>.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="first N items (smoke runs)")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    ds = cfg["dataset"]
    splits = by_split(load_examples(locale_file(resolve(ds["raw_dir"]), ds["locale"])))
    examples = eval_subset(cfg, splits)[: args.limit]
    build = prompt_builder(args.system, splits["train"], cfg["prompting"]["few_shot_k"])
    backend = make_backend(args.backend, adapter=args.adapter, max_tokens=args.max_tokens)
    out = resolve(RUNS_DIR) / f"{args.name}.jsonl"
    meta = {"name": args.name, "system": args.system, "backend": backend.name,
            "adapter": args.adapter, "max_tokens": args.max_tokens}
    meta_path = out.with_suffix(".meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    n = run(backend, examples, build, out)
    print(f"{out}: {n} new, {len(load_run(out))} total")


if __name__ == "__main__":
    main()
