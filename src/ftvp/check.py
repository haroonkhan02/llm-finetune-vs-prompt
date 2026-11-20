"""Dataset check: split sizes, label space, slot statistics, a retrieval-only baseline,
and the prompt length each system pays per request. Writes ``reports/data_check.md``.

The baseline labels a test utterance with the intent of its nearest training utterance
(TF-IDF cosine). It needs no LLM and sets the floor every model has to beat. The
"gold intent among the k shots" rate is the ceiling of what few-shot examples show.
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

from ftvp.config import load_config, resolve
from ftvp.download import locale_file
from ftvp.massive import Example, by_split, label_space, load_examples
from ftvp.prompts import Retriever, few_shot, finetuned, system_prompt, zero_shot
from ftvp.schema import example_target
from ftvp.sft import eval_subset

TokenCounter = Callable[[list[dict]], int]   # prompt messages -> input tokens
TextCounter = Callable[[str], int]           # reply text -> output tokens


def split_stats(splits: dict[str, list[Example]]) -> list[dict]:
    rows = []
    for name, exs in splits.items():
        n_slots = [len(e.slots) for e in exs]
        rows.append({
            "split": name, "n": len(exs),
            "intents": len({e.intent for e in exs}),
            "words": statistics.mean(len(e.utt.split()) for e in exs),
            "slots": statistics.mean(n_slots),
            "no_slot": sum(n == 0 for n in n_slots) / len(exs),
            "repeat_type": sum(len({t for t, _ in e.slots}) < len(e.slots) for e in exs)
            / len(exs),
        })
    return rows


def unseen_labels(train: Sequence[Example], other: Sequence[Example]) -> tuple[set, set]:
    intents, slot_types = label_space(train)
    o_int, o_slot = label_space(other)
    return set(o_int) - set(intents), set(o_slot) - set(slot_types)


def retrieval_baseline(retriever: Retriever, test: Sequence[Example], k: int) -> dict:
    hits1 = hitsk = 0
    for ex in test:
        near = retriever.nearest(ex.utt, k)
        hits1 += near[0].intent == ex.intent
        hitsk += any(n.intent == ex.intent for n in near)
    return {"nn_intent_acc": hits1 / len(test), "gold_in_shots": hitsk / len(test)}


def prompt_lengths(test: Sequence[Example], retriever: Retriever, system: str, k: int,
                   count: TokenCounter, count_text: TextCounter) -> dict[str, dict[str, float]]:
    """Median input tokens per request and output tokens of the gold answer."""
    sizes: dict[str, list[int]] = {"zero-shot": [], f"few-shot (k={k})": [], "fine-tuned": []}
    for ex in test:
        sizes["zero-shot"].append(count(zero_shot(ex.utt, system)))
        sizes[f"few-shot (k={k})"].append(
            count(few_shot(ex.utt, system, retriever.nearest(ex.utt, k))))
        sizes["fine-tuned"].append(count(finetuned(ex.utt)))
    out_tokens = statistics.median(count_text(example_target(e)) for e in test)
    return {name: {"input": statistics.median(v), "output": out_tokens}
            for name, v in sizes.items()}


def hf_token_counters(tokenizer_id: str) -> tuple[TokenCounter, TextCounter]:
    """Token counters from a model's tokenizer and chat template (needs the ``mlx`` extra)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_id)

    def count(msgs: list[dict]) -> int:
        # recent transformers return a BatchEncoding here, older ones a list of ids
        ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        return len(ids["input_ids"] if hasattr(ids, "keys") else ids)

    def count_text(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False))

    return count, count_text


def render(stats: list[dict], unseen: dict, base: dict, k: int, n_test: int,
           n_intents: int, n_slot_types: int, top_intents: list, lengths: dict | None,
           tokenizer_id: str | None) -> str:
    lines = ["# MASSIVE en-US: data check", "",
             f"Label space (train): {n_intents} intents, {n_slot_types} slot types.", "",
             "| Split | Utterances | Intents | Mean words | Mean slots | No slot "
             "| Repeated slot type |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in stats:
        lines.append(f"| {r['split']} | {r['n']:,} | {r['intents']} | {r['words']:.1f} | "
                     f"{r['slots']:.2f} | {r['no_slot']:.1%} | {r['repeat_type']:.1%} |")
    lines += ["", "Labels in dev/test that never occur in train: "
              f"{unseen or 'none'}.", "",
              "Most frequent train intents: "
              + ", ".join(f"{i} ({c})" for i, c in top_intents) + ".", "",
              f"## Retrieval-only baseline (evaluation subset, n = {n_test})", "",
              "| | Value |", "|---|---:|",
              f"| Intent of the nearest train utterance (1-NN) | {base['nn_intent_acc']:.1%} |",
              f"| Gold intent among the {k} retrieved shots | {base['gold_in_shots']:.1%} |"]
    if lengths:
        lines += ["", f"## Prompt length per request (median tokens, `{tokenizer_id}` "
                  "chat template)", "", "| System | Input | Output (gold answer) |",
                  "|---|---:|---:|"]
        for name, v in lengths.items():
            lines.append(f"| {name} | {v['input']:.0f} | {v['output']:.0f} |")
    lines += ["", "Produced by `ftvp-check`."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--tokenizer", default=None,
                    help="HF id of a tokenizer for prompt lengths, e.g. "
                         "mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    ds = cfg["dataset"]
    splits = by_split(load_examples(locale_file(resolve(ds["raw_dir"]), ds["locale"])))
    intents, slot_types = label_space(splits["train"])
    test = eval_subset(cfg, splits)
    k = cfg["prompting"]["few_shot_k"]
    retriever = Retriever(splits["train"])
    unseen = {}
    for name in ("dev", "test"):
        ui, us = unseen_labels(splits["train"], splits[name])
        if ui or us:
            unseen[name] = {"intents": sorted(ui), "slot_types": sorted(us)}
    lengths = None
    if args.tokenizer:
        lengths = prompt_lengths(test, retriever, system_prompt(intents, slot_types), k,
                                 *hf_token_counters(args.tokenizer))
    top = Counter(e.intent for e in splits["train"]).most_common(5)
    text = render(split_stats(splits), unseen, retrieval_baseline(retriever, test, k), k,
                  len(test), len(intents), len(slot_types), top, lengths, args.tokenizer)
    out = Path(resolve(cfg.get("reports_dir", "reports"))) / "data_check.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
