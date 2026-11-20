"""Write the LoRA training data in the chat format ``mlx_lm.lora`` reads.

``<out_dir>/train.jsonl``, ``valid.jsonl`` and ``test.jsonl``, one
``{"messages": [system, user, assistant]}`` per line, using the short fine-tuned prompt.
Train = the full MASSIVE train split; valid = a seeded dev subset (validation loss only);
test = the shared evaluation subset of the test split.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from ftvp.config import load_config, resolve
from ftvp.download import locale_file
from ftvp.massive import Example, by_split, load_examples, sample
from ftvp.prompts import finetuned
from ftvp.schema import example_target


def chat_record(ex: Example) -> dict:
    return {"messages": finetuned(ex.utt) + [{"role": "assistant",
                                              "content": example_target(ex)}]}


def write_jsonl(path: Path, examples: Sequence[Example]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(chat_record(ex), ensure_ascii=False) + "\n")
    return path


def eval_subset(cfg: dict, splits: dict[str, list[Example]]) -> list[Example]:
    ev = cfg["eval"]
    return sample(splits["test"], ev.get("test_subset"), ev.get("seed", 0))


def prepare(cfg: dict, splits: dict[str, list[Example]], out_dir: Path) -> dict[str, int]:
    seed = cfg["eval"].get("seed", 0)
    parts = {
        "train": splits["train"],
        "valid": sample(splits["dev"], cfg["sft"].get("valid_size"), seed),
        "test": eval_subset(cfg, splits),
    }
    for name, exs in parts.items():
        write_jsonl(out_dir / f"{name}.jsonl", exs)
    return {name: len(exs) for name, exs in parts.items()}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    ds = cfg["dataset"]
    splits = by_split(load_examples(locale_file(resolve(ds["raw_dir"]), ds["locale"])))
    out_dir = resolve(cfg["sft"]["out_dir"])
    for name, n in prepare(cfg, splits, out_dir).items():
        print(f"{out_dir / (name + '.jsonl')}: {n} examples")


if __name__ == "__main__":
    main()
