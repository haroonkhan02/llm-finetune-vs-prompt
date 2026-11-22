"""MASSIVE loader: one utterance -> intent label + list of (slot type, value) pairs.

Annotated utterances look like ``wake me up at [time : five am] [date : this week]``.
Slots are kept as an ordered list, not a dict, because about 3 % of English utterances repeat
a slot type (e.g. two ``date`` values).
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

SLOT_RE = re.compile(r"\[([^\]:]+?) : ([^\]]*)\]")
SPLITS = ("train", "dev", "test")


@dataclass(frozen=True)
class Example:
    id: str
    split: str
    utt: str
    intent: str
    slots: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @property
    def scenario(self) -> str:
        return self.intent.split("_", 1)[0]


def parse_annotation(annot_utt: str) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Return (plain utterance, slots) from an annotated MASSIVE utterance."""
    slots = tuple((t.strip(), v.strip()) for t, v in SLOT_RE.findall(annot_utt))
    plain = SLOT_RE.sub(lambda m: m.group(2), annot_utt)
    if "[" in plain or "]" in plain:
        raise ValueError(f"unbalanced slot brackets: {annot_utt!r}")
    return plain, slots


def example_from_record(rec: dict) -> Example:
    plain, slots = parse_annotation(rec["annot_utt"])
    if plain != rec["utt"]:
        raise ValueError(f"id {rec['id']}: annotation does not match utterance")
    return Example(id=str(rec["id"]), split=rec["partition"], utt=rec["utt"],
                   intent=rec["intent"], slots=slots)


def load_examples(path: str | Path) -> list[Example]:
    with open(path, encoding="utf-8") as f:
        return [example_from_record(json.loads(line)) for line in f if line.strip()]


def by_split(examples: Iterable[Example]) -> dict[str, list[Example]]:
    out: dict[str, list[Example]] = {s: [] for s in SPLITS}
    for ex in examples:
        out.setdefault(ex.split, []).append(ex)
    return out


def label_space(examples: Iterable[Example]) -> tuple[list[str], list[str]]:
    """Sorted intent labels and slot types seen in ``examples`` (use the train split)."""
    intents, slot_types = set(), set()
    for ex in examples:
        intents.add(ex.intent)
        slot_types.update(t for t, _ in ex.slots)
    return sorted(intents), sorted(slot_types)


def sample(examples: list[Example], n: int | None, seed: int) -> list[Example]:
    """A seeded subset, returned in original order so runs are easy to diff."""
    if n is None or n >= len(examples):
        return list(examples)
    idx = sorted(random.Random(seed).sample(range(len(examples)), n))
    return [examples[i] for i in idx]
