"""Prompts for each system, as chat messages (``[{"role": ..., "content": ...}]``).

- ``zero_shot``: instructions + the full label space (60 intents, 55 slot types).
- ``few_shot``: the same, plus k training examples nearest to the query (TF-IDF cosine
  over words) given as earlier user/assistant turns.
- ``finetuned``: a one-line instruction and the utterance. The LoRA model learns the
  labels and the format from training, so it does not pay for them on every request.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Sequence

from ftvp.massive import Example
from ftvp.schema import example_target

_TOKEN = re.compile(r"[a-z0-9']+")

FINETUNED_SYSTEM = "Extract the intent and slots of the user's request as JSON."

_INSTRUCTIONS = """\
You label requests sent to a voice assistant.
Return one line of JSON and nothing else, in this form:
{{"intent": "<intent>", "slots": [["<slot type>", "<value>"], ...]}}

Rules:
- "intent" is exactly one of the intents listed below.
- "slots" lists every slot in the order it appears; use [] when there is none.
- A value is copied word for word from the request. A slot type can appear twice.
- Use only the slot types listed below.

Intents: {intents}

Slot types: {slot_types}"""


def system_prompt(intents: Sequence[str], slot_types: Sequence[str]) -> str:
    return _INSTRUCTIONS.format(intents=", ".join(intents), slot_types=", ".join(slot_types))


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Retriever:
    """TF-IDF cosine nearest neighbours over a fixed pool (the training split).

    A query that shares too few words with the pool ("wooo", "knock knock") is topped up
    to k with pool examples in a fixed seeded order, so every prompt has k shots.
    """

    def __init__(self, pool: Sequence[Example], seed: int = 0):
        self.pool = list(pool)
        self._fallback = list(range(len(self.pool)))
        random.Random(seed).shuffle(self._fallback)
        df: Counter = Counter()
        bags = [Counter(tokens(ex.utt)) for ex in self.pool]
        for bag in bags:
            df.update(bag.keys())
        n = len(self.pool)
        self.idf = {w: math.log((1 + n) / (1 + c)) + 1.0 for w, c in df.items()}
        self.index: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for i, bag in enumerate(bags):
            vec = self._weights(bag)
            for w, x in vec.items():
                self.index[w].append((i, x))

    def _weights(self, bag: Counter) -> dict[str, float]:
        vec = {w: (1 + math.log(c)) * self.idf.get(w, 0.0) for w, c in bag.items()}
        norm = math.sqrt(sum(x * x for x in vec.values())) or 1.0
        return {w: x / norm for w, x in vec.items() if x}

    def nearest(self, utt: str, k: int, exclude_id: str | None = None,
                fill: bool = True) -> list[Example]:
        scores: dict[int, float] = defaultdict(float)
        for w, x in self._weights(Counter(tokens(utt))).items():
            for i, y in self.index.get(w, ()):
                scores[i] += x * y
        order = [i for i, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]
        if fill and len(order) < k + 1:
            seen = set(order)
            order += [i for i in self._fallback if i not in seen][: k + 1 - len(order)]
        return [self.pool[i] for i in order if self.pool[i].id != exclude_id][:k]


def zero_shot(utt: str, system: str) -> list[dict]:
    return [{"role": "system", "content": system}, {"role": "user", "content": utt}]


def few_shot(utt: str, system: str, shots: Sequence[Example]) -> list[dict]:
    """Shots are ordered most similar last, closest to the query."""
    msgs = [{"role": "system", "content": system}]
    for ex in reversed(list(shots)):
        msgs += [{"role": "user", "content": ex.utt},
                 {"role": "assistant", "content": example_target(ex)}]
    msgs.append({"role": "user", "content": utt})
    return msgs


def finetuned(utt: str) -> list[dict]:
    return [{"role": "system", "content": FINETUNED_SYSTEM}, {"role": "user", "content": utt}]
