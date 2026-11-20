"""Scores for intent + slot extraction, with bootstrap confidence intervals.

- intent accuracy
- slot precision / recall / F1, micro-averaged over (type, normalised value) multisets
- exact match: intent right and the slot multiset identical
- format: valid JSON, exact schema, clean reply (nothing around the JSON)
- grounded: share of predicted slot values that occur in the utterance (copying, not
  inventing; a proxy for hallucinated values)

Slot F1 here is value-level, not the span-level F1 of the MASSIVE paper: a generative
model returns strings, not token offsets, so a value is right when it matches the gold
value after normalisation.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from ftvp.massive import Example
from ftvp.schema import Prediction, norm_value


def slot_bag(slots) -> Counter:
    return Counter((t.strip().lower(), norm_value(v)) for t, v in slots)


@dataclass
class ItemScore:
    intent_ok: bool
    exact: bool
    tp: int
    n_pred: int
    n_gold: int
    valid_json: bool
    schema_ok: bool
    clean: bool
    grounded: int


def score_item(ex: Example, pred: Prediction) -> ItemScore:
    gold, got = slot_bag(ex.slots), slot_bag(pred.slots)
    tp = sum((gold & got).values())
    intent_ok = pred.intent == ex.intent
    utt = norm_value(ex.utt)
    grounded = sum(1 for _, v in pred.slots if norm_value(v) and norm_value(v) in utt)
    return ItemScore(intent_ok=intent_ok, exact=intent_ok and gold == got, tp=tp,
                     n_pred=sum(got.values()), n_gold=sum(gold.values()),
                     valid_json=pred.valid_json, schema_ok=pred.schema_ok,
                     clean=pred.clean, grounded=grounded)


def _f1(tp: int, n_pred: int, n_gold: int) -> tuple[float, float, float]:
    p = tp / n_pred if n_pred else (1.0 if n_gold == 0 else 0.0)
    r = tp / n_gold if n_gold else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def aggregate(items: Sequence[ItemScore]) -> dict[str, float]:
    n = len(items)
    if n == 0:
        raise ValueError("no items to aggregate")
    tp = sum(i.tp for i in items)
    n_pred = sum(i.n_pred for i in items)
    n_gold = sum(i.n_gold for i in items)
    p, r, f = _f1(tp, n_pred, n_gold)
    return {
        "n": n,
        "intent_acc": sum(i.intent_ok for i in items) / n,
        "slot_p": p, "slot_r": r, "slot_f1": f,
        "exact_match": sum(i.exact for i in items) / n,
        "valid_json": sum(i.valid_json for i in items) / n,
        "schema_ok": sum(i.schema_ok for i in items) / n,
        "clean": sum(i.clean for i in items) / n,
        "grounded": sum(i.grounded for i in items) / n_pred if n_pred else 1.0,
    }


def bootstrap_ci(items: Sequence[ItemScore], metric: str, n_boot: int = 1000,
                 seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap over utterances for any key returned by ``aggregate``."""
    rng = random.Random(seed)
    n = len(items)
    stats = sorted(aggregate([items[rng.randrange(n)] for _ in range(n)])[metric]
                   for _ in range(n_boot))
    lo = stats[int(alpha / 2 * n_boot)]
    hi = stats[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return lo, hi


def paired_bootstrap(a: Sequence[ItemScore], b: Sequence[ItemScore], metric: str,
                     n_boot: int = 1000, seed: int = 0) -> dict[str, float]:
    """Difference ``metric(a) - metric(b)`` on the same utterances, with a 95 % CI and
    the share of resamples where ``a`` is not better (a one-sided p-value)."""
    if len(a) != len(b):
        raise ValueError("paired bootstrap needs runs on the same items")
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(aggregate([a[i] for i in idx])[metric]
                     - aggregate([b[i] for i in idx])[metric])
    diffs.sort()
    return {"diff": aggregate(a)[metric] - aggregate(b)[metric],
            "lo": diffs[int(0.025 * n_boot)], "hi": diffs[min(n_boot - 1, int(0.975 * n_boot))],
            "p_not_better": sum(d <= 0 for d in diffs) / n_boot}
