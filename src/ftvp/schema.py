"""The output format every system must produce, and a tolerant parser for model replies.

Target (one line of JSON)::

    {"intent": "alarm_set", "slots": [["time", "five am"], ["date", "this week"]]}

Slots are ``[type, value]`` pairs in utterance order, values copied from the utterance.
The parser accepts surrounding prose or a code fence (prompted models add them) but
records what it had to forgive, so format compliance is a metric of its own.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ftvp.massive import Example

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_WS = re.compile(r"\s+")


def target_json(intent: str, slots) -> str:
    """Serialise a label as the exact string a fine-tuned model is trained to emit."""
    return json.dumps({"intent": intent, "slots": [[t, v] for t, v in slots]},
                      ensure_ascii=False)


def example_target(ex: Example) -> str:
    return target_json(ex.intent, ex.slots)


def norm_value(value: str) -> str:
    """Lower-case, collapse whitespace and trim punctuation at the ends."""
    return _WS.sub(" ", str(value).lower()).strip(" \t\n.,;:!?'\"")


@dataclass
class Prediction:
    intent: str | None = None
    slots: list[tuple[str, str]] = field(default_factory=list)
    valid_json: bool = False      # a JSON object was found
    schema_ok: bool = False       # it had the exact target shape
    clean: bool = False           # the whole reply was that object, nothing else
    errors: list[str] = field(default_factory=list)


def _first_object(text: str) -> tuple[dict | None, bool]:
    """Return the first decodable JSON object in ``text`` and whether it was the whole reply."""
    stripped = text.strip()
    candidates = [stripped]
    candidates += [m.group(1).strip() for m in _FENCE.finditer(text)]
    dec = json.JSONDecoder()
    for cand in candidates:
        for i, ch in enumerate(cand):
            if ch != "{":
                continue
            try:
                obj, end = dec.raw_decode(cand, i)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                whole = cand is stripped and i == 0 and end == len(stripped)
                return obj, whole
    return None, False


def _coerce_slots(raw, errors: list[str]) -> list[tuple[str, str]] | None:
    """Accept the target list of pairs; also a {type: value} dict or list of {type, value}."""
    if raw is None:
        return []
    if isinstance(raw, dict):
        errors.append("slots given as a dict")
        return [(str(t), str(v)) for t, v in raw.items()]
    if not isinstance(raw, list):
        return None
    out, shape_ok = [], True
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            out.append((str(item[0]), str(item[1])))
        elif isinstance(item, dict) and {"type", "value"} <= item.keys():
            out.append((str(item["type"]), str(item["value"])))
            shape_ok = False
        else:
            return None
    if not shape_ok:
        errors.append("slots given as objects")
    return out


def parse_reply(text: str) -> Prediction:
    pred = Prediction()
    obj, whole = _first_object(text or "")
    if obj is None:
        pred.errors.append("no JSON object")
        return pred
    pred.valid_json = True
    intent = obj.get("intent")
    if isinstance(intent, str) and intent.strip():
        pred.intent = intent.strip().lower()
    else:
        pred.errors.append("missing intent")
    slots = _coerce_slots(obj.get("slots"), pred.errors)
    if slots is None:
        pred.errors.append("bad slots")
        slots = []
    pred.slots = [(t.strip().lower(), v) for t, v in slots]
    pred.schema_ok = not pred.errors and set(obj) == {"intent", "slots"}
    pred.clean = pred.schema_ok and whole
    return pred
