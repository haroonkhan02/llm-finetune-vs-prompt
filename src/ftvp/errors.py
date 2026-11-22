"""Error analysis: what kind of mistakes each system makes, re-scored from stored replies.

Per run:

- intent errors split into *invented* (not one of the 60 training intents) and *wrong
  label* (a real intent, but not the gold one), plus the most frequent confusions;
- slot errors on utterances whose intent is right, split into *boundary* (same type,
  overlapping value: "five am" vs "five"), *wrong type* (right value, other type),
  *missed* and *extra* slots, and *wrong value*;
- slot F1 per slot type for the most frequent types.

Across two runs (default: the LoRA model and the 7B few-shot model) it counts where they
fail together and lists utterances where both agree with each other but not with the gold
label: the first place to look for annotation noise or genuinely ambiguous requests.

    ftvp-errors qwen05b-zero qwen05b-few qwen05b-lora qwen7b-zero qwen7b-few \
        --pair qwen05b-lora qwen7b-few
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ftvp.config import load_config, resolve
from ftvp.download import locale_file
from ftvp.massive import by_split, label_space, load_examples
from ftvp.metrics import slot_bag
from ftvp.schema import norm_value, parse_reply

SLOT_KINDS = ("boundary", "wrong_type", "missed", "extra", "wrong_value")


@dataclass
class ItemErrors:
    id: str
    utt: str
    gold_intent: str
    pred_intent: str | None
    format_ok: bool
    invented_intent: bool
    slot_errors: Counter
    exact: bool


def _overlaps(a: str, b: str) -> bool:
    a, b = norm_value(a), norm_value(b)
    return bool(a and b) and (a in b or b in a or bool(set(a.split()) & set(b.split())))


def slot_error_kinds(gold: Iterable, pred: Iterable) -> Counter:
    """Classify the differences between two slot lists (matched pairs are removed first).

    Each unmatched gold slot is paired with at most one unmatched predicted slot, in this
    order of preference: same type and overlapping value (boundary), same value and other
    type (wrong type), same type and unrelated value (wrong value). Gold slots left over are
    missed, predicted slots left over are extra.
    """
    g, p = slot_bag(gold), slot_bag(pred)
    common = g & p
    g_left = list((g - common).elements())
    p_left = list((p - common).elements())
    kinds: Counter = Counter()
    rules = (
        ("boundary", lambda gs, ps: gs[0] == ps[0] and _overlaps(gs[1], ps[1])),
        ("wrong_type", lambda gs, ps: gs[0] != ps[0] and gs[1] == ps[1]),
        ("wrong_value", lambda gs, ps: gs[0] == ps[0]),
    )
    for kind, match in rules:
        for gs in list(g_left):
            for ps in p_left:
                if match(gs, ps):
                    kinds[kind] += 1
                    g_left.remove(gs)
                    p_left.remove(ps)
                    break
    kinds["missed"] += len(g_left)
    kinds["extra"] += len(p_left)
    return +kinds


def item_errors(rec: dict, intents: set[str]) -> ItemErrors:
    pred = parse_reply(rec["reply"])
    gold_slots = [tuple(s) for s in rec["gold_slots"]]
    intent_ok = pred.intent == rec["gold_intent"]
    slots = slot_error_kinds(gold_slots, pred.slots) if intent_ok else Counter()
    return ItemErrors(
        id=rec["id"], utt=rec["utt"], gold_intent=rec["gold_intent"], pred_intent=pred.intent,
        format_ok=pred.valid_json and pred.intent is not None,
        invented_intent=pred.intent is not None and pred.intent not in intents,
        slot_errors=slots,
        exact=intent_ok and slot_bag(gold_slots) == slot_bag(pred.slots))


def per_type_f1(recs: Sequence[dict], types: Sequence[str]) -> dict[str, dict]:
    """Value-level slot F1 for each slot type (all utterances, intent right or not)."""
    tp, n_pred, n_gold = Counter(), Counter(), Counter()
    for r in recs:
        g, p = slot_bag(r["gold_slots"]), slot_bag(parse_reply(r["reply"]).slots)
        for (t, _), c in (g & p).items():
            tp[t] += c
        for (t, _), c in p.items():
            n_pred[t] += c
        for (t, _), c in g.items():
            n_gold[t] += c
    out = {}
    for t in types:
        prec = tp[t] / n_pred[t] if n_pred[t] else 0.0
        rec_ = tp[t] / n_gold[t] if n_gold[t] else 0.0
        out[t] = {"f1": 2 * prec * rec_ / (prec + rec_) if prec + rec_ else 0.0,
                  "n_gold": n_gold[t]}
    return out


def summarise_errors(recs: Sequence[dict], intents: set[str], top: int = 10) -> dict:
    items = [item_errors(r, intents) for r in recs]
    n = len(items)
    wrong_intent = [i for i in items if i.pred_intent != i.gold_intent]
    intent_right = [i for i in items if i.pred_intent == i.gold_intent]
    slot_kinds: Counter = Counter()
    for i in intent_right:
        slot_kinds.update(i.slot_errors)
    confusions = Counter((i.gold_intent, i.pred_intent or "<none>")
                         for i in wrong_intent if i.format_ok and not i.invented_intent)
    return {
        "n": n,
        "format_failures": sum(not i.format_ok for i in items),
        "intent_errors": len(wrong_intent),
        "invented_intent": sum(i.invented_intent for i in items),
        "wrong_label": sum(i.format_ok and not i.invented_intent for i in wrong_intent),
        "intent_right_slots_wrong": sum(not i.exact for i in intent_right),
        "slot_error_kinds": {k: slot_kinds.get(k, 0) for k in SLOT_KINDS},
        "top_confusions": [[g, p, c] for (g, p), c in confusions.most_common(top)],
        "invented_examples": Counter(i.pred_intent for i in items
                                     if i.invented_intent).most_common(top),
    }


def pair_overlap(a: Sequence[dict], b: Sequence[dict], intents: set[str],
                 max_examples: int = 20) -> dict:
    """Exact-match agreement between two runs, and items where both make the same mistake."""
    eb = {r["id"]: r for r in b}
    both_wrong, only_a, only_b, both_right, same_wrong = 0, 0, 0, 0, []
    for r in a:
        if r["id"] not in eb:
            continue
        ia, ib = item_errors(r, intents), item_errors(eb[r["id"]], intents)
        if ia.exact and ib.exact:
            both_right += 1
        elif ia.exact:
            only_b += 1          # b wrong only
        elif ib.exact:
            only_a += 1          # a wrong only
        else:
            both_wrong += 1
            pa, pb = parse_reply(r["reply"]), parse_reply(eb[r["id"]]["reply"])
            if pa.intent == pb.intent and slot_bag(pa.slots) == slot_bag(pb.slots):
                same_wrong.append({"id": r["id"], "utt": r["utt"],
                                   "gold": [r["gold_intent"], r["gold_slots"]],
                                   "both": [pa.intent, [list(s) for s in pa.slots]]})
    return {"both_right": both_right, "only_a_wrong": only_a, "only_b_wrong": only_b,
            "both_wrong": both_wrong, "same_wrong_answer": len(same_wrong),
            "same_wrong_examples": same_wrong[:max_examples]}


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def render(summaries: dict, types: dict, pair: dict | None, pair_names: Sequence[str]) -> str:
    names = list(summaries)
    lines = ["# Error analysis on the shared test subset", "",
             "Counts out of the same utterances for every run. Slot errors are counted only "
             "where the intent is right, so they are not double-counted with intent errors.",
             "", "## Intent and format errors", "",
             "| Run | Format failures | Intent wrong | of which invented label | Intent right, "
             "slots wrong |", "|---|---:|---:|---:|---:|"]
    for name, s in summaries.items():
        lines.append(f"| {name} | {s['format_failures']} | {s['intent_errors']} | "
                     f"{s['invented_intent']} | {s['intent_right_slots_wrong']} |")
    lines += ["", "## Slot errors where the intent is right", "",
              "| Run | " + " | ".join(k.replace("_", " ") for k in SLOT_KINDS) + " |",
              "|---|" + "---:|" * len(SLOT_KINDS)]
    for name, s in summaries.items():
        lines.append(f"| {name} | " + " | ".join(str(s["slot_error_kinds"][k])
                                                for k in SLOT_KINDS) + " |")
    lines += ["", "Boundary = right type, overlapping value; wrong type = right value, other "
              "type; wrong value = right type, unrelated value.", "",
              "## Slot F1 by type (most frequent gold types)", "",
              "| Slot type | Gold slots | " + " | ".join(names) + " |",
              "|---|---:|" + "---:|" * len(names)]
    first = next(iter(types.values()))
    for t in first:
        lines.append(f"| {t} | {first[t]['n_gold']} | " +
                     " | ".join(_pct(types[n][t]["f1"]) for n in names) + " |")
    for name, s in summaries.items():
        if not s["top_confusions"]:
            continue
        lines += ["", f"## Most frequent intent confusions: `{name}`", "",
                  "| Gold | Predicted | Count |", "|---|---|---:|"]
        lines += [f"| {g} | {p} | {c} |" for g, p, c in s["top_confusions"]]
    if pair:
        a, b = pair_names
        lines += ["", f"## Where `{a}` and `{b}` fail (exact match)", "",
                  "| Both right | Only " + a + " wrong | Only " + b + " wrong | Both wrong | "
                  "Both give the same wrong answer |", "|---:|---:|---:|---:|---:|",
                  f"| {pair['both_right']} | {pair['only_a_wrong']} | {pair['only_b_wrong']} | "
                  f"{pair['both_wrong']} | {pair['same_wrong_answer']} |", "",
                  "Two different models agreeing on the same non-gold answer is the first place "
                  "to look for label noise or ambiguous requests:", "",
                  "| Utterance | Gold | Both models |", "|---|---|---|"]
        for e in pair["same_wrong_examples"]:
            gold = f"{e['gold'][0]} {json.dumps(e['gold'][1])}"
            both = f"{e['both'][0]} {json.dumps(e['both'][1])}"
            lines.append(f"| {e['utt']} | `{gold}` | `{both}` |")
    lines += ["", "Produced by `ftvp-errors` from `reports/runs/*.jsonl`."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    from ftvp.runner import load_run
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", help="run names")
    ap.add_argument("--runs-dir", default="reports/runs")
    ap.add_argument("--pair", nargs=2, default=None, metavar=("A", "B"),
                    help="two runs to compare item by item")
    ap.add_argument("--top-types", type=int, default=12)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    ds = cfg["dataset"]
    train = by_split(load_examples(locale_file(resolve(ds["raw_dir"]), ds["locale"])))["train"]
    intents = set(label_space(train)[0])
    runs_dir = resolve(args.runs_dir)
    runs = {n: load_run(runs_dir / f"{n}.jsonl") for n in args.names}
    first = next(iter(runs.values()))
    gold_types = Counter(t for r in first for t, _ in r["gold_slots"])
    types = [t for t, _ in gold_types.most_common(args.top_types)]
    summaries = {n: summarise_errors(recs, intents) for n, recs in runs.items()}
    per_type = {n: per_type_f1(recs, types) for n, recs in runs.items()}
    pair = pair_overlap(runs[args.pair[0]], runs[args.pair[1]], intents) if args.pair else None
    out_dir = resolve(cfg.get("reports_dir", "reports"))
    text = render(summaries, per_type, pair, args.pair or ())
    (out_dir / "error_analysis.md").write_text(text, encoding="utf-8")
    (out_dir / "error_analysis.json").write_text(json.dumps(
        {"runs": summaries, "slot_f1_by_type": per_type, "pair": pair,
         "pair_names": args.pair}, indent=2) + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
