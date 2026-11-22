import json
from collections import Counter

from ftvp.errors import pair_overlap, per_type_f1, render, slot_error_kinds, summarise_errors

INTENTS = {"alarm_set", "alarm_query", "weather_query"}


def rec(i, intent, slots, gold_intent="alarm_set", gold_slots=(("time", "six am"),)):
    return {"id": str(i), "utt": "wake me at six am tomorrow", "gold_intent": gold_intent,
            "gold_slots": [list(s) for s in gold_slots],
            "reply": json.dumps({"intent": intent, "slots": [list(s) for s in slots]})}


def test_slot_error_kinds():
    gold = [("time", "six am"), ("date", "tomorrow")]
    assert slot_error_kinds(gold, gold) == Counter()
    assert slot_error_kinds(gold, [("time", "six"), ("date", "tomorrow")]) == {"boundary": 1}
    assert slot_error_kinds(gold, [("time", "six am"), ("time", "tomorrow")]) == {
        "wrong_type": 1}
    assert slot_error_kinds(gold, [("time", "six am")]) == {"missed": 1}
    assert slot_error_kinds([], [("date", "today")]) == {"extra": 1}
    assert slot_error_kinds([("place_name", "rome")], [("place_name", "paris")]) == {
        "wrong_value": 1}


def test_summarise_errors_splits_intent_and_slot_errors():
    recs = [rec(0, "alarm_set", [("time", "six am")]),
            rec(1, "alarm_query", []),                       # wrong real label
            rec(2, "set_alarm", [("time", "six am")]),       # invented label
            rec(3, "alarm_set", [("time", "six")]),          # boundary
            {**rec(4, "x", []), "reply": "not json"}]
    s = summarise_errors(recs, INTENTS)
    assert s["n"] == 5
    assert s["format_failures"] == 1
    assert s["intent_errors"] == 3
    assert s["invented_intent"] == 1
    assert s["wrong_label"] == 1
    assert s["intent_right_slots_wrong"] == 1
    assert s["slot_error_kinds"]["boundary"] == 1
    assert s["top_confusions"] == [["alarm_set", "alarm_query", 1]]


def test_per_type_f1():
    recs = [rec(0, "alarm_set", [("time", "six am")]), rec(1, "alarm_set", [])]
    f = per_type_f1(recs, ["time"])
    assert f["time"]["n_gold"] == 2
    assert abs(f["time"]["f1"] - 2 / 3) < 1e-9


def test_pair_overlap_and_render():
    a = [rec(0, "alarm_set", [("time", "six am")]), rec(1, "alarm_query", []),
         rec(2, "alarm_query", [])]
    b = [rec(0, "alarm_set", [("time", "six am")]), rec(1, "alarm_query", []),
         rec(2, "alarm_set", [("time", "six am")])]
    p = pair_overlap(a, b, INTENTS)
    assert (p["both_right"], p["only_a_wrong"], p["only_b_wrong"], p["both_wrong"]) == (
        1, 1, 0, 1)
    assert p["same_wrong_answer"] == 1
    summaries = {"a": summarise_errors(a, INTENTS), "b": summarise_errors(b, INTENTS)}
    types = {"a": per_type_f1(a, ["time"]), "b": per_type_f1(b, ["time"])}
    text = render(summaries, types, p, ("a", "b"))
    assert "Both give the same wrong answer" in text
    assert "| time | 3 |" in text
