import json

import pytest

from ftvp.report import compare, parse_lora_log, percentile, render, score_records, summarise


def rec(i, reply, gold_intent="alarm_set", gold_slots=(("time", "six am"),), lat=0.1):
    return {"id": str(i), "utt": "wake me at six am", "gold_intent": gold_intent,
            "gold_slots": [list(s) for s in gold_slots], "reply": reply,
            "latency_s": lat, "ttft_s": lat / 2, "input_tokens": 30, "output_tokens": 15,
            "stop_reason": "end_turn", "cost_usd": 0.0}


GOOD = json.dumps({"intent": "alarm_set", "slots": [["time", "six am"]]})
WRONG = json.dumps({"intent": "alarm_query", "slots": []})


def test_score_records_reparses_reply():
    items = score_records([rec(0, GOOD), rec(1, "```json\n" + WRONG + "\n```"), rec(2, "no")])
    assert [i.exact for i in items] == [True, False, False]
    assert [i.valid_json for i in items] == [True, True, False]


def test_summarise_quality_and_speed():
    recs = [rec(i, GOOD if i % 2 else WRONG, lat=0.1 * (i + 1)) for i in range(10)]
    s = summarise("r", recs, n_boot=200)
    assert s["intent_acc"] == pytest.approx(0.5)
    assert s["ci"]["intent_acc"][0] <= 0.5 <= s["ci"]["intent_acc"][1]
    assert s["latency_p50"] == pytest.approx(0.55)
    assert s["seconds_per_1k"] == pytest.approx(550)
    assert percentile([1, 2, 3, 4], 0.95) == 4


def test_compare_is_paired_on_shared_ids():
    ref = [rec(i, WRONG) for i in range(20)]
    other = [rec(i, GOOD) for i in range(20)] + [rec(99, WRONG)]
    c = compare(ref, other, n_boot=200)
    assert c["exact_match"]["diff"] == pytest.approx(1.0)
    assert c["exact_match"]["lo"] == pytest.approx(1.0)


LOG = """\x1b[1m  iter   train_loss     tok/s     tokens\x1b[0m
     1    val 3.264    8.32s
    20    1.416 ▼      635      7.2k
    40    0.504 ▼      684     14.2k
   120    val 0.210    7.90s
"""


def test_parse_lora_log():
    curve = parse_lora_log(LOG)
    assert curve["val"] == [(1, 3.264), (120, 0.21)]
    assert curve["train"] == [(20, 1.416), (40, 0.504)]


def test_render_has_tables():
    rows = [summarise("a", [rec(0, GOOD), rec(1, WRONG)], n_boot=50)]
    text = render(rows, {"a": {"system": "finetuned", "backend": "fake"}}, None, {},
                  parse_lora_log(LOG), 2)
    assert "| a | finetuned | fake | 50.0" in text and "| 120 | 0.210 |" in text
