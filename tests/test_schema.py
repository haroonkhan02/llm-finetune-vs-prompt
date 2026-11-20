import json

from ftvp.massive import Example
from ftvp.schema import example_target, norm_value, parse_reply

EX = Example(id="0", split="test", utt="wake me up at five am this week", intent="alarm_set",
             slots=(("time", "five am"), ("date", "this week")))


def test_target_round_trips():
    text = example_target(EX)
    assert json.loads(text) == {"intent": "alarm_set",
                                "slots": [["time", "five am"], ["date", "this week"]]}
    pred = parse_reply(text)
    assert pred.clean and pred.schema_ok and pred.valid_json
    assert pred.intent == "alarm_set"
    assert pred.slots == [("time", "five am"), ("date", "this week")]


def test_fenced_reply_is_parsed_but_not_clean():
    pred = parse_reply('Sure!\n```json\n{"intent": "alarm_set", "slots": []}\n```')
    assert pred.valid_json and pred.schema_ok and not pred.clean
    assert pred.intent == "alarm_set"


def test_dict_slots_accepted_with_error_recorded():
    pred = parse_reply('{"intent": "Alarm_Set", "slots": {"time": "five am"}}')
    assert pred.intent == "alarm_set"
    assert pred.slots == [("time", "five am")]
    assert not pred.schema_ok and "slots given as a dict" in pred.errors


def test_object_slots_accepted():
    pred = parse_reply('{"intent": "x", "slots": [{"type": "Time", "value": "9"}]}')
    assert pred.slots == [("time", "9")] and not pred.schema_ok


def test_garbage_and_missing_fields():
    assert parse_reply("I think it is an alarm").errors == ["no JSON object"]
    assert parse_reply("").valid_json is False
    pred = parse_reply('{"slots": [["time", "x", "y"]]}')
    assert pred.valid_json and not pred.schema_ok
    assert pred.intent is None and pred.slots == []
    assert "missing intent" in pred.errors and "bad slots" in pred.errors


def test_extra_keys_break_schema_only():
    pred = parse_reply('{"intent": "alarm_set", "slots": [], "confidence": 0.9}')
    assert pred.intent == "alarm_set" and not pred.schema_ok


def test_skips_braces_that_are_not_json():
    pred = parse_reply('Format: {intent}. Answer: {"intent": "play_music", "slots": []}')
    assert pred.intent == "play_music"


def test_norm_value():
    assert norm_value("  Five  AM. ") == "five am"
