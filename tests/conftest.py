import json

import pytest

RECORDS = [
    {"id": "0", "partition": "train", "intent": "alarm_set",
     "utt": "wake me up at five am this week",
     "annot_utt": "wake me up at [time : five am] [date : this week]"},
    {"id": "1", "partition": "train", "intent": "alarm_set", "utt": "set an alarm for nine am",
     "annot_utt": "set an alarm for [time : nine am]"},
    {"id": "2", "partition": "train", "intent": "weather_query",
     "utt": "is it raining in paris today",
     "annot_utt": "is it [weather_descriptor : raining] in [place_name : paris] [date : today]"},
    {"id": "3", "partition": "train", "intent": "play_music", "utt": "play some jazz",
     "annot_utt": "play some [music_genre : jazz]"},
    {"id": "4", "partition": "dev", "intent": "weather_query", "utt": "weather in rome",
     "annot_utt": "weather in [place_name : rome]"},
    {"id": "5", "partition": "dev", "intent": "general_joke", "utt": "tell me a joke",
     "annot_utt": "tell me a joke"},
    {"id": "6", "partition": "test", "intent": "calendar_set",
     "utt": "remind me on monday and on friday",
     "annot_utt": "remind me on [date : monday] and on [date : friday]"},
    {"id": "7", "partition": "test", "intent": "alarm_set", "utt": "wake me at six am",
     "annot_utt": "wake me at [time : six am]"},
]


@pytest.fixture
def records():
    return [dict(r, locale="en-US", scenario=r["intent"].split("_")[0]) for r in RECORDS]


@pytest.fixture
def jsonl_path(tmp_path, records):
    path = tmp_path / "en-US.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path
