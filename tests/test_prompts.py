import json

from ftvp.massive import by_split, label_space, load_examples
from ftvp.prompts import (
    FINETUNED_SYSTEM,
    Retriever,
    few_shot,
    finetuned,
    system_prompt,
    tokens,
    zero_shot,
)


def test_system_prompt_lists_labels():
    text = system_prompt(["alarm_set", "play_music"], ["date", "time"])
    assert "Intents: alarm_set, play_music" in text
    assert "Slot types: date, time" in text
    assert '{"intent": "<intent>"' in text  # braces survive str.format


def test_tokens():
    assert tokens("What's the WEATHER, in Paris?") == ["what's", "the", "weather", "in", "paris"]


def test_retriever_finds_nearest(jsonl_path):
    train = by_split(load_examples(jsonl_path))["train"]
    r = Retriever(train)
    top = r.nearest("will it be raining in paris", k=2)
    assert top[0].intent == "weather_query"
    assert len(top) <= 2
    assert all(e.id != "2" for e in r.nearest("is it raining in paris today", 3, exclude_id="2"))


def test_retriever_tops_up_unknown_queries(jsonl_path):
    r = Retriever(by_split(load_examples(jsonl_path))["train"])
    assert r.nearest("zzz qqq", k=3, fill=False) == []
    filled = r.nearest("zzz qqq", k=3)
    assert len(filled) == 3 and len({e.id for e in filled}) == 3
    assert filled == r.nearest("zzz qqq", k=3)  # same seeded order every time
    part = r.nearest("play jazz", k=3)
    assert part[0].intent == "play_music" and len(part) == 3


def test_few_shot_message_layout(jsonl_path):
    train = by_split(load_examples(jsonl_path))["train"]
    shots = train[:2]
    msgs = few_shot("wake me at six", "SYS", shots)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant",
                                         "user"]
    # most similar shot (first) sits right before the query
    assert msgs[-3]["content"] == shots[0].utt
    assert json.loads(msgs[-2]["content"])["intent"] == shots[0].intent
    assert msgs[-1]["content"] == "wake me at six"


def test_zero_shot_and_finetuned():
    assert zero_shot("hi", "SYS") == [{"role": "system", "content": "SYS"},
                                      {"role": "user", "content": "hi"}]
    assert finetuned("hi")[0]["content"] == FINETUNED_SYSTEM


def test_label_space_on_fixture(jsonl_path):
    intents, slots = label_space(by_split(load_examples(jsonl_path))["train"])
    assert len(intents) == 3 and "time" in slots
