import pytest

from ftvp.massive import (
    by_split,
    example_from_record,
    label_space,
    load_examples,
    parse_annotation,
    sample,
)


def test_parse_annotation_keeps_order_and_repeats():
    plain, slots = parse_annotation("remind me on [date : monday] and on [date : friday]")
    assert plain == "remind me on monday and on friday"
    assert slots == (("date", "monday"), ("date", "friday"))


def test_parse_annotation_without_slots():
    assert parse_annotation("tell me a joke") == ("tell me a joke", ())


def test_parse_annotation_rejects_unbalanced_brackets():
    with pytest.raises(ValueError):
        parse_annotation("wake me [time : five am")


def test_record_must_match_utterance(records):
    bad = dict(records[0], utt="something else")
    with pytest.raises(ValueError):
        example_from_record(bad)


def test_load_and_split(jsonl_path):
    exs = load_examples(jsonl_path)
    splits = by_split(exs)
    assert [len(splits[s]) for s in ("train", "dev", "test")] == [4, 2, 2]
    assert exs[2].scenario == "weather"
    intents, slot_types = label_space(splits["train"])
    assert intents == ["alarm_set", "play_music", "weather_query"]
    assert "music_genre" in slot_types and "place_name" in slot_types


def test_sample_is_seeded_and_ordered(jsonl_path):
    exs = load_examples(jsonl_path)
    a, b = sample(exs, 4, seed=1), sample(exs, 4, seed=1)
    assert a == b and len(a) == 4
    assert [e.id for e in a] == sorted((e.id for e in a), key=int)
    assert sample(exs, None, seed=0) == exs
