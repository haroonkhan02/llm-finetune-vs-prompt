from ftvp.check import prompt_lengths, render, retrieval_baseline, split_stats, unseen_labels
from ftvp.massive import by_split, load_examples
from ftvp.prompts import Retriever


def _splits(path):
    return by_split(load_examples(path))


def test_split_stats(jsonl_path):
    rows = {r["split"]: r for r in split_stats(_splits(jsonl_path))}
    assert rows["train"]["n"] == 4 and rows["train"]["intents"] == 3
    assert rows["test"]["repeat_type"] == 0.5     # "monday" and "friday" are both dates
    assert rows["dev"]["no_slot"] == 0.5


def test_unseen_labels(jsonl_path):
    s = _splits(jsonl_path)
    intents, slots = unseen_labels(s["train"], s["test"])
    assert intents == {"calendar_set"} and slots == set()


def test_retrieval_baseline(jsonl_path):
    s = _splits(jsonl_path)
    res = retrieval_baseline(Retriever(s["train"]), s["test"], k=2)
    # "wake me at six am" -> an alarm example; the calendar intent is not in train
    assert res["nn_intent_acc"] == 0.5
    assert res["gold_in_shots"] == 0.5


def test_prompt_lengths_and_render(jsonl_path):
    s = _splits(jsonl_path)
    words = lambda msgs: sum(len(m["content"].split()) for m in msgs)  # noqa: E731
    lengths = prompt_lengths(s["test"], Retriever(s["train"]), "SYS PROMPT", 2, words,
                             lambda t: len(t.split()))
    assert lengths["fine-tuned"]["input"] < lengths["few-shot (k=2)"]["input"]
    assert lengths["zero-shot"]["output"] > 0
    text = render(split_stats(s), {}, {"nn_intent_acc": 0.5, "gold_in_shots": 0.5}, 2, 2,
                  3, 5, [("alarm_set", 2)], lengths, "words")
    assert "| train | 4 |" in text and "few-shot (k=2)" in text and "50.0%" in text
