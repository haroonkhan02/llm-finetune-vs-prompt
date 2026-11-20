import json

from ftvp.config import load_config
from ftvp.massive import by_split, load_examples
from ftvp.prompts import FINETUNED_SYSTEM
from ftvp.sft import chat_record, prepare


def test_chat_record_layout(jsonl_path):
    ex = load_examples(jsonl_path)[0]
    msgs = chat_record(ex)["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[0]["content"] == FINETUNED_SYSTEM
    assert msgs[1]["content"] == ex.utt
    assert json.loads(msgs[2]["content"])["slots"] == [["time", "five am"],
                                                       ["date", "this week"]]


def test_prepare_writes_three_files(jsonl_path, tmp_path):
    cfg = load_config()
    cfg["eval"] = {"test_subset": 1, "seed": 0}
    cfg["sft"] = {"valid_size": 1}
    splits = by_split(load_examples(jsonl_path))
    counts = prepare(cfg, splits, tmp_path / "sft")
    assert counts == {"train": 4, "valid": 1, "test": 1}
    lines = (tmp_path / "sft" / "train.jsonl").read_text().splitlines()
    assert len(lines) == 4 and all("messages" in json.loads(x) for x in lines)
    test_line = json.loads((tmp_path / "sft" / "test.jsonl").read_text())
    assert test_line["messages"][1]["content"] in {e.utt for e in splits["test"]}
