import json

from ftvp.backends import FakeBackend
from ftvp.massive import by_split, load_examples
from ftvp.prompts import FINETUNED_SYSTEM
from ftvp.runner import load_run, prompt_builder, run


def gold_reply(messages):
    utt = messages[-1]["content"]
    return {"wake me at six am": 'ok: {"intent": "alarm_set", "slots": [["time", "six am"]]}',
            }.get(utt, '{"intent": "calendar_set", "slots": [["date", "monday"]]}')


def test_prompt_builders(jsonl_path):
    splits = by_split(load_examples(jsonl_path))
    ex = splits["test"][0]
    assert prompt_builder("finetuned", splits["train"], 2)(ex)[0]["content"] == FINETUNED_SYSTEM
    zero = prompt_builder("zero-shot", splits["train"], 2)(ex)
    few = prompt_builder("few-shot", splits["train"], 2)(ex)
    assert len(zero) == 2 and len(few) == 2 + 2 * 2
    assert "alarm_set" in zero[0]["content"]


def test_run_writes_records_and_resumes(jsonl_path, tmp_path):
    splits = by_split(load_examples(jsonl_path))
    build = prompt_builder("finetuned", splits["train"], 2)
    out = tmp_path / "runs" / "r.jsonl"
    backend = FakeBackend(gold_reply)
    assert run(backend, splits["test"][:1], build, out, log_every=0) == 1
    assert run(backend, splits["test"], build, out, log_every=0) == 1   # only the new one
    assert run(backend, splits["test"], build, out, log_every=0) == 0
    recs = load_run(out)
    assert [r["id"] for r in recs] == ["6", "7"]
    six = recs[1]
    assert six["pred_intent"] == "alarm_set" and six["pred_slots"] == [["time", "six am"]]
    assert six["valid_json"] and not six["clean"]          # prose before the JSON
    assert recs[0]["gold_slots"] == [["date", "monday"], ["date", "friday"]]
    assert json.loads(out.read_text().splitlines()[0])["model"] == "fake"
