import pytest

from ftvp.massive import Example
from ftvp.metrics import aggregate, bootstrap_ci, paired_bootstrap, score_item
from ftvp.schema import Prediction

EX = Example(id="0", split="test", utt="remind me on monday and on friday",
             intent="calendar_set", slots=(("date", "monday"), ("date", "friday")))


def pred(intent, slots, **kw):
    return Prediction(intent=intent, slots=list(slots), valid_json=True, schema_ok=True,
                      clean=True, **kw)


def test_perfect_prediction():
    s = score_item(EX, pred("calendar_set", [("date", "Friday"), ("date", "monday")]))
    assert s.intent_ok and s.exact and s.tp == 2 and s.grounded == 2


def test_duplicate_slot_counts_once():
    s = score_item(EX, pred("calendar_set", [("date", "monday"), ("date", "monday")]))
    assert s.tp == 1 and s.n_pred == 2 and not s.exact


def test_wrong_intent_breaks_exact_but_not_slots():
    s = score_item(EX, pred("calendar_query", EX.slots))
    assert not s.intent_ok and not s.exact and s.tp == 2


def test_hallucinated_value_not_grounded():
    s = score_item(EX, pred("calendar_set", [("date", "tuesday")]))
    assert s.grounded == 0 and s.tp == 0


def test_aggregate_micro_f1():
    no_slots = Example(id="1", split="test", utt="tell me a joke", intent="general_joke")
    items = [score_item(EX, pred("calendar_set", [("date", "monday")])),
             score_item(no_slots, pred("general_joke", []))]
    agg = aggregate(items)
    assert agg["intent_acc"] == 1.0
    assert agg["slot_p"] == 1.0 and agg["slot_r"] == 0.5
    assert agg["slot_f1"] == pytest.approx(2 / 3)
    assert agg["exact_match"] == 0.5


def test_aggregate_empty_raises():
    with pytest.raises(ValueError):
        aggregate([])


def test_bootstrap_brackets_point_estimate():
    good = score_item(EX, pred("calendar_set", EX.slots))
    bad = score_item(EX, Prediction())
    items = [good] * 70 + [bad] * 30
    lo, hi = bootstrap_ci(items, "intent_acc", n_boot=300)
    assert lo < 0.7 < hi
    res = paired_bootstrap([good] * 100, items, "exact_match", n_boot=300)
    assert res["diff"] == pytest.approx(0.3) and res["lo"] > 0 and res["p_not_better"] == 0


def test_paired_bootstrap_length_mismatch():
    s = score_item(EX, Prediction())
    with pytest.raises(ValueError):
        paired_bootstrap([s], [s, s], "intent_acc")
