import pytest

from ftvp.cost import cost_per_1k, render


def rec(n_in, n_out, lat):
    return {"input_tokens": n_in, "output_tokens": n_out, "latency_s": lat}


def test_cost_per_1k():
    r = cost_per_1k([rec(800, 20, 0.2), rec(800, 20, 0.4)])
    assert r["input_tokens_per_1k"] == pytest.approx(800_000)
    assert r["m5_seconds_per_1k"] == pytest.approx(300)
    assert r["requests_per_hour"] == pytest.approx(12_000)
    # Opus 5.5 at 4 / 20 USD per million tokens: (800*4 + 20*20) / 1e6 * 1000
    assert r["api_usd_per_1k"]["claude-opus-5-5"] == pytest.approx(3.6)
    assert "| a | 800,000 |" in render({"a": r})


def test_finetuned_runs_are_not_priced():
    r = cost_per_1k([rec(30, 20, 0.2)], priced=False)
    assert r["api_usd_per_1k"]["claude-opus-5-5"] is None
    assert "n/a" in render({"lora": r})


def test_cost_needs_records():
    with pytest.raises(ValueError):
        cost_per_1k([])
