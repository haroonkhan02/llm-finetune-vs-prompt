from types import SimpleNamespace

import pytest

from ftvp.backends import (
    AnthropicBackend,
    FakeBackend,
    MLXBackend,
    cost_usd,
    make_backend,
    split_system,
)

MSGS = [{"role": "system", "content": "label it"}, {"role": "user", "content": "play jazz"}]


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        return "|".join(m["content"] for m in messages)


def fake_loader(model, adapter_path=None):
    return SimpleNamespace(id=model, adapter=adapter_path), FakeTokenizer()


def fake_generator(model, tokenizer, prompt, max_tokens):
    pieces = ['{"intent": ', '"play_music", ', '"slots": []}']
    for i, p in enumerate(pieces, 1):
        yield SimpleNamespace(text=p, prompt_tokens=7, generation_tokens=i,
                              finish_reason="stop" if i == len(pieces) else None)


def test_split_system():
    system, turns = split_system(MSGS)
    assert system == "label it" and turns == MSGS[1:]
    assert split_system(MSGS[1:]) == ("", MSGS[1:])


def test_fake_backend_cycles_and_records_calls():
    b = FakeBackend(["a", "b"])
    assert [b.complete(MSGS).text for _ in range(3)] == ["a", "b", "a"]
    assert len(b.calls) == 3
    assert FakeBackend(lambda m: m[-1]["content"]).complete(MSGS).text == "play jazz"


def test_mlx_backend_joins_stream_and_times_it():
    b = MLXBackend("org/tiny", adapter="adapters/x", loader=fake_loader,
                   generator=fake_generator)
    assert b.name == "mlx:tiny+lora" and b.model.adapter == "adapters/x"
    assert b.render(MSGS) == "label it|play jazz"
    c = b.complete(MSGS)
    assert c.text == '{"intent": "play_music", "slots": []}'
    assert (c.input_tokens, c.output_tokens, c.stop_reason) == (7, 3, "end_turn")
    assert 0 <= c.ttft_s <= c.latency_s and c.cost_usd == 0


def test_mlx_backend_flags_truncation():
    def gen(model, tokenizer, prompt, max_tokens):
        yield SimpleNamespace(text="{", prompt_tokens=3, generation_tokens=1,
                              finish_reason="length")
    c = MLXBackend("org/tiny", loader=fake_loader, generator=gen).complete(MSGS)
    assert c.stop_reason == "max_tokens"


class FakeMessages:
    def __init__(self, stop="end_turn"):
        self.stop = stop
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            stop_reason=self.stop, model=kwargs["model"],
            content=[SimpleNamespace(type="thinking", thinking=""),
                     SimpleNamespace(type="text", text='{"intent": "x", "slots": []}')],
            usage=SimpleNamespace(input_tokens=1000, output_tokens=50))


def fake_client(stop="end_turn"):
    return SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages(stop)))


def test_anthropic_backend_request_and_cost():
    client = fake_client()
    c = AnthropicBackend(client=client).complete(MSGS)
    kw = client.beta.messages.kwargs
    assert kw["system"] == "label it" and kw["messages"] == MSGS[1:]
    assert kw["model"] == "claude-opus-5-5" and kw["output_config"] == {"effort": "low"}
    assert kw["fallbacks"] == "default" and "temperature" not in kw
    assert c.text == '{"intent": "x", "slots": []}'
    assert c.cost_usd == pytest.approx(cost_usd("claude-opus-5-5", 1000, 50))
    assert c.cost_usd == pytest.approx((1000 * 4 + 50 * 20) / 1e6)


def test_anthropic_refusal_gives_empty_text():
    c = AnthropicBackend(client=fake_client("refusal")).complete(MSGS)
    assert c.text == "" and c.stop_reason == "refusal"


def test_make_backend_specs():
    assert make_backend("fake").name == "fake"
    with pytest.raises(ValueError):
        make_backend("openai:gpt")
