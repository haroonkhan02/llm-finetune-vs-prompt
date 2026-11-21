"""Model backends behind one interface: ``complete(messages) -> Completion``.

- ``FakeBackend``: scripted replies, for tests and pipeline smoke runs.
- ``MLXBackend``: an open-weight model on Apple silicon via ``mlx-lm``, optionally with a
  LoRA adapter. Greedy decoding, so a run is deterministic. Records time to first token
  (prefill) and total time per request.
- ``AnthropicBackend``: Claude via the Anthropic SDK. The client is injectable so tests
  never touch the network; a live run needs an Anthropic API key.

Messages are chat dicts (``system`` first, then alternating user / assistant turns), the
format produced by :mod:`ftvp.prompts`.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

# USD per million tokens (input, output), Anthropic list prices.
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0),
          "claude-haiku-5-5": (0.10, 0.50)}


@dataclass
class Completion:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0        # request start -> last token
    ttft_s: float = 0.0           # request start -> first token (prefill)
    stop_reason: str = "end_turn"
    cost_usd: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


class Backend(Protocol):
    name: str

    def complete(self, messages: list[dict]) -> Completion: ...


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = PRICES.get(model, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1e6


def split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    """Anthropic takes the system prompt as a separate field."""
    if messages and messages[0]["role"] == "system":
        return messages[0]["content"], list(messages[1:])
    return "", list(messages)


class FakeBackend:
    """Returns scripted replies: a fixed list (cycled) or a function of the messages."""

    def __init__(self, replies: Sequence[str] | Callable[[list[dict]], str],
                 name: str = "fake"):
        self.name = name
        self._replies = replies
        self._i = 0
        self.calls: list[list[dict]] = []

    def complete(self, messages: list[dict]) -> Completion:
        self.calls.append(messages)
        if callable(self._replies):
            text = self._replies(messages)
        else:
            text = self._replies[self._i % len(self._replies)]
            self._i += 1
        n_in = sum(len(m["content"].split()) for m in messages)
        return Completion(text=text, model=self.name, input_tokens=n_in,
                          output_tokens=len(text.split()))


class MLXBackend:
    """Local model on Apple silicon. ``loader`` / ``generator`` default to
    ``mlx_lm.load`` / ``mlx_lm.stream_generate``; tests pass fakes. Cost is 0 here; the
    cost study prices local inference from measured throughput separately."""

    def __init__(self, model: str, adapter: str | None = None, max_tokens: int = 128,
                 loader=None, generator=None):
        if loader is None or generator is None:
            import mlx_lm

            loader = loader or mlx_lm.load
            generator = generator or mlx_lm.stream_generate
        self.model, self.tokenizer = loader(model, adapter_path=adapter)
        self._generate = generator
        self.model_id = model
        self.adapter = adapter
        self.max_tokens = max_tokens
        self.name = f"mlx:{model.split('/')[-1]}" + ("+lora" if adapter else "")

    def render(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

    def complete(self, messages: list[dict]) -> Completion:
        prompt = self.render(messages)
        t0 = time.perf_counter()
        ttft, parts, last = 0.0, [], None
        for last in self._generate(self.model, self.tokenizer, prompt,
                                   max_tokens=self.max_tokens):
            if not parts:
                ttft = time.perf_counter() - t0
            parts.append(last.text)
        latency = time.perf_counter() - t0
        finish = getattr(last, "finish_reason", None) if last is not None else None
        return Completion(
            text="".join(parts), model=self.model_id,
            input_tokens=int(getattr(last, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(last, "generation_tokens", 0) or 0),
            latency_s=latency, ttft_s=ttft,
            stop_reason="max_tokens" if finish == "length" else "end_turn",
        )


class AnthropicBackend:
    """Claude. Thinking is adaptive (always on for Opus 5.5); depth is set with ``effort``
    (``low`` suits a short extraction). Server-side refusal fallbacks are on; a refusal
    that survives the fallback chain comes back with empty text and
    ``stop_reason == "refusal"``, which the parser scores as no JSON."""

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "low",
                 max_tokens: int = 4000, client=None):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.name = f"anthropic:{model}:{effort}"

    def complete(self, messages: list[dict]) -> Completion:
        system, turns = split_system(messages)
        t0 = time.perf_counter()
        resp = self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=turns,
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        latency = time.perf_counter() - t0
        stop = resp.stop_reason or "end_turn"
        text = "" if stop == "refusal" else "".join(
            b.text for b in resp.content if getattr(b, "type", None) == "text")
        usage = resp.usage
        served_by = getattr(resp, "model", None) or self.model
        return Completion(
            text=text, model=served_by,
            input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
            latency_s=latency, ttft_s=latency, stop_reason=stop,
            cost_usd=cost_usd(served_by, usage.input_tokens, usage.output_tokens),
        )


def make_backend(spec: str, adapter: str | None = None, max_tokens: int = 128) -> Backend:
    """``mlx:<hf repo>``, ``anthropic:<model>[:effort]`` or ``fake``."""
    kind, _, rest = spec.partition(":")
    if kind == "mlx" and rest:
        return MLXBackend(rest, adapter=adapter, max_tokens=max_tokens)
    if kind == "anthropic":
        model, _, effort = (rest or "claude-opus-5-5").partition(":")
        return AnthropicBackend(model=model, effort=effort or "low")
    if kind == "fake":
        return FakeBackend(['{"intent": "general_quirky", "slots": []}'])
    raise ValueError(f"unknown backend spec {spec!r}")
