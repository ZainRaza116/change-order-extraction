"""The provider interface and its two implementations.

A provider turns an LLMRequest into an LLMResponse and raises LLMError for anything the
pipeline can't use (missing credentials, API errors, refusals, truncated output).
SDKs are imported lazily so neither is needed unless that provider is selected.
"""

import time
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class LLMRequest:
    model: str
    system: str
    messages: list[dict[str, str]]
    schema: dict[str, Any]
    temperature: float | None = 0.0
    effort: str | None = "medium"
    max_tokens: int = 16000


@dataclass
class LLMResponse:
    text: str
    model: str
    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    cached: bool = False


class LLMProvider(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse: ...


class LLMError(RuntimeError):
    """Any failure that means this document gets no usable LLM output."""


# --- Anthropic ------------------------------------------------------------------------

# Current Claude models reject sampling parameters (temperature/top_p/top_k) with a 400,
# so temperature is only sent to models that still accept it. Determinism for those
# models comes from the record/replay cache instead.
_ANTHROPIC_NO_SAMPLING = (
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-haiku-5",
    "claude-fable",
    "claude-opus-4-7",
    "claude-opus-4-8",
)
# Models that support server-side refusal fallbacks in the "default" form.
_ANTHROPIC_FALLBACKS = ("claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1")


def anthropic_kwargs(request: LLMRequest) -> dict[str, Any]:
    output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": request.schema}}
    if request.effort:
        output_config["effort"] = request.effort
    kwargs: dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.max_tokens,
        "system": request.system,
        "messages": request.messages,
        "output_config": output_config,
    }
    if request.temperature is not None and not request.model.startswith(_ANTHROPIC_NO_SAMPLING):
        kwargs["temperature"] = request.temperature
    if request.model.startswith(_ANTHROPIC_FALLBACKS):
        kwargs["betas"] = ["server-side-fallback-2026-07-01"]
        kwargs["fallbacks"] = "default"
    return kwargs


class AnthropicProvider:
    def __init__(self, client: Any = None):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client

    def complete(self, request: LLMRequest) -> LLMResponse:
        import anthropic

        kwargs = anthropic_kwargs(request)
        messages = self.client.beta.messages if "betas" in kwargs else self.client.messages
        start = time.monotonic()
        try:
            response = messages.create(**kwargs)
        except TypeError as exc:  # the SDK raises this when no credentials can be resolved
            raise LLMError(
                "no Anthropic credentials: set ANTHROPIC_API_KEY in .env, or use --replay"
            ) from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"could not reach the API: {exc}") from exc
        latency = time.monotonic() - start

        if response.stop_reason == "refusal":
            raise LLMError(f"model declined the request: {response.stop_details}")
        if response.stop_reason == "max_tokens":
            raise LLMError("response hit max_tokens before the JSON was complete")
        return LLMResponse(
            text=next((b.text for b in response.content if b.type == "text"), ""),
            model=response.model,
            stop_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_s=round(latency, 2),
        )


# --- OpenAI ---------------------------------------------------------------------------

# Reasoning models reject temperature but accept reasoning_effort; GPT-4-class models
# are the reverse.
_OPENAI_REASONING = ("o1", "o3", "o4", "gpt-5")


def openai_kwargs(request: LLMRequest) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": request.model,
        "messages": [{"role": "system", "content": request.system}, *request.messages],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "change_order", "schema": request.schema, "strict": True},
        },
        "max_completion_tokens": request.max_tokens,
    }
    if request.model.startswith(_OPENAI_REASONING):
        if request.effort:
            kwargs["reasoning_effort"] = request.effort
    elif request.temperature is not None:
        kwargs["temperature"] = request.temperature
    return kwargs


class OpenAIProvider:
    def __init__(self, client: Any = None):
        self._client = client  # created on first use, so a missing key fails per document

    @property
    def client(self) -> Any:
        if self._client is None:
            import openai

            try:
                self._client = openai.OpenAI()
            except openai.OpenAIError as exc:
                raise LLMError(
                    "no OpenAI credentials: set OPENAI_API_KEY in .env, or use --replay"
                ) from exc
        return self._client

    def complete(self, request: LLMRequest) -> LLMResponse:
        import openai

        start = time.monotonic()
        try:
            response = self.client.chat.completions.create(**openai_kwargs(request))
        except openai.APIStatusError as exc:
            raise LLMError(f"API error {exc.status_code}: {exc.message}") from exc
        except openai.APIConnectionError as exc:
            raise LLMError(f"could not reach the API: {exc}") from exc
        latency = time.monotonic() - start

        choice = response.choices[0]
        if choice.message.refusal:
            raise LLMError(f"model declined the request: {choice.message.refusal}")
        if choice.finish_reason == "length":
            raise LLMError("response hit the token limit before the JSON was complete")
        usage = response.usage
        return LLMResponse(
            text=choice.message.content or "",
            model=response.model,
            stop_reason=choice.finish_reason,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_s=round(latency, 2),
        )
