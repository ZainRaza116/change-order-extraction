"""The provider interface and the OpenAI implementation.

A provider turns an LLMRequest into an LLMResponse and raises LLMError for anything the
pipeline can't use (missing credentials, API errors, refusals, truncated output). The
interface keeps the pipeline independent of the vendor SDK and lets tests script responses.
"""

import time
from dataclasses import dataclass
from typing import Any, Protocol

import openai


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


# Reasoning models reject temperature but accept reasoning_effort; GPT-4-class models
# are the reverse.
_REASONING_MODELS = ("o1", "o3", "o4", "gpt-5")


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
    if request.model.startswith(_REASONING_MODELS):
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
            try:
                self._client = openai.OpenAI()
            except openai.OpenAIError as exc:
                raise LLMError(
                    "no OpenAI credentials: set OPENAI_API_KEY in .env, or use --replay"
                ) from exc
        return self._client

    def complete(self, request: LLMRequest) -> LLMResponse:
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
