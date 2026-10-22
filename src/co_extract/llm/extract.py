"""One schema-constrained call per document, with a single retry on schema failure."""

from dataclasses import dataclass, field
from typing import Literal

from pydantic import ValidationError

from co_extract.ingest import PageText
from co_extract.llm.output import LlmExtraction, output_schema
from co_extract.llm.prompt import RETRY_PROMPT, SYSTEM_PROMPT, user_prompt
from co_extract.llm.providers import LLMError, LLMProvider, LLMRequest, LLMResponse


class ExtractionFailed(LLMError):
    def __init__(self, message: str, responses: list[LLMResponse]):
        super().__init__(message)
        self.responses = responses


@dataclass
class LlmResult:
    extraction: LlmExtraction
    attempts: int
    responses: list[LLMResponse] = field(default_factory=list)

    @property
    def input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.responses)

    @property
    def output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.responses)


def _errors_for_prompt(exc: ValueError) -> str:
    if isinstance(exc, ValidationError):
        return "\n".join(
            f"- {'.'.join(str(p) for p in e['loc']) or '(root)'}: {e['msg']}" for e in exc.errors()
        )
    return f"- {exc}"


def extract_fields(
    pages: list[PageText],
    provider: LLMProvider,
    model: str,
    *,
    temperature: float | None = 0.0,
    effort: str | None = "medium",
    max_attempts: Literal[1, 2] = 2,
) -> LlmResult:
    """Call the model once; if its JSON fails validation, retry once with the errors."""
    messages = [{"role": "user", "content": user_prompt(pages)}]
    responses: list[LLMResponse] = []
    for attempt in range(1, max_attempts + 1):
        request = LLMRequest(
            model=model,
            system=SYSTEM_PROMPT,
            messages=list(messages),
            schema=output_schema(),
            temperature=temperature,
            effort=effort,
        )
        response = provider.complete(request)
        responses.append(response)
        try:
            extraction = LlmExtraction.model_validate_json(response.text)
        except ValueError as exc:  # pydantic's ValidationError is a ValueError
            if attempt == max_attempts:
                raise ExtractionFailed(
                    f"schema validation failed after {attempt} attempt(s): {exc}", responses
                ) from exc
            messages += [
                {"role": "assistant", "content": response.text},
                {"role": "user", "content": RETRY_PROMPT.format(errors=_errors_for_prompt(exc))},
            ]
            continue
        return LlmResult(extraction=extraction, attempts=attempt, responses=responses)
    raise AssertionError("unreachable")
