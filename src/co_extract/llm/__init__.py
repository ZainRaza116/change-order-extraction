"""LLM stage: one schema-constrained call that returns raw values with verbatim evidence.

The model copies values as written ("$2,250.00", "Sept. 18, 2026"); the deterministic
parsers in normalize.py turn them into numbers and dates, so the model never does
arithmetic.
"""

from co_extract.llm.cache import CachedProvider, CacheMiss, request_key
from co_extract.llm.extract import ExtractionFailed, LlmResult, extract_fields
from co_extract.llm.output import Extracted, LlmExtraction, output_schema, raw_value
from co_extract.llm.prompt import PROMPT_VERSION, SYSTEM_PROMPT, render_document
from co_extract.llm.providers import (
    AnthropicProvider,
    LLMError,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    OpenAIProvider,
    anthropic_kwargs,
    openai_kwargs,
)

__all__ = [
    "PROMPT_VERSION",
    "SYSTEM_PROMPT",
    "AnthropicProvider",
    "CacheMiss",
    "CachedProvider",
    "Extracted",
    "ExtractionFailed",
    "LLMError",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "LlmExtraction",
    "LlmResult",
    "OpenAIProvider",
    "anthropic_kwargs",
    "extract_fields",
    "openai_kwargs",
    "output_schema",
    "raw_value",
    "render_document",
    "request_key",
]
