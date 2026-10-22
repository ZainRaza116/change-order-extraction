"""Runtime settings read from the environment (.env supported)."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

LlmMode = Literal["auto", "replay", "refresh"]
Provider = Literal["anthropic", "openai"]

# (extraction model, escalation model) used when the env doesn't name one
DEFAULT_MODELS: dict[str, tuple[str, str]] = {
    "anthropic": ("claude-sonnet-5-5", "claude-opus-5-5"),
    "openai": ("gpt-4o-mini", "gpt-4o"),
}


@dataclass(frozen=True)
class Settings:
    model: str
    escalation_model: str
    vision_model: str
    effort: str
    temperature: float
    llm_mode: LlmMode
    escalate: bool
    review_threshold: float
    cache_dir: Path
    provider: Provider = "anthropic"


def load_settings() -> Settings:
    load_dotenv()
    mode = os.getenv("CO_EXTRACT_LLM_MODE", "auto")
    if mode not in ("auto", "replay", "refresh"):
        raise ValueError(f"CO_EXTRACT_LLM_MODE must be auto, replay or refresh, not {mode!r}")
    provider = os.getenv("CO_EXTRACT_PROVIDER", "anthropic")
    if provider not in DEFAULT_MODELS:
        raise ValueError(f"CO_EXTRACT_PROVIDER must be anthropic or openai, not {provider!r}")
    default_model, default_escalation = DEFAULT_MODELS[provider]
    return Settings(
        provider=provider,
        model=os.getenv("CO_EXTRACT_MODEL", default_model),
        escalation_model=os.getenv("CO_EXTRACT_ESCALATION_MODEL", default_escalation),
        vision_model=os.getenv("CO_EXTRACT_VISION_MODEL", "claude-opus-5-5"),
        effort=os.getenv("CO_EXTRACT_EFFORT", "medium"),
        temperature=float(os.getenv("CO_EXTRACT_TEMPERATURE", "0")),
        llm_mode=mode,
        escalate=os.getenv("CO_EXTRACT_ESCALATE", "0") == "1",
        review_threshold=float(os.getenv("CO_EXTRACT_REVIEW_THRESHOLD", "0.7")),
        cache_dir=Path(os.getenv("CO_EXTRACT_CACHE_DIR", "cassettes")),
    )
