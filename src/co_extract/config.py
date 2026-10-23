"""Runtime settings, read from the environment (a local .env file is loaded if present)."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

LlmMode = Literal["auto", "replay", "refresh"]


@dataclass(frozen=True)
class Settings:
    model: str
    effort: str
    temperature: float
    llm_mode: LlmMode
    review_threshold: float
    cache_dir: Path


def load_settings() -> Settings:
    load_dotenv()
    mode = os.getenv("CO_EXTRACT_LLM_MODE", "auto")
    if mode not in ("auto", "replay", "refresh"):
        raise ValueError(f"CO_EXTRACT_LLM_MODE must be auto, replay or refresh, not {mode!r}")
    return Settings(
        model=os.getenv("CO_EXTRACT_MODEL", "gpt-4o-mini"),
        effort=os.getenv("CO_EXTRACT_EFFORT", "medium"),
        temperature=float(os.getenv("CO_EXTRACT_TEMPERATURE", "0")),
        llm_mode=mode,
        review_threshold=float(os.getenv("CO_EXTRACT_REVIEW_THRESHOLD", "0.7")),
        cache_dir=Path(os.getenv("CO_EXTRACT_CACHE_DIR", "cassettes")),
    )
