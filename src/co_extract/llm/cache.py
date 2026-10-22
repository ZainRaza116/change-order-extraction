"""Record/replay cache: a provider wrapper that stores each response on disk, keyed by a
hash of the full request (model, prompt, document text, schema).

Modes:
    auto     replay if recorded, otherwise call the provider and record
    replay   recorded responses only; never call (offline runs, reviewers without a key)
    refresh  always call and overwrite the recording
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from co_extract.config import LlmMode
from co_extract.llm.prompt import PROMPT_VERSION
from co_extract.llm.providers import LLMError, LLMProvider, LLMRequest, LLMResponse


class CacheMiss(LLMError):
    pass


def request_key(request: LLMRequest) -> str:
    canonical = json.dumps(asdict(request), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


class CachedProvider:
    def __init__(self, inner: LLMProvider | None, cache_dir: Path, mode: LlmMode = "auto"):
        self.inner = inner
        self.dir = Path(cache_dir)
        self.mode = mode

    def complete(self, request: LLMRequest) -> LLMResponse:
        key = request_key(request)
        path = self.dir / f"{key}.json"
        if self.mode != "refresh" and path.exists():
            saved = json.loads(path.read_text())
            return LLMResponse(**{**saved["response"], "cached": True})
        if self.mode == "replay":
            raise CacheMiss(f"no recording for request {key[:12]} (mode=replay)")
        if self.inner is None:
            raise LLMError("no LLM provider configured")

        response = self.inner.complete(request)
        self.dir.mkdir(parents=True, exist_ok=True)
        record = {
            "key": key,
            "prompt_version": PROMPT_VERSION,
            "model": request.model,
            "response": asdict(response),
        }
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        return response
