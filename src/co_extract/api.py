"""HTTP API: POST /extract with a multipart file upload returns an ExtractionResult.

Run with `co-extract serve` (needs `uv sync --extra api`).
"""

import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, UploadFile

from co_extract import __version__
from co_extract.config import Settings, load_settings
from co_extract.ingest import SUPPORTED_SUFFIXES
from co_extract.llm import LLMProvider
from co_extract.pipeline import build_provider, run_pipeline
from co_extract.result import ExtractionResult

MAX_UPLOAD_BYTES = 20 * 1024 * 1024

app = FastAPI(title="co-extract", version=__version__)


@lru_cache
def get_settings() -> Settings:
    return load_settings()


@lru_cache
def get_provider() -> LLMProvider:
    return build_provider(get_settings())


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/extract", response_model=ExtractionResult)
async def extract(
    file: UploadFile,
    provider: Annotated[LLMProvider, Depends(get_provider)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ExtractionResult:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(415, f"unsupported file type {suffix or '(none)'}")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"file larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    if not data:
        raise HTTPException(400, "empty file")

    with tempfile.TemporaryDirectory() as tmp:
        # keep only the suffix from the client's filename; never use it as a path
        path = Path(tmp) / f"upload{suffix}"
        path.write_bytes(data)
        result = run_pipeline(path, provider, settings, escalate=settings.escalate)
    result.source_file = Path(file.filename or path.name).name
    if result.status == "error":
        raise HTTPException(422, detail=result.model_dump(mode="json"))
    return result
