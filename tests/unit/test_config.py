import pytest

from co_extract.config import load_settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("CO_EXTRACT_MODEL", "CO_EXTRACT_LLM_MODE", "CO_EXTRACT_REVIEW_THRESHOLD"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("co_extract.config.load_dotenv", lambda: None)  # ignore a local .env


def test_defaults_match_the_recordings():
    # the committed cassettes were recorded with gpt-4o-mini, so the default must match
    s = load_settings()
    assert (s.model, s.llm_mode, s.review_threshold) == ("gpt-4o-mini", "auto", 0.7)


def test_environment_overrides(monkeypatch):
    monkeypatch.setenv("CO_EXTRACT_MODEL", "gpt-4o")
    monkeypatch.setenv("CO_EXTRACT_REVIEW_THRESHOLD", "0.8")
    s = load_settings()
    assert (s.model, s.review_threshold) == ("gpt-4o", 0.8)


def test_invalid_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("CO_EXTRACT_LLM_MODE", "live")
    with pytest.raises(ValueError, match="CO_EXTRACT_LLM_MODE"):
        load_settings()
