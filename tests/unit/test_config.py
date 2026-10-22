import pytest

from co_extract.config import load_settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("CO_EXTRACT_PROVIDER", "CO_EXTRACT_MODEL", "CO_EXTRACT_ESCALATION_MODEL",
                 "CO_EXTRACT_LLM_MODE", "CO_EXTRACT_VISION_FALLBACK"):  # fmt: skip
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("co_extract.config.load_dotenv", lambda: None)  # ignore a local .env


def test_openai_is_the_default_provider():
    # the committed recordings were made with gpt-4o-mini, so defaults must match them
    s = load_settings()
    assert (s.provider, s.model, s.escalation_model) == ("openai", "gpt-4o-mini", "gpt-4o")


def test_anthropic_defaults(monkeypatch):
    monkeypatch.setenv("CO_EXTRACT_PROVIDER", "anthropic")
    s = load_settings()
    assert (s.model, s.escalation_model) == ("claude-sonnet-5-5", "claude-opus-5-5")


def test_vision_fallback_is_off_unless_enabled(monkeypatch):
    assert load_settings().ocr_vision_model is None
    monkeypatch.setenv("CO_EXTRACT_VISION_FALLBACK", "1")
    assert load_settings().ocr_vision_model == "claude-opus-5-5"


@pytest.mark.parametrize(
    ("name", "value"), [("CO_EXTRACT_PROVIDER", "gemini"), ("CO_EXTRACT_LLM_MODE", "live")]
)
def test_invalid_values_are_rejected(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        load_settings()
