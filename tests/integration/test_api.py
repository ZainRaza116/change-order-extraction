import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient  # noqa: E402

from co_extract import api  # noqa: E402
from tests.factories import SAMPLES, ScriptedProvider, co002_output, make_settings  # noqa: E402

DIGITAL = SAMPLES / "co_002_digital.pdf"


@pytest.fixture
def client(tmp_path):
    provider = ScriptedProvider(json.dumps(co002_output()))
    api.app.dependency_overrides[api.get_provider] = lambda: provider
    api.app.dependency_overrides[api.get_settings] = lambda: make_settings(tmp_path)
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_extract_pdf(client):
    response = client.post(
        "/extract", files={"file": ("CO-012.pdf", DIGITAL.read_bytes(), "application/pdf")}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["source_file"] == "CO-012.pdf"
    assert body["fields"]["change_amount"]["value"] == "10763.06"
    assert body["needs_review"] is False


def test_rejects_unsupported_type(client):
    response = client.post("/extract", files={"file": ("x.docx", b"PK..", "application/zip")})
    assert response.status_code == 415


def test_rejects_empty_file(client):
    assert (
        client.post("/extract", files={"file": ("x.pdf", b"", "application/pdf")}).status_code
        == 400
    )


def test_rejects_oversized_file(client, monkeypatch):
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 10)
    response = client.post("/extract", files={"file": ("x.pdf", b"0" * 11, "application/pdf")})
    assert response.status_code == 413


def test_path_in_filename_is_not_used(client):
    response = client.post(
        "/extract",
        files={"file": ("../../etc/CO-012.pdf", DIGITAL.read_bytes(), "application/pdf")},
    )
    assert response.status_code == 200
    assert response.json()["source_file"] == "CO-012.pdf"


def test_broken_pdf_returns_422(client):
    response = client.post("/extract", files={"file": ("x.pdf", b"not a pdf", "application/pdf")})
    assert response.status_code == 422
    assert response.json()["detail"]["status"] == "error"
