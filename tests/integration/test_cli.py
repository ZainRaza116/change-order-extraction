import json

import pytest
from typer.testing import CliRunner

from co_extract import __version__, cli
from co_extract.result import ExtractionResult
from tests.factories import ROOT, SAMPLES, ScriptedProvider, co002_output, make_settings

DIGITAL = SAMPLES / "co_002_digital.pdf"
GOOD = json.dumps(co002_output())


def invoke(*args: str):
    return CliRunner().invoke(cli.app, list(args))


def test_version():
    result = invoke("version")
    assert result.exit_code == 0 and __version__ in result.output


def test_run_rejects_empty_folder(tmp_path):
    assert invoke("run", str(tmp_path)).exit_code == 1


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    """Point the CLI at a scripted provider and a temp settings object."""
    provider = ScriptedProvider(GOOD, GOOD, GOOD)
    monkeypatch.setattr(cli, "load_settings", lambda: make_settings(tmp_path))
    monkeypatch.setattr(cli, "build_provider", lambda s, replay=False: provider)
    return provider


def test_cli_run_writes_one_json_per_document(tmp_path, cli_env):
    docs = tmp_path / "docs"
    docs.mkdir()
    for name in ("a.pdf", "b.pdf"):
        (docs / name).write_bytes(DIGITAL.read_bytes())
    (docs / "notes.md").write_text("ignored")
    out = tmp_path / "out"
    result = invoke(*["run", str(docs), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in out.iterdir()) == ["a.json", "b.json"]
    assert json.loads((out / "a.json").read_text())["status"] == "ok"
    assert "a.pdf" in result.output and "b.pdf" in result.output


def test_cli_run_exits_nonzero_when_a_document_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "load_settings", lambda: make_settings(tmp_path))
    monkeypatch.setattr(cli, "build_provider", lambda s, replay=False: ScriptedProvider("x", "y"))
    result = invoke(*["run", str(DIGITAL), "--out", str(tmp_path / "o")])
    assert result.exit_code == 1
    assert "ExtractionFailed" in result.output
    assert (tmp_path / "o" / "co_002_digital.json").exists()  # error result still written


def test_cli_replay_without_recordings_fails_cleanly(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "load_settings", lambda: make_settings(tmp_path))
    result = CliRunner().invoke(
        cli.app, ["run", str(DIGITAL), "--replay", "--out", str(tmp_path / "o")]
    )
    assert result.exit_code == 1
    assert "no recording" in result.output


def test_committed_schema_is_up_to_date(tmp_path):
    out = tmp_path / "schema.json"
    assert invoke("schema", "--out", str(out)).exit_code == 0
    committed = ROOT / "schema" / "extraction_result.schema.json"
    assert json.loads(out.read_text()) == json.loads(committed.read_text()), (
        "output contract changed: run `co-extract schema` and commit the result"
    )
    assert json.loads(out.read_text()) == ExtractionResult.model_json_schema()
