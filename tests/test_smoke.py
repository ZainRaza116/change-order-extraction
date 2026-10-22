from typer.testing import CliRunner

from co_extract import __version__
from co_extract.cli import app


def test_version_command():
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_run_rejects_empty_folder(tmp_path):
    result = CliRunner().invoke(app, ["run", str(tmp_path)])
    assert result.exit_code == 1
