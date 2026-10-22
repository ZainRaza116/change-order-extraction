"""Command-line entry point.

co-extract run samples/ --out outputs/      full pipeline, one JSON per document
co-extract run samples/ --replay            same, from recorded LLM responses only
co-extract serve                            start the HTTP API (needs the `api` extra)
co-extract text FILE / extract FILE         debugging: page text / raw LLM fields
"""

import json
from pathlib import Path
from typing import Annotated

import typer

from co_extract import __version__
from co_extract.config import load_settings
from co_extract.llm import LLMError, extract_fields
from co_extract.pipeline import build_provider, run_pipeline
from co_extract.text_extract import extract_text

app = typer.Typer(add_completion=False, help="Extract structured JSON from change orders.")

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}

ReplayOpt = Annotated[bool, typer.Option(help="Use recorded LLM responses only; no API calls.")]


def _write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload + "\n")


@app.command()
def run(
    path: Annotated[
        Path, typer.Argument(exists=True, help="A change-order file or a folder of them.")
    ],
    out: Annotated[Path, typer.Option("--out", "-o", help="Folder for JSON results.")] = Path(
        "outputs"
    ),
    replay: ReplayOpt = False,
    escalate: Annotated[
        bool | None,
        typer.Option(help="Re-run doubtful documents on the escalation model (extra cost)."),
    ] = None,
) -> None:
    """Run the full pipeline on PATH and write one JSON result per document into OUT."""
    files = (
        [path]
        if path.is_file()
        else sorted(p for p in path.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)
    )
    if not files:
        typer.echo(f"No supported files found in {path}")
        raise typer.Exit(1)

    settings = load_settings()
    provider = build_provider(settings, replay=replay)
    escalate = settings.escalate if escalate is None else escalate

    typer.echo(f"{'file':32} {'status':6} {'overall':>7}  {'review':6} {'cost':>8}  cached")
    errors = 0
    for file in files:
        result = run_pipeline(file, provider, settings, escalate=escalate)
        _write(out / f"{file.stem}.json", result.model_dump_json(indent=2))
        errors += result.status == "error"
        cost = f"${result.run.cost_usd:.4f}" if result.run.cost_usd is not None else "-"
        typer.echo(
            f"{file.name:32} {result.status:6} {result.overall_confidence:7.3f}  "
            f"{'yes' if result.needs_review else 'no':6} {cost:>8}  {result.run.cached}"
        )
        if result.error:
            typer.echo(f"  ! {result.error}")
    typer.echo(f"\nwrote {len(files)} result(s) to {out}/")
    if errors:
        raise typer.Exit(1)


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Interface to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8000,
) -> None:
    """Start the HTTP API: POST /extract with a file upload."""
    try:
        import uvicorn
    except ImportError as exc:
        typer.echo("The API needs extra packages: uv sync --extra api", err=True)
        raise typer.Exit(1) from exc
    uvicorn.run("co_extract.api:app", host=host, port=port)


@app.command(name="eval")
def eval_cmd(
    outputs: Annotated[Path, typer.Option(help="Pipeline results folder.")] = Path("outputs"),
    expected: Annotated[Path, typer.Option(help="Ground-truth folder.")] = Path("expected"),
    report_dir: Annotated[Path, typer.Option(help="Where to write the report.")] = Path("eval"),
) -> None:
    """Score pipeline outputs against ground truth; writes eval/report.md and results.json."""
    from co_extract.evaluate import evaluate, render_markdown

    report = evaluate(outputs, expected)
    if not report.documents:
        typer.echo(f"No results in {outputs}/ match ground truth in {expected}/")
        raise typer.Exit(1)
    markdown = render_markdown(report)
    _write(report_dir / "report.md", markdown)
    _write(report_dir / "results.json", json.dumps(report.to_dict(), indent=2, default=str))
    typer.echo(markdown)
    typer.echo(f"wrote {report_dir}/report.md and {report_dir}/results.json")


@app.command()
def text(
    path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="A PDF, image or .txt file.")
    ],
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Write JSON here.")] = None,
) -> None:
    """Debug: dump the extracted page text and tables as JSON."""
    pages = extract_text(path, vision_model=load_settings().vision_model)
    payload = json.dumps([p.model_dump() for p in pages], indent=2, ensure_ascii=False)
    if out:
        _write(out, payload)
        typer.echo(f"wrote {len(pages)} page(s) to {out}")
    else:
        typer.echo(payload)


@app.command()
def extract(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="A change order.")],
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Write JSON here.")] = None,
    replay: ReplayOpt = False,
) -> None:
    """Debug: dump the raw LLM field extraction (before parsing, rules and scoring)."""
    settings = load_settings()
    pages = extract_text(path, vision_model=settings.vision_model)
    try:
        result = extract_fields(
            pages,
            build_provider(settings, replay=replay),
            settings.model,
            temperature=settings.temperature,
            effort=settings.effort,
        )
    except LLMError as exc:
        typer.echo(f"extraction failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    payload = json.dumps(
        {
            "source_file": path.name,
            "model": settings.model,
            "attempts": result.attempts,
            "cached": all(r.cached for r in result.responses),
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "extraction": result.extraction.model_dump(mode="json"),
        },
        indent=2,
        ensure_ascii=False,
    )
    if out:
        _write(out, payload)
        typer.echo(f"wrote {out}")
    else:
        typer.echo(payload)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)
