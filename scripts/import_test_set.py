"""Import the externally labelled test set (co_test_set.zip) into testset/.

    uv run python scripts/import_test_set.py ~/Downloads/co_test_set

Copies the documents to testset/samples/ and converts each `<name>.expected.json` into
this repo's ground-truth format in testset/expected/:
- `reason: {category, text}` becomes `reason` + `reason_text`
- fields the pipeline does not extract (`source_file`, `substantial_completion_date`) and
  the signature `method` note are dropped
- money becomes a 2-decimal string, so Decimal comparison is exact
- one answer file that serves several inputs (s06 as .pdf and .txt) is written once per
  input, named after the result file the CLI writes for it (`<stem>.<ext>.json`)
"""

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "testset"
MONEY = ("subtotal", "change_amount", "original_contract_sum", "previous_changes_total",
         "revised_contract_sum")  # fmt: skip
DROPPED = ("source_file", "substantial_completion_date")


def money(value) -> str | None:
    return None if value is None else f"{value:.2f}"


def number(value) -> str | None:
    return None if value is None else (f"{value:g}" if isinstance(value, float) else str(value))


def convert(truth: dict) -> dict:
    out = {k: v for k, v in truth.items() if k not in DROPPED}
    reason = out.pop("reason", None) or {}
    out["reason"], out["reason_text"] = reason.get("category"), reason.get("text")
    for key in MONEY:
        out[key] = money(out.get(key))
    out["line_items"] = [
        {
            "description": li["description"],
            "quantity": number(li.get("quantity")),
            "unit": li.get("unit"),
            "unit_cost": money(li.get("unit_cost")),
            "amount": money(li["amount"]),
        }
        for li in out.get("line_items") or []
    ]
    out["markups"] = [
        {"description": m["description"], "amount": money(m["amount"])}
        for m in out.get("markups") or []
    ]
    out["signatures"] = [
        {k: s.get(k) for k in ("party", "name", "date", "signed")}
        for s in out.get("signatures") or []
    ]
    return out


def main(source: Path) -> None:
    samples, expected = DEST / "samples", DEST / "expected"
    samples.mkdir(parents=True, exist_ok=True)
    expected.mkdir(parents=True, exist_ok=True)

    docs = sorted(p for p in (source / "samples").iterdir() if p.suffix in (".pdf", ".txt"))
    stems = Counter(p.stem for p in docs)
    for doc in docs:
        shutil.copy2(doc, samples / doc.name)
        truth = json.loads((source / "expected" / f"{doc.stem}.expected.json").read_text())
        name = doc.stem if stems[doc.stem] == 1 else f"{doc.stem}.{doc.suffix[1:]}"
        (expected / f"{name}.json").write_text(json.dumps(convert(truth), indent=2) + "\n")
        print(f"{doc.name:30} -> testset/expected/{name}.json")
    shutil.copy2(source / "README.md", DEST / "README.md")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]).expanduser())
