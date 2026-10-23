"""Shared test builders: LLM outputs, page fixtures, settings and a scripted provider.

co002_output() is a realistic model response for samples/co_002_digital.pdf whose
evidence quotes come from that document's real extracted text.
"""

import json
from pathlib import Path

from co_extract.config import Settings
from co_extract.ingest import PageText
from co_extract.llm import LlmExtraction, LLMRequest, LLMResponse

ROOT = Path(__file__).parent.parent
SAMPLES = ROOT / "samples"
EXPECTED = ROOT / "expected"
TESTSET_EXPECTED = ROOT / "testset" / "expected"  # labelled externally
ALL_GROUND_TRUTH = sorted([*EXPECTED.glob("*.json"), *TESTSET_EXPECTED.glob("*.json")])
FIXTURES = Path(__file__).parent / "fixtures"

SCALAR_FIELDS = (
    "change_order_number", "project_name", "project_number", "date_issued", "contractor",
    "owner", "architect_engineer", "description", "reason", "reason_text", "subtotal",
    "change_amount", "currency", "original_contract_sum", "previous_changes_total",
    "revised_contract_sum", "schedule_impact_days", "status",
)  # fmt: skip

NULL = {"value": None, "evidence": None, "page": None, "alternatives": []}


def pages(name: str) -> list[PageText]:
    """Saved extractor output from tests/fixtures (digital_text.json, scanned_text.json)."""
    return [PageText.model_validate(p) for p in json.loads((FIXTURES / name).read_text())]


def f(value=None, evidence=None, page=1, alternatives=()) -> dict:
    """One Extracted field as the model would return it."""
    return {"value": value, "evidence": evidence, "page": page, "alternatives": list(alternatives)}


def minimal_output(**overrides) -> dict:
    """A valid model response with only the CO number and total filled in."""
    out = {name: dict(NULL) for name in SCALAR_FIELDS}
    out |= {"line_items": [], "markups": [], "signatures": [], "notes": []}
    out["change_order_number"] = f("007", "CHANGE ORDER No. 007")
    out["change_amount"] = f(
        "$10,763.06", "TOTAL THIS CHANGE ORDER $10,763.06", alternatives=["$11,240.00"]
    )
    return out | overrides


def item(description, quantity, unit, unit_cost, amount) -> dict:
    return {
        "description": description,
        "quantity": quantity,
        "unit": unit,
        "unit_cost": unit_cost,
        "amount": amount,
        "evidence": f"{description} {quantity} {unit} {unit_cost} {amount}",
        "page": 1,
    }


def signature(party, name, date, evidence) -> dict:
    return {
        "party": party,
        "name": name,
        "date": date,
        "signed": True,
        "evidence": evidence,
        "page": 2,
    }


def co002_output(**overrides) -> dict:
    return {
        "change_order_number": f("CO-012", "CHANGE ORDER NO.: CO-012"),
        "project_name": f(
            "Riverside Medical Office Building", "PROJECT: Riverside Medical Office Building"
        ),
        "project_number": f("24-207", "PROJECT NO.: 24-207"),
        "date_issued": f("October 6, 2026", "DATE: October 6, 2026"),
        "contractor": f("Ridgeline Builders Inc.", "CONTRACTOR: Ridgeline Builders Inc."),
        "owner": f("Riverside Health Partners LLC", "OWNER: Riverside Health Partners LLC"),
        "architect_engineer": f("Studio Lee Architects", "ARCHITECT: Studio Lee Architects"),
        "description": f(
            "Upgrade fire separation at Stair 2 and relocate floor drain FD-3",
            "Upgrade fire separation at Stair 2 and relocate floor drain FD-3",
        ),
        "reason": f("owner_request", "Reason: Owner request."),
        "reason_text": f("Owner request.", "Reason: Owner request."),
        "line_items": [
            item("Add 2-hr rated GWB at Stair 2", "640", "SF", "$14.25", "$9,120.00"),
            item("Relocate floor drain FD-3", "1", "LS", "$3,450.00", "$3,450.00"),
            item(
                "Credit: delete rubber base, storage rooms", "1", "LS", "($2,250.00)", "($2,250.00)"
            ),
            item("Bond and insurance", "1", "LS", "$443.06", "$443.06"),
        ],
        "subtotal": dict(NULL),
        "markups": [],
        "change_amount": f("$10,763.06", "Total this Change Order: $10,763.06"),
        "currency": f("$", "Total this Change Order: $10,763.06"),
        "original_contract_sum": f("$2,450,000.00", "The original Contract Sum was $2,450,000.00"),
        "previous_changes_total": f(
            "$72,420.50", "Net change by previously authorized Change Orders $72,420.50"
        ),
        "revised_contract_sum": f(
            "$2,533,183.56",
            "The new Contract Sum including this Change Order will be $2,533,183.56",
        ),
        "schedule_impact_days": f(
            "Six (6) calendar days",
            "The Contract Time will be increased by Six (6) calendar days.",
        ),
        "status": f("approved", "OWNER Riverside Health Partners LLC", 2),
        "signatures": [
            signature("architect", "Marcus Lee", "10/7/2026", "By: /s/ Marcus Lee Date: 10/7/2026"),
            signature(
                "contractor",
                "Dana Whitfield",
                "10/6/2026",
                "By: /s/ Dana Whitfield Date: 10/6/2026",
            ),
            signature("owner", "Priya Nair", "10/9/2026", "By: /s/ Priya Nair Date: 10/9/2026"),
        ],
        "notes": [],
    } | overrides


def co002_extraction(**overrides) -> LlmExtraction:
    return LlmExtraction.model_validate(co002_output(**overrides))


class ScriptedProvider:
    """Stands in for the API: returns canned response texts in order, records requests."""

    def __init__(self, *texts: str):
        self.texts = list(texts)
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(
            text=self.texts.pop(0), model=request.model, input_tokens=4000, output_tokens=2500
        )


def make_settings(tmp_path: Path, **overrides) -> Settings:
    base = dict(
        model="gpt-4o-mini",
        effort="medium",
        temperature=0.0,
        llm_mode="auto",
        review_threshold=0.7,
        cache_dir=tmp_path / "cassettes",
    )
    return Settings(**(base | overrides))
