"""A realistic LLM output for samples/co_002_digital.pdf, quoting its real extracted text."""

import json
from pathlib import Path

from co_extract.llm import LlmExtraction
from co_extract.text_extract import PageText

FIXTURES = Path(__file__).parent / "fixtures"


def pages(name: str) -> list[PageText]:
    return [PageText.model_validate(p) for p in json.loads((FIXTURES / name).read_text())]


def f(value, evidence, page=1, alternatives=()):
    return {"value": value, "evidence": evidence, "page": page, "alternatives": list(alternatives)}


NULL = {"value": None, "evidence": None, "page": None, "alternatives": []}


def co002_output() -> dict:
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
        "subtotal": NULL,
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
            sig("architect", "Marcus Lee", "10/7/2026", "By: /s/ Marcus Lee Date: 10/7/2026"),
            sig(
                "contractor",
                "Dana Whitfield",
                "10/6/2026",
                "By: /s/ Dana Whitfield Date: 10/6/2026",
            ),
            sig("owner", "Priya Nair", "10/9/2026", "By: /s/ Priya Nair Date: 10/9/2026"),
        ],
        "notes": [],
    }


def item(description, quantity, unit, unit_cost, amount):
    return {
        "description": description,
        "quantity": quantity,
        "unit": unit,
        "unit_cost": unit_cost,
        "amount": amount,
        "evidence": f"{description} {quantity} {unit} {unit_cost} {amount}",
        "page": 1,
    }


def sig(party, name, date, evidence):
    return {
        "party": party,
        "name": name,
        "date": date,
        "signed": True,
        "evidence": evidence,
        "page": 2,
    }


def co002_extraction(**overrides) -> LlmExtraction:
    return LlmExtraction.model_validate(co002_output() | overrides)


class ScriptedProvider:
    """Stands in for the API: returns canned response texts in order, records requests."""

    def __init__(self, *texts: str):
        self.texts = list(texts)
        self.requests = []

    def complete(self, request):
        from co_extract.llm import LLMResponse

        self.requests.append(request)
        return LLMResponse(
            text=self.texts.pop(0), model=request.model, input_tokens=4000, output_tokens=2500
        )


def settings(tmp_path, **overrides):
    from co_extract.config import Settings

    base = dict(
        model="claude-sonnet-5-5",
        escalation_model="claude-opus-5-5",
        vision_model="claude-opus-5-5",
        effort="medium",
        temperature=0.0,
        llm_mode="auto",
        escalate=False,
        review_threshold=0.7,
        cache_dir=tmp_path / "cassettes",
    )
    return Settings(**(base | overrides))
