"""The JSON schema the model must fill.

Values are raw strings copied from the document; normalize.py turns them into numbers and
dates. Structured outputs require every property to be listed as required with no extra
keys, so these models have no defaults: the model must write null explicitly.
"""

from enum import StrEnum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, model_validator

from co_extract.schema import Reason, Status

T = TypeVar("T")


class _Out(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Extracted(_Out, Generic[T]):
    value: T | None
    evidence: str | None  # verbatim quote from the document that contains the value
    page: int | None
    alternatives: list[str]  # other candidate values seen (superseded, proposed, conflicting)


class LineItemOut(_Out):
    description: str | None
    quantity: str | None
    unit: str | None
    unit_cost: str | None
    amount: str | None
    evidence: str | None
    page: int | None


class MarkupOut(_Out):
    description: str | None
    rate_percent: str | None
    amount: str | None
    evidence: str | None
    page: int | None


class Party(StrEnum):
    ARCHITECT = "architect"
    CONTRACTOR = "contractor"
    OWNER = "owner"
    OTHER = "other"


class SignatureOut(_Out):
    party: Party
    name: str | None
    date: str | None
    signed: bool
    evidence: str | None
    page: int | None


class LlmExtraction(_Out):
    change_order_number: Extracted[str]
    project_name: Extracted[str]
    project_number: Extracted[str]
    date_issued: Extracted[str]
    contractor: Extracted[str]
    owner: Extracted[str]
    architect_engineer: Extracted[str]
    description: Extracted[str]
    reason: Extracted[Reason]
    reason_text: Extracted[str]
    line_items: list[LineItemOut]
    subtotal: Extracted[str]
    markups: list[MarkupOut]
    change_amount: Extracted[str]
    currency: Extracted[str]
    original_contract_sum: Extracted[str]
    previous_changes_total: Extracted[str]
    revised_contract_sum: Extracted[str]
    schedule_impact_days: Extracted[str]
    status: Extracted[Status]
    signatures: list[SignatureOut]
    notes: list[str]  # unreadable areas, stamps over values, conflicts the model noticed

    @model_validator(mode="after")
    def _values_need_evidence(self) -> "LlmExtraction":
        missing = [
            name for name, item in self.scalars() if item.value is not None and not item.evidence
        ]
        if missing:
            raise ValueError(f"value given without evidence for: {', '.join(missing)}")
        return self

    def scalars(self) -> list[tuple[str, Extracted]]:
        """(name, field) for every single-valued field, in schema order."""
        return [(name, item) for name, item in self if isinstance(item, Extracted)]


def raw_value(item: Extracted) -> str | None:
    """The field's value as a plain string (enum members give their value)."""
    value = item.value
    return value.value if isinstance(value, StrEnum) else value


def output_schema() -> dict[str, Any]:
    return LlmExtraction.model_json_schema()
