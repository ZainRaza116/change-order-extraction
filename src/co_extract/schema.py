"""Pydantic models for a change order.

ChangeOrderRecord holds plain values and is what ground truth in expected/ is written in.
"""

import datetime as dt
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class Reason(StrEnum):
    OWNER_REQUEST = "owner_request"
    DESIGN_ERROR = "design_error"
    UNFORESEEN_CONDITION = "unforeseen_condition"
    CODE_REQUIREMENT = "code_requirement"
    OTHER = "other"


class Status(StrEnum):
    PROPOSED = "proposed"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LineItem(_Strict):
    description: str
    quantity: Decimal | None = None
    unit: str | None = None
    unit_cost: Decimal | None = None
    amount: Decimal


class Markup(_Strict):
    """Overhead & profit, bond, insurance or tax added on top of the line-item subtotal."""

    description: str
    rate_percent: Decimal | None = None
    amount: Decimal


class Signature(_Strict):
    party: str
    name: str | None = None
    date: dt.date | None = None
    signed: bool


class ChangeOrderDraft(_Strict):
    """Every field optional: what the pipeline holds before it knows what's missing.

    Validation rules run on a draft, so a missing required field is a rule failure
    rather than a crash.
    """

    change_order_number: str | None = None
    project_name: str | None = None
    project_number: str | None = None
    date_issued: dt.date | None = None
    contractor: str | None = None
    owner: str | None = None
    architect_engineer: str | None = None
    description: str | None = None
    reason: Reason | None = None
    reason_text: str | None = None
    line_items: list[LineItem] = []
    subtotal: Decimal | None = None
    markups: list[Markup] = []
    change_amount: Decimal | None = None  # = sum(line_items) + sum(markups) when both listed
    currency: str | None = None
    original_contract_sum: Decimal | None = None
    previous_changes_total: Decimal | None = None
    revised_contract_sum: Decimal | None = None
    schedule_impact_days: int | None = None
    status: Status | None = None
    signatures: list[Signature] = []


REQUIRED_FIELDS = (
    "change_order_number",
    "project_name",
    "date_issued",
    "contractor",
    "description",
    "change_amount",
    "currency",
)


class ChangeOrderRecord(ChangeOrderDraft):
    """A complete change order: the required fields are present. Ground truth uses this."""

    change_order_number: str
    project_name: str
    date_issued: dt.date
    contractor: str
    description: str
    change_amount: Decimal
    currency: str
