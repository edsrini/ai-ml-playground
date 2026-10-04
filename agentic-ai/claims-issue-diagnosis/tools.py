"""Deterministic evidence-gathering tools. Plain Python, so every check is exact and repeatable."""
import re
from datetime import datetime

from data import CLAIMS, RUNBOOK

REQUIRED_FIELDS = ["claim_id", "member_id", "service_date", "billed_amount", "procedure_code"]


def get_claim(claim_id: str | None) -> dict | None:
    return CLAIMS.get(claim_id) if claim_id else None


def validate_claim(claim: dict) -> list[str]:
    """Return every rule the claim breaks (empty list means the claim data is valid)."""
    problems = []
    for field in REQUIRED_FIELDS:
        if claim.get(field) in (None, ""):
            problems.append(f"required field '{field}' is missing or null")
    member_id = claim.get("member_id")
    if member_id and not re.fullmatch(r"M-\d{4}", member_id):
        problems.append(f"member_id '{member_id}' does not match M-####")
    service_date = claim.get("service_date")
    if service_date:
        try:
            datetime.strptime(service_date, "%Y-%m-%d")
        except ValueError:
            problems.append(f"service_date '{service_date}' is not a real YYYY-MM-DD date")
    amount = claim.get("billed_amount")
    if amount is not None and amount <= 0:
        problems.append(f"billed_amount {amount} must be greater than 0")
    return problems


def find_duplicates(claim: dict) -> list[str]:
    """Other claims with the same member, service date and amount."""
    return [c["claim_id"] for c in CLAIMS.values()
            if c["claim_id"] != claim["claim_id"]
            and (c["member_id"], c["service_date"], c["billed_amount"])
            == (claim["member_id"], claim["service_date"], claim["billed_amount"])]


def lookup_runbook(category: str) -> str:
    return RUNBOOK.get(category, RUNBOOK["unknown"])
