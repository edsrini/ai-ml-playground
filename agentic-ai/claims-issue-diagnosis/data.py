"""Synthetic claims data, ingestion-failure incidents, and runbook. All fake -- no real members or PHI."""

CLAIMS = {
    "CLM-001": {"claim_id": "CLM-001", "member_id": "M-1001", "service_date": "2025-13-45", "billed_amount": 120.00, "procedure_code": "99213"},
    "CLM-002": {"claim_id": "CLM-002", "member_id": None, "service_date": "2025-03-04", "billed_amount": 85.50, "procedure_code": "99214"},
    "CLM-003": {"claim_id": "CLM-003", "member_id": "M-3003", "service_date": "2025-04-10", "billed_amount": 310.00, "procedure_code": "73721"},
    "CLM-004": {"claim_id": "CLM-004", "member_id": "M-3003", "service_date": "2025-04-10", "billed_amount": 310.00, "procedure_code": "73721"},  # same member/date/amount as CLM-003
    "CLM-005": {"claim_id": "CLM-005", "member_id": "M-5521", "service_date": "2025-05-02", "billed_amount": 75.00, "procedure_code": "99212"},
    "CLM-006": {"claim_id": "CLM-006", "member_id": "M-6006", "service_date": "2025-06-15", "billed_amount": -250.00, "procedure_code": "97110"},
}

# expected_category is the ground truth used by evaluate.py to score the model's triage.
INCIDENTS = [
    {"incident_id": "INC-1001", "claim_id": "CLM-001", "expected_category": "invalid_value",
     "log": "ValidationError: service_date '2025-13-45' does not match format YYYY-MM-DD (batch 41, row 17)"},
    {"incident_id": "INC-1002", "claim_id": "CLM-002", "expected_category": "schema_error",
     "log": "SchemaError: required field 'member_id' is null (batch 41, row 18)"},
    {"incident_id": "INC-1003", "claim_id": "CLM-003", "expected_category": "duplicate_claim",
     "log": "IntegrityError: duplicate key (member_id, service_date, billed_amount) already exists; insert of CLM-003 rejected"},
    {"incident_id": "INC-1004", "claim_id": "CLM-005", "expected_category": "eligibility_timeout",
     "log": "TimeoutError: eligibility-service GET /members/M-5521 exceeded 30s (attempt 3/3); claim CLM-005 parked"},
    {"incident_id": "INC-1005", "claim_id": None, "expected_category": "unknown",
     "log": "Segmentation fault (core dumped) in vendor parser libclaimparse v2.3 while reading batch 88"},
    {"incident_id": "INC-1006", "claim_id": "CLM-006", "expected_category": "invalid_value",
     "log": "ValidationError: billed_amount -250.0 must be greater than 0 (batch 42, row 3)"},
]

RUNBOOK = {
    "schema_error": "Find the source record, confirm the required field is truly missing upstream, request a corrected record from the submitter, then re-ingest that row only.",
    "invalid_value": "Compare the value against the field's format or range rule, ask the submitter for a corrected claim (do not guess the value), then re-ingest that row only.",
    "duplicate_claim": "Confirm the existing claim matches member, service date and amount. If it does, reject the new claim as a duplicate; do not insert it. Notify the submitter.",
    "eligibility_timeout": "The claim data is fine; the eligibility service was slow or down. Check the service status, then retry the parked claim once it is healthy.",
    "unknown": "No automated diagnosis. Escalate to the on-call data engineer with the full log.",
}
