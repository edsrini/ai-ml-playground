"""Claims-ingestion issue diagnosis -- read-only: the LLM classifies and explains, code gathers evidence.

triage (LLM) -> unknown?  -> escalate (no LLM guess) -> END
              -> otherwise -> gather_evidence (plain Python) -> diagnose (LLM) -> review_gate -> END
"""
import sys
from typing import Literal, TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from config import OLLAMA_BASE_URL, OLLAMA_MODEL
from tools import find_duplicates, get_claim, lookup_runbook, validate_claim

sys.stdout.reconfigure(encoding="utf-8")

# reasoning=False skips qwen3's long hidden thinking; the JSON schema constrains the output anyway.
llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0, num_ctx=8192,
                 num_predict=500, reasoning=False)


class Triage(BaseModel):
    category: Literal["schema_error", "invalid_value", "duplicate_claim", "eligibility_timeout", "unknown"] = Field(
        description="schema_error: required field missing/null. invalid_value: field present but bad date/amount/format. "
                    "duplicate_claim: duplicate key / already exists. eligibility_timeout: eligibility service slow/down. "
                    "unknown: anything else, including crashes in vendor code.")
    summary: str = Field(description="One sentence describing what failed")


class Diagnosis(BaseModel):
    root_cause: str = Field(description="The cause, stated only from the evidence provided")
    recommended_fix: str = Field(description="Concrete next step for a human, following the runbook")
    confidence: Literal["low", "medium", "high"] = Field(description="low if the evidence does not clearly support the cause")


class IncidentState(TypedDict, total=False):  # total=False: keys appear as nodes fill them in
    incident: dict
    category: str
    triage_summary: str
    evidence: list[str]
    diagnosis: dict
    escalate: bool


def triage(state: IncidentState) -> IncidentState:
    result = llm.with_structured_output(Triage).invoke(
        f"Classify this claims-ingestion failure log:\n{state['incident']['log']}")
    return {"category": result.category, "triage_summary": result.summary}


def route_after_triage(state: IncidentState) -> str:
    return "escalate" if state["category"] == "unknown" else "gather_evidence"  # never let the LLM guess on unknowns


def escalate(state: IncidentState) -> IncidentState:
    return {"evidence": [], "escalate": True,
            "diagnosis": {"root_cause": "Not classified as a known failure type.",
                          "recommended_fix": lookup_runbook("unknown"), "confidence": "low"}}


def gather_evidence(state: IncidentState) -> IncidentState:
    claim = get_claim(state["incident"]["claim_id"])
    evidence = []
    if claim is None:
        evidence.append("no claim record found for this incident")
    else:
        problems = validate_claim(claim)
        evidence.append(f"validator on {claim['claim_id']}: " + ("; ".join(problems) if problems else "claim data is valid"))
        if state["category"] == "duplicate_claim":
            duplicates = find_duplicates(claim)
            evidence.append(f"duplicate check: matches {', '.join(duplicates)}" if duplicates else "duplicate check: no matching claim found")
    return {"evidence": evidence}


def diagnose(state: IncidentState) -> IncidentState:
    prompt = (f"Incident log: {state['incident']['log']}\nCategory: {state['category']}\n"
              "Evidence gathered by validators:\n- " + "\n- ".join(state["evidence"]) +
              f"\nRunbook for this category: {lookup_runbook(state['category'])}\n\n"
              "Diagnose the root cause using ONLY the log, evidence and runbook above. "
              "If the evidence does not support a clear cause, set confidence to low.")
    result = llm.with_structured_output(Diagnosis).invoke(prompt)
    return {"diagnosis": result.model_dump()}


def review_gate(state: IncidentState) -> IncidentState:
    # Plain-code safety net: low confidence always goes to a person, whatever the model said.
    return {"escalate": state["diagnosis"]["confidence"] == "low"}


builder = StateGraph(IncidentState)
builder.add_node("triage", triage)
builder.add_node("escalate", escalate)
builder.add_node("gather_evidence", gather_evidence)
builder.add_node("diagnose", diagnose)
builder.add_node("review_gate", review_gate)

builder.add_edge(START, "triage")
builder.add_conditional_edges("triage", route_after_triage, ["escalate", "gather_evidence"])
builder.add_edge("gather_evidence", "diagnose")
builder.add_edge("diagnose", "review_gate")
builder.add_edge("escalate", END)
builder.add_edge("review_gate", END)

graph = builder.compile()


def diagnose_incident(incident: dict) -> IncidentState:
    return graph.invoke({"incident": incident})


if __name__ == "__main__":
    from data import INCIDENTS

    for incident in INCIDENTS:
        result = diagnose_incident(incident)
        d = result["diagnosis"]
        print(f"\n{incident['incident_id']}  category={result['category']}  escalate={result['escalate']}")
        print(f"  log:       {incident['log']}")
        for item in result["evidence"]:
            print(f"  evidence:  {item}")
        print(f"  cause:     {d['root_cause']}")
        print(f"  fix:       {d['recommended_fix']}")
        print(f"  confidence: {d['confidence']}")
