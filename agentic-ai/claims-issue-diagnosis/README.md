# Claims Issue Diagnosis

A read-only LangGraph agent that diagnoses claims-ingestion failures. The LLM reads the raw error
log and writes the diagnosis; plain Python gathers the evidence and decides when to escalate.
**All data is synthetic** (fake members and claims, no PHI). The agent recommends a fix; a human applies it.

## Flow

```
failure log -> triage (LLM: classify) -+-> unknown -> escalate (no LLM guess) -> END
                                       +-> known   -> gather_evidence (code) -> diagnose (LLM)
                                                      -> review_gate (code: low confidence -> human) -> END
```

| Piece | Role |
|---|---|
| `triage` | LLM, structured output: one of 5 failure categories plus a one-line summary |
| `gather_evidence` | Deterministic validators, claim lookup, duplicate check (`tools.py`) |
| `diagnose` | LLM, structured output: root cause, recommended fix, confidence, using only the evidence |
| `review_gate` | Code forces escalation when confidence is low, whatever the model says |

## Run

```bash
pip install -r requirements.txt
ollama pull qwen3:4b
python diagnosis.py     # diagnose every synthetic incident
python evaluate.py      # score triage against the known category of each incident
```

## Not included yet

Runbook retrieval (RAG) over real documents, human approval before any write action, a persistent
checkpointer, tracing, and a larger evaluation set. A small local model is also a weak diagnostician;
expect to test a larger one before trusting it.
