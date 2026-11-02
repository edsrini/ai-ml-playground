# AI/ML Playground

[![tests](https://github.com/edsrini/ai-ml-playground/actions/workflows/tests.yml/badge.svg)](https://github.com/edsrini/ai-ml-playground/actions/workflows/tests.yml)

Hands-on projects in **MLOps** and **agentic AI**: reproducible ML pipelines, and LangChain / LangGraph agents running on local open-weight models (Ollama).

## Start here

Two projects that show the approach best: a local model handles the language, and plain, tested Python decides what is allowed.

- **[`agentic-ai/text2sql-agent`](agentic-ai/text2sql-agent)**: ask a database questions in plain English. Schema-aware tools, a read-only SQL guard, errors returned as data, 21 deterministic tests.
- **[`agentic-ai/log-search-lab`](agentic-ai/log-search-lab)**: an agent that investigates an incident across 1M synthetic logs in a local Elasticsearch + Kibana stack, using guarded search tools.

The groundwork behind them is in [`agentic-ai/agentic-ai-ollama`](agentic-ai/agentic-ai-ollama) (numbered LangChain / LangGraph labs) and [`mlops/reproducible-ml-pipeline`](mlops/reproducible-ml-pipeline) (data validation, artifact versioning, experiment tracking, a tagged release).

## Agentic AI (LangChain / LangGraph)

| Project | What it shows |
|---|---|
| [`agentic-ai/agentic-ai-ollama`](agentic-ai/agentic-ai-ollama) | Labs 01-20 on local models: chains, few-shot, structured output, tools, graphs, memory, agentic RAG, MLflow tracing, human-in-the-loop, RAGAS evaluation, long-term persistence |
| [`agentic-ai/ai-library-assistant`](agentic-ai/ai-library-assistant) | A library assistant as a LangGraph workflow and as a tool-calling agent, with a one-command `bootstrap.py` setup |
| [`agentic-ai/customer-service-state`](agentic-ai/customer-service-state) | State schema and reducers (`add_messages`, `operator.add`), department routing, per-thread memory |
| [`agentic-ai/ai-meal-suggester`](agentic-ai/ai-meal-suggester) | Per-user memory with a checkpointer: remembers diet preferences across runs, welcomes returning visitors, isolates users by `thread_id` |
| [`agentic-ai/text2sql-agent`](agentic-ai/text2sql-agent) | Text-to-SQL agent on a local model: schema-aware tools, read-only SQL guard, errors returned as data, pytest suite |
| [`agentic-ai/ai-knowledgebase-agent`](agentic-ai/ai-knowledgebase-agent) | Skeleton of a knowledge-base agent: documents loaded into a saved Chroma store, search and web-fallback tools, a local-model agent that cites its sources (same structure as text2sql-agent) |
| [`agentic-ai/log-search-lab`](agentic-ai/log-search-lab) | Local Elasticsearch + Kibana with 1M synthetic logs, guarded log-search tools, and a local-model agent that finds the cause of an incident (same structure as text2sql-agent) |
| [`agentic-ai/agentic-ai-healthcare-appointment-system`](agentic-ai/agentic-ai-healthcare-appointment-system) | Supervisor (coordinator) graph: doctor availability, patient verification, booking and notifications as separate agents, with retries routed through one coordinator |
| [`agentic-ai/text-action-router`](agentic-ai/text-action-router) | Conditional-edge routing with graceful handling of invalid actions |
| [`agentic-ai/ai-business-advisor`](agentic-ai/ai-business-advisor) | Multi-step LCEL workflow: industry, business idea, analysis, structured report |
| [`agentic-ai/claims-issue-diagnosis`](agentic-ai/claims-issue-diagnosis) | Prototype: read-only agent that diagnoses synthetic claims-ingestion failures. The evidence checks are tested; the LLM triage and diagnosis steps have not been evaluated yet |

## MLOps

| Project | What it shows |
|---|---|
| [`mlops/reproducible-ml-pipeline`](mlops/reproducible-ml-pipeline) | Artifact versioning, data validation, experiment tracking and pipeline orchestration with MLflow and Weights & Biases |

## Deep Learning (PyTorch)

| Project | What it shows |
|---|---|
| [`deep-learning`](deep-learning) | PyTorch exercises from logic gates to multi-layer networks, training and validation loops, tuning |

## Notes

- Many projects began as course exercises in agentic AI (LangChain / LangGraph) and MLOps, then were extended and documented here.
- All data is synthetic. No real claims, members, or personal information.
- Small local models are less reliable than hosted ones, so the code keeps exact decisions (validation, updates, escalation) in plain Python and uses the LLM for reading language.

---

**Srinivasan E** · [LinkedIn](https://www.linkedin.com/in/esrinivasan/) · [GitHub](https://github.com/edsrini)
