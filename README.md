# AI/ML Playground

Hands-on projects in **MLOps** and **agentic AI**: reproducible ML pipelines, and LangChain / LangGraph agents running on local open-weight models (Ollama).

## Start here

- **[`agentic-ai/agentic-ai-ollama`](agentic-ai/agentic-ai-ollama)** is the main agentic AI work. Numbered labs cover prompting, structured output, tool calling, LangGraph state graphs, checkpointer memory, time travel, and an LLM node with plain-code validation around it.
- **[`mlops/reproducible-ml-pipeline`](mlops/reproducible-ml-pipeline)** is the MLOps work: a reproducible pipeline with data validation, artifact versioning, experiment tracking, and a tagged release.

## Agentic AI (LangChain / LangGraph)

| Project | What it shows |
|---|---|
| [`agentic-ai/agentic-ai-ollama`](agentic-ai/agentic-ai-ollama) | Labs 01-11 on local models: chains, few-shot, structured output, tools, graphs, memory, LLM-assisted extraction |
| [`agentic-ai/ai-library-assistant`](agentic-ai/ai-library-assistant) | A library assistant as a LangGraph workflow and as a tool-calling agent, with a one-command `bootstrap.py` setup |
| [`agentic-ai/customer-service-state`](agentic-ai/customer-service-state) | State schema and reducers (`add_messages`, `operator.add`), department routing, per-thread memory |
| [`agentic-ai/ai-meal-suggester`](agentic-ai/ai-meal-suggester) | Per-user memory with a checkpointer: remembers diet preferences across runs, welcomes returning visitors, isolates users by `thread_id` |
| [`agentic-ai/text2sql-agent`](agentic-ai/text2sql-agent) | Text-to-SQL agent on a local model: schema-aware tools, read-only SQL guard, errors returned as data, pytest suite |
| [`agentic-ai/log-search-lab`](agentic-ai/log-search-lab) | Local Elasticsearch + Kibana with 1M synthetic logs, guarded log-search tools, and a local-model agent that finds the cause of an incident (same structure as text2sql-agent) |
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
| [`deep-learning`](deep-learning) | PyTorch exercises from logic gates to multi-layer networks, training and validation loops, tuning; plus a character-level GPT built from scratch (`tiny-gpt`, not yet run end to end) |

## Notes

- Many projects began as exercises from Udacity's Agentic AI Engineer (LangChain / LangGraph) and MLOps courses, then were extended and documented here.
- All data is synthetic. No real claims, members, or personal information.
- Small local models are less reliable than hosted ones, so the code keeps exact decisions (validation, updates, escalation) in plain Python and uses the LLM for reading language.

---

**Srinivasan E** · [LinkedIn](https://www.linkedin.com/in/esrinivasan/) · [GitHub](https://github.com/edsrini)
