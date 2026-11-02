# AI/ML Playground

[![tests](https://github.com/edsrini/ai-ml-playground/actions/workflows/tests.yml/badge.svg)](https://github.com/edsrini/ai-ml-playground/actions/workflows/tests.yml)

Hands-on projects in **MLOps** and **agentic AI**: reproducible ML pipelines, and LangChain / LangGraph agents running on local open-weight models (Ollama).

## Start here

Two projects that show the approach best: a local model handles the language, and plain, tested Python decides what is allowed.

- **[`agentic-ai/text2sql-agent`](agentic-ai/text2sql-agent)**: ask a database questions in plain English. Schema-aware tools, a read-only SQL guard, errors returned as data, 21 deterministic tests.
- **[`agentic-ai/log-search-lab`](agentic-ai/log-search-lab)**: an agent that investigates an incident across 1M synthetic logs in a local Elasticsearch + Kibana stack, using guarded search tools.

### Also in this repo

**Multi-agent systems**
- [`agentic-ai/agentic-ai-blog-post`](agentic-ai/agentic-ai-blog-post): a five-stage blog pipeline (intake, research, writer, reviewer, publisher) of tool-using agents with Tavily web search, SEO scoring and Markdown linting; also rebuilt as a plain LCEL chain for comparison.
- [`agentic-ai/agentic-ai-healthcare-appointment-system`](agentic-ai/agentic-ai-healthcare-appointment-system): a supervisor graph that coordinates doctor availability, patient verification, booking and notifications, with retries.
- [`agentic-ai/multi-agent-orchestration-demo`](agentic-ai/multi-agent-orchestration-demo): a supervisor agent that calls a research agent and a writer agent as tools.

**Agents, RAG and memory**
- [`agentic-ai/ai-knowledgebase-agent`](agentic-ai/ai-knowledgebase-agent): documents in a saved Chroma store, search and web-fallback tools, answers that cite their sources.
- [`agentic-ai/claims-issue-diagnosis`](agentic-ai/claims-issue-diagnosis): a read-only agent that diagnoses synthetic claims-ingestion failures, with tested evidence checks.
- [`agentic-ai/ai-library-assistant`](agentic-ai/ai-library-assistant): the same library assistant built as a LangGraph workflow and as a tool-calling agent.
- [`agentic-ai/ai-meal-suggester`](agentic-ai/ai-meal-suggester) and [`agentic-ai/customer-service-state`](agentic-ai/customer-service-state): per-user memory, state reducers and routing.
- [`agentic-ai/text-action-router`](agentic-ai/text-action-router) and [`agentic-ai/ai-business-advisor`](agentic-ai/ai-business-advisor): conditional routing and a multi-step LCEL workflow.

**Foundations**
- [`agentic-ai/agentic-ai-ollama`](agentic-ai/agentic-ai-ollama): 20 numbered LangChain / LangGraph labs, from a first chat call to agentic RAG, human-in-the-loop, RAGAS evaluation and long-term memory in PostgreSQL; plus a small chatbot with memory and a persona.
- [`mlops/reproducible-ml-pipeline`](mlops/reproducible-ml-pipeline): data validation, artifact versioning, experiment tracking and a tagged release with MLflow and Weights & Biases.
- [`mlops/mlflow-server`](mlops/mlflow-server): a standalone MLflow tracking server in Docker.
- [`deep-learning`](deep-learning): PyTorch exercises from logic gates to multi-layer networks, training loops and tuning.

## Tools used

| Area | Tools |
|---|---|
| Agents and LLMs | LangChain (LCEL), LangGraph, LangMem, Ollama (local open-weight models, e.g. Qwen3), Pydantic (structured output), Tavily web search |
| RAG and data stores | Chroma, Elasticsearch + Kibana, SQLite / SQLAlchemy, PostgreSQL (LangGraph checkpointer), pypdf / fpdf2 (PDF loading and generation) |
| Evaluation and observability | RAGAS, MLflow tracing, pytest |
| MLOps | MLflow (projects, tracking server), Weights & Biases, Hydra / OmegaConf, ydata-profiling, conda, JupyterLab |
| ML and deep learning | PyTorch, Hugging Face Datasets, scikit-learn, pandas, NumPy, SciPy, Matplotlib, Seaborn |
| Infrastructure | Docker, Docker Compose, GitHub Actions, python-dotenv (config) |
| Development | Python, PyCharm, VS Code, Git / GitHub, Claude Code |

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
| [`agentic-ai/agentic-ai-blog-post`](agentic-ai/agentic-ai-blog-post) | Blog-writing pipeline of five tool-using agents (intake, research, writer, reviewer, publisher): Tavily search, reference formatting, read time, Markdown lint and SEO score; an extended version with Pydantic-typed state, and the same pipeline as a plain LCEL chain |
| [`agentic-ai/multi-agent-orchestration-demo`](agentic-ai/multi-agent-orchestration-demo) | Supervisor pattern in miniature: a coordinating agent calls a research agent and a writer agent as tools |
| [`agentic-ai/agentic-ai-healthcare-appointment-system`](agentic-ai/agentic-ai-healthcare-appointment-system) | Supervisor (coordinator) graph: doctor availability, patient verification, booking and notifications as separate agents, with retries routed through one coordinator |
| [`agentic-ai/text-action-router`](agentic-ai/text-action-router) | Conditional-edge routing with graceful handling of invalid actions |
| [`agentic-ai/ai-business-advisor`](agentic-ai/ai-business-advisor) | Multi-step LCEL workflow: industry, business idea, analysis, structured report |
| [`agentic-ai/claims-issue-diagnosis`](agentic-ai/claims-issue-diagnosis) | Prototype: read-only agent that diagnoses synthetic claims-ingestion failures. The evidence checks are tested; the LLM triage and diagnosis steps have not been evaluated yet |

## MLOps

| Project | What it shows |
|---|---|
| [`mlops/reproducible-ml-pipeline`](mlops/reproducible-ml-pipeline) | Artifact versioning, data validation, experiment tracking and pipeline orchestration with MLflow and Weights & Biases |
| [`mlops/mlflow-server`](mlops/mlflow-server) | Standalone MLflow tracking server from the official Docker image, bound to localhost, data kept in `./mlruns` |

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
