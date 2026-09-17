# AgentOps-Lab

Hands-on exercises from Udacity's [Agentic AI Engineer with LangChain and LangGraph](https://www.udacity.com/course/ai-agents-with-langchain-and-langgraph--cd13764) Nanodegree (ND901), run against locally-hosted open-weight models via [Ollama](https://ollama.com) instead of a paid hosted API.

## Why local/open models

Running exercises against an open model surfaces things a hosted frontier model hides — smaller models are noticeably less reliable at structured tool-calling, which is exactly the kind of failure mode worth understanding before shipping an agent to production. Model choice is a single config value (`config.py`), so any exercise can be re-run against a hosted provider later for comparison.

## Setup

```bash
pip install -r requirements.txt
ollama pull qwen3:4b   # or any model already available: ollama list
```

Override the model without touching code:

```bash
# .env
OLLAMA_MODEL=llama3.2:1b
```

## Examples

| File | Covers |
|---|---|
| [`examples/01_basic_chat.py`](examples/01_basic_chat.py) | Direct chat call to a local model |
| [`examples/02_prompt_chain.py`](examples/02_prompt_chain.py) | Prompt template + output parser chain |
| [`examples/03_single_tool_agent.py`](examples/03_single_tool_agent.py) | LangGraph ReAct agent with one bound tool |

Run any example directly:

```bash
python examples/01_basic_chat.py
```

## Roadmap

Further exercises (multi-tool planning, RAG, memory, multi-agent collaboration) get added as separate `NN_*.py` files under `examples/` as the corresponding course lessons are worked through.
