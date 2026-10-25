# Agentic AI with Ollama

Hands-on exercises from an agentic AI course (LangChain and LangGraph), run against locally-hosted open-weight models via [Ollama](https://ollama.com) instead of a paid hosted API.

## Why local/open models

Running exercises against an open model surfaces things a hosted frontier model hides — smaller models are noticeably less reliable at structured tool-calling, which is exactly the kind of failure mode worth understanding before shipping an agent to production. Model choice is a single config value (`config.py`), so any exercise can be re-run against a hosted provider later for comparison.

## Setup

```bash
pip install -e .   # editable install (pyproject.toml) -- makes `config` importable
                    # from every script under langchain_langgraph_labs/ and chatbot/, no sys.path hacks
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
| [`langchain_langgraph_labs/01_basic_chat.py`](langchain_langgraph_labs/01_basic_chat.py) | Direct chat call to a local model |
| [`langchain_langgraph_labs/02_prompt_chain.py`](langchain_langgraph_labs/02_prompt_chain.py) | Prompt template + output parser chain |
| [`langchain_langgraph_labs/03_single_tool_agent.py`](langchain_langgraph_labs/03_single_tool_agent.py) | LangGraph ReAct agent with one bound tool (`create_agent`) |
| [`langchain_langgraph_labs/04_few_shot_prompt.py`](langchain_langgraph_labs/04_few_shot_prompt.py) | Few-shot prompting via worked examples |
| [`langchain_langgraph_labs/05_structured_output.py`](langchain_langgraph_labs/05_structured_output.py) | Schema-enforced output instead of free text |
| [`langchain_langgraph_labs/06_model_comparison.py`](langchain_langgraph_labs/06_model_comparison.py) | Scoring local models against a deterministic baseline |
| [`langchain_langgraph_labs/07_multistep_workflow.py`](langchain_langgraph_labs/07_multistep_workflow.py) | LCEL runnables: chains, batch, stream, parallel |
| [`langchain_langgraph_labs/08_functions_and_tools.py`](langchain_langgraph_labs/08_functions_and_tools.py) | Manual tool-binding mechanics behind `create_agent` |
| [`langchain_langgraph_labs/09_langgraph_basics.py`](langchain_langgraph_labs/09_langgraph_basics.py) | Hand-rolled ReAct loop: custom state, conditional edges |
| [`langchain_langgraph_labs/09_langgraph_workflow.py`](langchain_langgraph_labs/09_langgraph_workflow.py) | Deterministic graph vs. an LLM-backed graph, side by side |
| [`langchain_langgraph_labs/10_thread_isolation.py`](langchain_langgraph_labs/10_thread_isolation.py) | Checkpointer demos: thread isolation, config vs state, time travel, per-user memory |
| [`langchain_langgraph_labs/11_llm_preference_memory.py`](langchain_langgraph_labs/11_llm_preference_memory.py) | LLM node extracts food preferences from natural language; code validates and remembers them |
| [`langchain_langgraph_labs/12_tools_with_apis.py`](langchain_langgraph_labs/12_tools_with_apis.py) | `@tool` functions that call web APIs (quotes, Tavily search), bound to the local model |
| [`langchain_langgraph_labs/13_sql_toolkit.py`](langchain_langgraph_labs/13_sql_toolkit.py) | Database tools (list tables, schema, read-only SQL) for a text-to-SQL agent |
| [`langchain_langgraph_labs/14_agentic_rag.py`](langchain_langgraph_labs/14_agentic_rag.py) | Agentic RAG: embeddings and a vector store, with an agent that decides when to search and answers only from the passages (needs `ollama pull nomic-embed-text`) |
| [`langchain_langgraph_labs/14b_agentic_rag_structured.py`](langchain_langgraph_labs/14b_agentic_rag_structured.py) | The same agentic RAG, with a structured-output evaluator that returns a real true/false instead of a YES/NO string |
| [`langchain_langgraph_labs/15_pdf_without_embeddings.py`](langchain_langgraph_labs/15_pdf_without_embeddings.py) | Reading a PDF without embeddings: the whole text goes in the prompt, with a length guard that points to lab 14 when it is too long |
| [`langchain_langgraph_labs/16_agent_observability.py`](langchain_langgraph_labs/16_agent_observability.py) | MLflow tracing setup: sending an agent's steps to a local tracking server |
| [`langchain_langgraph_labs/17_human_in_the_loop.py`](langchain_langgraph_labs/17_human_in_the_loop.py) | Human-in-the-loop: the graph pauses before the tool node and a person approves or aborts the tool call |
| [`langchain_langgraph_labs/18_human_in_the_loop_edit_question.py`](langchain_langgraph_labs/18_human_in_the_loop_edit_question.py) | Human-in-the-loop: a person reviews the question before the agent starts and can replace it with `update_state` |

Run any example directly:

```bash
python langchain_langgraph_labs/01_basic_chat.py
```

## Roadmap

Further exercises (multi-tool planning, RAG, memory, multi-agent collaboration) get added as separate `NN_*.py` files under `langchain_langgraph_labs/` as the corresponding course lessons are worked through.
