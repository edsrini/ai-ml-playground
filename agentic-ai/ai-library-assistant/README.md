# AI Library Assistant

A tool-calling agent for a small library, running on a local open model via [Ollama](https://ollama.com).
The model decides when to call a tool; plain Python tools do the actual catalog work, and a
checkpointer remembers each member's conversation.

## Pieces

| File | Role |
|---|---|
| `library.py` | In-memory catalog plus three tools: `search_books`, `checkout_book`, `return_book` |
| `library_agent.py` | `create_agent` with a system prompt, the tools, and an `InMemorySaver`; demo conversation |
| `config.py` | Model name and Ollama URL (override with `OLLAMA_MODEL` in `.env`) |
| `bootstrap.py` | One-command setup: venv, pinned dependencies, `.env`, Ollama/model check, smoke test |
| `requirements.txt` | Direct dependencies pinned to the versions this project was tested with |

## How it works

```
member message -> agent (LLM) --tool call?--> tools (library.py) --result--> agent -> reply
                       thread_id = member id, so follow-ups ("is it available?") keep context
```

## Setup (for the class)

Prerequisites: Python 3.10+ and [Ollama](https://ollama.com/download) installed and running.

```bash
python bootstrap.py              # creates .venv, installs dependencies, downloads the model (~2.5 GB)
python bootstrap.py --skip-model # same, but don't download the model automatically
python bootstrap.py --model llama3.2:1b   # use a smaller model on a weak laptop
```

It is safe to re-run; finished steps are skipped. When it ends with "Smoke test: passed" and
"Ollama + model ready: yes", activate the environment and run the assistant:

```bash
.venv\Scripts\activate           # Windows   (macOS/Linux: source .venv/bin/activate)
python library_agent.py
```

Prefer to do it by hand? `python -m venv .venv`, activate it, `pip install -r requirements.txt`,
`ollama pull qwen3:4b`, then `python library_agent.py`.

## Notes

- Small local models are less reliable at multi-step tool use; the tools therefore refuse bad
  requests themselves (unknown title, no copies left) instead of trusting the model.
- Memory is in-process only (`InMemorySaver`); use a persistent checkpointer for a real service.
