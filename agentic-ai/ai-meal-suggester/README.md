# AI Meal Suggester

A small LangGraph app that greets a user, remembers their dietary preferences between runs, suggests matching dishes, and welcomes returning visitors. No LLM is involved: it shows how a checkpointer keeps per-user memory.

```
START -> greet_user -> remember_preferences -> END      (compiled with InMemorySaver)
```

| Piece | What it does |
|---|---|
| `MemoryState` | `messages`, `user_id`, and `user_memory = {"diet": [...], "visits": int}` |
| `greet_user` | Appends a greeting that uses `user_id` |
| `remember_preferences` | Detects vegan / vegetarian / gluten-free, stores each once, counts visits, suggests a dish, welcomes returning users |
| `thread_id` | One thread per user, so memory never mixes between users |

## Run

```bash
pip install -r requirements.txt
python meal_suggester.py
```

The script runs three checks: a first visit (`vegan`, visits 1), a second visit on the same thread (`vegan` + `gluten-free`, visits 2, welcome back), and a different user on a separate thread (starts clean).

## Notes

- The greeting node runs first, so the last message in state is the assistant's own. Assistant messages carry an `Assistant: ` prefix and the memory node reads the newest message without it.
- Keyword matching is simple. A message like "no meat or dairy" would be missed; an LLM extractor (see lab 11 in `agentic-ai-ollama`) handles that.
- `InMemorySaver` forgets everything when the program exits. A database-backed checkpointer would keep memory across restarts.
