# Customer Service State

A LangGraph exercise focused on the **state schema**: `CustomerServiceState` is the one shared
object every node reads from and writes to. No LLM involved -- triage is keyword-based so the
state behavior is the thing on display.

## State

```python
class CustomerServiceState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # reducer: append
    category: Literal["billing", "technical", "general"]  # no reducer: overwrite
    priority: Literal["low", "high"]                      # no reducer: overwrite
    escalated: bool                                       # no reducer: overwrite
```

- `messages` has a reducer, so each node's returned messages are **appended** -- the customer's
  message and every reply are all still there at the end.
- The other fields have no reducer, so a returned value **replaces** the old one.

## Graph

```
START -> triage -> route -+-> billing   -> END
                          +-> technical -> END
                          +-> general   -> END
                          +-> escalate  -> END   (any urgent message, regardless of category)
```

## Run

```bash
pip install -r requirements.txt
python service.py
```
