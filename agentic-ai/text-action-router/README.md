# Text Action Router

A LangGraph exercise: conditional-edge routing to one of several nodes based on a given
action, with graceful handling of an unrecognized action. No LLM involved -- the action
is given directly, so the routing decision is a plain conditional edge, not something
inferred by a model.

## Graph

```
         START
           |
   route_action(state)
      /    |    \
reverse  upper  invalid
      \    |    /
          END
```

- `reverse` -- reverses the input text
- `upper` -- uppercases the input text
- `invalid` -- any other action falls through here instead of crashing the graph

## Run

```bash
pip install -r requirements.txt
python router.py
```
