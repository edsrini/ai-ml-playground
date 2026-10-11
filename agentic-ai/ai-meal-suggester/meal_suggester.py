"""Meal Suggester: greets users, remembers dietary preferences across runs, personalizes dishes, welcomes returning visitors.

Quick checks:
- Role of a checkpointer: it saves the graph's state after each step under a thread_id, so a later invoke on the same
  thread can load that saved state and continue from it instead of starting empty.
- Isolation: each user gets their own thread_id, and the checkpointer stores and loads state per thread, so two users
  share one compiled workflow without their messages or memory ever mixing.
"""
from typing import Any, Dict, List, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

BOT = "Assistant: "  # prefix on every assistant message, so we can tell it apart from the user's messages


class MemoryState(TypedDict):
    messages: List[str]
    user_id: str
    user_memory: Dict[str, Any]  # long-term memory: {"diet": List[str], "visits": int}


# Node 1: greet the user by their id
def greet_user(state: MemoryState) -> MemoryState:
    state["messages"].append(f"{BOT}Hello {state['user_id']}! Welcome to the Meal Suggester.")
    return state


# Node 2: update user memory and personalize the response
def remember_preferences(state: MemoryState) -> MemoryState:
    # greet_user already appended its greeting, so messages[-1] is the bot's own. The user's message is the
    # newest one without the bot prefix.
    last_msg = next((m for m in reversed(state["messages"]) if not m.startswith(BOT)), "").lower()

    # First invocation starts with {}, so build the structure here: diet is a list, visits is an int
    diet = list(state["user_memory"].get("diet", []))

    if "vegan" in last_msg:
        diet.append("vegan")
    if "vegetarian" in last_msg:
        diet.append("vegetarian")
    if "gluten-free" in last_msg or "gluten free" in last_msg:  # accept both spellings
        diet.append("gluten-free")

    # Update the diet in user memory
    if diet:
        state["user_memory"]["diet"] = diet

    # Track visits
    count = state["user_memory"].get("visits", 0) + 1

    # Add a welcome-back message on return visits
    if count > 1:
        state["messages"].append(f"{BOT}Welcome back! This is visit #{count}")

    state["user_memory"]["visits"] = count

    # Personalize response based on preferences
    if diet:
        # Suggest a dish based on the first preference listed
        if "vegan" in diet:
            state["messages"].append(f"{BOT}May I recommend our vegan Buddha bowl?")
        elif "vegetarian" in diet:
            state["messages"].append(f"{BOT}How about a fresh vegetarian salad?")
        elif "gluten-free" in diet:
            state["messages"].append(f"{BOT}Try our gluten-free pasta.")

    return state


workflow = StateGraph(MemoryState)
workflow.add_node("remember_preferences", remember_preferences)
workflow.add_edge(START, "remember_preferences")
workflow.add_edge("remember_preferences", END)

app = workflow.compile(checkpointer=InMemorySaver())  # saved state is lost when the program exits


# config with a unique thread_id
config = {"configurable": {"thread_id": "user123"}}

# First interaction: user states a dietary preference
state1 = {
    "messages": ["I'm vegan and looking for options"],
    "user_id": "user123",
    "user_memory": {}
}

result1 = app.invoke(state1, config=config)
print("First interaction - User Memory:", result1["user_memory"])
# Expected: {'diet': ['vegan'], 'visits': 1}

print("First interaction - Messages:", result1["messages"])
# Expected to include a vegan dish suggestion

saved_state = app.get_state(config={"configurable": {"thread_id": "user123"}}).values  # .values is the state dict

saved_state["messages"].append("I am also gluten-free")
result2 = app.invoke(saved_state, config=config)
print("Second interaction - User Memory:", result2["user_memory"])
# Expected: {'diet': ['vegan', 'gluten-free'], 'visits': 2}

print("Second interaction - Messages:", result2["messages"])
