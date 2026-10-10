"""LLM food-preference memory, in the same function style as 10_thread_isolation.py.

Same idea as Demo 6 there, but "Chocolate is my favorite!" works because a local model extracts the foods
instead of taking the last word. Memory lives in the checkpointer, one thread per user.
"""
import re
import sys
from typing import Any as AnyType, Dict as DictType, List, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from config import OLLAMA_BASE_URL, OLLAMA_MODEL

sys.stdout.reconfigure(encoding="utf-8")

KNOWN_FOODS = {"chocolate", "cake", "cookies", "vanilla", "strawberry"}  # only used by the no-LLM fallback
ALIASES = {"cookie": "cookies", "cakes": "cake"}  # so singular/plural match what the dessert rules check
GENERIC_WORDS = {"dessert", "desserts", "food", "foods", "snack", "snacks", "meal", "meals"}  # categories, not foods
EXTRACT_RULES = (
    "You extract food preferences from a user's message.\n"
    "Return only foods or desserts the user says they like, in lowercase.\n"
    "Use the base food name: 'dark chocolate' becomes 'chocolate', 'a chocolate cake' becomes 'cake'.\n"
    "If the user does not name a food they like, return an empty list."
)


class MemoryState(TypedDict):
    messages: List[str]  # the conversation so far
    user_id: str  # who is talking
    user_memory: DictType[str, AnyType]  # things we remember about this user, e.g. food_preferences


class FoodPreferences(BaseModel):  # the shape we ask the LLM to answer in
    foods: List[str] = Field(description="Foods or desserts the user says they like; empty list if none")


# reasoning=False skips qwen3's long hidden thinking; the JSON schema constrains the output anyway
food_llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0, num_ctx=8192, reasoning=False)
extractor = food_llm.with_structured_output(FoodPreferences)


def normalize_food(food: str) -> str:
    # The model sometimes returns "dark chocolate" despite the rule; the last word is the main noun, so keep that
    words = food.strip().lower().split()
    head = words[-1] if words else ""
    if head in GENERIC_WORDS:  # the model sometimes lists "desserts" as if it were a food
        return ""
    return ALIASES.get(head, head)


# No-LLM fallback: pick out any known food word from the message
def scan_for_known_foods(message: str) -> List[str]:
    words = [ALIASES.get(word, word) for word in re.findall(r"[a-z]+", message.lower())]  # lowercase words only, no punctuation
    return [word for word in words if word in KNOWN_FOODS]


# The one place the LLM is used: ask the model which foods the user likes, then clean up its answer
def extract_foods(message: str) -> List[str]:
    try:
        result = extractor.invoke([SystemMessage(content=EXTRACT_RULES), HumanMessage(content=message)])
    except Exception:  # Ollama down or unusable output: fall back to scanning for known foods
        return scan_for_known_foods(message)
    text = message.lower()
    foods = [normalize_food(food) for food in result.foods]
    return [food for food in foods if food and food.rstrip("s") in text]  # drop anything the user never said


# Node 1: find the foods in the newest message (LLM) and add them to long-lived memory
def remember_user_preferences(state: MemoryState) -> MemoryState:
    preferences = state["user_memory"].get("food_preferences", [])  # empty list if nothing is remembered yet
    for food in extract_foods(state["messages"][-1]):
        if food not in preferences:  # never store the same food twice
            preferences.append(food)
    state["user_memory"]["food_preferences"] = preferences  # write the updated list back into memory
    return state


# Node 2: plain rules (no LLM) that pick a dessert from what is remembered
def suggest_a_dessert(state: MemoryState) -> MemoryState:
    preferences = state["user_memory"].get("food_preferences", [])

    if "chocolate" in preferences and "cookies" in preferences:
        state["messages"].append("How about chocolate cookies for dessert?")
    elif "chocolate" in preferences and "cake" in preferences:
        state["messages"].append("How about a chocolate cake for dessert?")
    elif "chocolate" in preferences:
        state["messages"].append("How about a chocolate brownie for dessert?")
    else:
        state["messages"].append("How about some ice cream for dessert?")
    return state


def demo_llm_memory():
    print("=== LLM food-preference memory ===\n")
    workflow = StateGraph(MemoryState)
    workflow.add_node("remember", remember_user_preferences)
    workflow.add_node("suggest", suggest_a_dessert)
    workflow.add_edge(START, "remember")  # always remember first...
    workflow.add_edge("remember", "suggest")  # ...then suggest using what was just remembered
    workflow.add_edge("suggest", END)
    app = workflow.compile(checkpointer=InMemorySaver())  # the checkpointer is what keeps memory between messages

    # Helper: send one message for a user, restoring their saved state first
    def say(user_id: str, text: str):
        config = {"configurable": {"thread_id": f"food_memory_{user_id}"}}  # one thread, so one memory, per user
        saved = app.get_state(config).values  # empty on that user's first message
        state = {"messages": list(saved.get("messages", [])) + [f"{user_id}: {text}"],
                 "user_id": user_id, "user_memory": saved.get("user_memory", {})}
        result = app.invoke(state, config=config)
        print(f"{user_id}: {text}")
        print(f"   AI: {result['messages'][-1]}   | remembered: {result['user_memory']['food_preferences']}")

    say("alice", "Dark chocolate is my absolute favorite!")  # "dark chocolate" should be stored as "chocolate"
    say("alice", "Honestly, I'd love some cake too.")
    say("alice", "I can't resist a good tiramisu.")  # a food the no-LLM fallback has never heard of
    say("alice", "Cookies are great, and I like vanilla.")
    say("bob", "I'm not picky about desserts.")  # should store nothing


if __name__ == "__main__":
    demo_llm_memory()
