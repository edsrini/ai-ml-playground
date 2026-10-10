"""Thread isolation -- each thread_id has its own saved state in the checkpointer, no LLM involved."""
import copy
from datetime import datetime
from typing import Any as AnyType, Dict as DictType, List, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


def demo_thread_isolation():
    print("=== Demo 3: Thread Isolation ===\n")
    memory = InMemorySaver()

    # A state that keeps track of how many times the user has visited support.
    class SupportState(TypedDict):
        messages: List[str]
        user_name: str
        visit: int

    def handle_support(state: SupportState) -> SupportState:
        # Increment the visit count stored in state and append a personalized message
        state['visit'] += 1
        state['messages'].append(f"Support visit {state['visit']} for {state['user_name']}")
        return state

    # Build the workflow graph using the SupportState type
    workflow = StateGraph(SupportState)
    workflow.add_node('support', handle_support)
    workflow.add_edge(START, 'support')
    workflow.add_edge('support', END)
    app = workflow.compile(checkpointer=memory)

    # Simulate the first visit for Alice in her own thread
    config_user1 = {"configurable": {"thread_id": 'alice_support'}}
    state_user1 = {"messages": [], "user_name": 'Alice', "visit": 0}
    result1 = app.invoke(state_user1, config=config_user1)
    print("User 'Alice' first visit:", result1['messages'])

    # Bob uses a different thread_id, so he starts from his own fresh state (visit 0 -> 1)
    config_user2 = {"configurable": {"thread_id": 'bob_support'}}
    state_user2 = {"messages": [], "user_name": 'Bob', "visit": 0}
    result2 = app.invoke(state_user2, config=config_user2)
    print("User 'Bob' first visit:  ", result2['messages'])

    # Simulate a second visit for Alice; because we use the same thread_id,
    # her previous state (visit count) is persisted
    saved_alice_state = app.get_state(config_user1).values
    result3 = app.invoke(saved_alice_state, config=config_user1)
    print("User 'Alice' second visit:", result3['messages'])

    # Show that Bob's visit count remains isolated
    saved_bob_state = app.get_state(config_user2).values
    result4 = app.invoke(saved_bob_state, config=config_user2)
    print("User 'Bob' second visit:", result4['messages'])


# Demo 4: Config vs State
class DetailedState(TypedDict):
    messages: List[str]
    current_issue: str
    resolved: bool
    timestamp: str


def process_with_config(state: DetailedState, config: RunnableConfig) -> DetailedState:
    # config carries per-run settings (read-only); state carries the data that flows through the graph
    settings = config.get("configurable", {})
    if settings.get("priority") == "high":
        state["messages"].append("HIGH PRIORITY!!!: Escalating immediately!")
    else:
        state["messages"].append("Standard processing...")
    if settings.get("language") == "es":
        state["messages"].append("Hola! ¿Cómo puedo ayudarte?")
    else:
        state["messages"].append("Hello! How can I help you?")
    state["timestamp"] = datetime.now().isoformat()
    return state


def demo_config_vs_state():
    print("\n=== Demo 4: Config vs State ===\n")
    workflow = StateGraph(DetailedState)
    workflow.add_node("process", process_with_config)
    workflow.add_edge(START, "process")
    workflow.add_edge("process", END)
    app = workflow.compile(checkpointer=InMemorySaver())
    initial_state = {"messages": ["Customer needs help"], "current_issue": "Login problem", "resolved": False, "timestamp": ""}

    config1 = {"configurable": {"thread_id": "test_1", "priority": "normal", "language": "en"}}
    # deepcopy, not .copy(): the node appends to messages in place, so a shallow copy would share that list across runs
    result1 = app.invoke(copy.deepcopy(initial_state), config=config1)
    print("Normal/English config messages:")
    for msg in result1["messages"][1:]:
        print(" -", msg)

    config2 = {"configurable": {"thread_id": "test_2", "priority": "high", "language": "es"}}
    result2 = app.invoke(copy.copy(initial_state), config=config2)
    print("High Priority/Spanish config messages:")
    for msg in result2["messages"][1:]:
        print(" -", msg)


# Demo 5: Time Travel with Checkpoints
class ConversationState(TypedDict):
    messages: List[str]
    conversation_topic: str
    mood: str


def demo_time_travel():
    print("\n=== Demo 5: Time Travel with Checkpoints ===\n")
    memory = InMemorySaver()

    def conversation_node(state: ConversationState) -> ConversationState:
        """Simulates a conversation response based on topic and mood"""
        topic = state["messages"][-1].split(" ")[-1]  # the last word of the latest message
        mood = state["mood"]

        if topic == "weather":
            if mood == "happy":
                response = "What a beautiful sunny day! Perfect for a picnic!"
            else:
                response = "Looks like it might rain today. Better stay inside."
        else:
            response = f"Let's talk about {topic}. I'm feeling {mood} today."

        state["messages"].append(f"AI: {response}")
        state["conversation_topic"] = topic
        return state

    workflow = StateGraph(ConversationState)
    workflow.add_node("conversation", conversation_node)
    workflow.add_edge(START, "conversation")
    workflow.add_edge("conversation", END)
    app = workflow.compile(checkpointer=memory)
    # Initial conversation
    config = {"configurable": {"thread_id": "time_travel_conversation"}}
    initial_state = {
        "messages": ["Human: Let's talk about the weather"],
        "mood": "happy"
    }

    print("=== Original Timeline ===")
    result1 = app.invoke(initial_state, config=config)
    print(f"Topic: {result1['conversation_topic']}, Mood: {result1['mood']}")
    print(f"Response: {result1['messages'][-1]}")

    print(f"\nFull conversation: {result1['messages']}")

    # Time travel: Go back to the first checkpoint and change the mood
    print("\n=== Time Travel: Going Back and Changing Mood ===")
    history = list(app.get_state_history(config))

    # The states are returned in reverse chronological order.
    for state in history:
        print(state.next)
        print(state.config["configurable"]["checkpoint_id"])
        print()

    # The checkpoint where the conversation node is still pending is the moment just before the reply
    before_reply = next(s for s in history if s.next == ("conversation",))

    # Fork from there with a different mood, then run forward again from that checkpoint
    fork_config = app.update_state(before_reply.config, {"mood": "sad"}, as_node="__start__")
    forked = app.invoke(None, config=fork_config)
    print("=== New Timeline (mood changed to 'sad') ===")
    print(f"Topic: {forked['conversation_topic']}, Mood: {forked['mood']}")
    print(f"Response: {forked['messages'][-1]}")
    print(f"\nHistory now has {len(list(app.get_state_history(config)))} checkpoints; the original timeline is still there.")


# Demo 6: Practical memory features
class MemoryState(TypedDict):
    messages: List[str]  # the conversation so far
    user_id: str  # who is talking
    user_memory: DictType[str, AnyType]  # things we remember about this user, e.g. food_preferences


# Node 1: save what the user just said into long-lived memory
def remember_user_preferences(state: MemoryState) -> MemoryState:
    # The last word of the latest user message is treated as the new preference
    new_preference = state["messages"][-1].split(" ")[-1]

    preferences = state["user_memory"].get("food_preferences", [])  # empty list if nothing is remembered yet
    preferences.append(new_preference)
    state["user_memory"]["food_preferences"] = preferences  # write the updated list back into memory
    return state


# Node 2: use everything remembered so far to pick a dessert
def suggest_a_dessert(state: MemoryState) -> MemoryState:
    preferences = state["user_memory"].get("food_preferences", [])

    if "chocolate" in preferences:
        if "cookies" in preferences:
            state["messages"].append("How about a chocolate cookies for dessert?")
            return state
        elif "cake" in preferences:
            state["messages"].append("How about a chocolate cake for dessert?")
            return state
        state["messages"].append("How about a chocolate brownie for dessert?")  # chocolate, nothing more specific yet
        return state

    state["messages"].append("How about some ice cream for dessert?")  # nothing known that narrows it down
    return state


def demo_practical_memory():
    print("\n=== Demo 6: Practical memory features ===\n")
    workflow = StateGraph(MemoryState)
    workflow.add_node("remember", remember_user_preferences)
    workflow.add_node("suggest", suggest_a_dessert)
    workflow.add_edge(START, "remember")  # always remember first...
    workflow.add_edge("remember", "suggest")  # ...then suggest using what was just remembered
    workflow.add_edge("suggest", END)
    app = workflow.compile(checkpointer=InMemorySaver())  # the checkpointer is what keeps memory between messages

    # Helper: send one message for a user, restoring their saved state first
    def say(user_id: str, text: str):
        config = {"configurable": {"thread_id": f"memory_{user_id}"}}  # one thread, so one memory, per user
        saved = app.get_state(config).values  # empty on that user's first message
        state = {"messages": list(saved.get("messages", [])) + [f"{user_id}: {text}"],
                 "user_id": user_id, "user_memory": saved.get("user_memory", {})}
        result = app.invoke(state, config=config)
        print(f"{user_id}: {text}")
        print(f"   AI: {result['messages'][-1]}   | remembered: {result['user_memory']['food_preferences']}")

    say("alice", "I like chocolate")    # first preference -> a chocolate suggestion
    say("alice", "I also like cake")    # memory now holds chocolate + cake
    say("alice", "Add cookies")         # memory now holds chocolate + cake + cookies
    say("bob", "I like vanilla")        # Bob has his own thread, so none of Alice's memory applies


if __name__ == "__main__":
    #demo_thread_isolation()
    #demo_config_vs_state()
    demo_time_travel()
    demo_practical_memory()
