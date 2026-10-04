"""AI Library Assistant -- a tool-calling agent (local Ollama model) with per-member conversation memory."""
import sys

from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

from config import OLLAMA_BASE_URL, OLLAMA_MODEL
from library import TOOLS

sys.stdout.reconfigure(encoding="utf-8")  # model replies may contain emoji

SYSTEM_PROMPT = (
    "You are a friendly library assistant. Use the tools to search the catalog and to check books "
    "out or in -- never guess availability. Ask for the member's name before checking a book out."
)

llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0, num_ctx=8192)

# The model decides when to call a tool; the checkpointer lets follow-ups like "is it available?" work.
agent = create_agent(llm, tools=TOOLS, system_prompt=SYSTEM_PROMPT, checkpointer=InMemorySaver())


def chat(message: str, member_id: str) -> str:
    config = {"configurable": {"thread_id": member_id}}  # one conversation thread per member
    result = agent.invoke({"messages": [("human", message)]}, config=config)
    return result["messages"][-1].content


if __name__ == "__main__":
    member = "member-1"
    for question in [
        "Do you have anything by Frank Herbert?",
        "Please check it out for Asha.",  # "it" only resolves because the thread remembers the last turn
        "Is it still available?",
        "What about Neuromancer?",
    ]:
        print(f"\nYou: {question}")
        print(f"Assistant: {chat(question, member)}")
