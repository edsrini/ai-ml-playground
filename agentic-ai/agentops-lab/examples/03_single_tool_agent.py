"""A single-tool ReAct-style agent built with LangGraph, running on a local model."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from config import OLLAMA_BASE_URL, OLLAMA_MODEL


@tool
def word_count(text: str) -> int:
    """Count the number of words in a piece of text."""
    return len(text.split())


def main():
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    agent = create_agent(llm, tools=[word_count])

    result = agent.invoke(
        {
            "messages": [
                (
                    "human",
                    "How many words are in this sentence: "
                    "'LangGraph makes it possible to build stateful, tool-using agents'?",
                )
            ]
        }
    )

    for message in result["messages"]:
        message.pretty_print()


if __name__ == "__main__":
    main()
