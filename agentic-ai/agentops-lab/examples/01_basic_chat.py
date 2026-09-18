"""Minimal call to a local open-weight model through LangChain."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_ollama import ChatOllama

from config import OLLAMA_BASE_URL, OLLAMA_MODEL


def main():
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    response = llm.invoke("In one sentence, what is the capital of France?")
    print(response.content)


if __name__ == "__main__":
    main()
