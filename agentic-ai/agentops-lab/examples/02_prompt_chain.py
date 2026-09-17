"""A prompt template + parser chain -- the LangChain fundamentals building block."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from config import OLLAMA_BASE_URL, OLLAMA_MODEL


def main():
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.3)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a terse release-notes writer. Max 2 sentences."),
            ("human", "Summarize this change for a changelog: {change}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    result = chain.invoke(
        {"change": "Added drift detection to the data validation step of the MLflow pipeline."}
    )
    print(result)


if __name__ == "__main__":
    main()
