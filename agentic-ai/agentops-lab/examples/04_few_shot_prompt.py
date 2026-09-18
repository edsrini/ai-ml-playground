"""Few-shot prompting -- teach a response format (Question/Thought/Response) via worked examples."""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama

from config import OLLAMA_BASE_URL, OLLAMA_MODEL

examples = [
    {
        "input": "What is the capital of France?",
        "thought": "This is a simple geography fact.",
        "output": "Paris",
    },
    {
        "input": "Is earth flat?",
        "thought": "This is a settled scientific fact, not a matter of opinion.",
        "output": "No, the earth is roughly a sphere.",
    },
]

# Formats a single example dict into "Question: ...\nThought: ...\nResponse: ..."
example_prompt = PromptTemplate(
    input_variables=["input", "thought", "output"],
    template="Question: {input}\nThought: {thought}\nResponse: {output}",
)

# Stitches multiple formatted examples together, then appends the real question
# using the same input/thought/output shape the examples demonstrated.
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {input}\nThought:",
    input_variables=["input"],
)


def main():
    filled = few_shot_prompt.format(input="Is the moon made of cheese?")
    print("--- Prompt sent to the model ---")
    print(filled)

    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    response = llm.invoke(filled)

    print("\n--- Model response ---")
    print(response.content)


if __name__ == "__main__":
    main()
