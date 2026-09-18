"""A prompt template + parser chain -- the LangChain fundamentals building block."""
import sys
from pathlib import Path

import output
import thought
from langchain_core.prompts import prompt, PromptTemplate, FewShotPromptTemplate
from langgraph.channels import topic

sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from config import OLLAMA_BASE_URL, OLLAMA_MODEL


def main():
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.3)
    result = llm.invoke("hello there!")
    print(result.content)

    messages = [
        SystemMessage(
            content=(
                "You are a Medical Insurance TPA adjudicator. "
                "Answer in 1 sentences. "
                "If you lack enough context to judge validity, say so explicitly rather than guessing."
            )
        ),
        HumanMessage(content="Is this data valid: 38/32/2025"),
        AIMessage(content="This data (`38/32/2025`) is invalid."),
        HumanMessage(content="Is this data valid: 8/32/2025"),
    ]
    #response = llm.invoke(messages)
    #print(response.content)

    topic = "Science"
    prompt = f"Talk about {topic}."
    #response = llm.invoke(prompt)
    #print(response.content)

    prompt_template = PromptTemplate(
        input_variables=["topic"],
        template="Talk about {topic}."
    )

    filled = prompt_template.format(topic="Astronomy")
    #response = llm.invoke(filled)
    #print(response.content)

    example_prompt = PromptTemplate(
        template="Question: {input}\nThought: {thought}\nResponse:{output}"
    )

    examples = [
        {
            "input": "What is the capital of France?",
            "thought": "I think it's Paris",
            "output": "Paris"
        },
        {
            "input": "Is earth flat?",
            "thought": "I think it's flat",
            "output": "Shpere"
        }

    ]
    print(example_prompt.invoke(examples[0]).to_string())

    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        suffix="Question: {input}",
        input_variables=["input"],
    )

    response = llm.invoke(few_shot_prompt.format(input="Is the moon made of cheese?"))
    print(response.content)

if __name__ == "__main__":
    main()
