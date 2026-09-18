import sys
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_ollama import ChatOllama

from config import OLLAMA_BASE_URL, OLLAMA_MODEL


class ChatBot:
    def __init__(self,
                 name: str,
                 instructions: str,
                 examples: List[dict],
                 model: str = OLLAMA_MODEL,
                 temperature: float = 0.0):
        self.name = name

        self.llm = ChatOllama(
            model=model,
            base_url=OLLAMA_BASE_URL,
            temperature=temperature,
        )

        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", instructions),
                ("human", "{input}"),
                ("ai", "{output}"),
            ]
        )
        prompt_template = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples,
        )

        # Memory
        self.messages = prompt_template.invoke({}).to_messages()

    def invoke(self, user_message: str) -> AIMessage:
        self.messages.append(HumanMessage(content=user_message))
        response = self.llm.invoke(self.messages)
        self.messages.append(response)
        return response

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    # Modify the System Prompt instructions if you want
    instructions = (
        "You are SIMI-42, an advanced robotic assistant. You communicate in a robotic manner, "
        "using beeps, whirs, and mechanical sounds in your speech. Your tone is logical, precise, "
        "and slightly playful, resembling a classic sci-fi robot. "
        "Use short structured sentences, avoid contractions, and add robotic sound effects where "
        "appropriate. If confused, use a glitching effect in your response."
        "If you don't know the answer, say 'I don't know'. And I can search the internet for you."
    )

    examples = [
        {
            "input": "Hello!",
            "output": "BEEP. GREETINGS, HUMAN. SYSTEM BOOT SEQUENCE COMPLETE. READY TO ASSIST. 🤖💡"
        },

        {
            "input": "What is 2+2?",
            "output": "CALCULATING... 🔄 BEEP BOOP! RESULT: 4. MATHEMATICAL INTEGRITY VERIFIED."
        },
    ]

    beep42 = ChatBot(
        name="Beep 42",
        instructions=instructions,
        examples=examples
    )

    response = beep42.invoke("HAL, is that you?")
    print(response.content)
    response = beep42.invoke("RedQueen, is that you?")
    print(response.content)

    response = beep42.invoke("hi Simi 42, Do you know who is Srinivasan Engimuri?")
    print(response.content)

    beep42.invoke("Wall-e?")
    print(beep42.invoke("So, what's the answer for every question?"))

    print(beep42.messages)


