"""Structured output -- force the model to return a typed object instead of free text.

This directly addresses the format-adherence failures from the few-shot experiments
(examples/04_few_shot_prompt.py): instead of asking the model to follow a text pattern
like "Thought: ...\\nResponse: ..." and hoping it complies, we define a schema and let
the model-provider's structured-output machinery enforce it.
"""
import sys
from pathlib import Path
from typing import Annotated, List, TypedDict

sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_classic.output_parsers import BooleanOutputParser, DatetimeOutputParser, OutputFixingParser
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from config import OLLAMA_BASE_URL, OLLAMA_MODEL

INTRO = "My name is Srinivasan and I am 42 years old. I am from India but live in the US."


# TypedDict schema: the model returns a plain dict and values are not validated.
class UserInfo(TypedDict):
    name: Annotated[str, "The user's name. Defaults to 'Anonymous'."]
    age: Annotated[int, "The user's age. Defaults to 0."]
    country: Annotated[str, "The user's country. Defaults to 'Unknown'."]


# Pydantic schema: the model returns an object with validated, typed fields.
class PydanticUserInfo(BaseModel):
    name: str = Field(default="Anonymous", description="The user's name. Defaults to 'Anonymous'.")
    age: int = Field(default=0, description="The user's age. Defaults to 0.")
    country: str = Field(default="Unknown", description="The user's country. Defaults to 'Unknown'.")


class Performer(BaseModel):
    name: str = Field(default="Anonymous", description="The actor/actress's name. Defaults to 'Anonymous'.")
    age: int = Field(default=0, description="The performer's age. Defaults to 0.")
    films: List[str] = Field(default_factory=list, description="List of films they starred in.")


def plain_text(llm):
    # StrOutputParser just returns the message text.
    print(StrOutputParser().invoke(llm.invoke("Hi Naughty!")))


def parse_date(llm):
    # The parser turns the model's text into a datetime; the format must match the prompt.
    parser = DatetimeOutputParser(format="%Y-%m-%d")
    print(parser.invoke(llm.invoke(
        "Output a random date in the format %Y-%m-%d, e.g. 2025-06-15. "
        "Don't say anything else."
    )))


def parse_boolean(llm):
    # Maps a YES/NO reply to True/False.
    print(BooleanOutputParser().invoke(llm.invoke("Are you a robot? Yes or no only.")))


def typed_dict_output(llm):
    structured = llm.with_structured_output(UserInfo)
    print(structured.invoke("What is your name?"))
    print(structured.invoke(INTRO))


def pydantic_output(llm):
    structured = llm.with_structured_output(PydanticUserInfo)
    print(structured.invoke("What is your name?"))
    print(structured.invoke(INTRO))


def performer_output(llm):
    # model_dump_json() serializes the Pydantic object to a JSON string.
    structured = llm.with_structured_output(Performer)
    print(structured.invoke("Power star films").model_dump_json())


def output_fixing(llm):
    # OutputFixingParser: when the wrapped parser fails, the LLM is asked to repair the text.
    bad_output = "15 June 2025"  # not %Y-%m-%d, so the plain parser rejects it
    date_parser = DatetimeOutputParser(format="%Y-%m-%d")
    try:
        date_parser.parse(bad_output)
    except Exception as e:
        print(f"Plain parser failed: {type(e).__name__}")

    fixing_parser = OutputFixingParser.from_llm(parser=date_parser, llm=llm)
    print(fixing_parser.parse(bad_output))


def main():
    # Default reasoning keeps the thinking trace out of `content` for plain calls.
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

    plain_text(llm)
    parse_date(llm)
    parse_boolean(llm)
    typed_dict_output(llm)
    pydantic_output(llm)
    performer_output(llm)
    output_fixing(llm)


if __name__ == "__main__":
    main()
