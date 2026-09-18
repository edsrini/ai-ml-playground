"""Structured output -- force the model to return a typed object instead of free text.

This directly addresses the format-adherence failures from the few-shot experiments
(examples/04_few_shot_prompt.py): instead of asking the model to follow a text pattern
like "Thought: ...\\nResponse: ..." and hoping it complies, we define a schema and let
the model-provider's structured-output machinery enforce it.
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from config import OLLAMA_BASE_URL, OLLAMA_MODEL


class DateValidation(BaseModel):
    """Result of validating a date-like field on an insurance claim."""

    is_valid: bool = Field(description="Whether the date is a real, valid calendar date")
    reason: str = Field(description="One short sentence explaining the verdict")


def main():
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    structured_llm = llm.with_structured_output(DateValidation)

    for candidate in ["38/32/2025", "15/06/2025", "8/32/2025"]:
        result = structured_llm.invoke(
            f"Validate this date field from a claim form: {candidate}"
        )
        print(f"{candidate!r:>14} -> valid={str(result.is_valid):<5} reason={result.reason}")


if __name__ == "__main__":
    main()
