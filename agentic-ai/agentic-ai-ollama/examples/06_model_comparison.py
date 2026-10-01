"""Compare local models on the same task: is this DD/MM/YYYY claim date a real date?

Ground truth comes from datetime.strptime, so the test set needs no hand labeling and
the deterministic check doubles as the baseline the LLMs are measured against.
"""
import sys
import time
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from config import OLLAMA_BASE_URL

# Any model from `ollama list` works here.
MODELS = ["qwen3:4b", "llama3.2:1b"]

CASES = [
    "15/06/2025", "01/01/2024", "31/12/2023", "29/02/2024",
    "38/32/2025", "8/32/2025", "31/04/2025", "29/02/2025",
    "00/05/2025", "10/13/2025", "15/06/25", "2025/06/15",
]


class DateValidation(BaseModel):
    reason: str = Field(description="One short sentence: check day, month and year ranges, then conclude")
    is_valid: bool = Field(description="Final verdict, consistent with the reason")


def ground_truth(value: str) -> bool:
    try:
        datetime.strptime(value, "%d/%m/%Y")
        return True
    except ValueError:
        return False


def evaluate(model: str) -> dict:
    llm = ChatOllama(model=model, base_url=OLLAMA_BASE_URL, temperature=0)
    structured = llm.with_structured_output(DateValidation)

    correct, errors, elapsed = 0, 0, 0.0
    for case in CASES:
        start = time.time()
        try:
            result = structured.invoke(f"Validate this date field (DD/MM/YYYY) from a claim form: {case}")
            correct += result.is_valid == ground_truth(case)
        except Exception:
            errors += 1
        elapsed += time.time() - start

    return {"model": model, "correct": correct, "errors": errors, "avg_s": elapsed / len(CASES)}


def main():
    start = time.time()
    for case in CASES:
        ground_truth(case)
    baseline_s = (time.time() - start) / len(CASES)
    # Correct by definition: it is the reference the models are scored against.
    rows = [{"model": "strptime (baseline)", "correct": len(CASES), "errors": 0, "avg_s": baseline_s}]

    for model in MODELS:
        print(f"Evaluating {model} ...")
        rows.append(evaluate(model))

    print(f"\n{'model':<22}{'accuracy':>10}{'errors':>8}{'avg sec/call':>14}")
    for r in rows:
        print(f"{r['model']:<22}{r['correct']:>4}/{len(CASES):<5}{r['errors']:>8}{r['avg_s']:>14.2f}")


if __name__ == "__main__":
    main()
