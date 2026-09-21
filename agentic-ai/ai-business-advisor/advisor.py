"""AI Business Advisor -- a multi-step LCEL workflow: industry -> idea -> analysis -> report."""
import sys

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.tracers.context import collect_runs
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from config import OLLAMA_BASE_URL, OLLAMA_MODEL

from typing import List, Annotated

sys.stdout.reconfigure(encoding="utf-8")

llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.7)
parser = StrOutputParser()


logs = []  # every raw model message is appended here, one per LLM call


def get_logs(message):
    logs.append(message)
    return {"model": message.response_metadata.get("model"), "tokens": message.usage_metadata}


# Two branches read the same model message: one extracts the text, one collects the logs.
# The result is {"output": ..., "logs": ...}.
parse_and_log_output_chain = RunnableParallel(
    output=StrOutputParser(),
    logs=RunnableLambda(get_logs),
)

# Step 1: industry -> business idea
idea_prompt = ChatPromptTemplate.from_template(
    "You are a creative business advisor. "
    "Generate one innovative business idea in the {industry} industry. "
    "Describe it in 2-3 sentences and give it a short name."
)

idea_chain = (idea_prompt | llm | parse_and_log_output_chain)

idea_result = idea_chain.invoke({"industry": "Service"})

#print(idea_result["output"])
#print(idea_result["logs"])

# Step 2: idea -> strengths and weaknesses
analysis_prompt = ChatPromptTemplate.from_template(
    "Analyze this business idea: {idea}\n\n"
    "List exactly 3 strengths and 3 weaknesses as short bullet points."
)

analysis_chain = (analysis_prompt | llm | parse_and_log_output_chain)
analysis_result = analysis_chain.invoke({"idea": idea_result["output"]})
#print(analysis_result["output"])
#print(analysis_result["logs"])

# Step 3: idea + analysis -> formatted report
report_prompt = PromptTemplate(
    template=(
        "Here is a business idea and its analysis.\n\n"
        "Strengths & Weaknesses: {output}\n\n"
        "Generate a structured report."
    )
)

class AnalysisResult(BaseModel):
    strengths: List[str] = Field(default=[], description="List of strengths")
    weaknesses: List[str] = Field(default=[], description="List of weaknesses")

report_chain = (report_prompt | llm.with_structured_output(AnalysisResult))

report_result = report_chain.invoke(analysis_result["output"])

#print(report_result.model_dump_json(indent=2))

# assign() keeps the existing keys and adds each step's result, so later prompts can use them.
e2e_chain = (
    RunnablePassthrough()
    | idea_chain
    | RunnableParallel(idea=RunnablePassthrough())
    | analysis_chain
    | report_chain
)
e2e_chain.get_graph().print_ascii()
print(e2e_chain.invoke({"industry": "Child Care"}))

print(logs)


def main():
    industry = sys.argv[1] if len(sys.argv) > 1 else "Healthcare"
    #print(f"Industry: {industry}\n")
    #print(workflow.invoke({"industry": industry}))


if __name__ == "__main__":
    main()
