"""Learning demo: the blog pipeline written with LCEL (the | operator) instead of a LangGraph StateGraph.

Same five stages as agentic_ai_blog_post_ext.py:
    intake -> research -> writer -> reviewer -> publisher

Differences, on purpose, to keep it easy to read:
    - No tools and no agents. Each stage is just  prompt | llm | parser  (a "chain").
    - "Research" here is the model's own notes, not a web search (a web search needs a tool-using agent).
    - State is a plain dict that grows as each stage adds one key (RunnablePassthrough.assign).
"""
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.0, num_ctx=8192)


# ============================================================
# PART 1: one chain = prompt | llm | parser
# ============================================================
# The | operator feeds the output of the left side into the right side:
#   prompt  : dict  -> list of chat messages
#   llm     : messages -> AIMessage
#   parser  : AIMessage -> plain str
def make_chain(system_text: str, human_template: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_text),
        ("human", human_template),
    ])
    return prompt | llm | StrOutputParser()


intake_chain = make_chain(
    "You are the Intake agent. Turn the request into a short plan: a title, the audience and 3 section headings.",
    "Request: {query}",
)

research_chain = make_chain(
    "You are the Researcher agent. Write 5 short bullet points of facts for this blog plan.",
    "Request: {query}\n\nPlan:\n{plan}",
)

writer_chain = make_chain(
    "You are the Writer agent. Write the blog post in Markdown: one '# ' title, then one '## ' heading per section.",
    "Plan:\n{plan}\n\nNotes:\n{research}",
)

reviewer_chain = make_chain(
    "You are the Reviewer agent. Fix clarity and grammar and return the full improved Markdown post only.",
    "Draft:\n{draft}",
)

publisher_chain = make_chain(
    "You are the Publisher agent. Reply with one line: 'Published: <title>'.",
    "Final post:\n{final}",
)


# ============================================================
# PART 2: the pipeline = chains joined with |
# ============================================================
# RunnablePassthrough.assign(name=chain) means:
#   "keep every key already in the dict, run `chain` on it, and add its result under `name`".
# That is how the shared state grows from stage to stage:
#   {query}  ->  {query, plan}  ->  {query, plan, research}  ->  ...
def tap(stage: str):
    """Prints which stage just finished, then passes the dict along unchanged (for learning only)."""
    def _print(state: dict) -> dict:
        print(f"[{stage}] done -> keys now: {list(state)}")
        return state
    return RunnableLambda(_print)


pipeline = (
    RunnablePassthrough.assign(plan=intake_chain) | tap("intake")
    | RunnablePassthrough.assign(research=research_chain) | tap("research")
    | RunnablePassthrough.assign(draft=writer_chain) | tap("writer")
    | RunnablePassthrough.assign(final=reviewer_chain) | tap("reviewer")
    | RunnablePassthrough.assign(status=publisher_chain) | tap("publisher")
)


# ============================================================
# PART 3: how an agent (with tools) would fit in the same pipeline
# ============================================================
# An agent from create_react_agent is also a runnable, but it wants {"messages": [...]} and returns
# {"messages": [...]}. A small adapter converts between the pipeline dict and the agent:
#
#   from langchain_core.messages import HumanMessage
#
#   def agent_stage(agent):
#       def _run(state: dict) -> str:
#           reply = agent.invoke({"messages": [HumanMessage(content=f"{state['query']}\n\n{state['plan']}")]})
#           return reply["messages"][-1].content
#       return RunnableLambda(_run)
#
#   RunnablePassthrough.assign(research=agent_stage(researcher_agent))
#
# This is the same job the nodes in agentic_ai_blog_post_ext.py do: build the context from state,
# call the agent, return the result for the state.


# ============================================================
# RUN
# ============================================================
result = pipeline.invoke({"query": "I want a blog post about AI Agents"})

print("\n=== STATE KEYS ===")
print(list(result))
print("\n=== FINAL MARKDOWN ===")
print(result["final"])
print("\n=== STATUS ===")
print(result["status"])
