"""The same text-to-SQL agent as agent.py, written as a plain loop instead of a LangGraph graph.
It prints every step so you can watch how the model and the tools take turns.

Usage:  python agent_loop.py "Which model earned the most revenue?"
"""
import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama

from database import get_engine
from tools import TOOLS

load_dotenv()  # reads .env into environment variables

SYSTEM_PROMPT = (
    "You are a Sr. SQL developer. First list the tables, then get the schema of the table you need, then write "
    "ONE SELECT query and run it with execute_sql. Before filtering on a text column, run SELECT DISTINCT on it to see "
    "the exact values. If a tool returns an error, fix the query and retry. "
    "Never guess or invent a result: answer only from the rows returned. Always answer in English."
)

llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL"),
    base_url=os.getenv("OLLAMA_BASE_URL"),
    temperature=0.0,
    num_ctx=8192,  # keeps the model on the GPU
)
llm_with_tools = llm.bind_tools(TOOLS)  # the model can now ask for these tools by name
tools_by_name = {t.name: t for t in TOOLS}  # so we can find a tool from the name the model gives


def ask(question, max_steps=10):
    # The database connection travels in config, so the model never sees it
    config = {"configurable": {"db_engine": get_engine()}}
    messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(question)]  # the whole conversation, grown each step

    for step in range(1, max_steps + 1):  # step cap: stops a model that keeps asking for tools
        reply = llm_with_tools.invoke(messages)  # the model reads everything so far
        messages.append(reply)

        if not reply.tool_calls:  # no tool requested, so this is the final answer
            print(f"\nStep {step}: the model answered without a tool:\n{reply.content}")
            return reply.content

        for call in reply.tool_calls:  # the model may ask for several tools in one reply
            print(f"\nStep {step}: the model asked for {call['name']}({call['args']})")
            tool = tools_by_name.get(call["name"])
            result = tool.invoke(call["args"], config=config) if tool else {"error": f"no tool named {call['name']}"}
            print(f"          result: {str(result)[:200]}")
            # tool_call_id links this result to the request it answers
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    print(f"\nStopped after {max_steps} steps without a final answer.")


if __name__ == "__main__":
    ask(" ".join(sys.argv[1:]) or "Which model earned the most revenue?")
