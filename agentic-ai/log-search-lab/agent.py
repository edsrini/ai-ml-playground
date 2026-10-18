"""Log-search agent: a local model searches the logs, finds what went wrong and explains it.

Same shape as the text2sql-agent graph: build the prompt, let the model call tools, run the tools, repeat.

Usage:  python agent.py "What caused the errors on 8 October?"
"""
import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.errors import GraphRecursionError
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode

from database import get_client
from message import MessagesState  # our own copy of langgraph's MessagesState (see message.py)
from tools import get_log_fields, log_stats, search_logs

# Instantiate your chat model
load_dotenv()
llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL"),
    base_url=os.getenv("OLLAMA_BASE_URL"),
    temperature=0.0,
    num_ctx=8192,  # keeps the model on the GPU
)


class State(MessagesState):
    user_query: str


log_tools = [get_log_fields, log_stats, search_logs]
log_llm = llm.bind_tools(log_tools)  # shows the model each tool's name, description and arguments

workflow = StateGraph(State)
workflow.add_node("log_tools", ToolNode(log_tools))  # runs whichever tool the model asked for


# Node: build the prompt for the model from the user's question
def message_builder(state: State):
    log_sys_msg = (
        "You are a site reliability engineer investigating a problem by searching application logs.\n"
        "The logs cover 2026-10-01 to 2026-10-15 (UTC). Always give start and end as ISO timestamps ending in +00:00.\n"
        "First, use log_stats to find which service and error code are behind the problem. "
        "Then use search_logs on that service and time window to read a few example lines. "
        "Finally, explain what happened: the service, the error code, the time window and an example message.\n"
        "Extra rules: if a tool returns an error, fix the arguments and retry. "
        "If you cannot find the cause, say so. Never guess or invent a result: answer only from what the tools "
        "returned. Always answer in English."
    )
    messages = [
        SystemMessage(log_sys_msg),
        HumanMessage(state["user_query"])
    ]
    return {"messages": messages}


# Node: the model reads the conversation so far and either asks for a tool or answers
def log_agent(state: State):
    ai_message = log_llm.invoke(state["messages"])
    ai_message.name = "log_agent"
    return {"messages": ai_message}


# Router: a tool call means run the tool; no tool call means the model has answered
def should_continue(state: State):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "log_tools"
    return END


workflow.add_node("message_builder", message_builder)
workflow.add_node("log_agent", log_agent)

workflow.add_edge(START, "message_builder")
workflow.add_edge("message_builder", "log_agent")
workflow.add_conditional_edges(
    source="log_agent",
    path=should_continue,
    path_map=["log_tools", END]
)
workflow.add_edge("log_tools", "log_agent")  # after a tool runs, the model reads the result and decides again

react_graph = workflow.compile()


def ask(question: str):
    # The client travels in config, so the model never sees it and the tools stay testable
    config = {
        "configurable": {"es_client": get_client()},
        "recursion_limit": 14,  # hard cap on graph steps: stops a model that keeps calling tools
    }
    try:
        messages = react_graph.invoke(input={"user_query": question}, config=config)
    except GraphRecursionError:
        print("The agent could not finish within the step limit; check the logs and the question.")
        return
    for message in messages["messages"]:
        message.pretty_print()


if __name__ == "__main__":
    ask(" ".join(sys.argv[1:]) or "What caused the errors on 8 October?")
