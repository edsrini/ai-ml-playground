"""Text-to-SQL agent: a local model looks at the tables, writes a SELECT, runs it and explains the rows.

Usage:  python agent.py "Which model earned the most revenue?"
"""
import os
import sys

import truststore

truststore.inject_into_ssl()  # trust the Windows certificate store so HTTPS (mermaid.ink) works
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from database import get_engine
from tools import TOOLS
from dotenv import load_dotenv

from langgraph.graph import START, END, StateGraph
from message import MessagesState  # our own copy of langgraph's MessagesState (see message.py)
from langgraph.prebuilt import ToolNode
from IPython.display import Image, display
from sqlalchemy import create_engine

from tools import (
    list_tables as list_tables_tool,
    get_table_schema as get_table_schema_tool,
    execute_sql as execute_sql_tool
)

SYSTEM_PROMPT = (
    "You answer questions about a SQLite database using tools.\n"
    "1. Call list_tables, then get_table_schema for the table you need. Never guess column names.\n"
    "2. Write ONE SELECT query and run it with execute_sql. If it returns an error, fix the query and retry.\n"
    "3. Answer using only the rows returned. If the rows do not answer the question, say so.\n"
    "Revenue is price * quantity. Always answer in English."
)


def build_agent():

    llm = ChatOllama(model=os.getenv("OLLAMA_MODEL"), base_url=os.getenv("OLLAMA_BASE_URL"), temperature=0, num_ctx=8192)  # 8192 keeps it on the GPU
    return create_agent(llm, tools=TOOLS, system_prompt=SYSTEM_PROMPT)

# Instantiate your chat model
load_dotenv()
llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL"),
    base_url=os.getenv("OLLAMA_BASE_URL"),
    temperature=0.0,
    num_ctx=8192,  # keeps the model on the GPU; not needed with a hosted model
)

#response = llm.invoke("What's Pokemon?")
#print(repr(response))  # repr shows AIMessage(content='...', ...) like a notebook cell; print(response.content) shows only the text

class State(MessagesState):
    user_query: str

workflow = StateGraph(State)

dba_tools = [list_tables_tool, get_table_schema_tool, execute_sql_tool]

workflow.add_node("dba_tools", ToolNode(dba_tools))

dba_llm = llm.bind_tools(dba_tools, tool_choice="auto")

# Node: build the prompt for the model from the user's question
def message_builder(state: State):
    dba_sys_msg = (
        "You are a Sr. SQL developer tasked with generating SQL queries. Perform the following steps:\n"
        "First, find out the appropriate table name based on all tables. "
        "Then get the table's schema to understand the columns. "
        "With the table name and the schema, generate the ANSI SQL query you think is applicable to the user question. "
        "Finally, use a tool to execute the above SQL query and output the result based on the user question.\n"
        # Added rules (not in the course prompt), after the wrong "Tablet" vs "Tablets" filter and the made-up "0 laptops":
        "Extra rules: "
        "Before filtering on a text column, run SELECT DISTINCT on it to see the exact values, which may be plural or "
        "capitalised differently. "
        "If a query returns no rows, try a broader one (for example LIKE '%word%') before saying nothing exists. "
        "If a tool returns an error, fix the query and retry; if you still cannot get data, say so. "
        "Never guess or invent a result: answer only from the rows returned. Always answer in English."
        "Provide all the information across all relevant tables if needed like orders, issues, returns etc"
    )
    messages = [
        SystemMessage(dba_sys_msg),
        HumanMessage(state["user_query"])
    ]
    return {"messages": messages}

def dba_agent(state: State):
    ai_message = dba_llm.invoke(state["messages"])
    ai_message.name = "dba_agent"
    return {"messages": ai_message}

workflow.add_node("message_builder", message_builder)
workflow.add_node("dba_agent", dba_agent)

def should_continue(state: State):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "dba_tools"
    return END

workflow.add_edge(START, "message_builder")
workflow.add_edge("message_builder", "dba_agent")
workflow.add_conditional_edges(
    source="dba_agent",
    path=should_continue,
    path_map=["dba_tools", END]
)

workflow.add_edge("dba_tools", "dba_agent")

react_graph = workflow.compile()

# In a notebook display(Image(...)) shows the picture; in a script we save it as a file instead.
# mermaid.ink renders the PNG over the internet, so fall back to an ASCII drawing if that fails.
try:
    os.makedirs("output", exist_ok=True)
    with open("output/text2sql_graph.png", "wb") as f:
        f.write(react_graph.get_graph().draw_mermaid_png())
    print("Saved output/text2sql_graph.png")
except Exception as error:
    print("Could not render the PNG:", type(error).__name__)
    react_graph.get_graph().print_ascii()

db_engine = get_engine()  # the sales.db path comes from .env (DB_PATH), so it never depends on the working directory

config = {
    "configurable": {
        "db_engine": db_engine
    }
}

inputs = {
    "user_query": "Pull all the information related Ravi Stone, I also want to see information related to orders for this user and related issue. Please fix the issue."
}

messages = react_graph.invoke(
    input=inputs,
    config=config
)

for message in messages["messages"]:
    message.pretty_print()