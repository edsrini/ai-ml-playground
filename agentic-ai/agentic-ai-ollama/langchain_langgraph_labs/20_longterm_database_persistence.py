import os
import json
from pathlib import Path
from pprint import pprint

import psycopg
from psycopg.rows import dict_row
from langchain_core.messages import (
    SystemMessage,
    AIMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_ollama import ChatOllama  # local replacement for ChatOpenAI
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver  # local replacement for SqliteSaver: saves state in Postgres
from IPython.display import Image, display

from dotenv import load_dotenv
from config import OLLAMA_BASE_URL, OLLAMA_MODEL

load_dotenv()

# The Postgres server running in Docker; the address and password are in .env as DATABASE_URL
DB_URI = os.getenv("DATABASE_URL")

# The course uses ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.0,
)


# Run the graph for one user message; the thread_id picks which saved conversation the message belongs to
def run_graph(query: str, graph: StateGraph, thread_id: int):
    output = graph.invoke(
        config={"configurable": {"thread_id": thread_id}},
        input={"messages": [HumanMessage(query)]}
    )
    return output


# ---- In-memory ----
# Graph node: send the whole conversation so far to the model and add its reply to the state
def chatbot(state: MessagesState):
    ai_message = llm.invoke(state["messages"])
    return {"messages": ai_message}


workflow = StateGraph(MessagesState)

workflow.add_node(chatbot)  # the node is named after the function: "chatbot"
workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot", END)

# MemorySaver keeps each thread's messages in RAM, so they are lost when the script ends
checkpointer = MemorySaver()
in_memory_graph = workflow.compile(checkpointer=checkpointer)

# A script cannot display the picture inline like a notebook, so save it as a file (nothing is drawn if offline)
try:
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "in_memory_graph.png").write_bytes(in_memory_graph.get_graph().draw_mermaid_png())
    print(f"Saved {output_dir / 'in_memory_graph.png'}")
except Exception as error:
    print("Could not render the PNG:", type(error).__name__)


# Send "Hi" on thread "1"; the notebook shows nothing for this cell, so print the model's reply to see it worked
output = run_graph(
    query="Hi",
    graph=in_memory_graph,
    thread_id="1"
)
output["messages"][-1].pretty_print()

# get_state returns a snapshot of the thread: its messages (values), the next node to run, config, metadata and more.
# list() turns that snapshot into a list of those fields, and a notebook cell would show it; a script needs print
print(list(
    in_memory_graph.get_state(
        config={"configurable": {"thread_id": "1"}}
    )
))


# ---- External memory (Postgres) ----
# The course saves to a SQLite file; here the Postgres server in Docker holds the checkpoints, as you would in production.
# PostgresSaver needs autocommit (each checkpoint is saved at once) and dict rows (it reads columns by name).
conn = psycopg.connect(DB_URI, autocommit=True, row_factory=dict_row)

memory = PostgresSaver(conn)
memory.setup()  # creates the checkpoint tables the first time; does nothing if they already exist
external_memory_graph = workflow.compile(checkpointer=memory)

try:
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "external_memory_graph.png").write_bytes(external_memory_graph.get_graph().draw_mermaid_png())
    print(f"Saved {output_dir / 'external_memory_graph.png'}")
except Exception as error:
    print("Could not render the PNG:", type(error).__name__)


# setup() only creates the empty tables; a row is written when the graph runs, so run it once on thread "2"
output = run_graph(
    query="Hi",
    graph=external_memory_graph,
    thread_id="2"
)
output["messages"][-1].pretty_print()

print(list(
    external_memory_graph.get_state(
        config={"configurable": {"thread_id": "2"}}
    )
))


# ---- Look inside the database ----
# SQLite lists its tables from sqlite_master; Postgres lists them in information_schema ("public" holds the checkpoint tables).
# The connection uses dict rows, so each row is read by column name instead of by position [0].
cursor = conn.cursor()
cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
tables = [row["table_name"] for row in cursor.fetchall()]
print(tables)

columns_map = []
results = []

for table in tables:
    cursor.execute(f"select * from {table}")
    results.append(cursor.fetchall())
    columns_map.append({table: [desc.name for desc in cursor.description]})

print(columns_map)  # the column names of each table
print({table: len(rows) for table, rows in zip(tables, results)})  # how many rows each table holds

# The metadata column of each checkpoint; Postgres stores it as JSON, so each value arrives as a Python dictionary
cursor.execute("select metadata from checkpoints")
metadata = cursor.fetchall()
print(metadata)

# SQLite hands the metadata over as JSON text, so the course parses it with json.loads(m[0]).
# Here each row is a dict and Postgres has already turned the JSON into a dict, so read the column by name
steps = [m["metadata"] for m in metadata]
print(steps)


# ---- Messages straight from the database ----
# The course's older langgraph put "writes" (what a node produced) inside the metadata column. This version keeps only
# step, source and parents there; what each step wrote is in the checkpoint_writes table, one row per write, stored
# as bytes. The saver's own serializer (memory.serde) turns those bytes back into message objects.
cursor.execute(
    "select checkpoint_id, idx, channel, type, blob from checkpoint_writes "
    "where thread_id = %s and channel = 'messages' order by checkpoint_id, idx",
    ("2",)
)
for row in cursor.fetchall():
    value = memory.serde.loads_typed((row["type"], bytes(row["blob"])))
    for message in value if isinstance(value, list) else [value]:  # a write holds one message or a list of them
        pprint({
            "checkpoint_id": row["checkpoint_id"],
            "channel": row["channel"],
            "message": message.model_dump(),  # text, response_metadata (model, timings) and token counts
        }, sort_dicts=False)
