"""Human-in-the-loop with a local Ollama model: the agent proposes a web search and a person decides what happens.

The graph pauses right before the web search runs. The person can:
  [a] approve   the search runs as the model asked
  [e] edit      the search runs with a query the person types instead
  [r] reject    the search does not run; the agent is told why and carries on
  [q] quit      the run stays paused

Same style as 14_agentic_rag.py. Every run is traced in MLflow (http://127.0.0.1:5000).
Needs the MLflow server:   docker compose up -d   (in mlops/mlflow-server)
"""
import mlflow
import os
import uuid
from pathlib import Path
from typing import Dict

from langgraph.checkpoint.memory import MemorySaver

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from tavily import TavilyClient

import truststore
truststore.inject_into_ssl()  # use the Windows certificate store so HTTPS calls work

from dotenv import load_dotenv
load_dotenv()

from config import OLLAMA_BASE_URL, OLLAMA_MODEL

# ---- MLflow tracing ----
tracking_uri = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(tracking_uri)
experiment = mlflow.set_experiment("srini's-human-in-the-loop")

# Newer langgraph calls on_interrupt when the graph pauses and on_resume when it continues, on every callback handler.
# This MLflow version has neither method, which logs "Error in MlflowLangchainTracer.on_interrupt callback". Accept and ignore both.
from mlflow.langchain.langchain_tracer import MlflowLangchainTracer
for _name in ("on_interrupt", "on_resume"):
    if not hasattr(MlflowLangchainTracer, _name):
        setattr(MlflowLangchainTracer, _name, lambda self, event: None)

mlflow.langchain.autolog()

# Chat model; temperature 0 gives repeatable answers
llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.0,
    num_ctx=8192,
)


# ---- Graph state ----
# Shared data every node reads and updates; "messages" comes from MessagesState
class State(MessagesState):
    question: str
    answer: str


# ---- Web search tool ----
# Tavily search; errors are returned as data so the graph doesn't crash
@tool
def web_search(question: str) -> Dict:
    """
    Return top search results for a given search query
    """
    try:
        tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = tavily_client.search(question)
    except Exception as error:
        return {"error": str(error)}
    return response


# Let the model request web_search by name
llm_with_tools = llm.bind_tools([web_search])


# Node: entry point; turns the question into the first message of the conversation
def entry_point(state: State):
    question = state["question"]
    human_message = HumanMessage(
        "Conduct a web research to find out the answer to the following question: "
        f"```{question}```"
    )
    return {"messages": [human_message]}


# Node: the model reads the conversation and either answers or asks for web_search
def agent(state: State):
    messages = state["messages"]
    ai_message = llm_with_tools.invoke(messages)
    messages.append(ai_message)
    return {"messages": messages, "answer": ai_message.content}


# Router: run the tools if the model asked for one, otherwise finish
def router(state: State):
    if state["messages"][-1].tool_calls:
        return "tools"
    return END


# ---- Build the graph ----
workflow = StateGraph(State)

workflow.add_node("entry_point", entry_point)
workflow.add_node("agent", agent)
workflow.add_node("tools", ToolNode([web_search]))  # runs the tool the model asked for

workflow.add_edge(START, "entry_point")
workflow.add_edge("entry_point", "agent")

# Research loop: agent -> tools -> agent, until no more tool calls
workflow.add_conditional_edges(
    source="agent",
    path=router,
    path_map=["tools", END]
)
workflow.add_edge("tools", "agent")

memory = MemorySaver()  # keeps the paused run's state so it can be resumed
graph = workflow.compile(
    interrupt_before=["tools"],  # pause before the tools run, so a person can decide
    checkpointer=memory
)


# ---- Save a picture of the graph (ASCII if offline) ----
try:
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "human_in_the_loop_graph.png").write_bytes(graph.get_graph().draw_mermaid_png())
    print(f"Saved {output_dir / 'human_in_the_loop_graph.png'}")
except Exception as error:
    print("Could not render the PNG:", type(error).__name__)
    graph.get_graph().print_ascii()


# ---- Run it with a human decision ----
# Streams the run step by step until the graph pauses before "tools", asks a person, then resumes or aborts.
def human_in_the_loop_run(graph: CompiledStateGraph, question: str, thread_id: str):
    input_question = {"question": question}
    config = {"recursion_limit": 25, "configurable": {"thread_id": thread_id}}

    # stream_mode="values" gives the whole state after every step; print the newest message each time
    for event in graph.stream(input=input_question, config=config, stream_mode="values"):
        if not event['messages']:
            continue
        event['messages'][-1].pretty_print()

    if not graph.get_state(config=config).next:  # nothing is paused (the model answered without tools): no decision needed
        return

    human_input = input("Do you approve the tool calling? (YES or NO): ")
    if human_input.lower() == "yes":
        for event in graph.stream(input=None, config=config, stream_mode="values"):  # None = carry on from the saved state
            if not event['messages']:
                continue
            event['messages'][-1].pretty_print()

    else:
        SystemMessage("Workflow aborted by a human").pretty_print()


# One thread_id per run, so each question is its own session in MLflow
human_in_the_loop_run(graph, "What is the tallest building in the world?", "hitl-" + uuid.uuid4().hex[:6])
