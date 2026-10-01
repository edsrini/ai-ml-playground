"""Node router -- LangGraph conditional routing: dispatch to reverse/upper nodes by action.

Two graphs share the same reverse/upper/invalid nodes and routing logic, built once via
wire_routing_nodes(); they differ only in how the action is decided before that point.
"""
import random
import sys
from pathlib import Path
from typing import TypedDict

sys.stdout.reconfigure(encoding="utf-8")

from IPython.display import display, Image

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from config import OLLAMA_BASE_URL, OLLAMA_MODEL

sys.stdout.reconfigure(encoding="utf-8")  # avoids a crash when model output


class RouterState(TypedDict):
    text: str
    action: str
    result: str


def reverse_node(state: RouterState) -> RouterState:
    return {"result": state["text"][::-1]}


def upper_node(state: RouterState) -> RouterState:
    return {"result": state["text"].upper()}


def invalid_node(state: RouterState) -> RouterState:
    return {"result": f"Invalid action: '{state['action']}'. Valid actions: reverse, upper."}  # graceful fallback, no crash

workflowLLM = StateGraph(state_schema=RouterState)

workflowLLM.add_node("reverse", reverse_node)
workflowLLM.add_node("upper", upper_node)
workflowLLM.add_node("invalid", invalid_node)


def route_action(state: RouterState) -> str:
    if state["action"] == "reverse":
        return "reverse"
    if state["action"] == "upper":
        return "upper"
    return "invalid"


workflowLLM.add_conditional_edges(
    source= START,
    path= route_action,
    path_map= ["reverse", "upper", "invalid"]
)

workflowLLM.add_edge("reverse", END)
workflowLLM.add_edge("upper", END)
workflowLLM.add_edge("invalid", END)

routergraph = workflowLLM.compile()

try:
    png_bytes = routergraph.get_graph().draw_mermaid_png()
    display(Image(png_bytes))

    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(exist_ok=True)
    png_path = output_dir / "router_workflow.png"
    png_path.write_bytes(png_bytes)
    print(f"GraphLLM image saved to {png_path}")
except Exception as e:
    print(f"Could not reach mermaid.ink to render PNG ({e}); showing ASCII graph instead:")
    routergraph.get_graph().print_ascii()

result = routergraph.invoke(
    input= {
        "text": "Hello world",
        "action": "reverse"
    }
)
print("reverse:", result)

result = routergraph.invoke(
    input= {
        "text": "Hello world",
        "action": "upper"
    }
)
print("upper:", result)

