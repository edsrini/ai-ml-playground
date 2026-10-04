"""Node router -- LangGraph conditional routing: dispatch to reverse/upper nodes by action.

Two graphs share the same reverse/upper/invalid nodes and routing logic, built once via
wire_routing_nodes(); they differ only in how the action is decided before that point.
"""
import random
import sys
from pathlib import Path
from typing import Literal, TypedDict

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

# Router node: normalizes the given action (so " Reverse " still matches) and writes it back to state.
def route_request(state: RouterState) -> RouterState:
    return {"action": state["action"].strip().lower()}


workflowLLM = StateGraph(state_schema=RouterState)

workflowLLM.add_node("router", route_request)  # entry node: cleans up the action before routing
workflowLLM.add_node("reverse", reverse_node)
workflowLLM.add_node("upper", upper_node)
workflowLLM.add_node("invalid", invalid_node)


# Routing function (not a node): only reads state["action"] and returns the next node's name.
def route_action(state: RouterState) -> str:
    if state["action"] == "reverse":
        return "reverse"
    if state["action"] == "upper":
        return "upper"
    return "invalid"


workflowLLM.add_edge(START, "router")  # always enter through the router node first
workflowLLM.add_conditional_edges(
    source= "router",  # branch after the router node, instead of straight from START
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


# --- Second graph: an LLM reasons about free text first, then the same conditional edge routes on its answer ---

class RouterStateLLM(RouterState):
    user_input: str  # the raw free-form message, before the LLM turns it into action/text


class Intent(BaseModel):
    action: Literal["reverse", "upper", "invalid"] = Field(
        description="'reverse' or 'upper' if the user clearly asked for one of those text operations, otherwise 'invalid'"
    )
    text: str = Field(description="The text the user wants the operation applied to")


llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0, num_ctx=8192)
classifier_llm = llm.with_structured_output(Intent)


def classify_node(state: RouterStateLLM) -> RouterStateLLM:  # must be typed RouterStateLLM, or user_input gets filtered out
    # The LLM reasons here and returns a decision as data (action, text) -- it has no
    # idea "reverse_node" or "upper_node" exist. route_action below is what actually
    # picks the next node, by reading this data, not by anything the LLM controls.
    intent = classifier_llm.invoke(f"Classify this request and extract the target text: {state['user_input']}")
    return {"action": intent.action, "text": intent.text}


workflowWithLLM = StateGraph(state_schema=RouterStateLLM)

workflowWithLLM.add_node("classify", classify_node)
workflowWithLLM.add_node("reverse", reverse_node)
workflowWithLLM.add_node("upper", upper_node)
workflowWithLLM.add_node("invalid", invalid_node)

workflowWithLLM.add_edge(START, "classify")
workflowWithLLM.add_conditional_edges(
    source="classify",
    path=route_action,  # same routing function as the graph above -- it only ever reads state["action"]
    path_map=["reverse", "upper", "invalid"]
)
workflowWithLLM.add_edge("reverse", END)
workflowWithLLM.add_edge("upper", END)
workflowWithLLM.add_edge("invalid", END)

routergraph_llm = workflowWithLLM.compile()

result = routergraph_llm.invoke(input={"user_input": "please reverse hello"})
print("llm reverse:", result)

result = routergraph_llm.invoke(input={"user_input": "make hello uppercase"})
print("llm upper:", result)

result = routergraph_llm.invoke(input={"user_input": "what's the weather like?"})
print("llm invalid:", result)

