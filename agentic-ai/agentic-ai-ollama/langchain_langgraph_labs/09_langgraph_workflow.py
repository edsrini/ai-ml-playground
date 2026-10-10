"""LangGraph exercise: a deterministic two-node workflow (graph), then a second,
separate graph (graphLLM) swapping in a single LLM-backed node instead."""
import random
import sys
from pathlib import Path
from typing import TypedDict

sys.stdout.reconfigure(encoding="utf-8")

from IPython.display import display, Image
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama

from config import OLLAMA_BASE_URL, OLLAMA_MODEL
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from dotenv import load_dotenv
load_dotenv()

llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0, num_ctx=8192)

class State(TypedDict):
    input:int
    output:int

def node_a(state: State)->State:
    input_value = state["input"]
    offset = random.randint(1, 10)
    output_value = input_value + offset
    print(
        f"NODE A:\n"
        f"\t{input_value}\n"
        f"\t{offset}\n"
        f"\t{output_value}\n"
    )
    return State(output=output_value)

def node_b(state: State)->State:
    input_value = state["output"]
    offset = random.randint(1, 10)
    output_value = input_value + offset
    print(
        f"NODE B:\n"
        f"\t{input_value}\n"
        f"\t{offset}\n"
        f"\t{output_value}\n"
    )
    return State(output=output_value)

workflow = StateGraph(state_schema=State)

workflow.add_node(node_a)
workflow.add_node(node_b)

workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

graph = workflow.compile()

# display(Image(...)) only renders inline in a real Jupyter notebook -- in a plain script
# it just prints the object's repr, so the PNG bytes are also saved to a file to view directly.
# draw_mermaid_png() calls the remote mermaid.ink API and has been intermittently failing on
# an SSL cert error (network/proxy TLS inspection) -- fall back to the local ASCII view instead
# of crashing when that happens.
try:
    png_bytes = graph.get_graph().draw_mermaid_png()
    display(Image(png_bytes))

    output_dir = Path(__file__).resolve().parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    png_path = output_dir / "09_langgraph_workflow.png"
    png_path.write_bytes(png_bytes)
    print(f"Graph image saved to {png_path}")
except Exception as e:
    print(f"Could not reach mermaid.ink to render PNG ({e}); showing ASCII graph instead:")
    graph.get_graph().print_ascii()

if __name__ == "__main__":
    result = graph.invoke({"input": 5})
    print("Final state:", result)


# Second, separate graph: a single LLM-backed node instead of the deterministic node_a/node_b.
class WorkflowState(TypedDict):
    question: str
    response: str

def modelLLM(state: WorkflowState) -> WorkflowState:
    # The param's type hint (WorkflowState) is what LangGraph uses to decide which
    # fields of the graph state this node receives -- get it wrong and "question"
    # silently never arrives, even though invoke() was called with it.
    question = state["question"]
    response = llm.invoke([
        SystemMessage("You are a subject matter expert!"),
        HumanMessage(question)
    ])
    # Must return the update dict; a node that computes but doesn't return loses the result.
    return {"response": response.content}

workflowLLM = StateGraph(state_schema=WorkflowState)

workflowLLM.add_node(modelLLM)

workflowLLM.add_edge(START, "modelLLM")
workflowLLM.add_edge("modelLLM", END)

graphLLM = workflowLLM.compile()


try:
    png_bytes = graphLLM.get_graph().draw_mermaid_png()
    display(Image(png_bytes))

    output_dir = Path(__file__).resolve().parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    png_path = output_dir / "09_langgraph_llm_workflow.png"
    png_path.write_bytes(png_bytes)
    print(f"GraphLLM image saved to {png_path}")
except Exception as e:
    print(f"Could not reach mermaid.ink to render PNG ({e}); showing ASCII graph instead:")
    graphLLM.get_graph().print_ascii()

result = graphLLM.invoke(input={"question": "Do you like Tamil?"},)
print("Final state:", result)