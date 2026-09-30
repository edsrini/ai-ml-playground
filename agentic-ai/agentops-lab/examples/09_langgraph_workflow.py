import random
from pathlib import Path
from typing import TypedDict

from IPython.display import display, Image
from langgraph.constants import START, END
from langgraph.graph import StateGraph


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

