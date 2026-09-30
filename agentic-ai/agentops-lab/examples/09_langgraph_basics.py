"""LangGraph fundamentals -- state, nodes, edges, conditional routing, compile, invoke.

Hand-rolls the same tool-call loop that create_agent() did automatically in
03_single_tool_agent.py, using the manual tool-execution mechanics from
08_functions_and_tools.py as the "tools" node. Seeing the loop built by hand once
is what makes create_agent() no longer feel like a black box.
"""
import sys
from pathlib import Path
from typing import TypedDict

sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from config import OLLAMA_BASE_URL, OLLAMA_MODEL


@tool
def add(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the result."""
    return a * b


tools = [add, multiply]
tool_map = {t.name: t for t in tools}

# num_ctx keeps the model fully on GPU -- see 08_functions_and_tools.py for why this matters.
# reasoning=False was tried here to fix a final-answer confusion bug (see below) but made
# it worse: the model looped through the same tool calls 13 times and still leaked its
# raw <think> block into .content, matching the reasoning=False finding in
# 01_basic_chat.py/05_structured_output.py earlier in this project. Left at the default.
llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0, num_ctx=8192)
llm_with_tools = llm.bind_tools(tools)


# A state schema built from scratch (a TypedDict), rather than the library's pre-built
# MessagesState. All nodes and edges read from and write to this same shared state.
class WorkflowState(TypedDict):
    messages: list[AnyMessage]


def call_model(state: WorkflowState) -> WorkflowState:
    """Node: ask the model what to do next, given the conversation so far."""
    ai_message = llm_with_tools.invoke(state["messages"])
    return {"messages": [ai_message]}


def call_tools(state: WorkflowState) -> WorkflowState:
    """Node: run every tool call the model just requested, one ToolMessage per call."""
    last_message = state["messages"][-1]
    results = []
    for call in last_message.tool_calls:
        tool_fn = tool_map[call["name"]]
        result = tool_fn.invoke(call["args"])
        print(f"  Ran {call['name']}({call['args']}) -> {result}")
        results.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    return {"messages": results}


def should_continue(state: WorkflowState) -> str:
    """Routing function: branch to the tools node, or stop, based on current state."""
    last_message = state["messages"][-1]
    return "tools" if last_message.tool_calls else END


# Build the graph: register nodes, wire normal and conditional edges, then compile.
# Compiling validates the structure -- the graph can't be invoked before this step.
graph_builder = StateGraph(WorkflowState)
graph_builder.add_node("agent", call_model)
graph_builder.add_node("tools", call_tools)

graph_builder.add_edge(START, "agent")  # entry point: always start at "agent"
graph_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "agent")  # loop back so the model sees the tool result

graph = graph_builder.compile()


def ask(question: str):
    # Known finding: the graph mechanics (tool execution, looping, message threading) are
    # reliable every run -- the correct tool calls always run in the right order. The final
    # natural-language synthesis after a *two*-hop tool chain is where qwen3:4b is flaky:
    # it sometimes claims "no query was given" despite the original question still being
    # earlier in `messages`. Tried three system prompt rewrites and reasoning=False; none
    # fixed it reliably. This mirrors every other qwen3:4b reliability finding in this
    # project -- small local models are strong at single-step tool use, weak at multi-step
    # synthesis without extra scaffolding (e.g. structured output, or a deterministic
    # final-answer extraction step instead of trusting free-form text).
    print(f"\nQuestion: {question}")
    initial_state = {
        "messages": [
            SystemMessage(
                content=(
                    "You are a helpful assistant with access to math tools. "
                    "Use the tools to fully compute the answer to the user's question, "
                    "calling them more than once if the question has multiple steps. "
                    "Once every calculation is done, state the final numeric answer in "
                    "one short sentence."
                )
            ),
            HumanMessage(content=question),
        ]
    }
    final_state = graph.invoke(initial_state)
    print("Final answer:", final_state["messages"][-1].content)


def main():
    print(graph.get_graph().draw_mermaid())

    ask("3 multiply by 4, then add 10 to that")  # needs two tool calls, looping twice
    ask("Who are you?")  # no tool needed, one pass through "agent" then straight to END


if __name__ == "__main__":
    main()
