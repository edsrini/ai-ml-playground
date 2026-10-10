"""Functions and tools -- the manual mechanics behind 03_single_tool_agent.py.

Turns two plain Python functions into tools, binds them to the model, and shows
step by step what "tool calling" actually does: the model doesn't run the
function, it only asks for it to be run and with what arguments.
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.output_parsers.openai_tools import parse_tool_calls
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from config import OLLAMA_BASE_URL, OLLAMA_MODEL


# @tool turns a plain function into something the model can be told about and asked to call.
# The docstring is not decoration -- the model reads it to decide when and how to call the tool.
@tool
def add(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the result."""
    return a * b

tools = [add, multiply]

tool_map = {tool.name: tool for tool in tools}

print(tool_map)

# num_ctx caps the context window. Newer Ollama versions default to a model's max
# supported context (262144 for qwen3) instead of a sane default, which no longer
# fits fully in VRAM and forces a slow CPU/GPU split. 8192 is plenty for these examples.
llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0, num_ctx=8192)

llm_with_tools = llm.bind_tools(tools)

def ask(question: str):
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=question),
    ]

    ai_message = llm_with_tools.invoke(messages)
    messages.append(ai_message)  # must precede any ToolMessage replying to its tool_calls

    print(f"\nQuestion: {question}")
    print("tool_calls:", ai_message.tool_calls)
    # Note: ai_message.additional_kwargs.get("tool_calls") is always None for ChatOllama --
    # it never populates that raw OpenAI-wire-format key, so parse_tool_calls can't be used
    # here. See the hand-built raw_tool_call demo below for what it actually expects.

    if not ai_message.tool_calls:
        print("Model answered directly:", ai_message.content)
        return

    # One ToolMessage per tool call, each tagged with that call's id so the model
    # can match each result back to the request that produced it.
    for call in ai_message.tool_calls:
        tool_fn = tool_map[call["name"]]
        result = tool_fn.invoke(call["args"])
        print(f"Ran {call['name']}({call['args']}) -> {result}")
        messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    final = llm_with_tools.invoke(messages)
    print("Final answer:", final.content)


question1 = "3 multiply by 4"
question2 = "Who are you?"

ask(question1)  # expects a tool call
ask(question2)  # expects a direct answer, no tool needed


# parse_tool_calls does NOT work on ai_message.tool_calls -- that's already parsed.
# It expects the raw wire format a provider's API actually returns, where "arguments"
# is a JSON *string*, not a dict. This is what LangChain's Ollama/OpenAI integrations
# parse internally, before you ever see ai_message.tool_calls.
raw_tool_call = {
    "id": "call_123",
    "type": "function",
    "function": {"name": "multiply", "arguments": '{"a": 3, "b": 4}'},
}
