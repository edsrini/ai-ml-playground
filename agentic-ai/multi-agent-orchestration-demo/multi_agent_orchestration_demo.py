# ---- 0. Setup ----
import os
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import create_react_agent


load_dotenv()


OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# The one model that all three agents below use
llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.0,
    num_ctx=8192,
)


# ---- 1. Sub Agents ----
# Worker 1: it has no tools, so it answers once with keywords and sources
research_worker = create_react_agent(
    name="research_worker",
    prompt=SystemMessage(
        content=(
            "You are a research specialist. Given a topic, find key information. "
            "Return ONLY JSON with: keywords, sources."
        )
    ),
    model=llm,
    tools=[],
)


# Worker 2: it has no tools either; it turns research keywords into written content
writing_worker = create_react_agent(
    name="writing_worker",
    prompt=SystemMessage(
        content=(
            "You are a content writer. "
            "Given research keywords, write clear and concise content. "
            "Return ONLY JSON with: content, summary."
        )
    ),
    model=llm,
    tools=[],
)


# ---- 2. Handoff Tools ----
# A handoff tool: the supervisor calls it like any other tool, and inside it the research worker runs
@tool
def call_researcher(topic: str) -> Dict[str, Any]:
    """Ask the research worker to find information about a topic."""
    message = HumanMessage(content=f"Research this topic: {topic}")

    result = research_worker.invoke(
        input={"messages": [message]}
    )

    # The worker's final answer is its last message
    last_message: AIMessage = result["messages"][-1]

    # Simple parsing (could use structured output for production)
    return {
        "task": "research",
        "topic": topic,
        "results": last_message.content
    }


# Same idea for the writer: the supervisor passes in the research keywords and gets the writer's answer back
@tool
def call_writer(research_keywords: str) -> Dict[str, Any]:
    """Ask the writing worker to create content based on research."""
    message = HumanMessage(
        content=f"Write a concise content based on these keywords: {research_keywords}"
    )

    result = writing_worker.invoke(
        input={"messages": [message]}
    )

    last_message: AIMessage = result["messages"][-1]

    return {
        "task": "writing",
        "research": research_keywords,
        "results": last_message.content
    }


# ---- 3. Supervisor Agent ----
# The supervisor is the only agent that decides anything: it chooses which handoff tool to call, with what input, and when to stop
supervisor_agent = create_react_agent(
    name="supervisor_agent",
    prompt=SystemMessage(
        content=(
            "You are a task supervisor. When given a request:\n"
            "1. First use call_researcher to gather information\n"
            "2. Then use call_writer to create content based on the research\n"
            "3. Combine everything into a final response\n\n"
            "Use the tools in this order: call_researcher → call_writer"
        )
    ),
    model=llm,
    tools=[call_researcher, call_writer],
)


# ---- 4. Run ----
# Start the run with one human message; the supervisor works out the rest
result = supervisor_agent.invoke(
    input={
        "messages": [
            HumanMessage(
                content="topic machine learning basics"
            )
        ]
    }
)

# Print the whole trail: the request, the tool calls, the tool results and the final answer
for message in result["messages"]:
    message.pretty_print()
