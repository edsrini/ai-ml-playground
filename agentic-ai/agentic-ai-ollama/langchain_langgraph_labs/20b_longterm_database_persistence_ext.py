from pathlib import Path

from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings  # local replacements for the OpenAI chat and embedding models

from dotenv import load_dotenv
from config import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL, OLLAMA_MODEL

load_dotenv()

# The store keeps long-term memories and finds them by meaning, so it needs an embedding model.
# The course uses "openai:text-embedding-3-small" with 1536 dimensions; nomic-embed-text makes vectors of 768 numbers.
store = InMemoryStore(
    index={
        "dims": 768,
        "embed": OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL),
    }
)

agent = create_react_agent(
    model=ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL),
    tools=[
        # Memory tools use LangGraph's BaseStore for persistence
        create_manage_memory_tool(namespace=("memories",)),  # lets the agent save, update and delete a memory
        create_search_memory_tool(namespace=("memories",)),  # lets the agent look memories up by meaning
    ],
    store=store,
)

# Draw the agent's graph and save it as a picture (a script cannot show it inline; nothing is drawn if offline)
try:
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "memory_agent_graph.png").write_bytes(agent.get_graph().draw_mermaid_png())
    print(f"Saved {output_dir / 'memory_agent_graph.png'}")
except Exception as error:
    print("Could not render the PNG:", type(error).__name__)

print(agent)  # a notebook cell would show this compiled graph; a script needs print
print(agent.get_input_jsonschema())  # the shape of the input the agent accepts


# Ask a random question to the agent
output = agent.invoke(
    input={
        "messages": [
            {
                "role": "user",
                "content": "What are my lighting preferences?"
            }
        ]
    },
    config={"configurable": {"thread_id": "11"}}
)

# A notebook cell would show output["messages"]; print each message so the memory tool calls are visible too
for message in output["messages"]:
    message.pretty_print()


messages = output["messages"]

messages.append(HumanMessage("Ok. Remember that I prefer dark mode."))

# Store a new memory in one Session
output = agent.invoke(
    input={
        "messages": messages,
    },
    config={"configurable": {"thread_id": "11"}}
)

for message in output["messages"]:
    message.pretty_print()


output = agent.invoke(
    input={
        "messages": [
            {
                "role": "user",
                "content": "What are my lighting preferences?"
            }
        ]
    },
    config={"configurable": {"thread_id": "12"}}
)

for message in output["messages"]:
    message.pretty_print()