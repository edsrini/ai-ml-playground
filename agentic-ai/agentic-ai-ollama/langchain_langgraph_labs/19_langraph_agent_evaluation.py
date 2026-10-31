"""Evaluating a LangGraph agent with RAGAS: retrieval and answer quality, tool-call accuracy and goal accuracy.

The course notebook uses ChatOpenAI and OpenAIEmbeddings; here ChatOllama and OllamaEmbeddings fill the same two roles,
so the agent and the judging both run on local models.
"""
import sys
import types

# ragas 0.4 imports ChatVertexAI from a langchain_community module that newer langchain_community versions no longer
# have, so `import ragas` fails. Google Vertex AI is not used here, so register an empty stand-in before the import.
try:
    import langchain_community.chat_models.vertexai  # noqa: F401
except ModuleNotFoundError:
    _stand_in = types.ModuleType("langchain_community.chat_models.vertexai")
    _stand_in.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules["langchain_community.chat_models.vertexai"] = _stand_in

import asyncio
import json
from pathlib import Path

from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings  # local replacements for ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from IPython.display import Image, display
from dotenv import load_dotenv
from ragas import EvaluationDataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from ragas.integrations.langgraph import convert_to_ragas_messages
from ragas import messages as ragas_messages
from ragas.dataset_schema import MultiTurnSample
from ragas.metrics import ToolCallAccuracy
from ragas.metrics import AgentGoalAccuracyWithReference
from ragas.run_config import RunConfig

load_dotenv()


# ---- Sample documents ----
# Five short news headlines, each tagged with the company and topic it is about
documents = [
    Document(
        page_content="Meta drops multimodal Llama 3.2 — here's why it's such a big deal",
        metadata={"company": "Meta", "topic": "llama"}
    ),
    Document(
        page_content="Chip giant Nvidia acquires OctoAI, a Seattle startup that helps companies run AI models",
        metadata={"company": "Nvidia", "topic": "acquisition"}
    ),
    Document(
        page_content="Google is bringing Gemini to all older Pixel Buds",
        metadata={"company": "Google", "topic": "gemini"}
    ),
    Document(
        page_content="The first Intel Battlmage GPU benchmarks have leaked",
        metadata={"company": "Intel", "topic": "gpu"}
    ),
    Document(
        page_content="Dell partners with Nvidia to accelerate AI adoption in telecoms",
        metadata={"company": "Dell", "topic": "partnership"}
    ),
]

# One id per document, so each can be found (or replaced) in the vector store later
ids = ["id1", "id2", "id3", "id4", "id5"]


# ---- Vector store ----
from config import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL, OLLAMA_JUDGE_MODEL, OLLAMA_MODEL

# Embedding model turns text into vectors; the course uses OpenAIEmbeddings(model="text-embedding-3-large")
embeddings_fn = OllamaEmbeddings(
    model=OLLAMA_EMBED_MODEL,
    base_url=OLLAMA_BASE_URL,
)

# In-memory vector store, rebuilt on every run
vector_store = Chroma(
    collection_name="agent-eval",
    embedding_function=embeddings_fn,
)

# Embed the documents and store them under their ids; adding the same ids again replaces them instead of duplicating
_ = vector_store.add_documents(documents=documents, ids=ids)
print(f"Stored {len(vector_store.get()['ids'])} documents in the vector store")

retriever = vector_store.as_retriever(search_kwargs={"k": 1})  # return only the single closest document


# ---- Chat model ----
# The course uses ChatOpenAI(model="gpt-4o-mini", temperature=0.0); temperature 0 gives repeatable answers
llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.0,
    num_ctx=8192,  # keeps the model on the GPU
)


# ---- Answer chain ----
prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
{context}

Question: {query}
"""
)

chain = prompt | llm | StrOutputParser()  # LCEL: fill the prompt, call the model, return plain text


print(chain)


# Join the retrieved documents into one block of text for the {context} slot
def format_docs(relevant_docs):
    return "\n".join(doc.page_content for doc in relevant_docs)


query = "Who is partnering with Nvidia?"
relevant_docs = retriever.invoke(query)
print(chain.invoke({"context": format_docs(relevant_docs), "query": query}))


# ---- Ragas Evaluation ----
# Five test questions, one per document
sample_queries = [
    "What is Meta's latest development in AI?",
    "Which company did Nvidia acquire?",
    "What AI feature is Google adding to older Pixel Buds?",
    "What recent information has leaked about Intel GPUs?",
    "Which company did Dell partner with to accelerate AI adoption?"
]

# The answer each question should get (the reference RAGAS compares the agent's answers with), in the same order
expected_responses = [
    "Meta drops multimodal Llama 3.2 — here's why it's such a big deal",
    "Chip giant Nvidia acquires OctoAI, a Seattle startup that helps companies run AI models",
    "Google is bringing Gemini to all older Pixel Buds",
    "The first Intel Battlemage GPU benchmarks have leaked",
    "Dell partners with Nvidia to accelerate AI adoption in telecoms"
]

# Run every question through the system and record what RAGAS needs for each one:
# the question, the passages retrieved, the answer given, and the reference answer
dataset = []

for query, reference in zip(sample_queries, expected_responses):
    relevant_docs = retriever.invoke(query)
    response = chain.invoke(
        {
            "context": format_docs(relevant_docs),
            "query": query
        }
    )
    dataset.append(
        {
            "user_input": query,
            "retrieved_contexts": [doc.page_content for doc in relevant_docs],
            "response": response,
            "reference": reference,
        }
    )
"""
evaluation_dataset = EvaluationDataset.from_list(dataset)

# The judge: RAGAS asks a language model to grade each answer, so it needs the model wrapped in its own interface.
# It is a separate setting from the model that answers (OLLAMA_MODEL): a stronger non-thinking model, set in config.py
# or by OLLAMA_JUDGE_MODEL in .env. A non-thinking model gives clean JSON; qwen3:4b with reasoning=False still thinks,
# but writes the thinking into the answer text, which breaks the JSON that RAGAS expects.
evaluator_llm = LangchainLLMWrapper(ChatOllama(  # the wrapper lets RAGAS call any LangChain chat model as its judge
    model=OLLAMA_JUDGE_MODEL,  # the grader, not the model that wrote the answers
    base_url=OLLAMA_BASE_URL,  # the local Ollama server
    temperature=0.0,  # no randomness, so the same answer gets the same grade each time
    num_ctx=8192,  # context window in tokens: enough for question, context, answer and grading prompt
))

# evaluate() runs every metric on every question. Each (metric, question) pair is a job: 3 metrics x 5 questions = 15
result = evaluate(
    dataset=evaluation_dataset,  # the questions, retrieved passages, answers and reference answers built above
    # context_recall: did retrieval find what the reference answer needs (needs reference + retrieved contexts)
    # faithfulness: is every claim in the answer backed by the retrieved passages (needs response + retrieved contexts)
    # factual_correctness: does the answer match the reference answer (needs response + reference)
    metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
    llm=evaluator_llm,  # the judge that grades each job
    # Ollama serves about 4 requests at once (measured: 4 together took 4.0s against 10.5s one by one), so 4 workers use
    # that speed without piling up a queue; more than that only makes jobs wait. Each job gets up to 10 minutes.
    run_config=RunConfig(timeout=600, max_workers=4),
)

print(result)  # the average score of each metric over the five questions
"""

# ---- Pokemon types ----
# Lookup table from a Pokemon's name to its type (dual types are written "type/type")
pokemon_types_map = {
    "pikachu": "electric",
    "eevee": "normal",
    "bulbasaur": "grass/poison",
    "squirtle": "water",
    "charizard": "fire/flying",
    "jigglypuff": "normal/fairy",
    "meowth": "normal",
    "psyduck": "water",
    "machamp": "fighting",
    "gengar": "ghost/poison",
    "alakazam": "psychic",
    "snorlax": "normal",
    "dragonite": "dragon/flying",
}


# The tool the agent calls to look a Pokemon's type up in the table above.
# @tool turns the function into a LangChain tool: the model sees its name, its argument and the docstring below,
# and decides from the docstring when to call it, so the docstring is part of the tool's instructions.
@tool
def get_pokemon_type(pokemon_name: str) -> str:
    """Fetches the type of the specified Pokémon.

    Args:
        pokemon_name : The name of the Pokémon (e.g., 'pikachu', 'charizard', 'eevee').

    Returns:
        str: The type(s) of the Pokémon.

    Raises:
        KeyError: If the specified Pokémon is not found in the data source.
    """
    try:
        # The model may send "Pikachu " or "PIKACHU"; the table keys are lowercase with no spaces
        pokemon_name = pokemon_name.lower().strip()
        if pokemon_name not in pokemon_types_map:
            # The message lists every valid name, so the caller can see what it should have asked for
            raise KeyError(
                f"Pokémon '{pokemon_name}' not found. Available Pokémon: {', '.join(pokemon_types_map.keys())}"
            )
        return pokemon_types_map[pokemon_name]  # e.g. "charizard" -> "fire/flying"
    except Exception as e:
        # Any failure, including the KeyError above, is re-raised with one clear prefix
        raise Exception(f"Error fetching Pokémon type: {str(e)}")


# ---- Agent graph pieces ----
tools = [get_pokemon_type]  # every tool the agent may call

# bind_tools tells the model which tools exist, so it can answer with a tool call instead of plain text
llm_with_tools = llm.bind_tools(tools)


# Graph node: send the whole conversation to the model and add its reply (text or a tool call) to the state
def agent(state: MessagesState):
    ai_message = llm_with_tools.invoke(state["messages"])
    return {"messages": ai_message}


# Conditional edge: a reply that asks for a tool goes to the "tools" node, anything else ends the run
def router(state: MessagesState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"

    return END


# ---- Build the graph ----
workflow = StateGraph(MessagesState)

workflow.add_node("agent", agent)
workflow.add_node("tools", ToolNode(tools))  # runs the tool the model asked for

workflow.add_edge(START, "agent")

# Loop: agent -> tools -> agent, until the model replies without a tool call
workflow.add_conditional_edges(
    source="agent",
    path=router,
    path_map=["tools", END]
)
workflow.add_edge("tools", "agent")

graph = workflow.compile()


# ---- Save a picture of the graph (ASCII if offline) ----
try:
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "pokemon_agent_graph.png").write_bytes(graph.get_graph().draw_mermaid_png())
    print(f"Saved {output_dir / 'pokemon_agent_graph.png'}")
except Exception as error:
    print("Could not render the PNG:", type(error).__name__)


# ---- Ask the agent ----
result = graph.invoke(
    {"messages": [HumanMessage(content="What is the Gengar's type?")]}
)

# Print the whole conversation: the question, the tool call, the tool result and the final answer
for message in result["messages"]:
    message.pretty_print()

# convert_to_ragas_messages reads tool calls from the OpenAI-style additional_kwargs["tool_calls"], but ChatOllama keeps
# them only in message.tool_calls. Without this copy the converted AI message has tool_calls=None, and RAGAS rejects
# the tool message that follows it.
for message in result["messages"]:
    if isinstance(message, AIMessage) and message.tool_calls:
        message.additional_kwargs["tool_calls"] = [
            {"function": {"name": call["name"], "arguments": json.dumps(call["args"])}}
            for call in message.tool_calls
        ]

ragas_trace = convert_to_ragas_messages(result["messages"])

print(ragas_trace)


# ---- Tool call accuracy ----
# The conversation to grade, plus the tool call the agent should have made
sample = MultiTurnSample(
    user_input=ragas_trace,
    reference_tool_calls=[
        ragas_messages.ToolCall(
            name="get_pokemon_type",
            args={"pokemon_name": "gengar"}
        )
    ],
)

scorer = ToolCallAccuracy()
scorer.llm = llm

# The notebook awaits this directly; a plain script has no running event loop, so asyncio.run starts one
print(asyncio.run(scorer.multi_turn_ascore(sample)))  # 1.0 when the agent's tool calls match the reference


# ---- Agent goal accuracy ----
# The same conversation, graded against the outcome the user wanted. The reference describes that outcome; the course
# uses the question itself, which a strict judge does not match to an answer (it scored 0.0 for a correct answer).
sample = MultiTurnSample(
    user_input=ragas_trace,
    reference="The type of Gengar is ghost/poison",
)

# Unlike tool call accuracy, this metric asks a model to judge whether the goal was reached, so it needs the wrapped LLM
scorer = AgentGoalAccuracyWithReference()
scorer.llm = LangchainLLMWrapper(llm)

print(asyncio.run(scorer.multi_turn_ascore(sample)))  # 1.0 if the judge decides the goal was reached, else 0.0





