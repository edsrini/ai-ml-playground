"""Agentic RAG with a local Ollama model: answer from a PDF, or search the web when the PDF is not relevant.

Needs an embedding model:  ollama pull nomic-embed-text
"""
import mlflow
import os
from pathlib import Path
from typing import List, Dict

from langgraph.checkpoint.memory import MemorySaver

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from IPython.display import Image, display
from tavily import TavilyClient

import truststore
truststore.inject_into_ssl()  # use the Windows certificate store so HTTPS calls work

tracking_uri = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(tracking_uri)

experiment = mlflow.set_experiment("srini's-new_agent")
from dotenv import load_dotenv
load_dotenv()


from config import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL, OLLAMA_MODEL

# Chat model; temperature 0 gives repeatable answers
llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.0,
    num_ctx=8192,
)

# ---- Load the PDF into a vector store ----
# Embedding model turns text into vectors; the same one must embed the chunks and the questions
embeddings_fn = OllamaEmbeddings(
    model=OLLAMA_EMBED_MODEL,
    base_url=OLLAMA_BASE_URL,
)

# In-memory vector store, rebuilt on every run
vector_store = Chroma(
    collection_name="srini-test",
    embedding_function=embeddings_fn,
)

# Use the course PDF, or the sample PDF if the course one is missing
file_path = Path(__file__).parent / "data" / "compact-guide-to-large-language-models.pdf"
sample_path = Path(__file__).parent / "data" / "sample-llm-notes.pdf"
if not file_path.exists() and sample_path.exists():
    print(f"Course PDF not found, using the stand-in {sample_path.name}")
    file_path = sample_path
if not file_path.exists():
    raise SystemExit(f"Put the course PDF at {file_path} and run again (or run data/make_sample_pdf.py).")
loader = PyPDFLoader(str(file_path))
pages = loader.load()

# Split into 1000-character chunks with 200 overlap, then embed and store them
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(pages)

_ = vector_store.add_documents(documents=all_splits)
print(f"Loaded {len(pages)} pages as {len(all_splits)} pieces into the vector store")


# ---- Graph state ----
# Shared data every node reads and updates; "messages" comes from MessagesState
class State(MessagesState):
    question: str
    documents: List[Document]
    search_required: str = "NO"
    answer: str


# Node: get the 4 chunks closest in meaning to the question
def retrieve(state: State):
    question = state["question"]
    retrieved_docs = vector_store.similarity_search(question)
    return {"documents": retrieved_docs}


# Node: ask the LLM if the chunks are relevant; YES means search the web
def evaluator(state: State):
    question = state["question"]
    documents = state["documents"]
    docs_content = "\n\n".join(doc.page_content for doc in documents)

    template = ChatPromptTemplate([
        ("system", "You are an assistant for evaluating context."),
        ("human", "Use the following pieces of retrieved context to determine if an "
                  "additional search is required. If the context is relevant to the "
                  "question, your response should be 'NO', i.e. no need for search. "
                  "If the context is not relevant, say 'YES', i.e. search the web for an answer. "
                  "Don't explain anything, if it's to search the web, say YES, otherwise, NO. "
                  "\n# Question: \n-> {question} "
                  "\n# Context: \n-> {context} "
                  "\n# Answer: "),
    ])

    # LCEL chain: fill prompt -> call model -> return plain text
    chain = template | llm | StrOutputParser()

    search_required = chain.invoke(
        {"context": docs_content, "question": question}
    )

    # Force the reply to exactly YES or NO
    search_required = "YES" if "YES" in search_required.upper() else "NO"
    return {"search_required": search_required}


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


# Node: the model researches the question, calling web_search when it needs to
def researcher(state: State):
    question = state["question"]
    messages = state["messages"]
    if not messages:  # first visit: start the conversation
        human_message = HumanMessage(
            "Conduct a web research to findout the answer to the following question: "
            f"```{question}```"
        )
        messages = [human_message]
    ai_message = llm_with_tools.invoke(messages)
    messages.append(ai_message)
    return {"messages": messages, "answer": ai_message.content}


# Node (Augment): put the chunks and the question into one prompt
def augment(state: State):
    question = state["question"]
    documents = state["documents"]
    docs_content = "\n\n".join(doc.page_content for doc in documents)

    template = ChatPromptTemplate([
        ("system", "You are an assistant for question-answering tasks."),
        ("human", "Use the following pieces of retrieved context to answer the question. "
                  "If you don't know the answer, just say that you don't know. "
                  "Use three sentences maximum and keep the answer concise. "
                  "\n# Question: \n-> {question} "
                  "\n# Context: \n-> {context} "
                  "\n# Answer: "),
    ])

    messages = template.invoke(
        {"context": docs_content, "question": question}
    ).to_messages()

    return {"messages": messages}


# Node (Generate): the model writes the answer from the augmented prompt
def generate(state: State):
    ai_message = llm.invoke(state["messages"])
    return {"answer": ai_message.content, "messages": ai_message}


# Router: YES goes to web research, NO answers from the PDF
def rag_router(state: State):
    search_required = state["search_required"]
    if search_required.lower() == "yes":
        return "researcher"

    return "augment"


# Router: run the tool if the model asked for one, otherwise finish
def tool_router(state: State):
    if state["messages"][-1].tool_calls:
        return "web_search"
    return END


# ---- Build the graph ----
workflow = StateGraph(State)

workflow.add_node("retrieve", retrieve)
workflow.add_node("augment", augment)
workflow.add_node("generate", generate)
workflow.add_node("evaluator", evaluator)
workflow.add_node("researcher", researcher)
workflow.add_node("web_search", ToolNode([web_search]))  # runs the tool the model asked for

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "evaluator")

# After grading: answer from the PDF or go research
workflow.add_conditional_edges(
    source="evaluator",
    path=rag_router,
    path_map=["augment", "researcher"]
)

# Research loop: researcher -> web_search -> researcher, until no more tool calls
workflow.add_conditional_edges(
    source="researcher",
    path=tool_router,
    path_map=["web_search", END]
)
workflow.add_edge("web_search", "researcher")

workflow.add_edge("augment", "generate")
workflow.add_edge("generate", END)

memory = MemorySaver()
rag_graph = workflow.compile(
    interrupt_before=["web_search"],  # must be a node name from add_node; pauses before the web search runs
    checkpointer = memory
)


# ---- Save a picture of the graph (ASCII if offline) ----
try:
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "agentic_rag_graph.png").write_bytes(rag_graph.get_graph().draw_mermaid_png())
    print(f"Saved {output_dir / 'agentic_rag_graph.png'}")
except Exception as error:
    print("Could not render the PNG:", type(error).__name__)
    rag_graph.get_graph().print_ascii()


# Newer langgraph calls on_interrupt when the graph pauses and on_resume when it continues, on every callback handler.
# This MLflow version has neither method, which logs "Error in MlflowLangchainTracer.on_interrupt callback". Accept and ignore both.
from mlflow.langchain.langchain_tracer import MlflowLangchainTracer
for _name in ("on_interrupt", "on_resume"):
    if not hasattr(MlflowLangchainTracer, _name):
        setattr(MlflowLangchainTracer, _name, lambda self, event: None)

mlflow.langchain.autolog()
config = {"recursion_limit": 15, "configurable": {"thread_id": "rag-demo1123"}}
# ---- Run it ----
# recursion_limit caps graph steps so the research loop can't run forever
output = rag_graph.invoke(
    input={"question": "is Ambeo Soundbar Max with Subwoofer good?"},
    config=config  # a checkpointer needs a thread_id
)
print(output["answer"])

state = rag_graph.get_state(config=config)
state.next

for message in output["messages"]:
    message.pretty_print()

print(output["search_required"])


# ---- Human approval: resume a paused run ----
# The graph pauses before web_search (interrupt_before). get_state shows which node is waiting.
state = rag_graph.get_state(config=config)
print("waiting before:", state.next)  # ('web_search',) means the run is paused; () means it already finished
if state.next:
    if input("Approve the web search? [y/N] ").strip().lower() == "y":
        output = rag_graph.invoke(None, config=config)  # None = carry on from the saved state of this thread_id
        print(output["answer"])
    else:
        print("Web search not approved: the run stays paused.")
