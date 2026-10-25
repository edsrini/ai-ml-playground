"""Agentic RAG with a local model: load documents, embed them into a vector store, and let an agent search them.

The course notebook uses ChatOpenAI and OpenAIEmbeddings; here the same two roles are filled by local Ollama models
(ChatOllama for the answers, OllamaEmbeddings for the vectors), so no API key is needed.

Needs an embedding model:  ollama pull nomic-embed-text

Same as 14_agentic_rag.py, except the evaluator uses llm.with_structured_output(Grade) instead of
template | llm | StrOutputParser(): the model returns a real True/False, so there is no "YES"/"NO" string to clean up.
"""
import os
from pathlib import Path
from typing import List, Dict
from langchain_ollama import ChatOllama, OllamaEmbeddings  # local replacements for ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from IPython.display import Image, display
from tavily import TavilyClient

import truststore
truststore.inject_into_ssl()  # trust the Windows certificate store, so HTTPS calls (Tavily) work

from dotenv import load_dotenv
load_dotenv()

from config import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL, OLLAMA_MODEL

# Instantiate your chat model (the course uses ChatOpenAI(model="gpt-4o-mini", temperature=0.0))
llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.0,
    num_ctx=8192,  # keeps the model on the GPU; not needed with a hosted model
)

# ---- Loading Documents to VectorDB ----
# Why a separate embedding model: the vector store searches by meaning, so every piece of text is first turned into a
# list of numbers (a vector) and texts with similar meaning get similar vectors. A chat model like qwen3:4b writes
# text; it is poor at making vectors that match by meaning. An embedding model is small and made for exactly this.
# The SAME embedding model must embed the pieces now and the questions later, because vectors from two different
# models cannot be compared.
embeddings_fn = OllamaEmbeddings(  # the course uses OpenAIEmbeddings(model="text-embedding-3-large")
    model=OLLAMA_EMBED_MODEL,
    base_url=OLLAMA_BASE_URL,
)

vector_store = Chroma(
    collection_name="srini-test",
    embedding_function=embeddings_fn,  # kept in memory, so it is rebuilt on every run
)

file_path = Path(__file__).parent / "data" / "compact-guide-to-large-language-models.pdf"
sample_path = Path(__file__).parent / "data" / "sample-llm-notes.pdf"  # made by data/make_sample_pdf.py
if not file_path.exists() and sample_path.exists():  # no course PDF yet: use the stand-in so the lab still runs
    print(f"Course PDF not found, using the stand-in {sample_path.name}")
    file_path = sample_path
if not file_path.exists():  # the PDF is not in the repo: download it from the course and put it in the data folder
    raise SystemExit(f"Put the course PDF at {file_path} and run again (or run data/make_sample_pdf.py).")
loader = PyPDFLoader(str(file_path))
pages = loader.load()  # the notebook uses `async for page in loader.alazy_load()`, which only works inside a notebook

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # 1,000-character pieces that overlap by 200
all_splits = text_splitter.split_documents(pages)

_ = vector_store.add_documents(documents=all_splits)  # embeds every piece and stores the vectors
print(f"Loaded {len(pages)} pages as {len(all_splits)} pieces into the vector store")


# ---- Graph state and the retrieval node ----
class State(MessagesState):
    question: str
    documents: List[Document]
    search_required: bool  # True when the PDF pieces do not answer the question and the web must be searched
    answer: str


# Node: find the pieces of the PDF closest in meaning to the question
def retrieve(state: State):
    question = state["question"]
    retrieved_docs = vector_store.similarity_search(question)  # returns the 4 nearest pieces by default
    return {"documents": retrieved_docs}


# The shape of the evaluator's answer. with_structured_output makes the model fill in this class,
# so the code gets a real True/False instead of a "YES"/"NO" string it has to clean up.
class Grade(BaseModel):
    needs_search: bool = Field(
        description="True if the context does NOT answer the question and the web must be searched; "
                    "False if the context is relevant to the question."
    )


# Node: ask the model whether the retrieved pieces are relevant enough, or a web search is needed
def evaluator(state: State):
    question = state["question"]
    documents = state["documents"]
    docs_content = "\n\n".join(doc.page_content for doc in documents)  # the retrieved pieces as one block of text

    if not docs_content.strip():  # nothing retrieved: no need to ask the model, go straight to the web
        return {"search_required": True}

    template = ChatPromptTemplate([
        ("system", "You are an assistant for evaluating context."),
        ("human", "Decide whether the retrieved context below is relevant to the question "
                  "and enough to answer it. If it is, no web search is needed. "
                  "If it is not, a web search is needed."
                  "\n# Question: \n-> {question} "
                  "\n# Context: \n-> {context} "),
    ])

    chain = template | llm.with_structured_output(Grade)  # prompt, then model, then a Grade object out

    try:
        grade = chain.invoke({"context": docs_content, "question": question})
        search_required = grade.needs_search
    except Exception as error:  # a small local model can return broken JSON: treat it as "no search" instead of crashing
        print("Evaluator could not read the model's answer:", type(error).__name__)
        search_required = False

    return {"search_required": search_required}


# ---- Web search fallback ----
@tool
def web_search(question: str) -> Dict:
    """
    Return top search results for a given search query
    """
    try:
        tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))  # the key lives in .env, never in the code
        response = tavily_client.search(question)
    except Exception as error:  # no key, or Tavily is down: return the problem as data so the graph does not crash
        return {"error": str(error)}
    return response


llm_with_tools = llm.bind_tools([web_search])  # the model may now ask for a web search by name


# Node: when the PDF could not answer, ask the model to research the question on the web
def researcher(state: State):
    question = state["question"]
    messages = state["messages"]
    if not messages:  # first visit: start the research conversation
        human_message = HumanMessage(
            "Conduct a web research to findout the answer to the following question: "
            f"```{question}```"
        )
        messages = [human_message]
    # The remainder of this node was cut off in the video frame: the model reads the messages and either asks
    # for web_search or answers. Returning the new messages lets add_messages append them to the state.
    ai_message = llm_with_tools.invoke(messages)
    messages.append(ai_message)
    return {"messages": messages, "answer": ai_message.content}


# Node: the "augment" step of RAG: put the retrieved pieces and the question into one prompt for the model
def augment(state: State):
    question = state["question"]
    documents = state["documents"]
    docs_content = "\n\n".join(doc.page_content for doc in documents)  # the retrieved pieces as one block of text

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
    ).to_messages()  # turn the filled template into a list of chat messages

    return {"messages": messages}


# Node: the "generate" step of RAG: the model writes the answer from the prompt that augment built
def generate(state: State):
    ai_message = llm.invoke(state["messages"])
    return {"answer": ai_message.content, "messages": ai_message}


# Router: after the evaluator, search the web if the PDF pieces were not relevant, otherwise go straight to augment
def rag_router(state: State):  # the video hints MessagesState, but that hides search_required here: KeyError
    if state["search_required"]:  # already a True/False, so no string comparison
        return "researcher"

    return "augment"


# Router: after the researcher, run the web search if the model asked for it, otherwise the research is done.
# (Not visible in the video frames: written in the same style as should_continue in the text2sql agent.)
def tool_router(state: State):
    if state["messages"][-1].tool_calls:
        return "web_search"
    return END


# ---- The graph ----
workflow = StateGraph(State)

workflow.add_node("retrieve", retrieve)
workflow.add_node("augment", augment)
workflow.add_node("generate", generate)
workflow.add_node("evaluator", evaluator)
workflow.add_node("researcher", researcher)
workflow.add_node("web_search", ToolNode([web_search]))  # runs the web_search tool when the researcher asks for it

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "evaluator")

workflow.add_conditional_edges(
    source="evaluator",
    path=rag_router,
    path_map=["augment", "researcher"]
)

workflow.add_conditional_edges(
    source="researcher",
    path=tool_router,
    path_map=["web_search", END]
)
workflow.add_edge("web_search", "researcher")  # after a search, the researcher reads the results

workflow.add_edge("augment", "generate")
workflow.add_edge("generate", END)

rag_graph = workflow.compile()  # the video calls this `graph`


# ---- Draw the graph ----
# In a notebook display(Image(...)) shows the picture; in a script we save it as a file instead.
# mermaid.ink renders the PNG over the internet, so fall back to an ASCII drawing if that fails.
try:
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "agentic_rag_graph.png").write_bytes(rag_graph.get_graph().draw_mermaid_png())
    print(f"Saved {output_dir / 'agentic_rag_graph.png'}")
except Exception as error:
    print("Could not render the PNG:", type(error).__name__)
    rag_graph.get_graph().print_ascii()


# ---- Run it ----
output = rag_graph.invoke(
    input={"question": "What is Tomatoe?"},
    config={"recursion_limit": 15},  # hard cap on graph steps: stops a researcher that keeps searching
)
print(output["answer"])

for message in output["messages"]:
    message.pretty_print()

print(output["search_required"])


