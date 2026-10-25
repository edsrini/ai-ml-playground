"""Knowledge-base agent: a local model decides when to search the documents or the web and answers from what it finds.

Same graph shape as text2sql-agent: build the prompt, let the model call tools, run the tools, repeat.

Usage:  python agent.py "What is a transformer?"
"""
import asyncio
import os
import sys
from pathlib import Path
from typing import Annotated, TypedDict

import truststore

truststore.inject_into_ssl()  # trust the Windows certificate store, so the graph picture can be drawn over HTTPS
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langgraph.errors import GraphRecursionError
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from knowledge_base import get_vector_store, ingest
from tools import TOOLS

from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import MessagesState
from IPython.display import Image, display

# Instantiate your chat model
load_dotenv()
llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL"),
    base_url=os.getenv("OLLAMA_BASE_URL"),
    temperature=0.0,
    num_ctx=8192,  # keeps the model on the GPU
)

embeddings_fn = OllamaEmbeddings(
    model=os.getenv("OLLAMA_EMBED_MODEL"),  # nomic-embed-text, set in .env
    base_url=os.getenv("OLLAMA_BASE_URL"),
)

# The course builds an empty in-memory store here: Chroma(collection_name="...", embedding_function=embeddings_fn).
# This project opens its saved collection instead (same Chroma, with the name and folder from .env), so the documents
# In-memory vector store, rebuilt on every run
vector_store = Chroma(
    collection_name="srini-test",
    embedding_function=embeddings_fn,
)

file_paths = [  # add more files here: every file in this list is loaded into the same vector store
    "compact-guide-to-large-language-models.pdf",
    "open-source-models.pdf",
]

async def load_pages():  # a notebook allows `async for` at the top level; a .py file needs it inside an async function
    pages = []
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        async for page in loader.alazy_load():
            pages.append(page)  # each page carries its file name in page.metadata["source"]
    return pages


pages = asyncio.run(load_pages())  # starts the async function and waits for it

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

all_splits = text_splitter.split_documents(pages)

_ = vector_store.add_documents(documents=all_splits)

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    documents: List[Document]
    answer: str

def retrieve(state: State):
    question = state["question"]
    retrieved_docs = vector_store.similarity_search(question)
    return {"documents": retrieved_docs}

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

def generate(state: State):
    ai_message = llm.invoke(state["messages"])
    return {"answer": ai_message.content, "messages": ai_message}

workflow = StateGraph(State)

workflow.add_node("retrieve", retrieve)
workflow.add_node("augment", augment)
workflow.add_node("generate", generate)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "augment")
workflow.add_edge("augment", "generate")
workflow.add_edge("generate", END)

graph = workflow.compile()

# ---- Save a picture of the graph (ASCII if offline) ----
try:
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "ai-knowledgebase-agent.png").write_bytes(graph.get_graph().draw_mermaid_png())
    print(f"Saved {output_dir / 'ai-knowledgebase-agent.png'}")
except Exception as error:
    print("Could not render the PNG:", type(error).__name__)
    graph.get_graph().print_ascii()


output = graph.invoke(
    {"question": "what is the document all about?"}
)

print(output["answer"])
