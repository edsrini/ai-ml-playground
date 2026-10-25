"""Two tools an LLM can call. Neither hardcodes a connection: the vector store is read from the run config
(config["configurable"]["vector_store"]), so a test can pass a stand-in store.

Plain code decides what comes back: how many passages, what fields, and a clear error instead of a crash.
The model only chooses which tool to call and what to search for.
"""
import os
from typing import Dict

import truststore
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from tavily import TavilyClient

truststore.inject_into_ssl()  # trust the Windows certificate store, so HTTPS calls (Tavily) work
load_dotenv()  # reads .env into environment variables
TOP_K = int(os.getenv("TOP_K"))  # how many passages one search returns


@tool
def search_knowledge_base(query: str, config: RunnableConfig) -> Dict:
    """Search the knowledge base (the company's own documents) and return the passages closest in meaning to the
    query, each with its source file, page and a distance (lower means closer). Use this first."""
    try:
        found = config["configurable"]["vector_store"].similarity_search_with_score(query, k=TOP_K)
    except Exception as error:  # e.g. Ollama is down: hand the problem back as data so the model can react
        return {"error": str(error)}
    return {"passages": [
        {"source": doc.metadata.get("source"), "page": doc.metadata.get("page"), "text": doc.page_content,
         "distance": round(float(distance), 2)}
        for doc, distance in found
    ]}


@tool
def web_search(question: str) -> Dict:
    """Search the web. Use only when the knowledge base does not contain the answer."""
    if not os.getenv("TAVILY_API_KEY"):
        return {"error": "TAVILY_API_KEY is not set in .env, so the web cannot be searched"}
    try:
        return TavilyClient(api_key=os.getenv("TAVILY_API_KEY")).search(question, max_results=3)
    except Exception as error:
        return {"error": str(error)}


TOOLS = [search_knowledge_base, web_search]
