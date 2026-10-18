"""Tools that call web APIs: a keyless public quotes API and a Tavily web search, in the same function style as 10 and 11.

A @tool turns a plain function into something an LLM can call. Its docstring becomes the description the model reads.
Keys come from a .env file (see .env.example), never from the code.
"""
import os
from typing import Dict

import requests
import truststore
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from tavily import TavilyClient

from config import OLLAMA_BASE_URL, OLLAMA_MODEL

truststore.inject_into_ssl()  # trust the Windows certificate store, so HTTPS works behind a proxy or antivirus that re-signs traffic
load_dotenv()  # reads .env into environment variables; returns True if a .env file was found


# Tool 1: a public API that needs no key
@tool
def random_quote_tool() -> Dict:
    """
    Return a random quote and the author who said it
    """
    response = requests.get("https://dummyjson.com/quotes/random", timeout=10)  # timeout so a dead API cannot hang us
    response.raise_for_status()  # turn an HTTP error into an exception instead of returning an error page
    return response.json()


print(random_quote_tool.invoke({}))  # tools are called with a dict of their arguments; {} because there are none

tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")  # the key lives in .env, never in the code
)

@tool
def web_search_tool(query: str) -> Dict:
    """
    Search the web for current information and return the top results
    """
    return tavily_client.search(query, max_results=3)

result = web_search_tool.invoke(
    {
        "query": "Who wrote A Man Should Never Neglect His Family For Business?"
    }
)

print(result)

tools = [random_quote_tool, web_search_tool]

# The local model. num_ctx=8192 keeps it on the GPU; temperature=0 makes tool choices repeatable.
llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0, num_ctx=8192)

# bind_tools shows the model each tool's name, description and args. It still only *asks* for a tool;
# our code has to run it.
llm_with_tools = llm.bind_tools(tools)
