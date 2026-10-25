"""Observability for an agent with MLflow: send what the agent does to a tracking server and look at it in the browser.

The course notebook uses ChatOpenAI; here ChatOllama fills the same role, so the agent runs on the local model.
Needs the standalone MLflow server running (mlops/mlflow-server, then `docker compose up -d`) at http://127.0.0.1:5000
"""
import mlflow
import os
from typing import Dict

from mlflow.entities import experiment
from tavily import TavilyClient
from langchain_ollama import ChatOllama  # local replacement for ChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage
from IPython.display import Image, display

tracking_uri = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(tracking_uri)

experiment = mlflow.set_experiment("srini's-agent")

@mlflow.trace
def add(a,b):
    return a+b
add(1,3)

print(f"MLflow {mlflow.__version__} -> {tracking_uri}: {len(mlflow.search_experiments())} experiment(s) on the server")  # connection check
