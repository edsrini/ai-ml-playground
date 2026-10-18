"""Three log tools an LLM can call. None of them hardcodes a connection: the Elasticsearch client is read from
the run config (config["configurable"]["es_client"]), so a test can pass a stub instead.

Plain code decides what is allowed. The model only chooses which tool to call and fills in the arguments.
"""
import os
from typing import Dict

from dotenv import load_dotenv
from elasticsearch import ApiError, TransportError
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

load_dotenv()  # reads .env into environment variables
LOG_INDEX = os.getenv("LOG_INDEX")
MAX_LINES = 20  # most log lines one answer may return


def build_filters(start, end, service, level, error_code):
    """The Elasticsearch filters shared by both search tools: the time range plus optional service, level and error code."""
    filters = [{"range": {"@timestamp": {"gte": start, "lte": end}}}]
    if service:
        filters.append({"term": {"service": service}})
    if level:
        filters.append({"term": {"level": level.upper()}})
    if error_code:
        filters.append({"term": {"error_code": error_code.upper()}})
    return filters


@tool
def get_log_fields(config: RunnableConfig) -> Dict:
    """Return the fields every log line has, with their types."""
    try:
        mapping = config["configurable"]["es_client"].indices.get_mapping(index=LOG_INDEX)
    except (ApiError, TransportError) as error:  # errors come back as data so the model can react
        return {"error": str(error)}
    fields = mapping[LOG_INDEX]["mappings"]["properties"]
    return {name: info["type"] for name, info in fields.items()}


@tool
def log_stats(start: str, end: str, group_by: str, service: str = "", level: str = "", error_code: str = "",
              *, config: RunnableConfig) -> Dict:
    """Count logs between start and end (ISO timestamps like 2026-10-08T00:00:00+00:00), grouped by one field:
    service, level, error_code, endpoint or host. Optional filters: service, level (INFO, WARN, ERROR) and error_code.
    Use it to find which service or error code is behind a problem."""
    try:
        result = config["configurable"]["es_client"].search(
            index=LOG_INDEX, size=0, track_total_hits=True,
            query={"bool": {"filter": build_filters(start, end, service, level, error_code)}},
            aggs={"groups": {"terms": {"field": group_by, "size": 10}}})
    except (ApiError, TransportError) as error:
        return {"error": str(error)}
    buckets = result["aggregations"]["groups"]["buckets"]
    return {"total": result["hits"]["total"]["value"], "counts": {b["key"]: b["doc_count"] for b in buckets}}


@tool
def search_logs(start: str, end: str, query: str = "", service: str = "", level: str = "", error_code: str = "",
                *, config: RunnableConfig) -> Dict:
    """Return the newest log lines between start and end (ISO timestamps like 2026-10-08T00:00:00+00:00).
    Optional filters: service, level (INFO, WARN, ERROR), error_code and query (words to find in the message).
    Use it to read example lines once log_stats has shown where the problem is."""
    filters = build_filters(start, end, service, level, error_code)
    if query:
        filters.append({"match": {"message": query}})
    try:
        result = config["configurable"]["es_client"].search(
            index=LOG_INDEX, size=MAX_LINES, track_total_hits=True, query={"bool": {"filter": filters}},
            sort=[{"@timestamp": "desc"}],
            source=["@timestamp", "service", "level", "error_code", "trace_id", "message"])
    except (ApiError, TransportError) as error:
        return {"error": str(error)}
    return {"total": result["hits"]["total"]["value"], "lines": [hit["_source"] for hit in result["hits"]["hits"]]}


TOOLS = [get_log_fields, log_stats, search_logs]
