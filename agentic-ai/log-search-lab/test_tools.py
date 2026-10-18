"""Deterministic tests for the log tools with a stub Elasticsearch client: no server, no LLM. Run with: pytest"""
import pytest
from elasticsearch import TransportError

from tools import MAX_LINES, build_filters, get_log_fields, log_stats, search_logs

DAY = {"start": "2026-10-08T00:00:00+00:00", "end": "2026-10-09T00:00:00+00:00"}


class StubES:
    """Remembers what it was asked and answers with fixed data."""

    def __init__(self):
        self.last = None

    def search(self, **kwargs):
        self.last = kwargs
        return {"hits": {"total": {"value": 3}, "hits": [{"_source": {"message": "boom", "level": "ERROR"}}]},
                "aggregations": {"groups": {"buckets": [{"key": "claims-ingest", "doc_count": 3}]}}}


class DownES:
    """A server that is not reachable."""

    def __init__(self):
        self.indices = self  # so es.indices.get_mapping(...) lands on the method below

    def search(self, **kwargs):
        raise TransportError("connection refused")

    def get_mapping(self, **kwargs):
        raise TransportError("connection refused")


def run(tool, es, **args):
    return tool.invoke(args, config={"configurable": {"es_client": es}})


def test_filters_always_have_the_time_range():
    filters = build_filters("a", "b", "", "", "")
    assert filters == [{"range": {"@timestamp": {"gte": "a", "lte": "b"}}}]


def test_service_level_and_error_code_filters_are_added_and_uppercased():
    filters = build_filters("a", "b", "claims-ingest", "error", "db_timeout")
    assert {"term": {"service": "claims-ingest"}} in filters and {"term": {"level": "ERROR"}} in filters
    assert {"term": {"error_code": "DB_TIMEOUT"}} in filters


def test_search_returns_the_total_and_the_lines():
    es = StubES()
    result = run(search_logs, es, **DAY, level="error", query="timed out")
    assert result == {"total": 3, "lines": [{"message": "boom", "level": "ERROR"}]}
    assert {"match": {"message": "timed out"}} in es.last["query"]["bool"]["filter"]


def test_search_never_returns_more_than_the_cap():
    es = StubES()
    run(search_logs, es, **DAY)
    assert es.last["size"] == MAX_LINES


def test_stats_returns_counts_by_the_chosen_field():
    es = StubES()
    result = run(log_stats, es, **DAY, group_by="service")
    assert result == {"total": 3, "counts": {"claims-ingest": 3}}
    assert es.last["aggs"]["groups"]["terms"]["field"] == "service"


@pytest.mark.parametrize("tool, args", [
    (search_logs, DAY), (log_stats, {**DAY, "group_by": "service"}), (get_log_fields, {}),
])
def test_a_server_failure_comes_back_as_data(tool, args):
    assert "error" in run(tool, DownES(), **args)
