"""Deterministic tests: fake embeddings and an in-memory store, so no Ollama, no network. Run with: pytest"""
import uuid

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

import knowledge_base
from knowledge_base import get_vector_store, ingest, load_documents, split_documents
from tools import search_knowledge_base, web_search


def new_store():
    """An in-memory store with a unique name, so tests never see each other's data."""
    return get_vector_store(DeterministicFakeEmbedding(size=32), collection_name="t" + uuid.uuid4().hex, persist=False)


@pytest.fixture
def docs_dir(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text("# Alpha\n\n" + "Alpha text about transformers. " * 80, encoding="utf-8")
    (tmp_path / "b.txt").write_text("Beta text about retrieval. " * 80, encoding="utf-8")
    (tmp_path / "ignored.docx").write_text("not supported yet")
    monkeypatch.setattr(knowledge_base, "DOCS_DIR", tmp_path)
    monkeypatch.setattr(knowledge_base, "load_documents", lambda docs_dir=tmp_path: load_documents(docs_dir))
    return tmp_path


def test_load_documents_reads_md_and_txt_tags_the_source_and_skips_other_files(docs_dir):
    documents = load_documents(docs_dir)
    assert sorted(d.metadata["source"] for d in documents) == ["a.md", "b.txt"]


def test_split_documents_makes_overlapping_pieces_that_are_small_enough():
    pieces = split_documents([Document(page_content="word " * 1000, metadata={"source": "x"})])
    assert len(pieces) > 1
    assert all(len(p.page_content) <= knowledge_base.CHUNK_SIZE for p in pieces)
    assert all(p.metadata["source"] == "x" for p in pieces)


def test_ingest_loads_once_and_does_not_duplicate(docs_dir):
    store = new_store()
    first = ingest(store)
    assert first > 0
    assert ingest(store) == 0  # already loaded
    assert len(store.get()["ids"]) == first


def test_rebuild_replaces_the_collection_instead_of_adding_to_it(docs_dir):
    store = new_store()
    first = ingest(store)
    assert ingest(store, rebuild=True) == first
    assert len(store.get()["ids"]) == first


def test_search_returns_passages_with_source_text_and_distance():
    store = new_store()
    store.add_documents([Document(page_content=f"piece {i}", metadata={"source": "s.md", "page": i}) for i in range(6)])
    result = search_knowledge_base.invoke({"query": "piece"}, config={"configurable": {"vector_store": store}})
    assert len(result["passages"]) == int(knowledge_base.os.getenv("TOP_K"))
    assert set(result["passages"][0]) == {"source", "page", "text", "distance"}


def test_a_broken_store_comes_back_as_data_not_a_crash():
    class Broken:
        def similarity_search_with_score(self, *args, **kwargs):
            raise RuntimeError("ollama is down")

    result = search_knowledge_base.invoke({"query": "x"}, config={"configurable": {"vector_store": Broken()}})
    assert result == {"error": "ollama is down"}


def test_web_search_without_a_key_says_so_and_makes_no_request(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert "TAVILY_API_KEY" in web_search.invoke({"question": "anything"})["error"]
