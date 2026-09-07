"""DocStore ingest: placeholder-count regression + soft-disable behavior.

Regression test for the live bug where the chunk INSERT named 6 columns
with 5 placeholders, failing every ingest and permanently disabling RAG.
"""

import asyncio

import docs
from docs import DocStore


class FakeEmbeddings:
    def __init__(self, *a, **k):
        pass

    def embed_query(self, text):
        return [0.5] * 8

    def embed_documents(self, texts):
        return [[float((len(t) % 7) + 1)] * 8 for t in texts]


def _store(tmp_path, monkeypatch):
    monkeypatch.setattr(docs, "OllamaEmbeddings", FakeEmbeddings)
    return DocStore(tmp_path / "docs.db")


def test_ingest_search_roundtrip(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    body = ("The quick brown fox jumps over the lazy dog. " * 40).encode()
    res = asyncio.run(store.ingest("u1", "notes.txt", body))
    assert res["status"] == "added" and res["chunks"] >= 1, res
    assert store.enabled
    hits = asyncio.run(store.search("u1", "quick brown fox"))
    assert hits and hits[0][0] == "notes.txt", hits
    docs_list = asyncio.run(store.list_docs("u1"))
    assert docs_list and docs_list[0]["source"] == "notes.txt"
    text, _truncated = asyncio.run(store.doc_text("u1", "notes.txt"))
    assert "quick brown fox" in text


def test_reingest_dedupes(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    body = ("hello world, this is a test document. " * 20).encode()
    first = asyncio.run(store.ingest("u1", "a.txt", body))
    assert first["status"] == "added"
    again = asyncio.run(store.ingest("u1", "a.txt", body))
    assert again == {"status": "unchanged", "chunks": 0}


def test_doc_error_keeps_store_enabled(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    bad = asyncio.run(store.ingest("u1", "file.exe", b"x" * 100))
    assert bad["status"] == "error"
    assert store.enabled  # document-level failure must not brick RAG
    good = asyncio.run(store.ingest("u1", "ok.txt", ("fine content here. " * 20).encode()))
    assert good["status"] == "added"


def test_db_error_disables_store(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    store._conn.close()  # simulate store-level breakage
    res = asyncio.run(store.ingest("u1", "a.txt", ("some content here. " * 20).encode()))
    assert res["status"] == "error"
    assert not store.enabled
