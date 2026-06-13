"""Tests for the ChromaDB vector store backend.

Two separate test groups:
  1. Offline / no-chromadb path (always runs): verifies that
     ``get_vector_store("chromadb")`` falls back to MemoryVectorStore when
     chromadb is not installed — zero extra dependencies required.
  2. Online / chromadb-installed path (skipped when chromadb is absent):
     exercises ChromaVectorStore.add / retrieve and asserts the
     RetrievedExample shape.
"""
from __future__ import annotations

import importlib

from src.vector_store import MemoryVectorStore, RetrievedExample, get_vector_store

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chromadb_available() -> bool:
    """Return True if the chromadb package can be imported."""
    try:
        importlib.import_module("chromadb")
        return True
    except ImportError:
        return False


CHROMADB_AVAILABLE = _chromadb_available()


# ---------------------------------------------------------------------------
# 1. Graceful fallback — MUST pass even when chromadb is not installed
# ---------------------------------------------------------------------------

def test_chromadb_backend_falls_back_to_memory_when_not_installed():
    """get_vector_store('chromadb') must never crash; returns MemoryVectorStore
    when chromadb is absent."""
    if CHROMADB_AVAILABLE:
        # chromadb IS installed — we can't test the ImportError path, but we
        # can still verify the factory returns something with the correct interface.
        vs = get_vector_store("chromadb")
        assert hasattr(vs, "add") and hasattr(vs, "retrieve") and hasattr(vs, "add_many")
        return

    # chromadb NOT installed — factory must return a MemoryVectorStore.
    vs = get_vector_store("chromadb")
    assert isinstance(vs, MemoryVectorStore), (
        f"Expected MemoryVectorStore fallback, got {type(vs).__name__}"
    )


# ---------------------------------------------------------------------------
# 2. Full ChromaVectorStore tests — skipped offline
# ---------------------------------------------------------------------------

def test_chromadb_add_and_retrieve():
    """Add two docs, retrieve top-1, assert RetrievedExample shape."""
    chromadb = pytest_importorskip_chromadb()

    import tempfile

    from src.vector_store import ChromaVectorStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ChromaVectorStore(
            collection_name="test_col",
            persist_directory=tmpdir,
        )

        store.add("Our team shipped a brand-new RAG pipeline today.",
                  {"domain": "social"})
        store.add("Quarterly revenue exceeded expectations by 20%.",
                  {"domain": "email"})

        assert len(store) == 2

        results = store.retrieve("we shipped a new feature", k=1)
        assert len(results) == 1

        ex = results[0]
        assert isinstance(ex, RetrievedExample), f"Expected RetrievedExample, got {type(ex)}"
        assert isinstance(ex.text, str) and ex.text
        assert isinstance(ex.metadata, dict)
        assert isinstance(ex.score, float)
        assert 0.0 <= ex.score <= 1.0


def test_chromadb_domain_filter():
    """Retrieve with domain filter returns only matching-domain docs."""
    pytest_importorskip_chromadb()

    import tempfile

    from src.vector_store import ChromaVectorStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ChromaVectorStore(
            collection_name="test_domain_filter",
            persist_directory=tmpdir,
        )

        store.add("Social post about shipping features.", {"domain": "social"})
        store.add("Email about quarterly business results.", {"domain": "email"})

        social_results = store.retrieve("features", k=5, domain="social")
        for ex in social_results:
            assert ex.metadata.get("domain") == "social", (
                f"Domain filter returned non-social doc: {ex.metadata}"
            )


def test_chromadb_retrieve_empty_store():
    """Retrieving from an empty store returns an empty list without errors."""
    pytest_importorskip_chromadb()

    import tempfile

    from src.vector_store import ChromaVectorStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ChromaVectorStore(
            collection_name="test_empty",
            persist_directory=tmpdir,
        )
        assert len(store) == 0
        results = store.retrieve("anything", k=3)
        assert results == []


def test_chromadb_add_many():
    """add_many populates the store correctly."""
    pytest_importorskip_chromadb()

    import tempfile

    from src.vector_store import ChromaVectorStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ChromaVectorStore(
            collection_name="test_add_many",
            persist_directory=tmpdir,
        )
        examples = [
            {"text": "First example text.", "domain": "social"},
            {"caption": "Second example via caption key.", "domain": "email"},
            {"description": "Third example via description key.", "domain": "product"},
        ]
        store.add_many(examples)
        assert len(store) == 3


# ---------------------------------------------------------------------------
# Internal helper for skipping when chromadb is absent
# ---------------------------------------------------------------------------

def pytest_importorskip_chromadb():
    """Skip the calling test if chromadb is not installed."""
    import pytest
    return pytest.importorskip("chromadb")
