"""Vector store for RAG retrieval of brand examples.

Default backend is an in-memory cosine store (zero dependencies, great for
demos/tests). Set VECTOR_BACKEND=chromadb to use a persistent ChromaDB
collection backed by a local PersistentClient at ``./chroma_db``.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

from . import quality

logger = logging.getLogger(__name__)


@dataclass
class RetrievedExample:
    text: str
    metadata: Dict
    score: float


class MemoryVectorStore:
    """Tiny cosine-similarity store over example texts."""

    def __init__(self, embed_model: str = "all-MiniLM-L6-v2"):
        self.embed_model = embed_model
        self._items: List[Dict] = []

    def add(self, text: str, metadata: Optional[Dict] = None) -> None:
        self._items.append({"text": text, "metadata": metadata or {}})

    def add_many(self, examples: List[Dict]) -> None:
        for ex in examples:
            text = ex.get("text") or ex.get("caption") or ex.get("description", "")
            if text:
                self.add(text, {k: v for k, v in ex.items() if k != "text"})

    def retrieve(self, query: str, k: int = 3,
                 domain: Optional[str] = None) -> List[RetrievedExample]:
        scored = []
        for it in self._items:
            if domain and it["metadata"].get("domain") not in (None, domain):
                continue
            s = quality.similarity(query, it["text"], self.embed_model)
            scored.append(RetrievedExample(it["text"], it["metadata"], s))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]

    def __len__(self) -> int:
        return len(self._items)


class ChromaVectorStore:
    """Persistent ChromaDB-backed vector store mirroring the MemoryVectorStore API.

    Uses ``quality.embed_one()`` to produce embeddings when sentence-transformers
    is available; falls back to Chroma's built-in default embedding function
    otherwise so the store remains usable without extra packages.

    Args:
        collection_name: Name of the ChromaDB collection to use/create.
        persist_directory: Directory for the PersistentClient (default ``./chroma_db``).
        embed_model: Sentence-transformers model name forwarded to ``quality.embed_one``.
    """

    COLLECTION_NAME = "brand_examples"

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        persist_directory: str = "./chroma_db",
        embed_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        import chromadb  # intentionally un-caught — caller handles ImportError

        self.embed_model = embed_model
        self._client = chromadb.PersistentClient(path=persist_directory)
        # If sentence-transformers is available we supply embeddings manually so
        # we stay in control of the model; otherwise let Chroma pick its default.
        self._manual_embed = quality.embed_one("test", embed_model) is not None
        if self._manual_embed:
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        else:
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
            )

    # ------------------------------------------------------------------
    # Public interface (mirrors MemoryVectorStore)
    # ------------------------------------------------------------------

    def add(self, text: str, metadata: Optional[Dict] = None) -> None:
        """Add a single text with optional metadata to the collection."""
        doc_id = str(uuid.uuid4())
        meta = {k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                for k, v in (metadata or {}).items()}
        kwargs: Dict = dict(documents=[text], metadatas=[meta], ids=[doc_id])
        if self._manual_embed:
            emb = quality.embed_one(text, self.embed_model)
            if emb is not None:
                kwargs["embeddings"] = [emb]
        self._collection.add(**kwargs)

    def add_many(self, examples: List[Dict]) -> None:
        """Add a list of example dicts, extracting ``text``/``caption``/``description``."""
        for ex in examples:
            text = ex.get("text") or ex.get("caption") or ex.get("description", "")
            if text:
                self.add(text, {k: v for k, v in ex.items() if k != "text"})

    def retrieve(
        self, query: str, k: int = 3, domain: Optional[str] = None
    ) -> List[RetrievedExample]:
        """Return up to *k* nearest examples, optionally filtered by domain.

        Chroma returns L2 or cosine *distances* (lower == more similar).
        We convert to a similarity score in [0, 1] via ``1 - distance`` for
        cosine space (distances are already in [0, 2] for cosine but in
        practice [0, 1] when embeddings are normalised).
        """
        total = len(self)
        if total == 0:
            return []

        query_kwargs: Dict = dict(query_texts=[query], n_results=min(k, total))
        if self._manual_embed:
            emb = quality.embed_one(query, self.embed_model)
            if emb is not None:
                query_kwargs = dict(
                    query_embeddings=[emb], n_results=min(k, total)
                )
        if domain is not None:
            query_kwargs["where"] = {"domain": domain}

        try:
            results = self._collection.query(**query_kwargs)
        except Exception as exc:
            logger.warning("ChromaVectorStore.retrieve failed: %s", exc)
            return []

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        out: List[RetrievedExample] = []
        for doc, meta, dist in zip(docs, metas, distances):
            # Convert cosine distance [0, 2] → similarity [0, 1].
            score = float(max(0.0, min(1.0, 1.0 - dist)))
            out.append(RetrievedExample(doc, meta or {}, score))
        return out

    def __len__(self) -> int:
        return self._collection.count()


def get_vector_store(backend: str = "memory",
                     embed_model: str = "all-MiniLM-L6-v2",
                     config=None):
    """Factory that returns the requested vector store backend.

    Graceful degradation is enforced: if the requested backend is unavailable
    (missing package or construction error) the factory logs a warning and
    returns a :class:`MemoryVectorStore` instead of raising.
    """
    if backend == "supabase" and config is not None:
        from .db import get_supabase_vector_store  # lazy: avoids circular import
        store = get_supabase_vector_store(config)
        if store is not None:
            return store  # else fall through to memory
    if backend == "chromadb":
        try:
            return ChromaVectorStore(embed_model=embed_model)
        except ImportError:
            logger.warning(
                "chromadb package not installed — falling back to MemoryVectorStore"
            )
        except Exception as exc:
            logger.warning(
                "ChromaVectorStore construction failed (%s) — falling back to MemoryVectorStore",
                exc,
            )
    return MemoryVectorStore(embed_model)
