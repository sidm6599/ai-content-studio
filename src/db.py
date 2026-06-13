"""Supabase persistence layer (optional).

Two responsibilities:
  1. HistoryStore  — persist generations so the app "doesn't forget" across
     sessions/refreshes (tables: `generations`).
  2. SupabaseVectorStore — store brand examples with pgvector embeddings and
     retrieve them for RAG (table: `brand_examples` + RPC `match_brand_examples`).

Everything degrades gracefully: with no Supabase creds (or no `supabase`
package) you get an in-memory NullHistoryStore, and the app keeps working.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from . import quality
from .vector_store import MemoryVectorStore, RetrievedExample

logger = logging.getLogger(__name__)


def _make_client(url: str, key: str):
    try:
        from supabase import create_client
        return create_client(url, key)
    except Exception as e:  # package missing or bad creds
        logger.warning("Supabase client unavailable: %s", e)
        return None


# ----------------------------------------------------------------------
# History persistence
# ----------------------------------------------------------------------
class NullHistoryStore:
    """In-memory fallback (lost on restart) — keeps the app working offline."""
    enabled = False

    def __init__(self):
        self._rows: List[Dict[str, Any]] = []

    def save_generation(self, row: Dict[str, Any]) -> None:
        self._rows.insert(0, row)

    def list_generations(self, limit: int = 20) -> List[Dict[str, Any]]:
        return self._rows[:limit]


class SupabaseHistoryStore:
    enabled = True

    def __init__(self, client, table: str = "generations"):
        self.client = client
        self.table = table

    def save_generation(self, row: Dict[str, Any]) -> None:
        try:
            self.client.table(self.table).insert(row).execute()
        except Exception as e:
            logger.error("save_generation failed: %s", e)

    def list_generations(self, limit: int = 20) -> List[Dict[str, Any]]:
        try:
            res = (self.client.table(self.table)
                   .select("*").order("created_at", desc=True)
                   .limit(limit).execute())
            return res.data or []
        except Exception as e:
            logger.error("list_generations failed: %s", e)
            return []


def get_history_store(config) -> Any:
    if not (config.persist_history and config.supabase_enabled()):
        return NullHistoryStore()
    client = _make_client(config.supabase_url, config.supabase_key)
    return SupabaseHistoryStore(client) if client else NullHistoryStore()


# ----------------------------------------------------------------------
# pgvector-backed RAG store (same interface as MemoryVectorStore)
# ----------------------------------------------------------------------
class SupabaseVectorStore:
    def __init__(self, client, embed_model: str = "all-MiniLM-L6-v2",
                 table: str = "brand_examples"):
        self.client = client
        self.embed_model = embed_model
        self.table = table

    def add(self, text: str, metadata: Optional[Dict] = None) -> None:
        emb = quality.embed_one(text, self.embed_model)
        row = {"content": text, "metadata": metadata or {}}
        if emb is not None:
            row["embedding"] = emb
        try:
            self.client.table(self.table).insert(row).execute()
        except Exception as e:
            logger.error("vector add failed: %s", e)

    def add_many(self, examples: List[Dict]) -> None:
        for ex in examples:
            text = ex.get("text") or ex.get("caption") or ex.get("description", "")
            if text:
                self.add(text, {k: v for k, v in ex.items() if k != "text"})

    def retrieve(self, query: str, k: int = 3,
                 domain: Optional[str] = None) -> List[RetrievedExample]:
        emb = quality.embed_one(query, self.embed_model)
        # Preferred path: pgvector similarity via RPC
        if emb is not None:
            try:
                res = self.client.rpc("match_brand_examples", {
                    "query_embedding": emb,
                    "match_count": k,
                    "filter_domain": domain,
                }).execute()
                return [RetrievedExample(r["content"], r.get("metadata", {}),
                                         float(r.get("similarity", 0.0)))
                        for r in (res.data or [])]
            except Exception as e:
                logger.warning("RPC match failed, falling back to local: %s", e)
        # Fallback: pull rows and cosine locally
        return self._local_fallback(query, k, domain)

    def _local_fallback(self, query, k, domain):
        try:
            res = self.client.table(self.table).select("content, metadata").execute()
            rows = res.data or []
        except Exception:
            return []
        mem = MemoryVectorStore(self.embed_model)
        for r in rows:
            mem.add(r["content"], r.get("metadata", {}))
        return mem.retrieve(query, k, domain)

    def __len__(self) -> int:
        try:
            res = self.client.table(self.table).select("id", count="exact").execute()
            return res.count or 0
        except Exception:
            return 0


def get_supabase_vector_store(config):
    client = _make_client(config.supabase_url, config.supabase_key)
    if client is None:
        return None
    return SupabaseVectorStore(client, config.embed_model)
