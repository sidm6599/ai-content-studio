"""Exercises the REAL Supabase code paths (SupabaseHistoryStore,
SupabaseVectorStore) against an in-memory fake client — no network, no creds.

This proves the DB logic works end-to-end before you plug in a live key.
"""
from src.db import SupabaseHistoryStore, SupabaseVectorStore


# ---- a tiny chainable fake of the supabase-py client ------------------
class _Resp:
    def __init__(self, data, count=None):
        self.data = data
        self.count = count


class _Query:
    def __init__(self, rows):
        self._rows = rows
        self._pending = None
        self._desc = False
        self._limit = None

    def insert(self, row):
        self._pending = row
        return self

    def select(self, *a, **k):
        return self

    def order(self, col, desc=False):
        self._desc = desc
        return self

    def limit(self, n):
        self._limit = n
        return self

    def execute(self):
        if self._pending is not None:
            row = dict(self._pending)
            row["_seq"] = len(self._rows)
            self._rows.append(row)
            return _Resp([row])
        rows = sorted(self._rows, key=lambda r: r.get("_seq", 0), reverse=self._desc)
        if self._limit:
            rows = rows[: self._limit]
        return _Resp(rows, count=len(rows))


class FakeClient:
    def __init__(self):
        self.store = {}

    def table(self, name):
        self.store.setdefault(name, [])
        return _Query(self.store[name])

    def rpc(self, *a, **k):          # force the local-fallback retrieval path
        raise RuntimeError("rpc not supported in fake")


# ---- tests ------------------------------------------------------------
def test_history_save_and_list_roundtrip():
    store = SupabaseHistoryStore(FakeClient())
    store.save_generation({"domain": "social", "topic": "first", "items": [1]})
    store.save_generation({"domain": "email", "topic": "second", "items": [1, 2]})
    rows = store.list_generations(limit=10)
    assert [r["topic"] for r in rows] == ["second", "first"]   # newest first


def test_history_list_respects_limit():
    store = SupabaseHistoryStore(FakeClient())
    for i in range(5):
        store.save_generation({"domain": "social", "topic": f"t{i}", "items": []})
    assert len(store.list_generations(limit=3)) == 3


def test_vector_store_add_and_retrieve_via_fallback():
    vs = SupabaseVectorStore(FakeClient())
    vs.add_many([
        {"text": "we shipped a new feature today", "domain": "social"},
        {"text": "buy our noise cancelling earbuds", "domain": "product"},
    ])
    hits = vs.retrieve("we shipped a feature", k=1)
    assert hits and "shipped" in hits[0].text


def test_vector_store_len():
    vs = SupabaseVectorStore(FakeClient())
    vs.add("hello world", {"domain": "social"})
    assert len(vs) == 1
