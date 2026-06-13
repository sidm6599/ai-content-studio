"""Offline test suite — runs with no API keys and no network (mock provider)."""
import pytest

from src import quality
from src.config import Config
from src.domains import DOMAINS, get_domain, list_domains
from src.generator import ContentGenerator, ContentRequest, _loads_lenient
from src.vector_store import MemoryVectorStore


# ---------- config ----------
def test_config_validation_ok():
    Config(temperature=0.5, max_generations=3, provider="mock").validate()


@pytest.mark.parametrize("bad", [
    {"temperature": 2.0},
    {"max_generations": 0},
    {"provider": "nope"},
])
def test_config_validation_rejects_bad(bad):
    with pytest.raises(ValueError):
        Config(**bad).validate()


def test_resolved_provider_prefers_keys():
    assert Config(provider="auto", gemini_api_key="x").resolved_provider() == "gemini"
    assert Config(provider="auto", openrouter_api_key="x").resolved_provider() == "openrouter"
    assert Config(provider="auto").resolved_provider() == "mock"


# ---------- domains ----------
def test_all_domains_have_fields_and_instruction():
    for d in list_domains():
        assert d.output_fields and d.instruction
    assert {"social", "product", "email", "resume", "blog", "video"}.issubset(set(DOMAINS))


def test_get_domain_unknown_raises():
    with pytest.raises(KeyError):
        get_domain("does-not-exist")


# ---------- generator (mock provider) ----------
def _gen():
    return ContentGenerator(Config(provider="mock"))


def test_generate_returns_requested_count_and_fields():
    items = _gen().generate(ContentRequest(domain="social", topic="new app launch",
                                           num_items=3, use_rag=False))
    assert len(items) == 3
    for it in items:
        assert set(it.fields) == set(get_domain("social").output_fields)


def test_generate_resume_bullets():
    items = _gen().generate(ContentRequest(domain="resume", topic="built a RAG pipeline",
                                           num_items=1, use_rag=False))
    assert isinstance(items[0].fields["bullets"], list)


def test_rag_attaches_brand_match_score():
    vs = MemoryVectorStore()
    vs.add("Thrilled to share our team shipped a feature.", {"domain": "social"})
    gen = ContentGenerator(Config(provider="mock"), vs)
    items = gen.generate(ContentRequest(domain="social", topic="we shipped a feature",
                                        num_items=1, use_rag=True))
    assert items[0].brand_match is not None
    assert 0.0 <= items[0].brand_match <= 1.0


# ---------- parsing ----------
def test_loads_lenient_handles_markdown_fence():
    assert _loads_lenient('```json\n[{"a": 1}]\n```') == [{"a": 1}]


def test_loads_lenient_handles_prose_wrapping():
    assert _loads_lenient('Here you go: [{"x": 2}] thanks') == [{"x": 2}]


# ---------- quality ----------
def test_similarity_bounds_and_identity():
    assert quality.similarity("hello world", "hello world") > 0.9
    s = quality.similarity("apples", "completely unrelated tokens")
    assert 0.0 <= s <= 1.0


def test_compare_rag_vs_norag_shape():
    out = quality.compare_rag_vs_norag(
        ["we shipped a feature fast"], ["random unrelated text"],
        ["we shipped a feature"])
    assert set(out) == {"rag_brand_match", "norag_brand_match", "delta", "rag_helps"}


# ---------- persistence (Supabase) fallback ----------
def test_config_supabase_disabled_by_default():
    assert Config().supabase_enabled() is False


def test_history_store_falls_back_without_creds():
    from src.db import NullHistoryStore, get_history_store
    store = get_history_store(Config())          # no Supabase creds
    assert isinstance(store, NullHistoryStore)
    assert store.enabled is False


def test_null_history_store_saves_and_lists_in_memory():
    from src.db import NullHistoryStore
    store = NullHistoryStore()
    store.save_generation({"topic": "a", "domain": "social", "items": [1]})
    store.save_generation({"topic": "b", "domain": "email", "items": [1, 2]})
    rows = store.list_generations()
    assert [r["topic"] for r in rows] == ["b", "a"]   # newest first


def test_vector_store_supabase_backend_degrades_to_memory():
    # No creds → factory must not crash, returns a usable store.
    from src.vector_store import MemoryVectorStore, get_vector_store
    vs = get_vector_store("supabase", config=Config())
    assert isinstance(vs, MemoryVectorStore)
