"""Offline pytest tests proving deploy-readiness for AI Content Studio.

All tests are hermetic: env vars that might supply real API keys are cleared
via the `clean_env` fixture before any import-time side effects can use them.
No network calls, no disk writes, no LLM API keys required.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# ── Ensure project root is importable ─────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Env-var names that may supply real credentials ────────────────────────────
_CRED_VARS = (
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
    "NVIDIA_API_KEY",
    "SUPABASE_URL",
    "SUPABASE_KEY",
    "LLM_PROVIDER",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Strip all credential env vars so every test is hermetic."""
    for var in _CRED_VARS:
        monkeypatch.delenv(var, raising=False)
    yield


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Import smoke tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_import_config():
    """src.config imports cleanly with no env vars set."""
    import importlib
    mod = importlib.import_module("src.config")
    assert hasattr(mod, "Config")


def test_import_generator():
    """src.generator imports cleanly with no env vars set."""
    import importlib
    mod = importlib.import_module("src.generator")
    assert hasattr(mod, "ContentGenerator")
    assert hasattr(mod, "ContentRequest")
    assert hasattr(mod, "GeneratedItem")


def test_import_domains():
    """src.domains imports cleanly with no env vars set."""
    import importlib
    mod = importlib.import_module("src.domains")
    assert hasattr(mod, "list_domains")
    assert hasattr(mod, "get_domain")


def test_import_exporters():
    """src.exporters imports cleanly with no env vars set."""
    import importlib
    mod = importlib.import_module("src.exporters")
    assert hasattr(mod, "to_json")
    assert hasattr(mod, "to_csv")
    assert hasattr(mod, "to_markdown")
    assert hasattr(mod, "EXPORTERS")


def test_import_engagement():
    """src.engagement imports cleanly with no env vars set."""
    import importlib
    mod = importlib.import_module("src.engagement")
    assert hasattr(mod, "predict_engagement")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Config.from_env() resolves to "mock" with no keys
# ═══════════════════════════════════════════════════════════════════════════════

def test_config_resolves_to_mock_without_keys():
    """With no GEMINI_API_KEY, no OPENROUTER_API_KEY, and no LLM_PROVIDER,
    Config.from_env().resolved_provider() must return 'mock'."""
    from src.config import Config

    cfg = Config.from_env()
    assert cfg.resolved_provider() == "mock", (
        f"Expected 'mock', got '{cfg.resolved_provider()}'"
    )


def test_config_from_env_explicit_mock(monkeypatch):
    """LLM_PROVIDER=mock is honoured directly."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    from src.config import Config

    cfg = Config.from_env()
    assert cfg.provider == "mock"
    assert cfg.resolved_provider() == "mock"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Generation works in mock mode for every domain
# ═══════════════════════════════════════════════════════════════════════════════

def _make_generator():
    """Return a ContentGenerator configured for mock + in-memory vector store."""
    from src.config import Config
    from src.generator import ContentGenerator
    from src.vector_store import MemoryVectorStore

    cfg = Config(provider="mock")
    vs = MemoryVectorStore()
    return ContentGenerator(cfg, vector_store=vs)


def test_generation_mock_all_domains():
    """generate() returns items with the correct output_fields for every domain."""
    from src.domains import list_domains
    from src.generator import ContentRequest

    gen = _make_generator()
    domains = list_domains()
    assert len(domains) > 0, "list_domains() returned empty — check domains.py"

    for domain in domains:
        req = ContentRequest(
            domain=domain.key,
            topic="cloud infrastructure cost optimisation",
            channel=domain.channels[0] if domain.channels else "",
            tone="professional",
            num_items=1,
            use_rag=False,  # vector store is empty in this fixture
        )
        items = gen.generate(req)

        assert len(items) > 0, (
            f"[{domain.key}] generate() returned no items"
        )

        for item in items:
            for field_name in domain.output_fields:
                assert field_name in item.fields, (
                    f"[{domain.key}] field '{field_name}' missing from item.fields. "
                    f"Got: {list(item.fields.keys())}"
                )


def test_generation_primary_text_non_empty():
    """GeneratedItem.primary_text() returns a non-empty string for mock output."""
    from src.domains import list_domains
    from src.generator import ContentRequest

    gen = _make_generator()
    domain = list_domains()[0]  # just test one
    req = ContentRequest(
        domain=domain.key,
        topic="test topic",
        num_items=1,
        use_rag=False,
    )
    items = gen.generate(req)
    assert items
    assert items[0].primary_text().strip() != "", (
        "primary_text() returned an empty string"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Vector store factory
# ═══════════════════════════════════════════════════════════════════════════════

def test_get_vector_store_memory():
    """get_vector_store('memory') returns a usable MemoryVectorStore."""
    from src.vector_store import MemoryVectorStore, get_vector_store

    vs = get_vector_store("memory")
    assert isinstance(vs, MemoryVectorStore)

    vs.add("sample text", {"domain": "social"})
    assert len(vs) == 1

    results = vs.retrieve("sample", k=1)
    assert len(results) == 1
    assert results[0].text == "sample text"


def test_get_vector_store_supabase_degrades_to_memory():
    """get_vector_store('supabase', config=Config()) degrades to MemoryVectorStore
    when no Supabase credentials are present."""
    from src.config import Config
    from src.vector_store import MemoryVectorStore, get_vector_store

    cfg = Config()  # supabase_url and supabase_key are both "" by default
    assert not cfg.supabase_enabled(), "Expected no Supabase creds in clean Config()"

    vs = get_vector_store("supabase", config=cfg)
    assert isinstance(vs, MemoryVectorStore), (
        f"Expected MemoryVectorStore fallback, got {type(vs).__name__}"
    )


def test_vector_store_add_many_and_retrieve():
    """add_many() loads multiple example dicts; retrieve() returns relevant hits."""
    from src.vector_store import get_vector_store

    vs = get_vector_store("memory")
    examples = [
        {"text": "Exciting product launch for the summer season!",
         "domain": "social"},
        {"text": "10 tips to boost your LinkedIn engagement",
         "domain": "blog"},
        {"text": "Subject: Your exclusive offer awaits",
         "domain": "email"},
    ]
    vs.add_many(examples)
    assert len(vs) == len(examples)

    hits = vs.retrieve("LinkedIn post about tips", k=2)
    assert len(hits) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Exporters produce non-empty output
# ═══════════════════════════════════════════════════════════════════════════════

_SAMPLE_ITEMS = [
    {
        "caption": "Check out our new product launch! 🚀",
        "hashtags": ["#launch", "#AI", "#tech"],
        "emojis": "🚀✨",
        "brand_match": None,
    },
    {
        "caption": "Productivity tips that actually work.",
        "hashtags": ["#productivity", "#tips"],
        "emojis": "💡",
        "brand_match": 0.85,
    },
]


def test_to_json_non_empty():
    from src.exporters import to_json

    result = to_json(_SAMPLE_ITEMS)
    assert result.strip(), "to_json() returned empty string"
    # Must be valid JSON
    parsed = json.loads(result)
    assert isinstance(parsed, list)
    assert len(parsed) == len(_SAMPLE_ITEMS)


def test_to_csv_non_empty():
    from src.exporters import to_csv

    result = to_csv(_SAMPLE_ITEMS)
    assert result.strip(), "to_csv() returned empty string"
    lines = [ln for ln in result.splitlines() if ln.strip()]
    # At least a header row + one data row
    assert len(lines) >= 2, f"CSV has fewer than 2 lines: {lines}"


def test_to_markdown_non_empty():
    from src.exporters import to_markdown

    result = to_markdown(_SAMPLE_ITEMS)
    assert result.strip(), "to_markdown() returned empty string"
    assert "## Item 1" in result
    assert "## Item 2" in result


def test_exporters_registry_covers_json_csv_markdown():
    """EXPORTERS dict must have json, csv, and markdown entries."""
    from src.exporters import EXPORTERS

    for fmt in ("json", "csv", "markdown"):
        assert fmt in EXPORTERS, f"EXPORTERS missing '{fmt}' key"
        fn, ext = EXPORTERS[fmt]
        assert callable(fn), f"EXPORTERS['{fmt}'] function is not callable"
        assert isinstance(ext, str) and ext, f"EXPORTERS['{fmt}'] extension is empty"


def test_exporters_all_produce_non_empty_output():
    """All three exporters in EXPORTERS produce non-empty output for the sample."""
    from src.exporters import EXPORTERS

    for fmt, (exporter_fn, _ext) in EXPORTERS.items():
        output = exporter_fn(_SAMPLE_ITEMS)
        assert output and output.strip(), (
            f"Exporter '{fmt}' returned empty output"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Engagement predictor
# ═══════════════════════════════════════════════════════════════════════════════

def test_predict_engagement_returns_expected_shape():
    """predict_engagement() returns a dict with tier, score, signals."""
    from src.engagement import predict_engagement

    text = "Buy now and save 20%! Limited time offer — click the link in bio. 🔥"
    result = predict_engagement(text)

    assert isinstance(result, dict), "predict_engagement() did not return a dict"
    assert "tier" in result, "Missing 'tier' key"
    assert "score" in result, "Missing 'score' key"
    assert "signals" in result, "Missing 'signals' key"
    assert result["tier"] in ("low", "medium", "high"), (
        f"Unexpected tier: {result['tier']!r}"
    )
    assert 0.0 <= result["score"] <= 1.0, (
        f"Score out of [0,1]: {result['score']}"
    )
    assert isinstance(result["signals"], dict)
