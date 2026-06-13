#!/usr/bin/env python3
"""Pre-deploy sanity script for AI Content Studio.

Simulates the Hugging Face environment: no .env file, no real API keys.
The LLM provider is forced to 'mock' so all checks run fully offline.

Usage (from project root):
    python scripts/healthcheck.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ── Make sure the project root is on the path ─────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Simulate HF: strip real API keys from the environment ─────────────────────
for _key in ("GEMINI_API_KEY", "OPENROUTER_API_KEY", "SUPABASE_URL",
             "SUPABASE_KEY"):
    os.environ.pop(_key, None)
os.environ["LLM_PROVIDER"] = "mock"
os.environ["VECTOR_BACKEND"] = "memory"

# ── Imports (after path / env setup) ──────────────────────────────────────────
from src.config import Config  # noqa: E402
from src.domains import list_domains  # noqa: E402
from src.engagement import predict_engagement  # noqa: E402
from src.exporters import EXPORTERS  # noqa: E402
from src.generator import ContentGenerator, ContentRequest  # noqa: E402
from src.vector_store import get_vector_store  # noqa: E402

# ── Tiny check harness ────────────────────────────────────────────────────────
_failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    symbol = "✅" if ok else "❌"
    msg = f"{symbol}  {label}"
    if detail:
        msg += f"  [{detail}]"
    print(msg)
    if not ok:
        _failures.append(label)
    return ok


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Config & provider resolution
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Config ──────────────────────────────────────────────────────────────")
config = Config.from_env()
resolved = config.resolved_provider()
check("Config.from_env() loads without error", True)
check(
    "Resolved provider is 'mock' (no API keys present)",
    resolved == "mock",
    f"got '{resolved}'",
)
print(f"   provider={config.provider!r}  resolved={resolved!r}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Vector store — seed from knowledge/examples.json
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Vector store ────────────────────────────────────────────────────────")
examples_path = PROJECT_ROOT / "knowledge" / "examples.json"
vs = get_vector_store("memory", config.embed_model)

try:
    with open(examples_path, encoding="utf-8") as fh:
        examples = json.load(fh)
    vs.add_many(examples)
    check(
        f"Vector store seeded from examples.json ({len(vs)} docs)",
        len(vs) > 0,
    )
except Exception as exc:
    check("Vector store seeded from examples.json", False, str(exc))

# ═══════════════════════════════════════════════════════════════════════════════
# 3. ContentGenerator — every domain
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Generation (all domains) ────────────────────────────────────────────")
generator = ContentGenerator(config, vector_store=vs)
domains = list_domains()
check(
    f"list_domains() returns {len(domains)} domain(s)",
    len(domains) > 0,
)

first_items = None  # saved for exporters / engagement checks below

for domain in domains:
    try:
        req = ContentRequest(
            domain=domain.key,
            topic="AI productivity tools for remote teams",
            channel=domain.channels[0] if domain.channels else "",
            tone="professional",
            num_items=2,
            use_rag=True,
        )
        items = generator.generate(req)

        # Check items returned
        ok_count = len(items) > 0
        check(
            f"  [{domain.key}] generate() returns >0 items",
            ok_count,
            f"{len(items)} item(s)",
        )

        # Check every expected output_field is present in every item
        missing_fields: list[str] = []
        for item in items:
            for field_name in domain.output_fields:
                if field_name not in item.fields:
                    missing_fields.append(field_name)

        check(
            f"  [{domain.key}] all output_fields present: {domain.output_fields}",
            not missing_fields,
            f"missing: {missing_fields}" if missing_fields else "all present",
        )

        if first_items is None and items:
            first_items = items

    except Exception as exc:
        check(f"  [{domain.key}] generate() succeeded", False, str(exc))

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Engagement prediction
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Engagement predictor ────────────────────────────────────────────────")
if first_items:
    sample_text = first_items[0].primary_text()
    try:
        eng = predict_engagement(sample_text)
        check(
            "predict_engagement() returns tier/score/signals dict",
            all(k in eng for k in ("tier", "score", "signals")),
            f"tier={eng.get('tier')!r} score={eng.get('score')}",
        )
    except Exception as exc:
        check("predict_engagement() runs without error", False, str(exc))
else:
    check("predict_engagement() — skipped (no items generated)", False,
          "no items to test with")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Exporters
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Exporters ────────────────────────────────────────────────────────────")
if first_items:
    # Build the flat dicts the exporters expect (mirrors app.py behaviour)
    sample_dicts = [
        item.fields | {"brand_match": item.brand_match}
        for item in first_items
    ]

    for fmt, (exporter_fn, ext) in EXPORTERS.items():
        try:
            output = exporter_fn(sample_dicts)
            check(
                f"  [{fmt}] exporter produces non-empty output",
                bool(output and output.strip()),
                f"{len(output)} chars",
            )
        except Exception as exc:
            check(f"  [{fmt}] exporter runs without error", False, str(exc))
else:
    for fmt in EXPORTERS:
        check(f"  [{fmt}] exporter — skipped (no items)", False,
              "no items to test with")

# ═══════════════════════════════════════════════════════════════════════════════
# Final verdict
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 72)
if _failures:
    print(f"❌  FAILED — {len(_failures)} check(s) did not pass:")
    for f in _failures:
        print(f"     • {f}")
    sys.exit(1)
else:
    total_domains = len(domains)
    print(
        f"✅  All checks passed  "
        f"({total_domains} domains · 3 exporters · engagement predictor)"
    )
    sys.exit(0)
