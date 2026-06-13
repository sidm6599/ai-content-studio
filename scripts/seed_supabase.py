#!/usr/bin/env python3
"""One-command Supabase setup check + seed.

Run AFTER you've (1) put SUPABASE_URL + SUPABASE_KEY in .env and
(2) executed supabase/schema.sql in the Supabase SQL editor.

    python scripts/seed_supabase.py

It will:
  • verify the connection,
  • seed knowledge/examples.json into `brand_examples`,
  • write + read back a test row in `generations` (proves persistence).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config  # noqa: E402
from src.db import get_history_store, get_supabase_vector_store  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    cfg = Config.from_env()
    if not cfg.supabase_enabled():
        print("❌ SUPABASE_URL / SUPABASE_KEY not set in .env — nothing to do.")
        return 1
    print(f"🔌 Connecting to {cfg.supabase_url} ...")

    vs = get_supabase_vector_store(cfg)
    if vs is None:
        print("❌ Could not create Supabase client. Is `supabase` installed? "
              "(pip install supabase) Are the URL/key correct?")
        return 1

    # 1) seed brand examples
    examples = json.loads((ROOT / "knowledge" / "examples.json").read_text())
    print(f"📚 Seeding {len(examples)} brand examples into `brand_examples` ...")
    vs.add_many(examples)
    print(f"   brand_examples row count ≈ {len(vs)}")

    # 2) round-trip a generation
    hist = get_history_store(cfg)
    if not getattr(hist, "enabled", False):
        print("⚠️  History store not enabled (check PERSIST_HISTORY / creds).")
        return 1
    print("📝 Writing a test generation to `generations` ...")
    hist.save_generation({
        "domain": "social", "topic": "supabase setup check",
        "channel": "LinkedIn", "tone": "professional",
        "items": [{"caption": "It persists! ✅"}],
    })
    rows = hist.list_generations(limit=1)
    if rows and rows[0]["topic"] == "supabase setup check":
        print("✅ Round-trip OK — history is persisting in Supabase.")
        return 0
    print("⚠️  Wrote a row but couldn't read it back. Check the table/RLS.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
