"""AI Content Studio — Streamlit UI.

Run: streamlit run app.py
Works with zero API keys (mock provider). Add a key in .env for live output.
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src import exporters
from src.config import Config
from src.db import get_history_store
from src.domains import list_domains, list_tones
from src.engagement import predict_engagement
from src.generator import ContentGenerator, ContentRequest
from src.vector_store import get_vector_store

st.set_page_config(page_title="AI Content Studio", page_icon="✍️", layout="wide")


@st.cache_resource
def load_backend():
    cfg = Config.from_env()
    cfg.validate()
    vs = get_vector_store(cfg.vector_backend, cfg.embed_model, config=cfg)
    # Seed in-memory store from local JSON; Supabase store is seeded once via DB.
    if cfg.vector_backend != "supabase":
        examples_path = Path(__file__).parent / "knowledge" / "examples.json"
        if examples_path.exists():
            vs.add_many(json.loads(examples_path.read_text()))
    history = get_history_store(cfg)
    return cfg, ContentGenerator(cfg, vs), history


cfg, generator, history_store = load_backend()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("✍️ AI Content Studio")
    st.caption("Multi-domain RAG content generator")
    st.markdown(f"**LLM provider:** `{cfg.resolved_provider()}`")
    _persist = "Supabase ✅" if getattr(history_store, "enabled", False) else "session-only"
    st.markdown(f"**History:** `{_persist}`")
    if cfg.resolved_provider() == "mock":
        st.info("Running in **mock** mode (no API key). Add `GEMINI_API_KEY` "
                "or `OPENROUTER_API_KEY` to `.env` for live output.")
    if not getattr(history_store, "enabled", False):
        st.caption("💾 Add `SUPABASE_URL` + `SUPABASE_KEY` to `.env` to save "
                   "history permanently (it won't forget on refresh).")
    use_rag = st.toggle("Use RAG (brand context)", value=True)
    compare = st.toggle("Compare RAG vs No-RAG", value=False)
    num_items = st.slider("Variations", 1, cfg.max_generations, 3)

# ---------------- Main ----------------
domains = list_domains()
labels = [d.label for d in domains]
choice = st.selectbox("Content type", labels)
domain = next(d for d in domains if d.label == choice)

st.caption(domain.description)
col1, col2 = st.columns([3, 1])
with col1:
    topic = st.text_area("What's it about?",
                         placeholder="e.g. New AI-powered smartwatch with health tracking")
with col2:
    channel = st.selectbox("Channel", domain.channels or ["Generic"])
    tone = st.selectbox("Tone", list_tones())


def _render_item(i, item, limit):
    with st.container(border=True):
        st.markdown(f"**Variation {i + 1}**")
        for field, value in item.fields.items():
            if isinstance(value, list):
                st.markdown(f"**{field}:** " + ", ".join(map(str, value)))
            else:
                st.markdown(f"**{field}:** {value}")
        text = item.primary_text()
        eng = predict_engagement(text)
        cols = st.columns(4)
        if item.brand_match is not None:
            cols[0].metric("Brand match", f"{item.brand_match * 100:.0f}%")
        _tier_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(eng["tier"], "")
        cols[1].metric("Engagement", f"{_tier_icon} {eng['tier']}",
                       help=f"score {eng['score']:.2f}")
        cols[2].metric("Characters", len(text))
        if limit:
            ok = len(text) <= limit
            cols[3].metric(f"≤ {limit}?", "✅" if ok else "⚠️ over")


if st.button("🎨 Generate", type="primary", use_container_width=True):
    if not topic.strip():
        st.warning("Add a topic first.")
    else:
        limit = domain.char_limits.get(channel)
        base = dict(domain=domain.key, topic=topic, channel=channel,
                    tone=tone, num_items=num_items)
        if compare:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("With RAG")
                for i, it in enumerate(generator.generate(ContentRequest(**base, use_rag=True))):
                    _render_item(i, it, limit)
            with c2:
                st.subheader("No RAG")
                for i, it in enumerate(generator.generate(ContentRequest(**base, use_rag=False))):
                    _render_item(i, it, limit)
        else:
            items = generator.generate(ContentRequest(**base, use_rag=use_rag))
            for i, it in enumerate(items):
                _render_item(i, it, limit)
            # export + persist to DB (Supabase if configured, else in-memory)
            export = [it.fields | {"brand_match": it.brand_match} for it in items]
            history_store.save_generation({
                "domain": domain.key, "topic": topic, "channel": channel,
                "tone": tone, "items": export,
            })
            # Multi-format export (json / csv / markdown)
            dl = st.columns(len(exporters.EXPORTERS))
            for col, (fmt, (fn, ext)) in zip(dl, exporters.EXPORTERS.items()):
                col.download_button(f"⬇️ {fmt.upper()}", fn(export),
                                    file_name=f"content.{ext}",
                                    key=f"dl_{fmt}")

# History (loaded from Supabase when configured — persists across refreshes)
_rows = history_store.list_generations(limit=10)
if _rows:
    label = "saved in Supabase" if getattr(history_store, "enabled", False) else "this session"
    with st.expander(f"🕘 History — {label} ({len(_rows)})"):
        for h in _rows:
            st.markdown(f"- **{h['topic']}** · _{h.get('domain', '')}_ — "
                        f"{len(h.get('items', []))} item(s)")
