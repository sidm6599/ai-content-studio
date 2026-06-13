---
title: AI Content Studio
emoji: ✍️
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
license: mit
---

# ✍️ AI Content Studio

A **multi-domain, RAG-powered content generator** — write social posts, product descriptions, marketing emails, and resume bullets, with an **embeddings-based brand-voice match score**, a **RAG vs No-RAG** comparison, and **multi-provider LLM fallback**.

Built by **Siddhesh Mishra** as an original project (architecture inspired by, but not copied from, common RAG content tools).

**Live demo:** &lt;add your HF Space URL&gt;

![status](https://img.shields.io/badge/status-active-green) ![python](https://img.shields.io/badge/python-3.10+-blue) ![tests](https://img.shields.io/badge/tests-pytest-success)

## ✨ Highlights
- **6 pluggable domains** (social / product / email / resume / blog / video) — adding one is a single `Domain` entry.
- **Tone presets** and an **engagement predictor** (scikit-learn with heuristic fallback) scoring each variation low/medium/high.
- **Multi-format export** — download as JSON, CSV, or Markdown.
- **RAG** over brand examples — **in-memory**, **Supabase pgvector**, or **ChromaDB** backends (graceful fallback).
- **Brand-voice match score** via sentence-transformer embeddings (cosine to brand examples).
- **RAG vs No-RAG** side-by-side — the evaluation pattern from my MSc dissertation.
- **Multi-LLM with fallback**: Gemini → OpenRouter (Llama) → offline **mock** so it always runs.
- **Persistent memory via Supabase** (Postgres + pgvector): saves generation **history** and brand examples so it **doesn't forget** across sessions — with graceful in-memory fallback when no DB is configured.
- **Export** (JSON), **history**, and per-channel **length checks** (e.g. X/Twitter 280).
- **Runs with zero API keys** (mock mode) — perfect for demos and CI.

## 🚀 Quickstart
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # optional: add a key for live LLM output
streamlit run app.py        # http://localhost:8501
pytest -q                   # run the offline test suite
```

## 🧱 Architecture
```
app.py (Streamlit)
  ├── ContentGenerator (src/generator.py)
  │     ├── DomainRegistry  (src/domains.py)      prompt templates per domain
  │     ├── VectorStore     (src/vector_store.py) RAG retrieval (memory | supabase | chromadb)
  │     ├── LLMClient       (src/llm_client.py)   gemini → openrouter → mock
  │     └── Quality         (src/quality.py)      brand-match, rag-vs-norag
  └── Persistence      (src/db.py)                Supabase history + pgvector (or in-memory)
```

## 🗄️ Persistence with Supabase (optional)
1. Create a Supabase project, open **SQL editor**, run [`supabase/schema.sql`](supabase/schema.sql).
2. Put `SUPABASE_URL` + `SUPABASE_KEY` in `.env` (and `VECTOR_BACKEND=supabase` to store brand examples in the DB too).
3. History now persists across refreshes; brand examples are retrieved via **pgvector** cosine search.
Without these, the app keeps working with in-memory history.

## 🔌 Configuration
All via `.env` (see `.env.example`). With no keys it runs in **mock** mode; add
`GEMINI_API_KEY` or `OPENROUTER_API_KEY` for live generation.

## 🚀 Deploy

Step-by-step instructions for deploying to **Hugging Face Spaces** or **GitHub** are in [`DEPLOY.md`](DEPLOY.md).

The lean [`requirements.txt`](requirements.txt) (4 runtime deps, no heavy ML libs) is what HF Spaces installs. For local dev with ChromaDB and the full test suite, use [`requirements-dev.txt`](requirements-dev.txt).

## 🗺️ Roadmap
See [`ROADMAP.md`](ROADMAP.md). Done: 6 domains, 3 RAG backends (memory / Supabase pgvector / ChromaDB), engagement scoring, multi-format export, GitHub Actions CI, and HF-Spaces deploy config. Next up: multimodal image input and localization.

## 🧪 Tests
`pytest -q` — the suite is fully offline (mock provider + TF-cosine fallback), so it
runs anywhere with no keys or network.

## 📄 License
MIT.
