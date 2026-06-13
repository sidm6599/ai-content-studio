# AI Content Studio — Roadmap & Process Map

A multi-domain, RAG-powered content generator (social posts · product descriptions · marketing emails · resume bullets) built by **Siddhesh Mishra**. Reuses the architecture pattern from the reference project but written fresh, with **multi-LLM fallback**, **ChromaDB** RAG, and an **embeddings-based quality score**.

## Goal
A deployed, tested, demoable portfolio project that proves: RAG, embeddings, multi-provider LLM orchestration, evaluation, and clean Python engineering.

## Architecture
```
Streamlit UI (app.py)
   │  domain selector + inputs + RAG toggle
   ▼
ContentGenerator (src/generator.py)
   ├── DomainRegistry (src/domains.py)      ← prompt templates per domain
   ├── RAG (src/vector_store.py)            ← ChromaDB + embeddings, optional
   ├── LLMClient (src/llm_client.py)        ← Gemini → OpenRouter/Llama → mock fallback
   └── Quality (src/quality.py)             ← brand-voice similarity, RAG vs No-RAG
```

## Process map (6 phases)
| Phase | Deliverable | Status |
|---|---|---|
| 0 | Repo scaffold, README, `.env.example`, requirements | ✅ done |
| 1 | Core: config + multi-provider LLM client + generator | ✅ done (mock provider runs offline) |
| 2 | Domains: social / product / email / resume templates | ✅ done |
| 3 | Quality: embeddings brand-match + RAG-vs-No-RAG compare | ✅ done (TF fallback; ST optional) |
| 4 | RAG vector store: in-memory + **Supabase pgvector** (+ ChromaDB stub) | ✅ memory + Supabase done |
| 5 | UI polish: export (json), history, length rules | ✅ core done |
| 6 | **Persistence: Supabase** — history + brand examples (won't forget) | ✅ done (graceful offline fallback) |
| 7 | Tests (pytest) + CI + deploy (HF Spaces) | ✅ **125 tests, ruff-clean, GitHub Actions CI + CD, HF deploy config — DEPLOY-READY** |

> **Status: deploy-ready.** Git initialised with an initial commit. To go live: push to GitHub (CI runs automatically) and create a Hugging Face Streamlit Space — see [`DEPLOY.md`](DEPLOY.md).

### Supabase persistence (the "don't forget" layer)
- Run `supabase/schema.sql` once in the Supabase SQL editor → creates `generations`, `brand_examples`, a pgvector index, and the `match_brand_examples` RPC.
- Add `SUPABASE_URL` + `SUPABASE_KEY` to `.env`; set `VECTOR_BACKEND=supabase` to also store brand examples in the DB.
- No creds? History falls back to in-memory and nothing breaks.

## Task checklist (do next, in order)
- [ ] `pip install -r requirements.txt` and run `streamlit run app.py` (works offline in mock mode)
- [ ] Add a real `GEMINI_API_KEY` (or `OPENROUTER_API_KEY`) to `.env` to enable live generation
- [ ] Phase 4: implement ChromaDB persistence in `src/vector_store.py` (currently in-memory cosine)
- [ ] Seed `knowledge/examples.json` with 10–15 strong examples per domain
- [ ] Add GitHub Actions workflow: `pytest` + `ruff`
- [ ] Deploy to Streamlit Community Cloud or Hugging Face Spaces; put the link in README + CV
- [ ] Record a 30s GIF for the README

## How to run
```bash
cd 04_projects/ai-content-studio
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # optional: add a key for live LLM output
streamlit run app.py          # http://localhost:8501
pytest -q                     # run the test suite
```

## Shipped via parallel subagents (2026-06-13)
- ✅ **6 domains** — added `blog` and `video` (was social/product/email/resume).
- ✅ **Tone presets** — `TONE_PRESETS` + `list_tones()` (UI dropdown).
- ✅ **Engagement predictor** — `src/engagement.py` (scikit-learn if available, heuristic fallback) → per-item tier + score in the UI.
- ✅ **ChromaDB backend** — `ChromaVectorStore` wired into `get_vector_store("chromadb")` with graceful fallback. **Now installed & verified** (chromadb 1.5.9): all backend tests run (no longer skipped), real semantic retrieval works via Chroma's built-in ONNX embeddings even without sentence-transformers. Set `VECTOR_BACKEND=chromadb` to use it.
- ✅ **Multi-format export** — `src/exporters.py` (JSON / CSV / Markdown) → three download buttons.
- Test count: **103 passing, 4 skipped** (chromadb-only).

## Build wave 2 (2026-06-14) — shipped
- ✅ **NVIDIA NIM provider** (live, `meta/llama-3.3-70b-instruct`) + vision models verified.
- ✅ **Multimodal image input** — upload an image; Gemini/NVIDIA VLM factor it in.
- ✅ **8-language localization** (`src/i18n.py`) — English/German/Hindi/Spanish/French/…
- ✅ **Sample gallery** — `scripts/generate_samples.py` → `SAMPLES.md` (live-generated).
- ✅ **Docker** — `Dockerfile` + `.dockerignore` (HF Docker-Spaces option included).

## Still open (nice-to-have)
- **Brand kits** — save per-brand voice + examples; switch between clients.
- **pgvector activation** — `pip install sentence-transformers` + re-seed Supabase for real embeddings (currently local-cosine fallback).
- **Deploy** — push to GitHub + create the HF Space (needs your accounts).

## Resume bullet (fill in once deployed)
> "Built **AI Content Studio**, a multi-domain RAG content generator (Streamlit, ChromaDB, sentence-transformers, multi-provider LLM with Gemini/Llama fallback): added an embeddings-based **brand-voice match score** and a **RAG vs No-RAG** comparison, covered by a **pytest** suite and CI, deployed live at <link>."
