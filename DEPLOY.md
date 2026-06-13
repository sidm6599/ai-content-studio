# Deployment Guide — AI Content Studio

This guide covers two deployment targets: **Hugging Face Spaces** (one-click public demo) and **GitHub** (source hosting + CI).

---

## 1. Hugging Face Spaces

### Prerequisites
- A free account at [huggingface.co](https://huggingface.co).
- The repo's `requirements.txt` already contains only the 4 lean runtime deps that HF will install — no changes needed.

### Steps

#### Option A — Connect GitHub (recommended)
1. Go to **huggingface.co/new-space**.
2. Name your Space (e.g. `ai-content-studio`), pick **Streamlit** as the SDK, set visibility to Public or Private.
3. Under **Files**, choose **Link to a GitHub repository** and authorise HF to access your repo.
4. HF will clone the repo, install `requirements.txt`, and run `streamlit run app.py` automatically on every push to `main`.

#### Option B — Push directly to HF
```bash
# One-time: add the HF remote (replace <your-hf-username>)
git remote add hf https://huggingface.co/spaces/<your-hf-username>/ai-content-studio

# Push
git push hf main
```

### Adding Secrets (API keys)
The app reads keys from environment variables. In HF Spaces you set these as **Secrets** (not plain variables) so they are not exposed in logs.

Go to your Space → **Settings** → **Repository secrets** and add:

| Secret name          | Required? | Purpose                                      |
|----------------------|-----------|----------------------------------------------|
| `GEMINI_API_KEY`     | Optional  | Live generation via Google Gemini            |
| `OPENROUTER_API_KEY` | Optional  | Fallback live generation via OpenRouter      |
| `NVIDIA_API_KEY`     | Optional  | Live generation via NVIDIA NIM (OpenAI-compatible) |
| `SUPABASE_URL`       | Optional  | Persistent history + pgvector brand examples |
| `SUPABASE_KEY`       | Optional  | Supabase service-role or anon key            |

> **Zero-key mode:** if none of the above are set, the app starts in **mock mode** — it returns plausible placeholder content so the UI is fully explorable. This is the default for demos.

### Vector backend
Set the `VECTOR_BACKEND` secret (or leave it unset) to control RAG storage:

| Value      | Description                                        |
|------------|----------------------------------------------------|
| `memory`   | Default. Zero setup, resets on each Space restart. |
| `supabase` | Persistent pgvector store. Requires `SUPABASE_URL` and `SUPABASE_KEY`. |

ChromaDB (`chromadb` value) is **not** available on HF Spaces because the library is excluded from `requirements.txt` to keep the Space lean.

### What HF installs
HF runs `pip install -r requirements.txt`, which installs:
```
streamlit>=1.39.0
requests>=2.31.0
python-dotenv>=1.0.0
supabase>=2.4.0
```
Heavy ML libs (`sentence-transformers`, `chromadb`) are **not** installed; the app falls back to its built-in TF-IDF cosine similarity for brand-voice scoring.

---

## 2. GitHub

### First push
```bash
cd /path/to/ai-content-studio

# Initialise git (skip if already a repo)
git init
git add .
git commit -m "feat: initial commit — AI Content Studio"

# Create the remote repo (GitHub CLI)
gh repo create ai-content-studio --public --source=. --remote=origin --push

# Or manually:
# git remote add origin https://github.com/<you>/ai-content-studio.git
# git push -u origin main
```

### CI
`.github/workflows/` contains the CI configuration. On every push / pull request GitHub Actions will:
- Install `requirements-dev.txt` (includes `pytest`, `ruff`, and `chromadb`).
- Run `ruff check .` (lint).
- Run `pytest -q` (offline test suite — no API keys needed).

### Local development
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt   # full deps inc. chromadb + pytest
cp .env.example .env                  # add keys for live LLM output
streamlit run app.py
pytest -q
```

---

## Environment variables reference

| Variable             | Default  | Description                                      |
|----------------------|----------|--------------------------------------------------|
| `LLM_PROVIDER`       | `auto`   | `auto` \| `gemini` \| `openrouter` \| `mock`    |
| `GEMINI_API_KEY`     | —        | Google AI Studio key                             |
| `GEMINI_MODEL`       | `gemini-2.0-flash` | Model name                            |
| `OPENROUTER_API_KEY` | —        | OpenRouter key                                   |
| `VECTOR_BACKEND`     | `memory` | `memory` \| `supabase` (chromadb: local only)   |
| `SUPABASE_URL`       | —        | Supabase project URL                             |
| `SUPABASE_KEY`       | —        | Supabase anon or service-role key                |
| `TEMPERATURE`        | `0.7`    | LLM temperature (0–1)                            |
| `MAX_GENERATIONS`    | `5`      | Max variations per request                       |
