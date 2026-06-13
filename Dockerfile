# Dockerfile — AI Content Studio
# Production-ready image for running the Streamlit app.
#
# Also works as a Hugging Face Spaces "Docker" SDK option.
# HF Docker Spaces expect the app on port 7860 by default.
# To deploy on HF, uncomment the ENV / EXPOSE / CMD lines
# labelled "HF Spaces" and comment out the standard ones.

FROM python:3.11-slim

# Prevent .pyc files and enable unbuffered stdout/stderr for container logs.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# --- layer-cache friendly: install deps before copying the full app ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application (see .dockerignore for exclusions).
COPY . .

# ---------------------------------------------------------------------------
# Standard deployment — port 8501
# ---------------------------------------------------------------------------
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]

# ---------------------------------------------------------------------------
# Hugging Face Spaces (Docker SDK) — port 7860
# Uncomment the three lines below and comment out the three lines above.
# ---------------------------------------------------------------------------
# ENV PORT=7860
# EXPOSE 7860
# CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
