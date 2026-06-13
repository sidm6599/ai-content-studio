"""Quality scoring: embeddings-based brand-voice similarity and a
RAG vs No-RAG comparison helper.

Uses sentence-transformers if installed; otherwise falls back to a
dependency-free TF (bag-of-words) cosine so tests run offline.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List

_ST_MODEL = None
_ST_TRIED = False


def _try_sentence_transformer(model_name: str):
    global _ST_MODEL, _ST_TRIED
    if _ST_TRIED:
        return _ST_MODEL
    _ST_TRIED = True
    try:
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer(model_name)
    except Exception:
        _ST_MODEL = None
    return _ST_MODEL


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _tf_vector(text: str) -> Dict[str, float]:
    vec: Dict[str, float] = {}
    for tok in _tokenize(text):
        vec[tok] = vec.get(tok, 0.0) + 1.0
    return vec


def _tf_cosine(a: str, b: str) -> float:
    va, vb = _tf_vector(a), _tf_vector(b)
    if not va or not vb:
        return 0.0
    common = set(va) & set(vb)
    dot = sum(va[t] * vb[t] for t in common)
    na = math.sqrt(sum(v * v for v in va.values()))
    nb = math.sqrt(sum(v * v for v in vb.values()))
    return dot / (na * nb) if na and nb else 0.0


def embed_one(text: str, model_name: str = "all-MiniLM-L6-v2"):
    """Return a real embedding vector (list[float]) if sentence-transformers is
    available, else None. Needed for pgvector similarity search."""
    model = _try_sentence_transformer(model_name)
    if model is None:
        return None
    return model.encode([text], normalize_embeddings=True)[0].tolist()


def similarity(a: str, b: str, model_name: str = "all-MiniLM-L6-v2") -> float:
    """Cosine similarity in [0, 1] between two texts."""
    model = _try_sentence_transformer(model_name)
    if model is not None:
        import numpy as np
        emb = model.encode([a, b], normalize_embeddings=True)
        return float(max(0.0, min(1.0, np.dot(emb[0], emb[1]))))
    return _tf_cosine(a, b)


def brand_match(text: str, brand_examples: List[str],
                model_name: str = "all-MiniLM-L6-v2") -> float:
    """Best similarity of `text` to any brand example, as a 0-1 score."""
    if not brand_examples:
        return 0.0
    return max(similarity(text, ex, model_name) for ex in brand_examples)


def compare_rag_vs_norag(rag_texts: List[str], norag_texts: List[str],
                         brand_examples: List[str]) -> Dict[str, float]:
    """Average brand-match for each mode — the core dissertation-style metric."""
    def avg(texts):
        return round(sum(brand_match(t, brand_examples) for t in texts) / len(texts), 4) \
            if texts else 0.0
    rag, norag = avg(rag_texts), avg(norag_texts)
    return {
        "rag_brand_match": rag,
        "norag_brand_match": norag,
        "delta": round(rag - norag, 4),
        "rag_helps": rag >= norag,
    }
