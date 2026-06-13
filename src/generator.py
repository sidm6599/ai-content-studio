"""Orchestrates: request -> (optional RAG) -> prompt -> LLM -> parse -> quality."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from . import quality
from .config import Config
from .domains import Domain, get_domain
from .llm_client import LLMClient
from .vector_store import MemoryVectorStore

logger = logging.getLogger(__name__)


@dataclass
class ContentRequest:
    domain: str
    topic: str
    channel: str = ""
    tone: str = "neutral"
    num_items: int = 3
    use_rag: bool = True


@dataclass
class GeneratedItem:
    fields: Dict[str, Any]
    brand_match: Optional[float] = None

    def primary_text(self) -> str:
        """A representative text blob used for scoring/length checks."""
        parts = []
        for v in self.fields.values():
            if isinstance(v, str):
                parts.append(v)
            elif isinstance(v, list):
                parts.append(" ".join(map(str, v)))
        return " ".join(parts)


class ContentGenerator:
    def __init__(self, config: Config,
                 vector_store: Optional[MemoryVectorStore] = None):
        self.config = config
        self.client = LLMClient(config)
        self.vector_store = vector_store

    # ---- prompt construction -----------------------------------------
    def build_prompt(self, req: ContentRequest, domain: Domain,
                     context: Optional[List[str]] = None) -> str:
        lines = [
            domain.instruction,
            f"NUM_ITEMS: {req.num_items}",
            f"FIELDS: {', '.join(domain.output_fields)}",
            f"CHANNEL: {req.channel or 'generic'}",
            f"TONE: {req.tone}",
        ]
        if context:
            lines.append("BRAND CONTEXT (match this style):")
            lines.extend(f"- {c}" for c in context)
        lines.append(f"INPUT: {req.topic}")
        lines.append(
            "Return ONLY a JSON array of "
            f"{req.num_items} objects, each with keys: "
            f"{', '.join(domain.output_fields)}."
        )
        return "\n".join(lines)

    # ---- main entry ---------------------------------------------------
    def generate(self, req: ContentRequest) -> List[GeneratedItem]:
        domain = get_domain(req.domain)
        context, brand_examples = self._retrieve(req)
        prompt = self.build_prompt(req, domain, context)
        raw = self.client.generate(prompt)
        items = self._parse(raw, domain)
        # score brand match against retrieved/brand examples
        if brand_examples:
            for it in items:
                it.brand_match = round(
                    quality.brand_match(it.primary_text(), brand_examples,
                                        self.config.embed_model), 4)
        return items

    def _retrieve(self, req: ContentRequest):
        if not (req.use_rag and self.vector_store and len(self.vector_store)):
            return None, []
        hits = self.vector_store.retrieve(req.topic, k=3, domain=req.domain)
        context = [h.text for h in hits]
        return context, context

    def _parse(self, raw: str, domain: Domain) -> List[GeneratedItem]:
        data = _loads_lenient(raw)
        if isinstance(data, dict):
            data = [data]
        items: List[GeneratedItem] = []
        for obj in data:
            fields = {f: obj.get(f, "") for f in domain.output_fields}
            items.append(GeneratedItem(fields=fields))
        return items


def _loads_lenient(raw: str):
    """Parse JSON even if wrapped in markdown fences or prose."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[raw.find("\n") + 1:] if "\n" in raw else raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1:
            return json.loads(raw[start:end + 1])
        raise
