"""Multi-provider LLM client with automatic fallback.

Order of preference (when provider='auto'): Gemini -> OpenRouter -> mock.
The `mock` provider returns deterministic templated JSON so the whole app
(and the test suite) runs with zero API keys and no network.
"""
from __future__ import annotations

import json
import logging
from typing import List

import requests

from .config import Config

logger = logging.getLogger(__name__)


class LLMError(Exception):
    pass


class LLMClient:
    def __init__(self, config: Config):
        self.config = config
        self.provider = config.resolved_provider()

    # ---- public API ---------------------------------------------------
    def generate(self, prompt: str) -> str:
        """Return raw model text (expected to be a JSON array string)."""
        for provider in self._provider_chain():
            try:
                return getattr(self, f"_call_{provider}")(prompt)
            except Exception as e:  # try the next provider
                logger.warning("Provider %s failed: %s", provider, e)
        raise LLMError("All providers failed")

    def _provider_chain(self) -> List[str]:
        if self.provider == "mock":
            return ["mock"]
        chain = [self.provider]
        # graceful fallbacks
        if self.provider == "gemini" and self.config.openrouter_api_key:
            chain.append("openrouter")
        chain.append("mock")
        return chain

    # ---- providers ----------------------------------------------------
    def _call_gemini(self, prompt: str) -> str:
        if not self.config.gemini_api_key:
            raise LLMError("No Gemini key")
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.config.gemini_model}:generateContent"
            f"?key={self.config.gemini_api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.config.temperature,
                "responseMimeType": "application/json",
            },
        }
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]

    def _call_openrouter(self, prompt: str) -> str:
        if not self.config.openrouter_api_key:
            raise LLMError("No OpenRouter key")
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.config.openrouter_api_key}"}
        payload = {
            "model": self.config.openrouter_model,
            "temperature": self.config.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        r = requests.post(url, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def _call_mock(self, prompt: str) -> str:
        """Deterministic offline generator. Reads how many items and which
        fields were requested from the prompt and returns valid JSON."""
        n = _extract_int(prompt, "NUM_ITEMS", 1)
        fields = _extract_list(prompt, "FIELDS")
        topic = _extract_after(prompt, "INPUT:", "(demo)")
        items = []
        for i in range(max(1, n)):
            item = {}
            for f in fields:
                if f == "hashtags":
                    item[f] = ["AI", "ContentStudio", "Demo"]
                elif f == "bullet_points" or f == "bullets":
                    item[f] = [
                        f"Delivered {topic} with measurable impact ([X%])",
                        f"Built and shipped {topic} end to end",
                        "Collaborated cross-functionally to hit the deadline",
                    ]
                elif f == "emojis":
                    item[f] = "✨🚀"
                else:
                    item[f] = f"[{f}] for '{topic}' — variation {i + 1} (mock output)"
            items.append(item)
        return json.dumps(items)


# ---- tiny prompt-marker helpers (shared with generator.py) ------------
def _extract_int(text: str, marker: str, default: int) -> int:
    for line in text.splitlines():
        if line.strip().startswith(marker):
            try:
                return int(line.split(":", 1)[1].strip())
            except Exception:
                return default
    return default


def _extract_list(text: str, marker: str) -> List[str]:
    for line in text.splitlines():
        if line.strip().startswith(marker):
            raw = line.split(":", 1)[1].strip()
            return [x.strip() for x in raw.split(",") if x.strip()]
    return ["caption"]


def _extract_after(text: str, marker: str, default: str) -> str:
    idx = text.find(marker)
    if idx == -1:
        return default
    return text[idx + len(marker):].strip().splitlines()[0][:80] or default
