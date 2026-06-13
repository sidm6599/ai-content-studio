"""Configuration loaded from environment variables (.env supported)."""
from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # python-dotenv optional
    pass


@dataclass
class Config:
    provider: str = "auto"          # auto | gemini | openrouter | nvidia | mock
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    openrouter_api_key: str = ""
    openrouter_model: str = "meta-llama/llama-3.1-8b-instruct"
    nvidia_api_key: str = ""
    nvidia_model: str = "meta/llama-3.3-70b-instruct"
    temperature: float = 0.7
    max_generations: int = 5
    embed_model: str = "all-MiniLM-L6-v2"
    vector_backend: str = "memory"  # memory | chromadb | supabase
    # Supabase persistence (optional)
    supabase_url: str = ""
    supabase_key: str = ""
    persist_history: bool = True

    @classmethod
    def from_env(cls) -> "Config":
        def _f(name, default):
            return os.getenv(name, default)
        return cls(
            provider=_f("LLM_PROVIDER", "auto").lower(),
            gemini_api_key=_f("GEMINI_API_KEY", ""),
            gemini_model=_f("GEMINI_MODEL", "gemini-2.0-flash"),
            openrouter_api_key=_f("OPENROUTER_API_KEY", ""),
            openrouter_model=_f("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct"),
            nvidia_api_key=_f("NVIDIA_API_KEY", ""),
            nvidia_model=_f("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct"),
            temperature=float(_f("TEMPERATURE", "0.7")),
            max_generations=int(_f("MAX_GENERATIONS", "5")),
            embed_model=_f("EMBED_MODEL", "all-MiniLM-L6-v2"),
            vector_backend=_f("VECTOR_BACKEND", "memory").lower(),
            supabase_url=_f("SUPABASE_URL", ""),
            supabase_key=_f("SUPABASE_KEY", ""),
            persist_history=_f("PERSIST_HISTORY", "true").lower() == "true",
        )

    def supabase_enabled(self) -> bool:
        return bool(self.supabase_url and self.supabase_key)

    def validate(self) -> None:
        if not 0.0 <= self.temperature <= 1.0:
            raise ValueError("TEMPERATURE must be between 0 and 1")
        if not 1 <= self.max_generations <= 10:
            raise ValueError("MAX_GENERATIONS must be between 1 and 10")
        if self.provider not in {"auto", "gemini", "openrouter", "nvidia", "mock"}:
            raise ValueError(f"Unknown provider: {self.provider}")

    def resolved_provider(self) -> str:
        """Pick a concrete provider given available keys."""
        if self.provider != "auto":
            return self.provider
        if self.gemini_api_key:
            return "gemini"
        if self.openrouter_api_key:
            return "openrouter"
        if self.nvidia_api_key:
            return "nvidia"
        return "mock"
