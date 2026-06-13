"""Offline tests for the NVIDIA NIM provider wiring (no network)."""
from src.config import Config
from src.llm_client import LLMClient


def test_nvidia_is_valid_provider():
    Config(provider="nvidia").validate()  # must not raise


def test_auto_resolves_to_nvidia_when_only_nvidia_key():
    cfg = Config(provider="auto", nvidia_api_key="nvapi-test")
    assert cfg.resolved_provider() == "nvidia"


def test_gemini_and_openrouter_take_priority_over_nvidia():
    assert Config(provider="auto", gemini_api_key="x",
                  nvidia_api_key="y").resolved_provider() == "gemini"
    assert Config(provider="auto", openrouter_api_key="x",
                  nvidia_api_key="y").resolved_provider() == "openrouter"


def test_nvidia_falls_back_to_mock_in_chain():
    # No key → provider chain still ends at mock so generation never hard-fails.
    client = LLMClient(Config(provider="nvidia"))
    assert client._provider_chain()[-1] == "mock"


def test_nvidia_appended_as_fallback_for_gemini():
    cfg = Config(provider="gemini", gemini_api_key="g", nvidia_api_key="n")
    chain = LLMClient(cfg)._provider_chain()
    assert "nvidia" in chain and chain[-1] == "mock"
