"""Offline tests for multimodal image input and language localisation.

All tests use the ``mock`` provider — no API keys or network required.
"""
from __future__ import annotations

import pytest

from src.config import Config
from src.domains import get_domain
from src.generator import ContentGenerator, ContentRequest
from src.llm_client import LLMClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_gen() -> ContentGenerator:
    return ContentGenerator(Config(provider="mock"))


def _mock_client() -> LLMClient:
    return LLMClient(Config(provider="mock"))


_TINY_PNG = bytes(
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
    b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
    b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


# ---------------------------------------------------------------------------
# 1. ContentRequest field defaults
# ---------------------------------------------------------------------------

class TestContentRequestDefaults:
    def test_image_bytes_defaults_to_none(self):
        req = ContentRequest(domain="social", topic="test")
        assert req.image_bytes is None

    def test_image_mime_defaults_to_png(self):
        req = ContentRequest(domain="social", topic="test")
        assert req.image_mime == "image/png"

    def test_language_defaults_to_english(self):
        req = ContentRequest(domain="social", topic="test")
        assert req.language == "English"

    def test_can_set_image_bytes_and_mime(self):
        req = ContentRequest(
            domain="social",
            topic="test",
            image_bytes=_TINY_PNG,
            image_mime="image/jpeg",
        )
        assert req.image_bytes == _TINY_PNG
        assert req.image_mime == "image/jpeg"

    def test_can_set_language(self):
        req = ContentRequest(domain="social", topic="test", language="French")
        assert req.language == "French"

    def test_existing_fields_still_work(self):
        req = ContentRequest(
            domain="email",
            topic="onboarding",
            channel="newsletter",
            tone="friendly",
            num_items=2,
            use_rag=False,
        )
        assert req.channel == "newsletter"
        assert req.num_items == 2


# ---------------------------------------------------------------------------
# 2. build_prompt — language instruction
# ---------------------------------------------------------------------------

class TestBuildPromptLanguage:
    def setup_method(self):
        self.gen = _mock_gen()
        self.domain = get_domain("social")

    def test_english_omits_language_line(self):
        req = ContentRequest(domain="social", topic="launch", language="English")
        prompt = self.gen.build_prompt(req, self.domain)
        assert "Write ALL output in" not in prompt

    def test_default_language_omits_language_line(self):
        # default is "English" — same as explicit English
        req = ContentRequest(domain="social", topic="launch")
        prompt = self.gen.build_prompt(req, self.domain)
        assert "Write ALL output in" not in prompt

    def test_empty_language_omits_language_line(self):
        req = ContentRequest(domain="social", topic="launch", language="")
        prompt = self.gen.build_prompt(req, self.domain)
        assert "Write ALL output in" not in prompt

    def test_non_english_appends_language_line(self):
        req = ContentRequest(domain="social", topic="launch", language="French")
        prompt = self.gen.build_prompt(req, self.domain)
        assert "Write ALL output in French." in prompt

    def test_spanish_appended(self):
        req = ContentRequest(domain="social", topic="lanzamiento", language="Spanish")
        prompt = self.gen.build_prompt(req, self.domain)
        assert "Write ALL output in Spanish." in prompt

    def test_language_line_appears_after_json_instruction(self):
        """Language instruction must be the last line of the prompt."""
        req = ContentRequest(domain="social", topic="launch", language="German")
        prompt = self.gen.build_prompt(req, self.domain)
        lines = prompt.strip().splitlines()
        assert lines[-1] == "Write ALL output in German."

    def test_language_case_insensitive_english(self):
        # "english" (lower-case) should still be treated as English
        req = ContentRequest(domain="social", topic="launch", language="english")
        prompt = self.gen.build_prompt(req, self.domain)
        assert "Write ALL output in" not in prompt


# ---------------------------------------------------------------------------
# 3. generate() works without image (mock provider, multiple domains)
# ---------------------------------------------------------------------------

class TestGenerateNoImage:
    def test_social_domain_no_image(self):
        items = _mock_gen().generate(
            ContentRequest(domain="social", topic="product launch", num_items=2,
                           use_rag=False)
        )
        assert len(items) == 2
        for it in items:
            assert set(it.fields) == set(get_domain("social").output_fields)

    def test_email_domain_no_image(self):
        items = _mock_gen().generate(
            ContentRequest(domain="email", topic="welcome series", num_items=1,
                           use_rag=False)
        )
        assert len(items) == 1
        assert set(items[0].fields) == set(get_domain("email").output_fields)

    def test_resume_domain_no_image(self):
        items = _mock_gen().generate(
            ContentRequest(domain="resume", topic="built a RAG pipeline",
                           num_items=1, use_rag=False)
        )
        assert isinstance(items[0].fields["bullets"], list)

    def test_generate_with_language_still_returns_items(self):
        items = _mock_gen().generate(
            ContentRequest(domain="social", topic="test topic", num_items=1,
                           use_rag=False, language="French")
        )
        assert len(items) == 1


# ---------------------------------------------------------------------------
# 4. generate() works WITH image (mock provider ignores image gracefully)
# ---------------------------------------------------------------------------

class TestGenerateWithImage:
    def test_generate_with_image_bytes_returns_items(self):
        items = _mock_gen().generate(
            ContentRequest(
                domain="social",
                topic="product screenshot",
                num_items=2,
                use_rag=False,
                image_bytes=_TINY_PNG,
                image_mime="image/png",
            )
        )
        assert len(items) == 2

    def test_generate_with_jpeg_image(self):
        # Even non-PNG bytes are accepted; mock just ignores them
        items = _mock_gen().generate(
            ContentRequest(
                domain="blog",
                topic="travel photo",
                num_items=1,
                use_rag=False,
                image_bytes=b"\xff\xd8\xff\xe0some-jpeg-data",
                image_mime="image/jpeg",
            )
        )
        assert len(items) == 1


# ---------------------------------------------------------------------------
# 5. LLMClient.generate() accepts image kwargs in mock mode
# ---------------------------------------------------------------------------

class TestLLMClientImageKwargs:
    def test_generate_without_image_kwargs(self):
        client = _mock_client()
        result = client.generate("NUM_ITEMS: 1\nFIELDS: caption\nINPUT: hello")
        assert "caption" in result

    def test_generate_with_image_bytes_none(self):
        client = _mock_client()
        result = client.generate(
            "NUM_ITEMS: 1\nFIELDS: caption\nINPUT: hello",
            image_bytes=None,
        )
        assert "caption" in result

    def test_generate_with_image_bytes_populated(self):
        client = _mock_client()
        result = client.generate(
            "NUM_ITEMS: 1\nFIELDS: caption\nINPUT: photo",
            image_bytes=_TINY_PNG,
            image_mime="image/png",
        )
        # mock ignores image; result is still valid JSON
        import json
        data = json.loads(result)
        assert isinstance(data, list)
        assert "caption" in data[0]

    def test_generate_with_jpeg_mime(self):
        client = _mock_client()
        result = client.generate(
            "NUM_ITEMS: 1\nFIELDS: caption\nINPUT: jpeg",
            image_bytes=b"\xff\xd8\xff\xe0",
            image_mime="image/jpeg",
        )
        assert "caption" in result
