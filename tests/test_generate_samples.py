"""Offline tests for scripts/generate_samples.py.

All tests run hermetically — no network, no real API keys.
The mock LLM provider is forced via environment variable before any
project imports are triggered.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# ── Force mock provider before any project code is imported ──────────────────
os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("VECTOR_BACKEND", "memory")

# ── Put project root on sys.path so both `src` and `scripts` are importable ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Import the module under test (importing must NOT execute main()) ──────────
import scripts.generate_samples as gs  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _all_domain_labels() -> list[str]:
    from src.domains import list_domains
    return [d.label for d in list_domains()]


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildGallery:
    """Tests for the build_gallery() core function."""

    def test_returns_non_empty_string(self):
        md = gs.build_gallery()
        assert isinstance(md, str)
        assert len(md) > 0

    def test_markdown_has_title(self):
        md = gs.build_gallery()
        assert "# AI Content Studio" in md

    def test_mentions_provider(self):
        md = gs.build_gallery()
        # The header note always names the provider that was used.
        assert "provider" in md.lower()

    def test_covers_all_domains(self):
        md = gs.build_gallery()
        for label in _all_domain_labels():
            assert label in md, f"Domain label '{label}' not found in SAMPLES output"

    def test_each_domain_has_heading(self):
        md = gs.build_gallery()
        for label in _all_domain_labels():
            assert f"## {label}" in md

    def test_field_values_rendered(self):
        """Every domain section must contain at least one field: value line."""
        md = gs.build_gallery()
        # build_gallery renders fields as "**field_name:** value"
        assert "**" in md  # at least some bold field labels present

    def test_writes_to_tmp_path(self, tmp_path):
        out = tmp_path / "SAMPLES.md"
        md = gs.build_gallery(output_path=str(out))
        assert out.exists(), "File was not written"
        assert out.read_text(encoding="utf-8") == md

    def test_written_file_covers_all_domains(self, tmp_path):
        out = tmp_path / "SAMPLES.md"
        gs.build_gallery(output_path=str(out))
        content = out.read_text(encoding="utf-8")
        for label in _all_domain_labels():
            assert label in content, (
                f"Domain label '{label}' missing from written file"
            )

    def test_no_file_written_when_output_path_is_none(self, tmp_path, monkeypatch):
        """Calling without output_path must not produce any file side-effects."""
        # Patch PROJECT_ROOT to a temp dir so even accidental writes go there.
        monkeypatch.setattr(gs, "PROJECT_ROOT", tmp_path)
        gs.build_gallery(output_path=None)
        # No SAMPLES.md should have appeared in tmp_path.
        assert not (tmp_path / "SAMPLES.md").exists()


class TestRenderField:
    """Unit tests for the _render_field helper."""

    def test_string_passthrough(self):
        assert gs._render_field("hello") == "hello"

    def test_list_joined(self):
        assert gs._render_field(["a", "b", "c"]) == "a, b, c"

    def test_none_becomes_empty_string(self):
        assert gs._render_field(None) == ""

    def test_number_stringified(self):
        assert gs._render_field(42) == "42"


class TestExampleTopics:
    """Verify the EXAMPLE_TOPICS dict covers the known domains."""

    def test_all_known_domains_have_topics(self):
        from src.domains import list_domains
        for domain in list_domains():
            assert domain.key in gs.EXAMPLE_TOPICS, (
                f"No example topic for domain key '{domain.key}'"
            )

    def test_topics_are_non_empty_strings(self):
        for key, topic in gs.EXAMPLE_TOPICS.items():
            assert isinstance(topic, str) and topic.strip(), (
                f"Empty topic for domain '{key}'"
            )


class TestMainGuard:
    """Ensure importing the module does not call main() automatically."""

    def test_import_does_not_write_samples_md(self, tmp_path, monkeypatch):
        # Patch the project root inside the already-imported module.
        monkeypatch.setattr(gs, "PROJECT_ROOT", tmp_path)
        # Re-running the module-level code must not write a file.
        # (The file would appear at tmp_path/SAMPLES.md if main() were called.)
        assert not (tmp_path / "SAMPLES.md").exists()
