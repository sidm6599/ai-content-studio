"""Offline tests for src/i18n.py — no LLM calls, no network."""
from __future__ import annotations

import pytest

from src.i18n import (
    DEFAULT_LANGUAGE,
    SUPPORTED_LANGUAGES,
    is_supported,
    language_instruction,
    list_languages,
)

# ---------------------------------------------------------------------------
# SUPPORTED_LANGUAGES constant
# ---------------------------------------------------------------------------

class TestSupportedLanguages:
    def test_non_empty(self) -> None:
        assert len(SUPPORTED_LANGUAGES) > 0

    def test_english_is_first(self) -> None:
        assert SUPPORTED_LANGUAGES[0] == "English"

    def test_contains_required_languages(self) -> None:
        required = {"English", "German", "Hindi", "Spanish", "French",
                    "Portuguese", "Arabic", "Japanese"}
        assert required.issubset(set(SUPPORTED_LANGUAGES))

    def test_default_language_is_english(self) -> None:
        assert DEFAULT_LANGUAGE == "English"


# ---------------------------------------------------------------------------
# list_languages()
# ---------------------------------------------------------------------------

class TestListLanguages:
    def test_returns_supported_languages(self) -> None:
        assert list_languages() == SUPPORTED_LANGUAGES

    def test_returns_a_copy(self) -> None:
        # Mutating the return value must not alter the module constant.
        result = list_languages()
        result.append("Klingon")
        assert "Klingon" not in SUPPORTED_LANGUAGES

    def test_return_type_is_list(self) -> None:
        assert isinstance(list_languages(), list)


# ---------------------------------------------------------------------------
# language_instruction()
# ---------------------------------------------------------------------------

class TestLanguageInstruction:
    def test_english_returns_empty_string(self) -> None:
        assert language_instruction("English") == ""

    def test_empty_string_returns_empty_string(self) -> None:
        assert language_instruction("") == ""

    def test_german_contains_german(self) -> None:
        result = language_instruction("German")
        assert "German" in result

    def test_german_is_non_empty(self) -> None:
        assert language_instruction("German") != ""

    def test_spanish_contains_spanish(self) -> None:
        assert "Spanish" in language_instruction("Spanish")

    def test_hindi_contains_hindi(self) -> None:
        assert "Hindi" in language_instruction("Hindi")

    def test_french_contains_french(self) -> None:
        assert "French" in language_instruction("French")

    def test_unknown_language_is_non_empty(self) -> None:
        # Unknown languages still get an instruction (safe pass-through).
        assert language_instruction("Swahili") != ""

    def test_unknown_language_contains_language_name(self) -> None:
        assert "Swahili" in language_instruction("Swahili")

    def test_return_type_is_str(self) -> None:
        assert isinstance(language_instruction("Japanese"), str)

    def test_english_case_insensitive_passthrough(self) -> None:
        # Lower-case "english" should also be treated as pass-through.
        assert language_instruction("english") == ""


# ---------------------------------------------------------------------------
# is_supported()
# ---------------------------------------------------------------------------

class TestIsSupported:
    def test_english_is_supported(self) -> None:
        assert is_supported("English") is True

    def test_german_is_supported(self) -> None:
        assert is_supported("German") is True

    def test_all_required_languages_supported(self) -> None:
        for lang in ["English", "German", "Hindi", "Spanish", "French",
                     "Portuguese", "Arabic", "Japanese"]:
            assert is_supported(lang) is True, f"{lang!r} should be supported"

    def test_case_insensitive_lower(self) -> None:
        assert is_supported("english") is True
        assert is_supported("german") is True

    def test_case_insensitive_upper(self) -> None:
        assert is_supported("ENGLISH") is True

    def test_unsupported_language_returns_false(self) -> None:
        assert is_supported("Klingon") is False

    def test_empty_string_returns_false(self) -> None:
        assert is_supported("") is False

    def test_return_type_is_bool(self) -> None:
        result = is_supported("English")
        assert isinstance(result, bool)

    def test_unsupported_returns_bool_false(self) -> None:
        result = is_supported("Martian")
        assert isinstance(result, bool)
        assert result is False
