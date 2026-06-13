"""Localization helpers for AI Content Studio.

Provides a small, dependency-free i18n layer so the UI and prompt layer
can request output in a specific language without any external packages.
"""
from __future__ import annotations

from typing import List

SUPPORTED_LANGUAGES: List[str] = [
    "English",
    "German",
    "Hindi",
    "Spanish",
    "French",
    "Portuguese",
    "Arabic",
    "Japanese",
]

DEFAULT_LANGUAGE = "English"


def list_languages() -> List[str]:
    """Return the list of supported languages (English first)."""
    return list(SUPPORTED_LANGUAGES)


def language_instruction(language: str) -> str:
    """Return a prompt instruction string for the given language.

    Returns an empty string for English, an empty/unknown language, or any
    value that should be treated as pass-through (i.e. no explicit language
    instruction is needed).  For all other recognised and unrecognised non-
    English values a safe, clear instruction string is returned so the caller
    can append it to a system prompt without conditional logic.

    Args:
        language: A language name such as ``"German"`` or ``"Spanish"``.

    Returns:
        ``""`` when no explicit instruction is needed, otherwise a string of
        the form ``"Write ALL output (captions, hashtags, body, etc.) in
        {language}."``.
    """
    if not language or language.strip().lower() == DEFAULT_LANGUAGE.lower():
        return ""
    lang = language.strip()
    return f"Write ALL output (captions, hashtags, body, etc.) in {lang}."


def is_supported(language: str) -> bool:
    """Return ``True`` if *language* is in :data:`SUPPORTED_LANGUAGES`.

    The comparison is case-insensitive so that ``"german"`` and ``"German"``
    both return ``True``.

    Args:
        language: A language name to test.

    Returns:
        ``True`` when the language is explicitly supported, ``False`` otherwise.
    """
    if not language:
        return False
    lower = language.strip().lower()
    return any(lang.lower() == lower for lang in SUPPORTED_LANGUAGES)
