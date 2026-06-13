"""Offline tests for the blog/video domains and TONE_PRESETS additions."""
from __future__ import annotations

import pytest

from src.domains import DOMAINS, TONE_PRESETS, get_domain, list_tones

# ---------- blog domain ----------

def test_blog_domain_exists_in_domains_dict():
    assert "blog" in DOMAINS


def test_blog_domain_via_get_domain():
    domain = get_domain("blog")
    assert domain.key == "blog"


def test_blog_domain_has_non_empty_output_fields():
    domain = get_domain("blog")
    assert domain.output_fields, "blog output_fields must not be empty"
    assert all(isinstance(f, str) and f for f in domain.output_fields)


def test_blog_domain_has_expected_output_fields():
    domain = get_domain("blog")
    assert set(domain.output_fields) == {"title", "intro", "outline"}


def test_blog_domain_has_non_empty_instruction():
    domain = get_domain("blog")
    assert domain.instruction and len(domain.instruction) > 10


def test_blog_domain_has_expected_channels():
    domain = get_domain("blog")
    assert set(domain.channels) == {"Personal", "Company", "SEO"}


# ---------- video domain ----------

def test_video_domain_exists_in_domains_dict():
    assert "video" in DOMAINS


def test_video_domain_via_get_domain():
    domain = get_domain("video")
    assert domain.key == "video"


def test_video_domain_has_non_empty_output_fields():
    domain = get_domain("video")
    assert domain.output_fields, "video output_fields must not be empty"
    assert all(isinstance(f, str) and f for f in domain.output_fields)


def test_video_domain_has_expected_output_fields():
    domain = get_domain("video")
    assert set(domain.output_fields) == {"hook", "script", "caption"}


def test_video_domain_has_non_empty_instruction():
    domain = get_domain("video")
    assert domain.instruction and len(domain.instruction) > 10


def test_video_domain_has_expected_channels():
    domain = get_domain("video")
    assert set(domain.channels) == {"TikTok", "YouTube Shorts", "Reels"}


# ---------- TONE_PRESETS and list_tones ----------

def test_tone_presets_is_non_empty_list():
    assert isinstance(TONE_PRESETS, list)
    assert len(TONE_PRESETS) > 0


def test_tone_presets_contains_strings():
    for tone in TONE_PRESETS:
        assert isinstance(tone, str) and tone


def test_list_tones_returns_tone_presets():
    assert list_tones() == list(TONE_PRESETS)


def test_list_tones_returns_new_list():
    """list_tones() should return a copy, not the original list object."""
    assert list_tones() is not TONE_PRESETS


def test_tone_presets_contains_expected_values():
    expected = {"professional", "playful", "concise", "inspirational",
                "informative", "luxury", "gen-z"}
    assert expected.issubset(set(TONE_PRESETS))
