"""Tests for src/engagement.py — run offline, no extra packages required.

All tests pass whether or not scikit-learn is installed.
"""
from __future__ import annotations

import pytest

from src.engagement import (
    EngagementPredictor,
    extract_features,
    predict_engagement,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def predictor() -> EngagementPredictor:
    """A single predictor instance shared across the module-scoped tests."""
    return EngagementPredictor()


# Texts used across multiple tests
PUNCHY_TEXT = (
    "🚀 This changes EVERYTHING! Comment below if you want early access — "
    "only 50 spots left! #Launch #AI #BuildInPublic #MustHave"
)

BLAND_TEXT = (
    "Please note that scheduled maintenance will occur this Sunday. "
    "The service will be unavailable from 2am to 4am UTC."
)

# ---------------------------------------------------------------------------
# Return-shape tests
# ---------------------------------------------------------------------------

class TestReturnShape:
    """The predict() dict must always have the right shape and value ranges."""

    @pytest.mark.parametrize("text", [
        PUNCHY_TEXT,
        BLAND_TEXT,
        "Hello world",
        "",  # edge case: empty string
        "Just one word",
        "Numbers: 42% increase and 3x ROI achieved! 🎉 #Win",
    ])
    def test_predict_keys(self, predictor, text):
        result = predictor.predict(text)
        assert set(result.keys()) == {"tier", "score", "signals"}, (
            f"Missing or extra keys in result: {result.keys()}"
        )

    @pytest.mark.parametrize("text", [
        PUNCHY_TEXT,
        BLAND_TEXT,
        "",
        "Hello",
    ])
    def test_tier_is_valid_value(self, predictor, text):
        tier = predictor.predict(text)["tier"]
        assert tier in ("low", "medium", "high"), f"Unexpected tier: {tier!r}"

    @pytest.mark.parametrize("text", [
        PUNCHY_TEXT,
        BLAND_TEXT,
        "",
        "Hello",
    ])
    def test_score_in_unit_interval(self, predictor, text):
        score = predictor.predict(text)["score"]
        assert isinstance(score, float), f"score is not float: {type(score)}"
        assert 0.0 <= score <= 1.0, f"score out of [0,1]: {score}"

    @pytest.mark.parametrize("text", [
        PUNCHY_TEXT,
        BLAND_TEXT,
        "",
    ])
    def test_signals_is_dict(self, predictor, text):
        signals = predictor.predict(text)["signals"]
        assert isinstance(signals, dict), "signals must be a dict"
        assert len(signals) > 0, "signals dict must not be empty"

    def test_signals_contains_expected_keys(self, predictor):
        signals = predictor.predict(PUNCHY_TEXT)["signals"]
        expected_keys = {
            "char_len", "word_count", "hashtag_count",
            "emoji_count", "has_emoji",
            "exclamation_count", "question_count",
            "cta_count", "has_cta", "number_count",
        }
        assert expected_keys.issubset(signals.keys()), (
            f"Missing signal keys: {expected_keys - signals.keys()}"
        )

    def test_signals_all_numeric(self, predictor):
        signals = predictor.predict(PUNCHY_TEXT)["signals"]
        for k, v in signals.items():
            assert isinstance(v, (int, float)), (
                f"Signal '{k}' has non-numeric value: {v!r}"
            )


# ---------------------------------------------------------------------------
# Relative ordering test (punchy >= bland)
# ---------------------------------------------------------------------------

class TestRelativeOrdering:
    """A clearly punchy text must score at least as high as a clearly bland one."""

    def test_punchy_scores_higher_than_bland(self, predictor):
        punchy_score = predictor.predict(PUNCHY_TEXT)["score"]
        bland_score = predictor.predict(BLAND_TEXT)["score"]
        assert punchy_score >= bland_score, (
            f"Expected punchy ({punchy_score:.4f}) >= bland ({bland_score:.4f})"
        )

    def test_punchy_is_not_low_tier(self, predictor):
        tier = predictor.predict(PUNCHY_TEXT)["tier"]
        assert tier != "low", f"Punchy text should not be 'low', got {tier!r}"

    def test_bland_is_not_high_tier(self, predictor):
        tier = predictor.predict(BLAND_TEXT)["tier"]
        assert tier != "high", f"Bland text should not be 'high', got {tier!r}"


# ---------------------------------------------------------------------------
# Feature extraction tests
# ---------------------------------------------------------------------------

class TestFeatureExtraction:
    """extract_features() must return correct signal counts."""

    def test_hashtag_counting(self):
        feats = extract_features("Check this out! #AI #ML #Data")
        assert feats["hashtag_count"] == 3.0

    def test_exclamation_counting(self):
        feats = extract_features("Wow! Amazing! Incredible!")
        assert feats["exclamation_count"] == 3.0

    def test_question_counting(self):
        feats = extract_features("What do you think? Is this good? Really?")
        assert feats["question_count"] == 3.0

    def test_emoji_detection(self):
        feats = extract_features("Hello 🎉 World 🚀")
        assert feats["has_emoji"] == 1.0
        assert feats["emoji_count"] >= 2.0

    def test_no_emoji(self):
        feats = extract_features("Plain text with no emoji here.")
        assert feats["has_emoji"] == 0.0
        assert feats["emoji_count"] == 0.0

    def test_cta_detection(self):
        feats = extract_features("Click here and share with your friends!")
        assert feats["has_cta"] == 1.0
        assert feats["cta_count"] >= 2.0

    def test_number_detection(self):
        feats = extract_features("We grew 42% and acquired 1000 new users!")
        assert feats["number_count"] >= 2.0

    def test_char_and_word_count(self):
        text = "Hello world foo"
        feats = extract_features(text)
        assert feats["char_len"] == float(len(text))
        assert feats["word_count"] == 3.0

    def test_empty_text_does_not_crash(self):
        feats = extract_features("")
        assert feats["word_count"] == 0.0
        assert feats["char_len"] == 0.0


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

class TestModuleLevelFunction:
    """predict_engagement() must behave identically to EngagementPredictor.predict()."""

    def test_returns_correct_shape(self):
        result = predict_engagement(PUNCHY_TEXT)
        assert set(result.keys()) == {"tier", "score", "signals"}

    def test_tier_valid(self):
        assert predict_engagement(BLAND_TEXT)["tier"] in ("low", "medium", "high")

    def test_score_valid(self):
        score = predict_engagement(PUNCHY_TEXT)["score"]
        assert 0.0 <= score <= 1.0

    def test_singleton_reuse(self):
        """Calling twice should return consistent results (same model)."""
        r1 = predict_engagement(PUNCHY_TEXT)
        r2 = predict_engagement(PUNCHY_TEXT)
        assert r1["score"] == r2["score"]
        assert r1["tier"] == r2["tier"]

    def test_punchy_beats_bland(self):
        assert predict_engagement(PUNCHY_TEXT)["score"] >= predict_engagement(BLAND_TEXT)["score"]
