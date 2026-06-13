"""Engagement predictor: estimates how engaging a piece of content is.

Extracts transparent text features and either:
- Trains a small sklearn classifier (LogisticRegression) on bundled samples
  if scikit-learn is importable, or
- Falls back to a deterministic weighted heuristic over the same features.

Both paths return the same dict shape from ``predict()``.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Optional sklearn import — detect once at module load time
# ---------------------------------------------------------------------------
try:
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore

    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False

# Path to the bundled labeled samples
_SAMPLES_PATH = Path(__file__).parent.parent / "knowledge" / "engagement_samples.json"

# Tier ordering used for numeric encoding
_TIER_ORDER = {"low": 0, "medium": 1, "high": 2}
_TIER_LABELS = ["low", "medium", "high"]

# Call-to-action keyword patterns (case-insensitive)
_CTA_PATTERNS = re.compile(
    r"\b(comment|reply|share|tag|click|tap|swipe|subscribe|follow|join|"
    r"grab|get|shop|buy|sign up|register|download|try|learn more|"
    r"find out|save this|check out|link in bio|dm me|drop a)\b",
    re.IGNORECASE,
)

# Emoji detection via Unicode category (So = Symbol-Other, covers most emoji blocks)
def _count_emoji(text: str) -> int:
    """Count emoji characters using Unicode category and supplemental ranges."""
    count = 0
    for ch in text:
        cp = ord(ch)
        # Common emoji ranges
        if (
            0x1F300 <= cp <= 0x1FAFF  # Misc symbols, emoticons, transport, etc.
            or 0x2600 <= cp <= 0x27BF  # Misc symbols, dingbats
            or 0x1F000 <= cp <= 0x1F02F  # Mahjong/domino tiles (sometimes emoji)
            or unicodedata.category(ch) in ("So", "Sm")
            and cp > 0x2000  # Symbol-other, above ASCII math
        ):
            count += 1
    return count


def extract_features(text: str) -> Dict[str, float]:
    """Extract interpretable engagement signals from *text*.

    Returns a dict of feature_name -> numeric value, suitable for use in both
    the heuristic and ML paths.
    """
    char_len = len(text)
    words = re.findall(r"\S+", text)
    word_count = len(words)
    hashtag_count = len(re.findall(r"#\w+", text))
    emoji_count = _count_emoji(text)
    exclamation_count = text.count("!")
    question_count = text.count("?")
    cta_count = len(_CTA_PATTERNS.findall(text))
    # numbers / percentages — digits or explicit "%" / "X times" patterns
    number_count = len(re.findall(r"\d+(?:[.,]\d+)?%?", text))

    return {
        "char_len": float(char_len),
        "word_count": float(word_count),
        "hashtag_count": float(hashtag_count),
        "emoji_count": float(emoji_count),
        "has_emoji": float(emoji_count > 0),
        "exclamation_count": float(exclamation_count),
        "question_count": float(question_count),
        "cta_count": float(cta_count),
        "has_cta": float(cta_count > 0),
        "number_count": float(number_count),
    }


def _features_to_list(feats: Dict[str, float]) -> list[float]:
    """Stable ordering for sklearn feature vector."""
    return [
        feats["char_len"],
        feats["word_count"],
        feats["hashtag_count"],
        feats["emoji_count"],
        feats["has_emoji"],
        feats["exclamation_count"],
        feats["question_count"],
        feats["cta_count"],
        feats["has_cta"],
        feats["number_count"],
    ]


# ---------------------------------------------------------------------------
# Heuristic scorer (dependency-free fallback)
# ---------------------------------------------------------------------------

# Weights assigned to each signal — calibrated so a score near 0 = bland,
# near 1 = highly engaging. The final score is clipped to [0, 1].
_WEIGHTS: Dict[str, float] = {
    "has_emoji": 0.15,
    "emoji_count": 0.03,        # per emoji, up to saturation
    "exclamation_count": 0.06,
    "question_count": 0.07,
    "has_cta": 0.18,
    "cta_count": 0.04,
    "hashtag_count": 0.05,
    "number_count": 0.04,
}

# Length sweet-spot: 60-280 chars (social-post sweet-spot) gets a bonus
_LEN_BONUS_LOW = 60
_LEN_BONUS_HIGH = 320


def _heuristic_score(feats: Dict[str, float]) -> float:
    """Return a raw score in roughly [0, 1] using hand-tuned weights."""
    raw = 0.0

    # Categorical / boolean signals
    raw += _WEIGHTS["has_emoji"] * feats["has_emoji"]
    raw += _WEIGHTS["emoji_count"] * math.log1p(feats["emoji_count"]) * 0.5
    raw += _WEIGHTS["exclamation_count"] * math.log1p(feats["exclamation_count"])
    raw += _WEIGHTS["question_count"] * math.log1p(feats["question_count"])
    raw += _WEIGHTS["has_cta"] * feats["has_cta"]
    raw += _WEIGHTS["cta_count"] * math.log1p(feats["cta_count"]) * 0.5
    raw += _WEIGHTS["hashtag_count"] * math.log1p(feats["hashtag_count"]) * 0.4
    raw += _WEIGHTS["number_count"] * math.log1p(feats["number_count"]) * 0.3

    # Length bonus
    char_len = feats["char_len"]
    if _LEN_BONUS_LOW <= char_len <= _LEN_BONUS_HIGH:
        raw += 0.08

    # Baseline floor so non-zero text always has some score
    if feats["word_count"] > 0:
        raw += 0.05

    return float(min(max(raw, 0.0), 1.0))


def _score_to_tier(score: float) -> str:
    if score >= 0.55:
        return "high"
    if score >= 0.28:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# EngagementPredictor class
# ---------------------------------------------------------------------------

class EngagementPredictor:
    """Predicts content engagement tier and score.

    If scikit-learn is available the predictor trains a LogisticRegression
    classifier on the bundled ``knowledge/engagement_samples.json`` at
    initialisation time and uses it for inference.

    Otherwise it falls back to a transparent weighted heuristic that has no
    third-party dependencies and always produces the same dict shape.

    Args:
        samples_path: Path to a JSON file with ``{"text": ..., "engagement": ...}``
            records.  Defaults to the bundled ``knowledge/engagement_samples.json``.
    """

    def __init__(self, samples_path: Path | str | None = None) -> None:
        self._path = Path(samples_path) if samples_path else _SAMPLES_PATH
        self._use_sklearn = _SKLEARN_AVAILABLE
        self._clf = None
        self._scaler = None

        if self._use_sklearn:
            self._train()

    # ------------------------------------------------------------------
    # Training (sklearn path)
    # ------------------------------------------------------------------

    def _load_samples(self) -> tuple[list[list[float]], list[int]]:
        """Load and featurise the labeled JSON samples."""
        with open(self._path, encoding="utf-8") as fh:
            samples = json.load(fh)

        X: list[list[float]] = []
        y: list[int] = []
        for row in samples:
            feats = extract_features(row["text"])
            X.append(_features_to_list(feats))
            y.append(_TIER_ORDER[row["engagement"]])
        return X, y

    def _train(self) -> None:
        """Fit a LogisticRegression on the bundled samples."""
        try:
            X, y = self._load_samples()
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)
            self._clf = LogisticRegression(
                max_iter=500,
                multi_class="multinomial",
                solver="lbfgs",
                random_state=42,
            )
            self._clf.fit(X_scaled, y)
        except Exception:
            # If training fails for any reason (e.g. missing file), degrade
            # gracefully to the heuristic path.
            self._use_sklearn = False
            self._clf = None
            self._scaler = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, text: str) -> Dict[str, Any]:
        """Predict engagement for *text*.

        Returns:
            A dict with keys:

            * ``"tier"``    – ``"low"``, ``"medium"``, or ``"high"``
            * ``"score"``   – float in ``[0, 1]`` (higher = more engaging)
            * ``"signals"`` – dict of raw feature values used for the prediction
        """
        feats = extract_features(text)

        if self._use_sklearn and self._clf is not None:
            tier, score = self._sklearn_predict(feats)
        else:
            score = _heuristic_score(feats)
            tier = _score_to_tier(score)

        return {
            "tier": tier,
            "score": round(score, 4),
            "signals": feats,
        }

    def _sklearn_predict(
        self, feats: Dict[str, float]
    ) -> tuple[str, float]:
        """Use the trained classifier to produce a tier and a calibrated score."""
        vec = [_features_to_list(feats)]
        vec_scaled = self._scaler.transform(vec)
        proba = self._clf.predict_proba(vec_scaled)[0]  # shape (3,) for low/med/high

        # The classifier classes are sorted: 0=low, 1=medium, 2=high
        # Use probability of "high" class (index of class label 2) as the score.
        classes = list(self._clf.classes_)
        high_idx = classes.index(2) if 2 in classes else -1
        med_idx = classes.index(1) if 1 in classes else -1

        if high_idx >= 0 and med_idx >= 0:
            # Weighted combination: 0.5*p(high) + 0.25*p(medium) as base,
            # then map to [0,1].
            raw_score = proba[high_idx] * 0.7 + proba[med_idx] * 0.3
        else:
            raw_score = float(proba.max())

        score = float(min(max(raw_score, 0.0), 1.0))

        pred_class = int(self._clf.predict(vec_scaled)[0])
        tier = _TIER_LABELS[pred_class]
        return tier, score


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

_default_predictor: EngagementPredictor | None = None


def _get_default_predictor() -> EngagementPredictor:
    """Lazily initialise the module-level predictor singleton."""
    global _default_predictor
    if _default_predictor is None:
        _default_predictor = EngagementPredictor()
    return _default_predictor


def predict_engagement(text: str) -> Dict[str, Any]:
    """Module-level convenience wrapper around :class:`EngagementPredictor`.

    Lazily creates a shared predictor instance on first call.

    Args:
        text: The content string to evaluate.

    Returns:
        Same dict as :meth:`EngagementPredictor.predict`:
        ``{"tier": str, "score": float, "signals": dict}``.
    """
    return _get_default_predictor().predict(text)
