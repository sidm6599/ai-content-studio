"""Pluggable content domains. Each domain defines how to prompt the LLM
and what fields it should return. Adding a domain = adding one Domain entry."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class Domain:
    key: str
    label: str
    description: str
    # Output JSON fields the model should produce per item
    output_fields: List[str]
    # System-style instruction injected into the prompt
    instruction: str
    # Optional channel/platform choices shown in the UI
    channels: List[str] = field(default_factory=list)
    # Soft character limit per channel (for the UI length check)
    char_limits: Dict[str, int] = field(default_factory=dict)


TONE_PRESETS: List[str] = [
    "professional",
    "playful",
    "concise",
    "inspirational",
    "informative",
    "luxury",
    "gen-z",
]


def list_tones() -> List[str]:
    """Return the list of available tone presets."""
    return list(TONE_PRESETS)


DOMAINS: Dict[str, Domain] = {
    "social": Domain(
        key="social",
        label="Social Media Post",
        description="Engaging captions + hashtags for social platforms.",
        output_fields=["caption", "hashtags", "emojis"],
        instruction=(
            "You are a senior social media copywriter. Write scroll-stopping, "
            "platform-appropriate captions with relevant hashtags and a few emojis."
        ),
        channels=["Instagram", "LinkedIn", "X/Twitter", "Facebook", "TikTok"],
        char_limits={"X/Twitter": 280, "Instagram": 2200, "LinkedIn": 3000,
                     "Facebook": 2000, "TikTok": 2200},
    ),
    "product": Domain(
        key="product",
        label="Product Description",
        description="Persuasive e-commerce product copy.",
        output_fields=["title", "description", "bullet_points"],
        instruction=(
            "You are an e-commerce copywriter. Write a benefit-led product title, "
            "a persuasive description, and 3-5 scannable bullet points."
        ),
        channels=["Generic", "Amazon", "Shopify"],
        char_limits={"Amazon": 2000},
    ),
    "email": Domain(
        key="email",
        label="Marketing Email",
        description="Subject line + body for a marketing email.",
        output_fields=["subject", "preview_text", "body", "cta"],
        instruction=(
            "You are an email marketer. Write a high-open-rate subject line, a short "
            "preview text, a concise persuasive body, and one clear call-to-action."
        ),
        channels=["Newsletter", "Promotional", "Onboarding"],
    ),
    "resume": Domain(
        key="resume",
        label="Resume Bullets",
        description="Quantified, action-led resume bullet points.",
        output_fields=["bullets"],
        instruction=(
            "You are a technical resume coach. Rewrite the input into 3-5 concise, "
            "action-led, quantified resume bullet points using strong verbs. Do not "
            "invent numbers; use placeholders like [X%] when a metric is unknown."
        ),
        channels=["Software", "Data/ML", "General"],
    ),
    "blog": Domain(
        key="blog",
        label="Blog Post",
        description="Structured blog content with a compelling title, intro, and outline.",
        output_fields=["title", "intro", "outline"],
        instruction=(
            "You are an experienced content strategist and blogger. Write a magnetic, "
            "SEO-aware title, a punchy introduction that hooks the reader in the first "
            "two sentences, and a clear section-by-section outline (3-6 sections) with "
            "one-line descriptions of what each section covers. Match the tone to the "
            "channel: Personal blogs should feel authentic and conversational; Company "
            "blogs should be authoritative and value-driven; SEO blogs should be "
            "keyword-conscious and structured for featured snippets."
        ),
        channels=["Personal", "Company", "SEO"],
    ),
    "video": Domain(
        key="video",
        label="Short Video Script",
        description="Hook, script, and caption for short-form vertical video.",
        output_fields=["hook", "script", "caption"],
        instruction=(
            "You are a viral short-form video scriptwriter. Craft a thumb-stopping "
            "opening hook (first 3 seconds — one punchy sentence that raises a question "
            "or drops a bold claim), a tight spoken script (60-90 words, conversational "
            "pacing, one idea per line to aid teleprompter reading), and a platform "
            "caption with relevant hashtags. TikTok scripts should feel raw and "
            "trend-aware; YouTube Shorts should be informative and retention-optimised; "
            "Reels should be visually descriptive with cue notes for B-roll."
        ),
        channels=["TikTok", "YouTube Shorts", "Reels"],
        char_limits={"TikTok": 2200, "YouTube Shorts": 5000, "Reels": 2200},
    ),
}


def get_domain(key: str) -> Domain:
    if key not in DOMAINS:
        raise KeyError(f"Unknown domain '{key}'. Available: {list(DOMAINS)}")
    return DOMAINS[key]


def list_domains() -> List[Domain]:
    return list(DOMAINS.values())
