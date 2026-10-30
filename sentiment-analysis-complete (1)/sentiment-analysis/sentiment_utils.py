"""Shared text-normalization utilities for the sentiment-analysis pipeline."""

import html
import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_NON_WORD_RE = re.compile(r"[^a-z0-9\s']")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalize review text while retaining negation words and apostrophes.

    The function is deliberately conservative: it decodes HTML entities,
    strips HTML tags, replaces URLs with a token, lowercases text, removes
    most punctuation, and collapses whitespace. Words such as ``not`` and
    ``no`` are retained because they can reverse sentiment.
    """
    text = "" if text is None else str(text)
    text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" urltoken ", text)
    text = text.lower()
    text = _NON_WORD_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text
