"""Deterministic, network-free heuristics for detecting garbled model output.

Each check inspects a response string and, if it looks wrong, contributes a
`Flag` explaining what tripped it. `evaluate()` runs all checks and returns a
verdict. Thresholds have sensible defaults and can be overridden from config.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

DEFAULT_THRESHOLDS: dict[str, float] = {
    "max_non_ascii_ratio": 0.15,
    "max_symbol_ratio": 0.30,
    "max_repetition_ratio": 0.50,
    "min_wordlike_ratio": 0.60,
}

# Minimum length before ratio-based checks kick in — very short valid replies
# (e.g. "Paris.") shouldn't be penalized by noisy ratios.
_MIN_LEN_FOR_RATIOS = 12
_MIN_TOKENS_FOR_RATIOS = 4

_TOKEN_RE = re.compile(r"\S+")
_VOWEL_RE = re.compile(r"[aeiouyAEIOUY]")
# A "word-like" token: mostly alphabetic, contains a vowel, plausible length.
_ALPHA_RE = re.compile(r"[A-Za-z]")


@dataclass
class Flag:
    check: str
    reason: str


@dataclass
class Verdict:
    passed: bool
    flags: list[Flag]


def _threshold(overrides: dict[str, float] | None, name: str) -> float:
    if overrides and name in overrides:
        return overrides[name]
    return DEFAULT_THRESHOLDS[name]


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def _is_wordlike(token: str) -> bool:
    """Structural test for a plausible English word — no dictionary needed."""
    core = re.sub(r"[^A-Za-z]", "", token)
    if not core:
        # Pure punctuation/number tokens are neutral, not counted against word-likeness.
        return True
    if len(core) > 20:
        return False
    if not _VOWEL_RE.search(core):
        return False
    # Reject long consonant runs like "bcdfg" (gibberish signature).
    if re.search(r"[^aeiouyAEIOUY]{5,}", core):
        return False
    return True


def check_empty(text: str) -> Flag | None:
    if not text or not text.strip():
        return Flag("empty", "Response was empty or whitespace only.")
    return None


def check_non_ascii(text: str, overrides: dict[str, float] | None = None) -> Flag | None:
    letters = [c for c in text if c.isalpha()]
    if len(letters) < _MIN_LEN_FOR_RATIOS:
        return None
    non_ascii = sum(1 for c in letters if ord(c) > 127)
    ratio = non_ascii / len(letters)
    limit = _threshold(overrides, "max_non_ascii_ratio")
    if ratio > limit:
        return Flag(
            "non_ascii",
            f"{ratio:.0%} of letters are non-ASCII (limit {limit:.0%}); "
            f"unexpected for an English prompt.",
        )
    return None


def check_symbols(text: str, overrides: dict[str, float] | None = None) -> Flag | None:
    stripped = text.strip()
    if len(stripped) < _MIN_LEN_FOR_RATIOS:
        return None
    noise = sum(
        1
        for c in stripped
        if not (c.isalnum() or c.isspace() or c in ".,!?;:'\"()-–—[]{}/%$&@#*+=<>|\\`~^_")
    )
    ratio = noise / len(stripped)
    limit = _threshold(overrides, "max_symbol_ratio")
    if ratio > limit:
        return Flag(
            "symbols",
            f"{ratio:.0%} of characters are noise/replacement symbols "
            f"(limit {limit:.0%}); looks like mojibake or control junk.",
        )
    return None


def check_repetition(text: str, overrides: dict[str, float] | None = None) -> Flag | None:
    tokens = [t.lower() for t in _tokens(text)]
    if len(tokens) < _MIN_TOKENS_FOR_RATIOS:
        return None
    most_common_token, count = Counter(tokens).most_common(1)[0]
    ratio = count / len(tokens)
    limit = _threshold(overrides, "max_repetition_ratio")
    if ratio > limit:
        return Flag(
            "repetition",
            f'Token "{most_common_token}" is {ratio:.0%} of all tokens '
            f"(limit {limit:.0%}); likely a degenerate repetition loop.",
        )
    return None


def check_wordlike(text: str, overrides: dict[str, float] | None = None) -> Flag | None:
    tokens = _tokens(text)
    # Only consider tokens that contain at least one letter.
    alpha_tokens = [t for t in tokens if _ALPHA_RE.search(t)]
    if len(alpha_tokens) < _MIN_TOKENS_FOR_RATIOS:
        return None
    wordlike = sum(1 for t in alpha_tokens if _is_wordlike(t))
    ratio = wordlike / len(alpha_tokens)
    limit = _threshold(overrides, "min_wordlike_ratio")
    if ratio < limit:
        return Flag(
            "wordlike",
            f"Only {ratio:.0%} of word tokens look like real words "
            f"(need {limit:.0%}); likely consonant-soup gibberish.",
        )
    return None


def evaluate(text: str, thresholds: dict[str, float] | None = None) -> Verdict:
    """Run all heuristics and return a pass/fail verdict with explanatory flags."""
    text = text or ""

    empty = check_empty(text)
    if empty:
        # Nothing else is meaningful on an empty response.
        return Verdict(passed=False, flags=[empty])

    flags = [
        f
        for f in (
            check_non_ascii(text, thresholds),
            check_symbols(text, thresholds),
            check_repetition(text, thresholds),
            check_wordlike(text, thresholds),
        )
        if f is not None
    ]
    return Verdict(passed=not flags, flags=flags)
