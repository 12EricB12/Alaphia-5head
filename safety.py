"""
Safety checker — runs before model inference.

Two-tier approach:
  1. Keyword/regex  — catches explicit phrases and common variants (near-zero latency)
  2. Semantic similarity — catches paraphrases using sentence embeddings (~50ms on CPU)

All flagged entries return a single unified result regardless of category; the caller
is responsible for routing to the appropriate help resource.

Usage:
    checker = SafetyChecker()
    result = checker.check(text)
    if result["flagged"]:
        # redirect to help page
        ...
"""

from __future__ import annotations

import re
import warnings
from typing import Optional

# sentence-transformers is optional; falls back to keyword-only if not installed
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

# ── Reference phrases (used for semantic similarity tier) ─────────────────────

_SELF_HARM_PHRASES = [
    "I want to kill myself",
    "I've been saving my pills",
    "I've written my suicide note",
    "I wish I was dead",
    "I think about suicide a lot",
    "Sometimes I think everyone would be better off without me",
    "I'm so tired of existing",
    "I've been cutting again",
    "I hurt myself last night",
    "The only thing that helps is pain",
    "I want to end my life",
    "I don't want to be alive anymore",
    "I've been thinking about suicide",
    "I just want it all to stop",
]

_HARM_TO_OTHERS_PHRASES = [
    "I have access to a gun",
    "I have access to a weapon",
    "I want to hurt someone",
    "I'm going to hurt them",
    "I have a gun and I'm angry",
    "I could kill them",
]

_HOUSING_CRISIS_PHRASES = [
    "I'm being evicted",
    "I got my eviction notice",
    "I'm sleeping in my car",
    "I lost my housing",
    "I'm living on the street",
    "I'm couch surfing and running out of options",
    "My landlord changed the locks",
    "We got a notice to vacate",
    "I can't pay rent and I'm about to lose my place",
    "I have 48 hours to leave",
    "I'm two months behind on rent",
    "I've used up all my friends' couches",
    "I can't afford anywhere to live",
    "I'm about to be on the street",
    "I have nowhere to go tonight",
    "I'm about to be homeless",
]

_CROSS_CATEGORY_PHRASES = [
    "My family will be better off with the life insurance",
    "At least if I'm gone they get the insurance money",
    "My life is over, my debts will die with me, and my family will be better off",
    "My family gets the insurance if I'm dead",
    "They'd be better off collecting on my life insurance",
    "The debt dies with me and my family will finally be free",
]

# All reference phrases flattened, with category labels
_ALL_REFERENCE: list[tuple[str, str]] = (
    [(p, "self_harm")       for p in _SELF_HARM_PHRASES]
    + [(p, "harm_to_others") for p in _HARM_TO_OTHERS_PHRASES]
    + [(p, "housing_crisis") for p in _HOUSING_CRISIS_PHRASES]
    + [(p, "cross_category") for p in _CROSS_CATEGORY_PHRASES]
)


# ── Keyword / regex patterns ───────────────────────────────────────────────────
# Each pattern is (compiled_regex, category). Patterns are checked sentence-by-sentence.

_WEAPON_NOUN = (
    r"(?:gun|firearm|rifle|pistol|handgun|shotgun|weapon|knife|blade|"
    r"explosives?|bomb|machete|axe|hatchet|crossbow|taser)"
)

_KEYWORD_PATTERNS: list[tuple[re.Pattern, str]] = [p for p in [
    # ── Self-harm / suicidal ──────────────────────────────────────────────────
    (re.compile(r"\bkill\s+my\s*self\b", re.I), "self_harm"),
    (re.compile(r"\bend\s+my\s+(own\s+)?life\b", re.I), "self_harm"),
    (re.compile(r"\btake\s+my\s+(own\s+)?life\b", re.I), "self_harm"),
    (re.compile(r"\bsuicid(?:e|al|ing)\b", re.I), "self_harm"),
    (re.compile(r"\bself[\s\-]?harm\b", re.I), "self_harm"),
    (re.compile(r"\bself[\s\-]?injur\b", re.I), "self_harm"),
    (re.compile(r"\bcutting\s+(again|myself)\b", re.I), "self_harm"),
    (re.compile(r"\bhurt\s+myself\b", re.I), "self_harm"),
    (re.compile(r"\bsaving\s+(up\s+)?(my\s+)?pills\b", re.I), "self_harm"),
    (re.compile(r"\b(written|wrote)\s+(my\s+)?suicide\s+note\b", re.I), "self_harm"),
    (re.compile(r"\bwish\s+i\s+(was|were)\s+dead\b", re.I), "self_harm"),
    (re.compile(r"\btired\s+of\s+(existing|living|being\s+alive)\b", re.I), "self_harm"),
    (re.compile(r"\bdon.t\s+want\s+to\s+(be\s+)?(alive|exist)\b", re.I), "self_harm"),
    (re.compile(r"\bno\s+(longer\s+want|reason)\s+to\s+(live|be\s+alive)\b", re.I), "self_harm"),
    (re.compile(r"\beveryone\s+would\s+be\s+better\s+off\s+without\s+me\b", re.I), "self_harm"),

    # ── Harm to others ────────────────────────────────────────────────────────
    (re.compile(rf"\b(i\s+)?(have|own|got|carry|bought|purchased)\s+(a\s+|an\s+|access\s+to\s+(a\s+)?)?{_WEAPON_NOUN}\b", re.I), "harm_to_others"),
    (re.compile(rf"\baccess\s+to\s+(a\s+|an\s+)?{_WEAPON_NOUN}\b", re.I), "harm_to_others"),
    (re.compile(r"\bi('m|\s+am|\s+will|\s+want\s+to)\s+(going\s+to\s+)?hurt\s+(him|her|them|someone|everybody|everyone)\b", re.I), "harm_to_others"),
    (re.compile(r"\bi\s+(could|will|want\s+to)\s+kill\s+(him|her|them|someone|everyone|my)\b", re.I), "harm_to_others"),

    # ── Housing crisis ────────────────────────────────────────────────────────
    (re.compile(r"\bevict(ed|ion|ing)\b", re.I), "housing_crisis"),
    (re.compile(r"\bsleeping\s+in\s+(my\s+)?(car|truck|vehicle|van)\b", re.I), "housing_crisis"),
    (re.compile(r"\bliving\s+(on\s+the\s+street|in\s+(my\s+)?(car|van|truck))\b", re.I), "housing_crisis"),
    (re.compile(r"\b(lost|losing)\s+(my\s+)?(home|housing|apartment|place\s+to\s+live)\b", re.I), "housing_crisis"),
    (re.compile(r"\blandlord\s+changed\s+the\s+locks\b", re.I), "housing_crisis"),
    (re.compile(r"\bnotice\s+to\s+vacate\b", re.I), "housing_crisis"),
    (re.compile(r"\b(48|24|72)\s+hours\s+to\s+(leave|vacate|get\s+out)\b", re.I), "housing_crisis"),
    (re.compile(r"\bcouch[\s\-]?surf(ing)?\b", re.I), "housing_crisis"),
    (re.compile(r"\b(becoming|going|about\s+to\s+be)\s+homeless\b", re.I), "housing_crisis"),
    (re.compile(r"\bnowhere\s+to\s+(go|live|sleep)\b", re.I), "housing_crisis"),
    (re.compile(r"\b(two|2)\s+months?\s+behind\s+on\s+rent\b", re.I), "housing_crisis"),
    (re.compile(r"\bno\s+(place|where)\s+to\s+(live|go|sleep)\b", re.I), "housing_crisis"),
    (re.compile(r"\bcan.t\s+afford\s+(anywhere|a\s+place)\s+to\s+live\b", re.I), "housing_crisis"),

    # ── Cross-category (finance + death) ──────────────────────────────────────
    (re.compile(r"\blife\s+insurance\b.{0,60}\b(gone|dead|die|death|better\s+off)\b", re.I | re.S), "cross_category"),
    (re.compile(r"\b(gone|dead|die|death|better\s+off)\b.{0,60}\blife\s+insurance\b", re.I | re.S), "cross_category"),
    (re.compile(r"\bdebts?\s+(will\s+)?die\s+with\s+me\b", re.I), "cross_category"),
    (re.compile(r"\bfamily\s+(would\s+be|will\s+be|be)\s+better\s+off\s+(without\s+me|if\s+i\s+(was|were|am)\s+gone)\b", re.I), "cross_category"),
    (re.compile(r"\binsurance\s+(money|payout|policy)\b.{0,60}\b(gone|dead|die)\b", re.I | re.S), "cross_category"),
]]


# ── Sentence splitter ─────────────────────────────────────────────────────────

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def _split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries; also keep the full text as one item so
    multi-sentence patterns (cross-category regex with re.S) are checked whole."""
    parts = _SENT_SPLIT.split(text.strip())
    if len(parts) > 1:
        parts.append(text)  # add full text for cross-sentence patterns
    return [p.strip() for p in parts if p.strip()]


# ── SafetyChecker class ───────────────────────────────────────────────────────

class SafetyChecker:
    """
    Instantiate once at app startup; reuse across all inference calls.

    Parameters
    ----------
    semantic_threshold : float
        Cosine similarity cutoff for the semantic tier (default 0.72).
        Raise to reduce false positives; lower to catch more paraphrases.
    model_name : str
        Sentence-transformers model to use for semantic matching.
    """

    def __init__(
        self,
        semantic_threshold: float = 0.72,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.threshold = semantic_threshold
        self._encoder = None
        self._ref_embeddings = None
        self._ref_labels: list[str] = []
        self._ref_phrases: list[str] = []

        if _ST_AVAILABLE:
            try:
                self._encoder = SentenceTransformer(model_name)
                phrases = [p for p, _ in _ALL_REFERENCE]
                labels  = [c for _, c in _ALL_REFERENCE]
                self._ref_embeddings = self._encoder.encode(
                    phrases, convert_to_numpy=True, show_progress_bar=False
                )
                self._ref_labels  = labels
                self._ref_phrases = phrases
            except Exception as exc:
                warnings.warn(
                    f"SafetyChecker: could not load sentence-transformer model "
                    f"({exc}). Falling back to keyword-only mode."
                )
                self._encoder = None
        else:
            warnings.warn(
                "SafetyChecker: sentence-transformers not installed. "
                "Running in keyword-only mode. "
                "Install with: pip install sentence-transformers"
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def check(self, text: str) -> dict:
        """
        Check text for safety signals.

        Returns
        -------
        dict with keys:
            flagged  : bool
            category : str | None   ('self_harm', 'harm_to_others',
                                     'housing_crisis', 'cross_category')
            trigger  : str | None   (the phrase or sentence that matched)
            method   : str | None   ('keyword' or 'semantic')
            score    : float | None (cosine similarity; only for semantic matches)
        """
        if not text or not text.strip():
            return self._safe()

        sentences = _split_sentences(text)

        # ── Tier 1: keywords ──────────────────────────────────────────────────
        for sentence in sentences:
            for pattern, category in _KEYWORD_PATTERNS:
                if pattern.search(sentence):
                    return {
                        "flagged":  True,
                        "category": category,
                        "trigger":  sentence[:200],
                        "method":   "keyword",
                        "score":    None,
                    }

        # ── Tier 2: semantic similarity ───────────────────────────────────────
        if self._encoder is None or self._ref_embeddings is None:
            return self._safe()

        sent_embeddings = self._encoder.encode(
            sentences, convert_to_numpy=True, show_progress_bar=False
        )

        # cosine similarity: dot product of L2-normalised vectors
        sent_norms = np.linalg.norm(sent_embeddings, axis=1, keepdims=True)
        ref_norms  = np.linalg.norm(self._ref_embeddings, axis=1, keepdims=True)
        sent_normed = sent_embeddings / np.maximum(sent_norms, 1e-9)
        ref_normed  = self._ref_embeddings / np.maximum(ref_norms, 1e-9)

        # (num_sentences × num_references) similarity matrix
        sim_matrix = sent_normed @ ref_normed.T

        best_idx = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
        best_score = float(sim_matrix[best_idx])

        if best_score >= self.threshold:
            sent_idx, ref_idx = best_idx
            return {
                "flagged":  True,
                "category": self._ref_labels[ref_idx],
                "trigger":  sentences[sent_idx][:200],
                "method":   "semantic",
                "score":    round(best_score, 4),
            }

        return self._safe()

    @staticmethod
    def _safe() -> dict:
        return {
            "flagged":  False,
            "category": None,
            "trigger":  None,
            "method":   None,
            "score":    None,
        }
