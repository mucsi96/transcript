"""Shared configuration: filter settings and POS label mapping."""

from __future__ import annotations

from dataclasses import dataclass

# Universal POS tags (UPOS) -> human-readable word type for flash cards.
POS_LABELS: dict[str, str] = {
    "ADJ": "adjective",
    "ADP": "preposition",
    "ADV": "adverb",
    "AUX": "auxiliary verb",
    "CCONJ": "conjunction",
    "DET": "determiner",
    "INTJ": "interjection",
    "NOUN": "noun",
    "NUM": "numeral",
    "PART": "particle",
    "PRON": "pronoun",
    "PROPN": "proper noun",
    "PUNCT": "punctuation",
    "SCONJ": "subordinating conjunction",
    "SYM": "symbol",
    "VERB": "verb",
    "X": "other",
    "SPACE": "space",
}

# Closed word classes: articles, pronouns, prepositions, auxiliaries and
# conjunctions ("der", "sie", "auf", "sein", "und"). German has a few hundred
# of them in total, every learner meets them in the first weeks, and they
# dominate any frequency list — so they are never flash-card material.
# `--keep-pos DET` etc. puts a class back.
FUNCTION_POS = frozenset({"ADP", "AUX", "CCONJ", "DET", "PART", "PRON", "SCONJ"})

DEFAULT_EXCLUDE_POS = frozenset(
    {"PROPN", "PUNCT", "SYM", "NUM", "X", "SPACE"} | FUNCTION_POS
)
# MISC is deliberately kept: German NER tags nationality adjectives like
# "deutsch" as MISC, and those are worth learning.
DEFAULT_EXCLUDE_ENT_TYPES = frozenset({"PER", "LOC", "ORG"})


@dataclass(frozen=True)
class FilterConfig:
    """Which tokens/words are excluded from the learn list."""

    exclude_pos: frozenset[str] = DEFAULT_EXCLUDE_POS
    exclude_ent_types: frozenset[str] = DEFAULT_EXCLUDE_ENT_TYPES
    min_token_len: int = 2
    require_alpha: bool = True
    drop_stopwords: bool = True
    min_count: int = 1

    def to_dict(self) -> dict:
        return {
            "exclude_pos": sorted(self.exclude_pos),
            "exclude_ent_types": sorted(self.exclude_ent_types),
            "min_token_len": self.min_token_len,
            "require_alpha": self.require_alpha,
            "drop_stopwords": self.drop_stopwords,
            "min_count": self.min_count,
        }


def pos_label(pos: str) -> str:
    return POS_LABELS.get(pos, pos.lower())
