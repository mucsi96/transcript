"""Stage 5: decide which tokens are worth learning.

Excluded by default:
  - POS in {PROPN, PUNCT, SYM, NUM, X, SPACE}: proper nouns (person /
    geographical names are not vocabulary), punctuation, symbols, numerals,
    foreign-word gibberish, whitespace tokens.
  - POS in FUNCTION_POS (DET, PRON, ADP, AUX, CCONJ, SCONJ, PART): the closed
    classes. "der/sie/auf/sein/und" are the top of every German frequency
    list and are learned as grammar, not as flash cards.
  - Stopwords: the open-class leftovers of the same idea — very common
    adverbs, verbs and adjectives ("so", "schon", "machen", "gut") that the
    POS rules above cannot catch, since ADV/VERB/ADJ are otherwise the words
    worth learning. `--keep-stopwords` turns this off.
  - Named entities tagged PER / LOC / ORG: NER catch-net for names the POS
    tagger missed (e.g. a surname misclassified as NOUN, or multi-word
    entities). MISC is kept because German NER tags learnable nationality
    adjectives ("deutsch") as MISC.
  - Non-alphabetic tokens ("2:30", "€") and tokens shorter than 2 chars
    (ASR debris).

Extension points (not implemented, by design):
  - Frequency filter: wordfreq.zipf_frequency(lemma, "de") to drop ultra-rare
    words (< 1.5, likely ASR errors) or ultra-common ones (> 6.0). Would be a
    post-aggregation filter over WordEntry, and a tunable alternative to the
    fixed stopword list.
  - LLM review pass: batch the candidate word list to GPT-5 / Claude asking
    "which of these are worth a flash card for a B1 learner?" — again a
    post-aggregation WordEntry filter, kept out of the core path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import FilterConfig

if TYPE_CHECKING:
    from .analyze import TokenRecord


def is_learnable(tok: "TokenRecord", cfg: FilterConfig) -> tuple[bool, str]:
    """Return (keep, reason). The reason string names the first rule that
    rejected the token, for --verbose debugging."""
    if tok.pos in cfg.exclude_pos:
        return False, f"pos:{tok.pos}"
    if cfg.require_alpha and not tok.is_alpha:
        return False, "non-alpha"
    if len(tok.text) < cfg.min_token_len:
        return False, "too-short"
    if tok.ent_type in cfg.exclude_ent_types:
        return False, f"entity:{tok.ent_type}"
    if cfg.drop_stopwords and tok.is_stop:
        return False, "stopword"
    return True, "keep"
