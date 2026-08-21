"""Stages 4-6: match against known words, filter, and aggregate into the
final flash-card word list."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .analyze import SentenceRecord
from .artifacts import SCHEMA_VERSION, save_json, utc_timestamp
from .config import FilterConfig, pos_label
from .filters import is_learnable
from .known_words import KnownWords, is_known

log = logging.getLogger(__name__)


@dataclass
class WordEntry:
    lemma: str
    pos: str
    word_type: str
    count: int = 0
    sentences: list[str] = field(default_factory=list)


def build_word_list(
    sentences: list[SentenceRecord],
    known: KnownWords,
    cfg: FilterConfig,
    *,
    verbose: bool = False,
) -> list[WordEntry]:
    """Group learnable, unknown lemmas into WordEntry items with counts and
    deduplicated example sentences.

    Aggregation key is (casefolded lemma, POS) so homonyms across word types
    stay separate: laufen/VERB vs Laufen/NOUN. The display lemma is the
    first-seen form, preserving spaCy's noun capitalization.
    """
    entries: dict[tuple[str, str], WordEntry] = {}
    for sent in sentences:
        for tok in sent.tokens:
            keep, reason = is_learnable(tok, cfg)
            if not keep:
                if verbose:
                    log.debug("drop %-20r %s", tok.text, reason)
                continue
            if is_known(tok.lemma, known):
                continue
            key = (tok.lemma.casefold(), tok.pos)
            entry = entries.get(key)
            if entry is None:
                entry = entries[key] = WordEntry(
                    lemma=tok.lemma, pos=tok.pos, word_type=pos_label(tok.pos)
                )
            entry.count += 1
            if sent.text not in entry.sentences:
                entry.sentences.append(sent.text)

    result = [e for e in entries.values() if e.count >= cfg.min_count]
    result.sort(key=lambda e: (-e.count, e.lemma.casefold(), e.pos))
    return result


def write_output(
    entries: list[WordEntry],
    output: Path,
    *,
    known_words_source: str | None,
    cfg: FilterConfig,
) -> dict:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_timestamp(),
        "known_words_source": known_words_source,
        "filter_config": cfg.to_dict(),
        "total_words": len(entries),
        "words": [
            {
                "lemma": e.lemma,
                "pos": e.pos,
                "word_type": e.word_type,
                "count": e.count,
                "sentences": e.sentences,
            }
            for e in entries
        ],
    }
    save_json(output, payload)
    return payload
