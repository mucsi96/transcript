"""Stages 5-6: match the LLM-extracted words against known words and
aggregate them into the final flash-card word list.

The heavy lifting — deciding what counts as a word worth learning and
producing its dictionary form — happened in the LLM stage; this one only
drops what the learner already knows, counts occurrences, and collects
example sentences.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .artifacts import SCHEMA_VERSION, save_json, utc_timestamp
from .known_words import KnownWords, is_known
from .llm import SentenceWords

log = logging.getLogger(__name__)


@dataclass
class WordEntry:
    lemma: str
    word_type: str
    article: str | None = None
    count: int = 0
    sentences: list[str] = field(default_factory=list)


def build_word_list(
    sentence_words: list[SentenceWords],
    known: KnownWords,
    *,
    min_count: int = 1,
    verbose: bool = False,
) -> list[WordEntry]:
    """Group unknown lemmas into WordEntry items with counts and example
    sentences.

    Aggregation key is (casefolded lemma, word type) so homonyms across
    word types stay separate: laufen/verb vs Laufen/noun. The display lemma
    is the first-seen form. A sentence's count is how often the source
    contains it — the LLM saw each distinct sentence once, but the word was
    heard every time.
    """
    entries: dict[tuple[str, str], WordEntry] = {}
    for sent in sentence_words:
        for word in sent.words:
            if is_known(word.lemma, known):
                if verbose:
                    log.debug("drop %-20r known", word.lemma)
                continue
            key = (word.lemma.casefold(), word.word_type)
            entry = entries.get(key)
            if entry is None:
                entry = entries[key] = WordEntry(
                    lemma=word.lemma, word_type=word.word_type, article=word.article
                )
            if entry.article is None and word.article is not None:
                entry.article = word.article
            entry.count += sent.count
            if sent.text not in entry.sentences:
                entry.sentences.append(sent.text)

    result = [e for e in entries.values() if e.count >= min_count]
    result.sort(key=lambda e: (-e.count, e.lemma.casefold(), e.word_type))
    return result


def write_output(
    entries: list[WordEntry],
    output: Path,
    *,
    known_words_source: str | None,
    llm_model: str | None,
    min_count: int,
) -> dict:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_timestamp(),
        "known_words_source": known_words_source,
        "llm_model": llm_model,
        "min_count": min_count,
        "total_words": len(entries),
        "words": [
            {
                "lemma": e.lemma,
                "word_type": e.word_type,
                "article": e.article,
                "count": e.count,
                "sentences": e.sentences,
            }
            for e in entries
        ],
    }
    save_json(output, payload)
    return payload
