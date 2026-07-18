"""In-memory state for a single learning session.

Speechmatics real-time emits finalized transcript *chunks* that are often just
a word or two long. Showing those raw would look like a word list, and using a
chunk as a word's "context" is useless. So we buffer the incoming chunks and
only surface a chunk once it forms a complete sentence — that sentence is what
we show in the transcript and attach to each new word as its context.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field

# Keep German umlauts/ß; drop surrounding punctuation and digits-only tokens.
_WORD_STRIP = re.compile(r"^[^0-9A-Za-zÄÖÜäöüß]+|[^0-9A-Za-zÄÖÜäöüß]+$")
_HAS_LETTER = re.compile(r"[A-Za-zÄÖÜäöüß]")
# A sentence ends at . ! ? or … (one or more), followed by whitespace or EOS.
_SENTENCE_END = re.compile(r"[.!?…]+(?=\s|$)")


def normalize(token: str) -> str:
    """Lowercase and strip leading/trailing punctuation from a token."""
    return _WORD_STRIP.sub("", token).lower()


def is_wordlike(token: str) -> bool:
    return bool(token) and bool(_HAS_LETTER.search(token))


def tokenize(sentence: str) -> list[str]:
    return sentence.split()


@dataclass
class WordEntry:
    word: str  # normalized form
    surface: str  # first-seen surface form (original casing)
    context: str  # full sentence it first appeared in
    count: int = 0


@dataclass
class Session:
    words: dict[str, WordEntry] = field(default_factory=dict)
    sentences: list[str] = field(default_factory=list)
    _buffer: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # -- ingestion ---------------------------------------------------------
    def add_final(self, text: str, words: list[str]) -> tuple[list[str], list[WordEntry]]:
        """Append a finalized chunk. Returns (completed_sentences, new_entries).

        Words are only recorded once the sentence they belong to is complete,
        so every word's context is a full sentence.
        """
        chunk = text.strip() or " ".join(words).strip()
        with self._lock:
            if chunk:
                self._buffer = f"{self._buffer} {chunk}".strip() if self._buffer else chunk
            completed = self._drain_sentences()
            new_entries = self._record(completed)
        return completed, new_entries

    def flush(self) -> tuple[list[str], list[WordEntry]]:
        """Finalize whatever is left in the buffer (called when a session ends)."""
        with self._lock:
            remainder = self._buffer.strip()
            self._buffer = ""
            completed = [remainder] if remainder else []
            new_entries = self._record(completed)
        return completed, new_entries

    # -- internals (must hold _lock) ---------------------------------------
    def _drain_sentences(self) -> list[str]:
        completed: list[str] = []
        while True:
            match = _SENTENCE_END.search(self._buffer)
            if not match:
                break
            end = match.end()
            sentence = self._buffer[:end].strip()
            self._buffer = self._buffer[end:].lstrip()
            if sentence:
                completed.append(sentence)
        return completed

    def _record(self, completed: list[str]) -> list[WordEntry]:
        new_entries: list[WordEntry] = []
        for sentence in completed:
            self.sentences.append(sentence)
            for raw in tokenize(sentence):
                norm = normalize(raw)
                if not is_wordlike(norm):
                    continue
                entry = self.words.get(norm)
                if entry is None:
                    entry = WordEntry(word=norm, surface=raw.strip(".,!?…;:\"'»«"), context=sentence, count=1)
                    self.words[norm] = entry
                    new_entries.append(entry)
                else:
                    entry.count += 1
        return new_entries

    # -- reads -------------------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            return {
                "unique_words": len(self.words),
                "transcript": list(self.sentences),
                "words": [
                    {
                        "word": e.word,
                        "surface": e.surface,
                        "context": e.context,
                        "count": e.count,
                    }
                    for e in sorted(self.words.values(), key=lambda x: (-x.count, x.word))
                ],
            }

    def unique_entries(self) -> list[WordEntry]:
        with self._lock:
            return list(self.words.values())

    def reset(self) -> None:
        with self._lock:
            self.words.clear()
            self.sentences.clear()
            self._buffer = ""
