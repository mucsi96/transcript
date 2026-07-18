"""In-memory state for a single learning session."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field

# Keep German umlauts/ß; drop surrounding punctuation and digits-only tokens.
_WORD_STRIP = re.compile(r"^[^0-9A-Za-zÄÖÜäöüß]+|[^0-9A-Za-zÄÖÜäöüß]+$")
_HAS_LETTER = re.compile(r"[A-Za-zÄÖÜäöüß]")


def normalize(token: str) -> str:
    """Lowercase and strip leading/trailing punctuation from a token."""
    return _WORD_STRIP.sub("", token).lower()


def is_wordlike(token: str) -> bool:
    return bool(token) and bool(_HAS_LETTER.search(token))


@dataclass
class WordEntry:
    word: str  # normalized form
    surface: str  # first-seen surface form (original casing)
    context: str  # sentence/segment it first appeared in
    count: int = 0


@dataclass
class Session:
    words: dict[str, WordEntry] = field(default_factory=dict)
    transcript_lines: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_final(self, text: str, words: list[str]) -> list[WordEntry]:
        """Record a finalized transcript segment. Returns newly-added entries."""
        new_entries: list[WordEntry] = []
        with self._lock:
            if text:
                self.transcript_lines.append(text)
            context = text or " ".join(words)
            for raw in words:
                norm = normalize(raw)
                if not is_wordlike(norm):
                    continue
                entry = self.words.get(norm)
                if entry is None:
                    entry = WordEntry(word=norm, surface=raw, context=context, count=1)
                    self.words[norm] = entry
                    new_entries.append(entry)
                else:
                    entry.count += 1
        return new_entries

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "unique_words": len(self.words),
                "total_lines": len(self.transcript_lines),
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
            self.transcript_lines.clear()
