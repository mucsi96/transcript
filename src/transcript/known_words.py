"""Stage 4: load the user's known-words list and match lemmas against it.

The list is plain text, one lemma (dictionary form) per line. Blank lines and
lines starting with '#' are ignored.

Matching uses str.casefold(), not lower(): casefold maps German ß -> ss, so a
list entry "Strasse" matches the lemma "Straße". This conflates rare pairs
like Maße/Masse, which is an acceptable trade-off for a vocabulary filter.
"""

from __future__ import annotations

from pathlib import Path


def normalize_word(word: str) -> str:
    return word.strip().casefold()


def load_known_words(path: Path) -> frozenset[str]:
    words = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            words.add(normalize_word(line))
    return frozenset(words)


def is_known(lemma: str, known: frozenset[str]) -> bool:
    return normalize_word(lemma) in known
