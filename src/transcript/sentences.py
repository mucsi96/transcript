"""Stage 3: sentence splitting with syntok.

Input is the text artifact of stage 2 — Whisper segments from `transcribe`
or book paragraphs from `epub`; both are lists of short text segments, so
this stage does not care which produced them. Segments are joined with
spaces before splitting, because a Whisper segment can end mid-sentence.

The output is deliberately just the ordered list of sentence strings
(duplicates included — a film repeats its lines, and the counts matter
later): an intermediary artifact that is easy to eyeball when a word in
the final list looks wrong. Everything token-level (lemmas, word types)
lives in the LLM stage, which sees whole sentences and can therefore
reunite German separable verbs ("Er fängt ... an" -> "anfangen") that a
per-token tagger takes apart.

syntok is a small pure-Python segmenter with German abbreviation handling
("z. B.", "Dr.", "usw."). Its one systematic German mistake is splitting
after an ordinal number that precedes a capitalized word — "im 2. Stock",
"am 24. Dezember" (its month list only covers abbreviations like "Okt") —
so such splits are merged back: a real sentence ending in a bare one- or
two-digit number is rare, and for vocabulary extraction an occasional
too-long sentence is harmless where a fragment like "Stock." is noise.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from .artifacts import SCHEMA_VERSION, load_json, require_list, save_json, should_skip

log = logging.getLogger(__name__)

# A sentence "ending" in an ordinal: a standalone 1-2 digit number + period.
# Years ("Das war 1998.") have four digits and stay sentence ends.
_ORDINAL_END = re.compile(r"(?:^|\s)\d{1,2}\.$")


def join_segments(segment_texts: list[str]) -> str:
    """One document out of the transcript segments. Joined with spaces, not
    newlines: a segment boundary (a Whisper pause) is not a sentence
    boundary."""
    return " ".join(text for text in (t.strip() for t in segment_texts) if text)


def merge_ordinal_splits(sentences: list[str]) -> list[str]:
    """Fold syntok's after-the-ordinal splits back into one sentence:
    ["Er wohnt im 2.", "Stock."] -> ["Er wohnt im 2. Stock."]."""
    merged: list[str] = []
    for sentence in sentences:
        if merged and _ORDINAL_END.search(merged[-1]):
            merged[-1] = f"{merged[-1]} {sentence}"
        else:
            merged.append(sentence)
    return merged


def split_text(text: str) -> list[str]:
    """Whitespace-normalized sentences of a document."""
    import syntok.segmenter as segmenter

    sentences: list[str] = []
    for paragraph in segmenter.process(text):
        for sentence in paragraph:
            joined = " ".join(
                "".join(tok.spacing + tok.value for tok in sentence).split()
            )
            if joined:
                sentences.append(joined)
    return merge_ordinal_splits(sentences)


def sentences_from_payload(payload: dict, source: Path | str = "sentences") -> list[str]:
    """Rebuild the sentence list from a sentences.json payload."""
    sentences = require_list(payload, "sentences", Path(source))
    for sent in sentences:
        if not isinstance(sent, str):
            raise ValueError(
                f"{source}: malformed sentence record ({sent!r} is not a string); "
                f"re-run the sentences stage with --force."
            )
    return sentences


def split_sentences(
    transcript_path: Path,
    output: Path,
    *,
    force: bool = False,
) -> dict:
    if should_skip(output, force):
        return load_json(output)

    transcript = load_json(transcript_path)
    segments = require_list(transcript, "segments", transcript_path)
    try:
        segment_texts = [seg["text"] for seg in segments]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"{transcript_path}: a segment has no 'text' field "
            f"({type(exc).__name__}: {exc}); re-run the text stage with --force."
        ) from exc

    text = join_segments(segment_texts)
    if not text:
        raise ValueError(
            f"{transcript_path} contains no text to analyze ({len(segments)} "
            f"segment(s), all empty)."
        )

    sentences = split_text(text)
    log.info("Found %d sentences (%d distinct)", len(sentences), len(set(sentences)))

    from importlib.metadata import PackageNotFoundError, version

    try:
        syntok_version = version("syntok")
    except PackageNotFoundError:  # pragma: no cover - installed as a dep
        syntok_version = "unknown"

    payload = {
        "schema_version": SCHEMA_VERSION,
        "segmenter": "syntok",
        "segmenter_version": syntok_version,
        "source": str(transcript_path),
        "total_sentences": len(sentences),
        "sentences": sentences,
    }
    save_json(output, payload)
    return payload
