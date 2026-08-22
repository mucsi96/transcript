"""Stage 3: sentence splitting with spaCy.

Input is the text artifact of stage 2 — Whisper segments from `transcribe`
or book paragraphs from `epub`; both are lists of short text segments, so
this stage does not care which produced them.

The output is deliberately just the ordered list of sentence strings
(duplicates included — a film repeats its lines, and the counts matter
later): an intermediary artifact that is easy to eyeball when a word in
the final list looks wrong. Everything token-level (lemmas, word types)
moved to the LLM stage, which sees whole sentences and can therefore
reunite German separable verbs ("Er fängt ... an" -> "anfangen") that a
per-token tagger takes apart.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .artifacts import SCHEMA_VERSION, load_json, require_list, save_json, should_skip

log = logging.getLogger(__name__)

DEFAULT_SPACY_MODEL = "de_core_news_lg"

# Only the parser (which sets sentence boundaries) and its tok2vec are
# needed; the tagging/lemmatizing/NER components are excluded for speed
# since the LLM stage replaces them.
EXCLUDED_PIPES = ("tagger", "morphologizer", "lemmatizer", "attribute_ruler", "ner")

# nlp() memory grows with document size, so the text is chunked at segment
# boundaries: Whisper segments end at pauses and book segments at paragraph
# ends, so sentences are not cut mid-chunk the way a fixed character split
# would cut them.
MAX_CHUNK_CHARS = 40_000


def load_nlp(model_name: str = DEFAULT_SPACY_MODEL):
    import spacy

    try:
        return spacy.load(model_name, exclude=list(EXCLUDED_PIPES))
    except OSError as exc:
        raise SystemExit(
            f"spaCy model {model_name!r} is not installed.\n"
            f"Install it with: python -m spacy download {model_name}"
        ) from exc


def doc_to_sentences(doc) -> list[str]:
    """Whitespace-normalized sentence strings of a spaCy Doc. Pure
    conversion — testable with fake doc objects."""
    sentences = []
    for sent in doc.sents:
        text = " ".join(sent.text.split())
        if text:
            sentences.append(text)
    return sentences


def chunk_segments(segment_texts: list[str], max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Join transcript segments into chunks of at most max_chars, splitting
    only at segment boundaries."""
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for text in segment_texts:
        text = text.strip()
        if not text:
            continue
        if current and size + len(text) + 1 > max_chars:
            chunks.append(" ".join(current))
            current, size = [], 0
        current.append(text)
        size += len(text) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


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
    model_name: str = DEFAULT_SPACY_MODEL,
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

    chunks = chunk_segments(segment_texts)
    if not chunks:
        raise ValueError(
            f"{transcript_path} contains no text to analyze ({len(segments)} "
            f"segment(s), all empty)."
        )
    log.info("Splitting %d chunk(s) into sentences with %s ...", len(chunks), model_name)

    nlp = load_nlp(model_name)
    sentences: list[str] = []
    for doc in nlp.pipe(chunks):
        sentences.extend(doc_to_sentences(doc))
    log.info("Found %d sentences (%d distinct)", len(sentences), len(set(sentences)))

    import spacy

    payload = {
        "schema_version": SCHEMA_VERSION,
        "spacy_model": model_name,
        "spacy_version": spacy.__version__,
        "source": str(transcript_path),
        "total_sentences": len(sentences),
        "sentences": sentences,
    }
    save_json(output, payload)
    return payload
