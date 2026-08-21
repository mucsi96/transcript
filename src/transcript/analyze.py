"""Stage 3: spaCy analysis — sentence splitting, lemmatization, POS, NER.

Input is the text artifact of stage 2 — Whisper segments from `transcribe`
or book paragraphs from `epub`; both are lists of short text segments, so
this stage does not care which produced them.

spaCy objects are converted to plain dataclasses immediately; the later
stages (matching, filtering, aggregation) never touch spaCy.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from .artifacts import SCHEMA_VERSION, load_json, save_json, should_skip

log = logging.getLogger(__name__)

DEFAULT_SPACY_MODEL = "de_core_news_lg"

# nlp() memory grows with document size, so the text is chunked at segment
# boundaries: Whisper segments end at pauses and book segments at paragraph
# ends, so sentences are not cut mid-chunk the way a fixed character split
# would cut them.
MAX_CHUNK_CHARS = 40_000


@dataclass(frozen=True)
class TokenRecord:
    text: str
    lemma: str
    pos: str
    ent_type: str
    is_alpha: bool
    is_stop: bool


@dataclass(frozen=True)
class SentenceRecord:
    text: str
    tokens: tuple[TokenRecord, ...]


def load_nlp(model_name: str = DEFAULT_SPACY_MODEL):
    import spacy

    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise SystemExit(
            f"spaCy model {model_name!r} is not installed.\n"
            f"Install it with: python -m spacy download {model_name}"
        ) from exc


def doc_to_sentences(doc) -> list[SentenceRecord]:
    """Convert a spaCy Doc into plain records. Pure conversion — testable
    with fake doc objects."""
    sentences = []
    for sent in doc.sents:
        text = " ".join(sent.text.split())
        if not text:
            continue
        tokens = tuple(
            TokenRecord(
                text=tok.text,
                lemma=tok.lemma_,
                pos=tok.pos_,
                ent_type=tok.ent_type_,
                is_alpha=tok.is_alpha,
                is_stop=tok.is_stop,
            )
            for tok in sent
        )
        sentences.append(SentenceRecord(text=text, tokens=tokens))
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


def sentences_from_payload(payload: dict) -> list[SentenceRecord]:
    """Rebuild records from an analysis.json payload."""
    return [
        SentenceRecord(
            text=sent["text"],
            tokens=tuple(TokenRecord(**tok) for tok in sent["tokens"]),
        )
        for sent in payload["sentences"]
    ]


def analyze(
    transcript_path: Path,
    output: Path,
    *,
    model_name: str = DEFAULT_SPACY_MODEL,
    force: bool = False,
) -> dict:
    if should_skip(output, force):
        return load_json(output)

    transcript = load_json(transcript_path)
    segment_texts = [seg["text"] for seg in transcript["segments"]]
    chunks = chunk_segments(segment_texts)
    log.info("Analyzing %d chunk(s) with %s ...", len(chunks), model_name)

    nlp = load_nlp(model_name)
    sentences: list[SentenceRecord] = []
    for doc in nlp.pipe(chunks):
        sentences.extend(doc_to_sentences(doc))
    log.info("Found %d sentences", len(sentences))

    import spacy

    payload = {
        "schema_version": SCHEMA_VERSION,
        "spacy_model": model_name,
        "spacy_version": spacy.__version__,
        "source": str(transcript_path),
        "sentences": [
            {"text": sent.text, "tokens": [asdict(tok) for tok in sent.tokens]}
            for sent in sentences
        ],
    }
    save_json(output, payload)
    return payload
