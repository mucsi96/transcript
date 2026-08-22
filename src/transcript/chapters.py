"""Stage 3 (EPUB): chapter classification and sentence extraction with an
OpenAI model.

Each spine document's raw XHTML goes to the LLM, which decides whether the
document is real reading content at all — as opposed to a cover, title
page, copyright notice, table of contents, index or advertisement — and,
for content, returns the sentences of the running text in a structured
answer. Nothing here parses the HTML: judging what a document *is* and
what its prose says is exactly the kind of reading only the model can do,
and it makes the manual `--chapters` selection an optional override rather
than a necessity.

The stage produces the same sentences.json artifact the syntok path
produces for audio transcripts, so the downstream `words` and `build`
stages do not care where the sentences came from. Per-chapter decisions
(kind, content or not, sentence count) are recorded in the artifact for
debugging, and finished chapters are checkpointed to
``<output>.partial.jsonl`` so an interrupted run resumes without paying
for them again — same mechanics as the word-extraction stage.

An oversized document (a badly converted book with one huge file) is split
at block-tag boundaries and sent in parts; the chapter counts as content
if any part is judged content, and the parts' sentences are concatenated.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from . import llm
from .artifacts import SCHEMA_VERSION, load_json, require_list, save_json, should_skip
from .llm import (
    DEFAULT_CONCURRENCY,
    DEFAULT_LLM_MODEL,
    DEFAULT_RPM,
    LLMError,
    RateLimiter,
    checkpoint_path,
    completion_json,
    read_checkpoint,
)

log = logging.getLogger(__name__)

# One request per part keeps the answer (the full sentence list) well under
# any output token limit; ~60k chars of markup is roughly 15-20k tokens in.
MAX_CHAPTER_HTML_CHARS = 60_000

_BLOCK_START = re.compile(r"(?=<(?:p|div|h[1-6]|section|article)\b)", re.IGNORECASE)

KINDS = (
    "chapter",
    "foreword",
    "afterword",
    "cover",
    "title_page",
    "copyright",
    "dedication",
    "toc",
    "index",
    "navigation",
    "advertisement",
    "other",
)

SYSTEM_PROMPT = """\
You read one XHTML document from the spine of a German EPUB book and
prepare it for vocabulary extraction.

First decide what the document is. Only real reading content — a chapter
or section of the book's actual text, including a foreword or afterword
written as prose — is worth processing. A cover, title page, copyright or
imprint page, dedication, table of contents, index, navigation page,
advertisement, or a page that is only images, tables or reference lists
is not.

Answer with:
- "kind": what the document is.
- "is_content": true only for real reading content.
- "sentences": for content, every sentence of the running text in reading
  order, as plain text — markup, footnote markers, page numbers and
  repeated page headers removed, whitespace normalized, end-of-line
  hyphenation undone. Include standalone headings as their own entries.
  For a non-content document, an empty list.

Never translate, summarize, correct or invent text: every entry must be a
verbatim sentence of the document.
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": list(KINDS)},
        "is_content": {"type": "boolean"},
        "sentences": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["kind", "is_content", "sentences"],
    "additionalProperties": False,
}

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "chapter_sentences",
        "strict": True,
        "schema": RESPONSE_SCHEMA,
    },
}


class ChapterExtractionError(LLMError):
    """The chapter-sentences stage's flavor of LLMError."""


@dataclass(frozen=True)
class ChapterResult:
    index: int
    href: str
    title: str
    kind: str
    is_content: bool
    sentences: tuple[str, ...]


def split_html(html: str, max_chars: int = MAX_CHAPTER_HTML_CHARS) -> list[str]:
    """Split an oversized document at block-tag starts into parts of at
    most max_chars; a document within the limit is returned whole."""
    if len(html) <= max_chars:
        return [html]
    parts: list[str] = []
    current = ""
    for piece in _BLOCK_START.split(html):
        while len(piece) > max_chars:  # no block boundary in sight: hard cut
            if current:
                parts.append(current)
                current = ""
            parts.append(piece[:max_chars])
            piece = piece[max_chars:]
        if not current:
            current = piece
        elif len(current) + len(piece) <= max_chars:
            current += piece
        else:
            parts.append(current)
            current = piece
    if current:
        parts.append(current)
    return parts


def chapter_answer_from_data(data, source: str) -> tuple[str, bool, tuple[str, ...]]:
    """Validate one model answer into (kind, is_content, sentences)."""
    if not isinstance(data, dict):
        raise ValueError(f"{source}: expected an object, got {data!r}")
    kind = data.get("kind")
    is_content = data.get("is_content")
    sentences = data.get("sentences")
    if not isinstance(kind, str) or not kind:
        raise ValueError(f"{source}: answer without a kind: {data!r}")
    if not isinstance(is_content, bool):
        raise ValueError(f"{source}: answer without an is_content flag: {data!r}")
    if not isinstance(sentences, list) or not all(
        isinstance(s, str) for s in sentences
    ):
        raise ValueError(f'{source}: answer without a "sentences" string array')
    cleaned = tuple(" ".join(s.split()) for s in sentences if s.strip())
    return kind, is_content, cleaned


def _combine_parts(
    parts: list[tuple[str, bool, tuple[str, ...]]],
) -> tuple[str, bool, tuple[str, ...]]:
    """One verdict for a document sent in several parts: content if any part
    is content, sentences concatenated from the content parts."""
    content_parts = [p for p in parts if p[1]]
    if not content_parts:
        return parts[0][0], False, ()
    kind = content_parts[0][0]
    sentences = tuple(s for _, _, sents in content_parts for s in sents)
    return kind, True, sentences


def _checkpoint_key(record: dict) -> tuple[int, str]:
    return int(record["index"]), str(record["href"])


def load_chapter_checkpoint(path: Path) -> dict[tuple[int, str], ChapterResult]:
    """Finished chapters of a previous, interrupted run."""
    done: dict[tuple[int, str], ChapterResult] = {}
    records = read_checkpoint(path)
    for pos, record in enumerate(records, 1):
        try:
            kind, is_content, sentences = chapter_answer_from_data(record, str(path))
            done[_checkpoint_key(record)] = ChapterResult(
                index=int(record["index"]),
                href=str(record["href"]),
                title=str(record.get("title", "")),
                kind=kind,
                is_content=is_content,
                sentences=sentences,
            )
        except (ValueError, KeyError, TypeError) as exc:
            if pos == len(records):
                log.warning("Ignoring torn last checkpoint line in %s", path)
                continue
            raise ValueError(
                f"{path}: not a checkpoint record "
                f"({type(exc).__name__}: {exc}); delete the file to restart "
                f"the stage from scratch."
            ) from exc
    return done


async def _extract_all(
    client,
    pending: list[dict],
    *,
    model: str,
    rpm: int,
    concurrency: int,
    checkpoint,
    done_so_far: int,
    total: int,
) -> dict[tuple[int, str], ChapterResult]:
    limiter = RateLimiter(rpm)
    semaphore = asyncio.Semaphore(concurrency)
    results: dict[tuple[int, str], ChapterResult] = {}
    completed = done_so_far

    async def ask(label: str, content: str):
        async with semaphore:
            await limiter.acquire()
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                response_format=RESPONSE_FORMAT,
            )
        data = completion_json(response, label, ChapterExtractionError)
        try:
            return chapter_answer_from_data(data, f"LLM answer for {label}")
        except ValueError as exc:
            raise ChapterExtractionError(str(exc)) from exc

    async def one(chapter: dict) -> None:
        nonlocal completed
        index, href = int(chapter["index"]), str(chapter["href"])
        parts = split_html(chapter["html"], MAX_CHAPTER_HTML_CHARS)
        answers = []
        for number, part in enumerate(parts, 1):
            label = f"chapter {index} ({href})"
            if len(parts) > 1:
                label += f" part {number}/{len(parts)}"
                part = f"Document {href}, part {number} of {len(parts)}:\n\n{part}"
            else:
                part = f"Document {href}:\n\n{part}"
            answers.append(await ask(label, part))
        kind, is_content, sentences = _combine_parts(answers)
        result = ChapterResult(
            index=index,
            href=href,
            title=str(chapter.get("title", "")),
            kind=kind,
            is_content=is_content,
            sentences=sentences,
        )
        results[(index, href)] = result
        checkpoint.write(
            json.dumps(
                {
                    "index": index,
                    "href": href,
                    "title": result.title,
                    "kind": kind,
                    "is_content": is_content,
                    "sentences": list(sentences),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        checkpoint.flush()
        completed += 1
        verdict = kind if not is_content else f"{kind}, {len(sentences)} sentences"
        log.info("Chapter %d/%d: %s -> %s", completed, total, href, verdict)

    tasks = [asyncio.create_task(one(c)) for c in pending]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return results


def extract_chapter_sentences(
    chapters_path: Path,
    output: Path,
    *,
    model: str = DEFAULT_LLM_MODEL,
    rpm: int = DEFAULT_RPM,
    concurrency: int = DEFAULT_CONCURRENCY,
    force: bool = False,
) -> dict:
    """The sentences.json artifact for an EPUB: LLM-judged chapters."""
    if should_skip(output, force):
        return load_json(output)

    payload = load_json(chapters_path)
    chapters = require_list(payload, "chapters", chapters_path)
    for chapter in chapters:
        if not isinstance(chapter, dict) or not isinstance(chapter.get("html"), str):
            raise ValueError(
                f"{chapters_path}: chapter record without 'html' "
                f"({chapter!r:.120}); re-run the epub stage with --force."
            )
    if not chapters:
        raise ValueError(
            f"{chapters_path} contains no chapters; re-run the epub stage "
            f"with --force."
        )

    keys = [(int(c["index"]), str(c["href"])) for c in chapters]
    ckpt_path = checkpoint_path(output)
    done = load_chapter_checkpoint(ckpt_path)
    done = {key: result for key, result in done.items() if key in set(keys)}
    pending = [c for c, key in zip(chapters, keys) if key not in done]
    log.info(
        "Judging %d chapter document(s) with %s; %d already done from a "
        "previous run",
        len(chapters), model, len(done),
    )

    if pending:
        client = llm._make_client(concurrency)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ckpt_path, "a", encoding="utf-8") as checkpoint:
            try:
                results = asyncio.run(
                    _extract_all(
                        client,
                        pending,
                        model=model,
                        rpm=rpm,
                        concurrency=concurrency,
                        checkpoint=checkpoint,
                        done_so_far=len(done),
                        total=len(chapters),
                    )
                )
            except ChapterExtractionError as exc:
                raise ChapterExtractionError(
                    f"{exc} Finished chapters are checkpointed in "
                    f"{ckpt_path}; re-run the stage to resume."
                ) from exc
            except Exception as exc:
                raise ChapterExtractionError(
                    f"chapter processing failed ({type(exc).__name__}: {exc}). "
                    f"Finished chapters are checkpointed in {ckpt_path}; "
                    f"re-run the stage to resume."
                ) from exc
        done.update(results)

    ordered = [done[key] for key in keys]
    sentences = [s for r in ordered if r.is_content for s in r.sentences]
    if not sentences:
        skipped = ", ".join(f"{r.href} ({r.kind})" for r in ordered)
        raise ValueError(
            f"the LLM judged no chapter of {chapters_path} to be book content: "
            f"{skipped}. If that is wrong, narrow the input with --chapters "
            f"and re-run with --force."
        )

    content_count = sum(1 for r in ordered if r.is_content)
    log.info(
        "Found %d sentences in %d content chapter(s); skipped %d other document(s)",
        len(sentences), content_count, len(ordered) - content_count,
    )

    result_payload = {
        "schema_version": SCHEMA_VERSION,
        "segmenter": "llm",
        "llm_model": model,
        "source": str(chapters_path),
        "total_sentences": len(sentences),
        "chapters": [
            {
                "index": r.index,
                "href": r.href,
                "title": r.title,
                "kind": r.kind,
                "is_content": r.is_content,
                "sentences": len(r.sentences),
            }
            for r in ordered
        ],
        "sentences": sentences,
    }
    save_json(output, result_payload)
    ckpt_path.unlink(missing_ok=True)
    return result_payload
