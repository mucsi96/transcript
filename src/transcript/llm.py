"""Stage 4: per-sentence word extraction with an OpenAI model.

Each *distinct* sentence is sent to the LLM once (a film repeats its lines;
paying twice for "Na gut." would be silly), asking for every word worth
learning in its dictionary form. The whole sentence goes to the model
because only something that reads the sentence can put a German separable
verb back together: in "Er fängt gerade an" the word is "anfangen", which
no per-token tagger can produce from "fängt" + "an". The rules for which
words to ignore (names, numbers, function words, fillers) live in the
system prompt.

One API call per sentence means thousands of calls per film, so the stage

  - runs the calls concurrently (``--llm-concurrency`` at a time),
  - respects the provider's rate limit with a client-side sliding-window
    requests-per-minute cap (``--llm-rpm``) on top of the SDK's own
    backoff-and-retry on 429/5xx responses, and
  - appends every finished sentence to a ``<output>.partial.jsonl``
    checkpoint, so an interrupted or failed run resumes where it stopped
    instead of paying for the finished sentences again. The checkpoint is
    folded into the final artifact and deleted when the stage completes.

The API key comes from the environment (typically a .env file, loaded at
CLI startup):

    OPENAI_API_KEY=sk-...

The output artifact keeps the per-sentence results — which words the model
extracted from which sentence — so a surprising entry in the final word
list can be traced back to the exact sentence and model answer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

from .artifacts import SCHEMA_VERSION, load_json, require_list, save_json, should_skip
from .sentences import sentences_from_payload

log = logging.getLogger(__name__)

API_KEY_ENV_VAR = "OPENAI_API_KEY"

DEFAULT_LLM_MODEL = "gpt-5-mini"
DEFAULT_RPM = 60
DEFAULT_CONCURRENCY = 8

# How often to log progress, in completed sentences.
PROGRESS_EVERY = 25

WORD_TYPES = (
    "noun",
    "verb",
    "adjective",
    "adverb",
    "expression",
    "other",
)

SYSTEM_PROMPT = """\
You extract the German vocabulary worth learning from one sentence of a film
or book, for a learner's flash cards. Return every word of the sentence that
belongs on a flash card, each in its dictionary form (Grundform):

- Verbs in the infinitive. Reunite separable verbs whose prefix stands
  elsewhere in the sentence: "Er fängt gerade an" contains "anfangen",
  never "fangen". Reflexive verbs as the bare infinitive ("sich freuen"
  -> "freuen").
- Nouns in the nominative singular, capitalized, with the definite article
  ("der"/"die"/"das") in the "article" field; plural-only nouns keep "die".
  Non-noun words get article null.
- Adjectives and adverbs in the positive base form ("besser" -> "gut",
  "am liebsten" -> "gern").
- A fixed multi-word expression whose meaning is not the sum of its words
  ("auf jeden Fall") may be returned whole, as word_type "expression".

Ignore — do not return:

- Proper names of people, places, brands, and fictional characters.
- Numbers, dates, times, punctuation, and symbols.
- Function words that are learned as grammar, not vocabulary: articles,
  pronouns, prepositions, conjunctions, question words, auxiliary and modal
  verbs (sein, haben, werden, können, müssen, wollen, sollen, dürfen,
  mögen), and particles — including the separated prefix of a separable
  verb, which belongs inside the verb's infinitive instead.
- Interjections, fillers, and greetings ("äh", "hm", "na", "ja", "hallo").
- Absolute beginner (A1) words every learner meets in the first weeks
  ("gut", "machen", "gehen", "kommen", "groß").
- Anything garbled, misspelled beyond recognition, or not German.

If nothing in the sentence is worth learning, return an empty list.
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "words": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "lemma": {
                        "type": "string",
                        "description": "Dictionary form (Grundform) of the word",
                    },
                    "word_type": {"type": "string", "enum": list(WORD_TYPES)},
                    "article": {
                        "type": ["string", "null"],
                        "enum": ["der", "die", "das", None],
                        "description": "Definite article for nouns, null otherwise",
                    },
                },
                "required": ["lemma", "word_type", "article"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["words"],
    "additionalProperties": False,
}

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "extracted_words",
        "strict": True,
        "schema": RESPONSE_SCHEMA,
    },
}


class LLMError(RuntimeError):
    """An LLM stage could not call the API, or the answer was unusable.
    Messages are written to be shown to the user as-is."""


class WordExtractionError(LLMError):
    """The word-extraction stage's flavor of LLMError."""


@dataclass(frozen=True)
class WordRecord:
    lemma: str
    word_type: str
    article: str | None = None


@dataclass(frozen=True)
class SentenceWords:
    """One distinct sentence, how often the source contains it, and the
    words the LLM extracted from it."""

    text: str
    count: int
    words: tuple[WordRecord, ...]


def words_from_data(data, source: str) -> tuple[WordRecord, ...]:
    """Validate one model answer (already JSON-decoded) into WordRecords.

    Repeats of the same (lemma, word_type) within one sentence are dropped —
    the aggregation counts sentence occurrences, not the model's stutters.
    """
    if not isinstance(data, dict) or not isinstance(data.get("words"), list):
        raise ValueError(f'{source}: expected an object with a "words" array, got {data!r}')
    records: list[WordRecord] = []
    seen: set[tuple[str, str]] = set()
    for item in data["words"]:
        if not isinstance(item, dict):
            raise ValueError(f"{source}: word entry is not an object: {item!r}")
        lemma = item.get("lemma")
        word_type = item.get("word_type")
        article = item.get("article")
        if not isinstance(lemma, str) or not lemma.strip():
            raise ValueError(f"{source}: word entry without a lemma: {item!r}")
        if not isinstance(word_type, str) or not word_type:
            raise ValueError(f"{source}: word entry without a word_type: {item!r}")
        if article is not None and article not in ("der", "die", "das"):
            raise ValueError(f"{source}: unexpected article {article!r} in {item!r}")
        lemma = " ".join(lemma.split())
        key = (lemma.casefold(), word_type)
        if key in seen:
            continue
        seen.add(key)
        records.append(WordRecord(lemma=lemma, word_type=word_type, article=article))
    return tuple(records)


def completion_json(response, label: str, error: type[LLMError] = LLMError):
    """The JSON payload of one chat-completion response. Every way a
    response can be unusable — malformed object, refusal, truncation, empty
    or non-JSON content — is raised as `error` with the given label."""
    try:
        choice = response.choices[0]
        message = choice.message
    except (AttributeError, IndexError) as exc:
        raise error(
            f"unexpected LLM response for {label}: {type(exc).__name__}: {exc}"
        ) from exc
    if getattr(choice, "finish_reason", None) == "length":
        raise error(
            f"the LLM answer for {label} was cut off at the output token limit; "
            f"the result would be incomplete."
        )
    refusal = getattr(message, "refusal", None)
    if refusal:
        raise error(f"the LLM refused {label}: {refusal}")
    if not message.content:
        raise error(f"the LLM returned no content for {label}")
    try:
        return json.loads(message.content)
    except ValueError as exc:
        raise error(
            f"the LLM did not return valid JSON for {label}: {exc}. "
            f"Response: {message.content[:200]}"
        ) from exc


def parse_completion(response, sentence: str) -> tuple[WordRecord, ...]:
    """Pull the word list out of one chat-completion response."""
    data = completion_json(response, repr(sentence), WordExtractionError)
    try:
        return words_from_data(data, f"LLM answer for {sentence!r}")
    except ValueError as exc:
        raise WordExtractionError(str(exc)) from exc


class RateLimiter:
    """Sliding-window limiter: at most `rpm` request starts per minute.

    The SDK's own retry-with-backoff still handles the 429s that slip
    through (several processes, or the provider counting tokens rather
    than requests); this cap just keeps a healthy run from provoking them.
    """

    WINDOW_S = 60.0

    def __init__(self, rpm: int) -> None:
        if rpm < 1:
            raise ValueError(f"rpm must be >= 1, got {rpm}")
        self.rpm = rpm
        self._starts: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                while self._starts and now - self._starts[0] >= self.WINDOW_S:
                    self._starts.popleft()
                if len(self._starts) < self.rpm:
                    self._starts.append(now)
                    return
                wait = self.WINDOW_S - (now - self._starts[0])
            await asyncio.sleep(wait)


def checkpoint_path(output: Path) -> Path:
    return output.with_name(output.name + ".partial.jsonl")


def read_checkpoint(path: Path) -> list[dict]:
    """Records of a previous, interrupted run's JSONL checkpoint.

    A torn last line (the run died mid-write) is skipped; any other
    malformed line fails the stage rather than silently dropping paid-for
    results. Shared by every LLM stage; each caller validates its own
    record shape (and should treat a bad *last* record like a torn line).
    """
    records: list[dict] = []
    try:
        with open(path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    except FileNotFoundError:
        return records
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"expected an object, got {type(record).__name__}")
            records.append(record)
        except ValueError as exc:
            if lineno == len(lines):
                log.warning("Ignoring torn last checkpoint line in %s", path)
                continue
            raise ValueError(
                f"{path}:{lineno} is not a checkpoint record "
                f"({type(exc).__name__}: {exc}); delete the file to restart "
                f"the LLM stage from scratch."
            ) from exc
    return records


def load_checkpoint(path: Path) -> dict[str, tuple[WordRecord, ...]]:
    """Word-stage checkpoint: sentence text -> words."""
    done: dict[str, tuple[WordRecord, ...]] = {}
    records = read_checkpoint(path)
    for pos, record in enumerate(records, 1):
        try:
            done[record["text"]] = words_from_data(record, str(path))
        except (ValueError, KeyError, TypeError) as exc:
            if pos == len(records):
                log.warning("Ignoring torn last checkpoint line in %s", path)
                continue
            raise ValueError(
                f"{path}: not a checkpoint record "
                f"({type(exc).__name__}: {exc}); delete the file to restart "
                f"the LLM stage from scratch."
            ) from exc
    return done


async def _extract_all(
    client,
    pending: list[str],
    *,
    model: str,
    rpm: int,
    concurrency: int,
    checkpoint,
    done_so_far: int,
    total: int,
) -> dict[str, tuple[WordRecord, ...]]:
    """Run the LLM over `pending` sentences: concurrent, rate-limited,
    checkpointing each result as it lands. Returns sentence -> words."""
    limiter = RateLimiter(rpm)
    semaphore = asyncio.Semaphore(concurrency)
    results: dict[str, tuple[WordRecord, ...]] = {}
    completed = done_so_far

    async def one(sentence: str) -> None:
        nonlocal completed
        async with semaphore:
            await limiter.acquire()
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": sentence},
                ],
                response_format=RESPONSE_FORMAT,
            )
        words = parse_completion(response, sentence)
        results[sentence] = words
        checkpoint.write(
            json.dumps(
                {"text": sentence, "words": [vars(w) for w in words]},
                ensure_ascii=False,
            )
            + "\n"
        )
        checkpoint.flush()
        completed += 1
        if completed % PROGRESS_EVERY == 0 or completed == total:
            log.info("LLM: %d/%d sentences", completed, total)

    tasks = [asyncio.create_task(one(s)) for s in pending]
    try:
        await asyncio.gather(*tasks)
    finally:
        # On failure: stop the still-running calls, and collect their
        # cancellations so the loop shuts down without warnings.
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return results


def _make_client(concurrency: int):
    """The AsyncOpenAI client, imported lazily so the tests (and the
    non-LLM stages) run without the openai package installed."""
    try:
        import openai
    except ImportError as exc:
        raise WordExtractionError(
            "the openai package is not installed; run pip install -e . again."
        ) from exc

    if not os.environ.get(API_KEY_ENV_VAR):
        raise WordExtractionError(
            f"{API_KEY_ENV_VAR} is not set (configure it in .env); the LLM "
            f"word-extraction stage needs an OpenAI API key."
        )
    # The SDK retries 429/5xx with backoff on its own; give it more room
    # than the default 2 attempts since a long run will brush the limit.
    return openai.AsyncOpenAI(max_retries=5)


def extract_words(
    sentences_path: Path,
    output: Path,
    *,
    model: str = DEFAULT_LLM_MODEL,
    rpm: int = DEFAULT_RPM,
    concurrency: int = DEFAULT_CONCURRENCY,
    force: bool = False,
) -> dict:
    if should_skip(output, force):
        return load_json(output)

    payload = load_json(sentences_path)
    sentences = sentences_from_payload(payload, sentences_path)
    counts = Counter(sentences)  # insertion-ordered: first-seen order
    if not counts:
        raise ValueError(
            f"{sentences_path} contains no sentences; re-run the sentences "
            f"stage with --force."
        )

    ckpt_path = checkpoint_path(output)
    done = load_checkpoint(ckpt_path)
    done = {text: words for text, words in done.items() if text in counts}
    pending = [text for text in counts if text not in done]
    log.info(
        "Extracting words from %d distinct sentences (%d total) with %s; "
        "%d already done from a previous run",
        len(counts), len(sentences), model, len(done),
    )

    if pending:
        client = _make_client(concurrency)
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
                        total=len(counts),
                    )
                )
            except WordExtractionError as exc:
                raise WordExtractionError(
                    f"{exc} Finished sentences are checkpointed in "
                    f"{ckpt_path}; re-run the stage to resume."
                ) from exc
            except Exception as exc:
                raise WordExtractionError(
                    f"LLM word extraction failed ({type(exc).__name__}: {exc}). "
                    f"Finished sentences are checkpointed in {ckpt_path}; "
                    f"re-run the stage to resume."
                ) from exc
        done.update(results)

    result_payload = {
        "schema_version": SCHEMA_VERSION,
        "llm_model": model,
        "source": str(sentences_path),
        "total_sentences": len(counts),
        "sentences": [
            {
                "text": text,
                "count": count,
                "words": [vars(w) for w in done[text]],
            }
            for text, count in counts.items()
        ],
    }
    save_json(output, result_payload)
    ckpt_path.unlink(missing_ok=True)
    return result_payload


def sentence_words_from_payload(payload: dict, source: Path | str = "sentence-words") -> list[SentenceWords]:
    """Rebuild records from a sentence-words.json payload."""
    sentences = require_list(payload, "sentences", Path(source))
    result: list[SentenceWords] = []
    for sent in sentences:
        try:
            result.append(
                SentenceWords(
                    text=sent["text"],
                    count=int(sent["count"]),
                    words=tuple(WordRecord(**w) for w in sent["words"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{source}: malformed sentence record ({type(exc).__name__}: {exc}); "
                f"re-run the words stage with --force."
            ) from exc
    return result
