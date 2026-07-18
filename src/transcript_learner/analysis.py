"""Decide which recorded words are worth learning.

Three stages, cheapest first:
  1. Exact match against the user's known-words list (normalized).
  2. Lemma match (spaCy German) — "spielte" is known if "spielen" is known.
  3. GPT-5 judgement on the remainder — is this a real, learnable German
     word the learner doesn't already know (filters names, onomatopoeia,
     mis-recognitions, inflected duplicates the lemmatizer missed)?
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

from .session import WordEntry, normalize


@dataclass
class Candidate:
    word: str
    lemma: str
    context: str
    count: int
    reason: str = ""


# --------------------------------------------------------------------------
# Lemmatization
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_nlp():
    """Load the spaCy German pipeline once, or return None if unavailable."""
    try:
        import spacy

        return spacy.load("de_core_news_sm", disable=["ner", "parser"])
    except Exception:
        return None


def lemmatize(word: str) -> str:
    nlp = _load_nlp()
    if nlp is None:
        return word
    doc = nlp(word)
    if len(doc) == 0:
        return word
    return doc[0].lemma_.lower()


def lemmatize_many(words: list[str]) -> dict[str, str]:
    nlp = _load_nlp()
    if nlp is None:
        return {w: w for w in words}
    out: dict[str, str] = {}
    for w in words:
        doc = nlp(w)
        out[w] = doc[0].lemma_.lower() if len(doc) else w
    return out


# --------------------------------------------------------------------------
# GPT-5 filtering
# --------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You are a German-language tutor helping a learner build a vocabulary list "
    "from words heard while watching a German cartoon. You will receive a list "
    "of candidate words (each already checked to NOT be in the learner's known "
    "words, even after lemmatization) and the learner's known-words list.\n\n"
    "For each candidate decide whether it is worth adding as a NEW vocabulary "
    "item. Exclude a candidate if it is:\n"
    "  - a proper noun / character name / place name,\n"
    "  - onomatopoeia, an interjection, or filler (e.g. 'aha', 'hmm'),\n"
    "  - a likely speech-recognition error or non-word fragment,\n"
    "  - merely an inflected form of a word already known,\n"
    "  - a trivial function word the learner almost certainly knows.\n\n"
    "Keep it if it is a genuine, useful German word (noun, verb, adjective, "
    "adverb) that a learner would benefit from studying. Return the base/"
    "dictionary form in 'lemma' (e.g. infinitive for verbs, nominative "
    "singular for nouns, with correct article-implied capitalization for nouns)."
)


def _build_user_prompt(candidates: list[str], known: list[str]) -> str:
    known_sample = ", ".join(sorted(known)[:400]) if known else "(none provided)"
    return (
        "Known words (already learned):\n"
        f"{known_sample}\n\n"
        "Candidate words to evaluate:\n"
        f"{json.dumps(candidates, ensure_ascii=False)}\n\n"
        "Respond ONLY with a JSON object of the form:\n"
        '{"words": [{"word": "<candidate as given>", '
        '"worth_learning": true|false, "lemma": "<dictionary form>", '
        '"reason": "<short reason>"}]}\n'
        "Include every candidate exactly once."
    )


def gpt_filter(
    candidates: list[str],
    known: list[str],
    *,
    api_key: str,
    model: str,
) -> dict[str, dict]:
    """Return {candidate: {worth_learning, lemma, reason}} keyed by input word."""
    if not candidates:
        return {}
    if not api_key:
        # No key: be permissive — keep everything, mark it clearly.
        return {
            w: {"worth_learning": True, "lemma": w, "reason": "no AI key; unfiltered"}
            for w in candidates
        }

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(candidates, known)},
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        data = json.loads(content)
    except Exception as exc:
        # Fail open rather than losing the session's data.
        return {
            w: {"worth_learning": True, "lemma": w, "reason": f"AI error: {exc}"}
            for w in candidates
        }

    result: dict[str, dict] = {}
    for item in data.get("words", []):
        word = (item.get("word") or "").strip()
        if not word:
            continue
        result[normalize(word)] = {
            "worth_learning": bool(item.get("worth_learning", True)),
            "lemma": (item.get("lemma") or word).strip(),
            "reason": (item.get("reason") or "").strip(),
        }
    # Any candidate the model dropped — keep it, conservatively.
    for w in candidates:
        result.setdefault(
            normalize(w),
            {"worth_learning": True, "lemma": w, "reason": "not returned by AI"},
        )
    return result


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def parse_known_words(raw: str) -> set[str]:
    """Split a pasted known-words blob (commas / newlines / spaces) into a set."""
    tokens = raw.replace(",", "\n").split()
    return {normalize(t) for t in tokens if normalize(t)}


def analyze(
    entries: list[WordEntry],
    known_raw: str,
    *,
    api_key: str,
    model: str,
) -> dict:
    known = parse_known_words(known_raw)
    known_lemmas = {lemmatize(w) for w in known} if known else set()

    stage1_known: list[str] = []  # exact match
    stage2_known: list[str] = []  # lemma match
    remaining: list[WordEntry] = []

    for entry in entries:
        word = entry.word
        if word in known:
            stage1_known.append(word)
            continue
        lemma = lemmatize(word)
        if lemma in known or lemma in known_lemmas:
            stage2_known.append(word)
            continue
        remaining.append(entry)

    candidate_words = [e.word for e in remaining]
    verdicts = gpt_filter(candidate_words, sorted(known), api_key=api_key, model=model)

    words_to_learn: list[dict] = []
    ai_rejected: list[dict] = []
    for entry in remaining:
        verdict = verdicts.get(entry.word, {"worth_learning": True, "lemma": entry.word, "reason": ""})
        record = {
            "word": entry.word,
            "surface": entry.surface,
            "lemma": verdict.get("lemma", entry.word),
            "context": entry.context,
            "count": entry.count,
            "reason": verdict.get("reason", ""),
        }
        if verdict.get("worth_learning", True):
            words_to_learn.append(record)
        else:
            ai_rejected.append(record)

    words_to_learn.sort(key=lambda r: (-r["count"], r["word"]))

    return {
        "summary": {
            "total_unique": len(entries),
            "already_known_exact": len(stage1_known),
            "already_known_lemma": len(stage2_known),
            "ai_rejected": len(ai_rejected),
            "to_learn": len(words_to_learn),
            "model": model,
            "lemmatizer": "de_core_news_sm" if _load_nlp() is not None else "unavailable",
        },
        "words_to_learn": words_to_learn,
        "already_known_exact": sorted(stage1_known),
        "already_known_lemma": sorted(stage2_known),
        "ai_rejected": ai_rejected,
    }
