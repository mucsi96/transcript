import asyncio
import json
from types import SimpleNamespace

import pytest

import transcript.llm as llm
from transcript.artifacts import save_json
from transcript.llm import (
    RateLimiter,
    SentenceWords,
    WordExtractionError,
    WordRecord,
    checkpoint_path,
    extract_words,
    load_checkpoint,
    parse_completion,
    sentence_words_from_payload,
    words_from_data,
)

# ---------------------------------------------------------------- helpers


def completion(payload):
    """A chat-completion response answering with `payload` as JSON."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(payload, ensure_ascii=False), refusal=None
                )
            )
        ]
    )


class FakeClient:
    """Stands in for AsyncOpenAI: answers per-sentence from a dict, records
    every sentence it was asked about."""

    def __init__(self, answers, fail_on=()):
        self.calls = []
        self._answers = answers
        self._fail_on = set(fail_on)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    async def _create(self, *, model, messages, response_format):
        sentence = messages[1]["content"]
        self.calls.append(sentence)
        if sentence in self._fail_on:
            raise RuntimeError(f"boom on {sentence!r}")
        return completion({"words": self._answers.get(sentence, [])})


def write_sentences(tmp_path, sentences):
    path = tmp_path / "sentences.json"
    save_json(path, {"schema_version": 1, "sentences": sentences})
    return path


HUND = {"lemma": "Hund", "word_type": "noun", "article": "der"}
BELLEN = {"lemma": "bellen", "word_type": "verb", "article": None}
ANFANGEN = {"lemma": "anfangen", "word_type": "verb", "article": None}


# ---------------------------------------------------- response validation


def test_words_from_data_builds_records():
    words = words_from_data({"words": [HUND, BELLEN]}, "test")
    assert words == (
        WordRecord("Hund", "noun", "der"),
        WordRecord("bellen", "verb", None),
    )


def test_words_from_data_drops_repeats_within_a_sentence():
    words = words_from_data({"words": [HUND, dict(HUND, article=None)]}, "test")
    assert words == (WordRecord("Hund", "noun", "der"),)


def test_words_from_data_rejects_bad_shapes():
    with pytest.raises(ValueError, match='"words" array'):
        words_from_data(["Hund"], "test")
    with pytest.raises(ValueError, match="without a lemma"):
        words_from_data({"words": [{"word_type": "noun", "article": None}]}, "test")
    with pytest.raises(ValueError, match="without a word_type"):
        words_from_data({"words": [{"lemma": "Hund", "article": None}]}, "test")
    with pytest.raises(ValueError, match="unexpected article"):
        words_from_data({"words": [dict(HUND, article="den")]}, "test")


def test_parse_completion_reports_a_refusal():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, refusal="no"))]
    )
    with pytest.raises(WordExtractionError, match="refused"):
        parse_completion(response, "Der Hund bellt.")


def test_parse_completion_reports_non_json():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="<html>", refusal=None))]
    )
    with pytest.raises(WordExtractionError, match="did not return valid JSON"):
        parse_completion(response, "Der Hund bellt.")


# ------------------------------------------------------------ rate limiter


def test_rate_limiter_rejects_zero_rpm():
    with pytest.raises(ValueError):
        RateLimiter(0)


def test_rate_limiter_admits_up_to_rpm_without_waiting():
    async def run():
        limiter = RateLimiter(5)
        for _ in range(5):
            await asyncio.wait_for(limiter.acquire(), timeout=1)
        assert len(limiter._starts) == 5

    asyncio.run(run())


# -------------------------------------------------------------- checkpoint


def test_load_checkpoint_missing_file_is_empty(tmp_path):
    assert load_checkpoint(tmp_path / "nope.jsonl") == {}


def test_load_checkpoint_ignores_a_torn_last_line(tmp_path):
    path = tmp_path / "ckpt.jsonl"
    good = json.dumps({"text": "Der Hund bellt.", "words": [HUND]})
    path.write_text(good + '\n{"text": "Er f', encoding="utf-8")
    done = load_checkpoint(path)
    assert done == {"Der Hund bellt.": (WordRecord("Hund", "noun", "der"),)}


def test_load_checkpoint_rejects_corruption_before_the_end(tmp_path):
    path = tmp_path / "ckpt.jsonl"
    good = json.dumps({"text": "Der Hund bellt.", "words": []})
    path.write_text("not json\n" + good + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a checkpoint record"):
        load_checkpoint(path)


# ------------------------------------------------------------ extract_words


def test_extract_words_calls_once_per_distinct_sentence(tmp_path, monkeypatch):
    sentences = write_sentences(
        tmp_path, ["Der Hund bellt.", "Er fängt an.", "Der Hund bellt."]
    )
    client = FakeClient(
        {"Der Hund bellt.": [HUND, BELLEN], "Er fängt an.": [ANFANGEN]}
    )
    monkeypatch.setattr(llm, "_make_client", lambda concurrency: client)

    out = tmp_path / "sentence-words.json"
    payload = extract_words(sentences, out, model="test-model")

    assert sorted(client.calls) == ["Der Hund bellt.", "Er fängt an."]
    assert payload["llm_model"] == "test-model"
    assert payload["total_sentences"] == 2
    assert payload["sentences"] == [
        {"text": "Der Hund bellt.", "count": 2, "words": [HUND, BELLEN]},
        {"text": "Er fängt an.", "count": 1, "words": [ANFANGEN]},
    ]
    assert not checkpoint_path(out).exists()
    assert json.loads(out.read_text(encoding="utf-8")) == payload


def test_extract_words_resumes_from_the_checkpoint(tmp_path, monkeypatch):
    sentences = write_sentences(tmp_path, ["Der Hund bellt.", "Er fängt an."])
    out = tmp_path / "sentence-words.json"
    ckpt = checkpoint_path(out)
    ckpt.write_text(
        json.dumps({"text": "Der Hund bellt.", "words": [HUND]}) + "\n",
        encoding="utf-8",
    )
    client = FakeClient({"Er fängt an.": [ANFANGEN]})
    monkeypatch.setattr(llm, "_make_client", lambda concurrency: client)

    payload = extract_words(sentences, out)

    assert client.calls == ["Er fängt an."]  # the finished one is not re-paid
    assert payload["sentences"][0]["words"] == [HUND]
    assert payload["sentences"][1]["words"] == [ANFANGEN]
    assert not ckpt.exists()


def test_extract_words_failure_keeps_the_checkpoint(tmp_path, monkeypatch):
    sentences = write_sentences(tmp_path, ["Der Hund bellt.", "Er fängt an."])
    client = FakeClient(
        {"Der Hund bellt.": [HUND]}, fail_on=["Er fängt an."]
    )
    monkeypatch.setattr(llm, "_make_client", lambda concurrency: client)

    out = tmp_path / "sentence-words.json"
    with pytest.raises(WordExtractionError) as excinfo:
        extract_words(sentences, out, concurrency=1)

    assert "re-run the stage to resume" in str(excinfo.value)
    assert not out.exists()
    done = load_checkpoint(checkpoint_path(out))
    assert done == {"Der Hund bellt.": (WordRecord("Hund", "noun", "der"),)}


def test_extract_words_uses_the_cached_artifact(tmp_path, monkeypatch):
    out = tmp_path / "sentence-words.json"
    save_json(out, {"schema_version": 1, "sentences": []})
    monkeypatch.setattr(
        llm, "_make_client", lambda concurrency: pytest.fail("must not call the API")
    )
    payload = extract_words(tmp_path / "missing.json", out)
    assert payload["sentences"] == []


def test_extract_words_rejects_an_empty_sentence_list(tmp_path):
    sentences = write_sentences(tmp_path, [])
    with pytest.raises(ValueError, match="no sentences"):
        extract_words(sentences, tmp_path / "out.json")


# ------------------------------------------------------------ payload IO


def test_sentence_words_from_payload_roundtrip():
    payload = {
        "sentences": [
            {"text": "Er fängt an.", "count": 2, "words": [ANFANGEN]},
        ]
    }
    records = sentence_words_from_payload(payload)
    assert records == [
        SentenceWords(
            text="Er fängt an.",
            count=2,
            words=(WordRecord("anfangen", "verb", None),),
        )
    ]


def test_sentence_words_from_payload_reports_a_malformed_record():
    payload = {"sentences": [{"text": "Er fängt an."}]}  # no count/words
    with pytest.raises(ValueError) as excinfo:
        sentence_words_from_payload(payload, "work/sentence-words.json")
    assert "malformed sentence record" in str(excinfo.value)
    assert "--force" in str(excinfo.value)
