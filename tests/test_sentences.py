from types import SimpleNamespace

import pytest

from transcript.sentences import chunk_segments, doc_to_sentences, sentences_from_payload


def test_doc_to_sentences_normalizes_whitespace():
    class FakeSpan:
        def __init__(self, text):
            self.text = text

    doc = SimpleNamespace(sents=[FakeSpan("Der  Hund . "), FakeSpan("Er läuft.")])
    assert doc_to_sentences(doc) == ["Der Hund .", "Er läuft."]


def test_doc_to_sentences_skips_empty_sentences():
    doc = SimpleNamespace(sents=[SimpleNamespace(text="   ")])
    assert doc_to_sentences(doc) == []


def test_chunk_segments_splits_at_boundaries():
    segments = ["a" * 30, "b" * 30, "c" * 30]
    chunks = chunk_segments(segments, max_chars=70)
    assert chunks == ["a" * 30 + " " + "b" * 30, "c" * 30]


def test_chunk_segments_skips_empty_and_never_splits_a_segment():
    segments = ["", "  ", "x" * 100]
    assert chunk_segments(segments, max_chars=10) == ["x" * 100]


def test_sentences_from_payload_roundtrip():
    payload = {"sentences": ["Der Hund bellt.", "Der Hund bellt."]}
    assert sentences_from_payload(payload) == ["Der Hund bellt.", "Der Hund bellt."]


def test_sentences_from_payload_rejects_a_payload_without_sentences():
    with pytest.raises(ValueError) as excinfo:
        sentences_from_payload({"schema_version": 1}, "work/sentences.json")
    message = str(excinfo.value)
    assert "work/sentences.json" in message
    assert "'sentences' list" in message


def test_sentences_from_payload_reports_a_malformed_record():
    payload = {"sentences": ["Der Hund bellt.", {"text": "nope"}]}
    with pytest.raises(ValueError) as excinfo:
        sentences_from_payload(payload, "work/sentences.json")
    assert "malformed sentence record" in str(excinfo.value)
    assert "--force" in str(excinfo.value)


def test_real_german_sentence_split_smoke():
    spacy = pytest.importorskip("spacy")
    if not spacy.util.is_package("de_core_news_lg"):
        pytest.skip("de_core_news_lg not installed")
    from transcript.sentences import load_nlp

    nlp = load_nlp("de_core_news_lg")
    doc = nlp("Der Hund läuft schnell. Anna wohnt in Berlin.")
    assert doc_to_sentences(doc) == [
        "Der Hund läuft schnell.",
        "Anna wohnt in Berlin.",
    ]
