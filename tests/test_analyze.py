from types import SimpleNamespace

import pytest

from transcript.analyze import (
    SentenceRecord,
    TokenRecord,
    chunk_segments,
    doc_to_sentences,
    sentences_from_payload,
)


def fake_token(text, lemma, pos, ent_type="", is_alpha=True, is_stop=False):
    return SimpleNamespace(
        text=text,
        lemma_=lemma,
        pos_=pos,
        ent_type_=ent_type,
        is_alpha=is_alpha,
        is_stop=is_stop,
    )


def test_doc_to_sentences_with_fake_doc():
    tokens = [
        fake_token("Der", "der", "DET", is_stop=True),
        fake_token("Hund", "Hund", "NOUN"),
        fake_token(".", ".", "PUNCT", is_alpha=False),
    ]

    class FakeSpan:
        text = "Der  Hund . "

        def __iter__(self):
            return iter(tokens)

    doc = SimpleNamespace(sents=[FakeSpan()])
    sentences = doc_to_sentences(doc)
    assert sentences == [
        SentenceRecord(
            text="Der Hund .",
            tokens=(
                TokenRecord("Der", "der", "DET", "", True, True),
                TokenRecord("Hund", "Hund", "NOUN", "", True, False),
                TokenRecord(".", ".", "PUNCT", "", False, False),
            ),
        )
    ]


def test_doc_to_sentences_skips_empty_sentences():
    class EmptySpan:
        text = "   "

        def __iter__(self):
            return iter([])

    doc = SimpleNamespace(sents=[EmptySpan()])
    assert doc_to_sentences(doc) == []


def test_chunk_segments_splits_at_boundaries():
    segments = ["a" * 30, "b" * 30, "c" * 30]
    chunks = chunk_segments(segments, max_chars=70)
    assert chunks == ["a" * 30 + " " + "b" * 30, "c" * 30]


def test_chunk_segments_skips_empty_and_never_splits_a_segment():
    segments = ["", "  ", "x" * 100]
    assert chunk_segments(segments, max_chars=10) == ["x" * 100]


def test_sentences_from_payload_roundtrip():
    payload = {
        "sentences": [
            {
                "text": "Der Hund bellt.",
                "tokens": [
                    {
                        "text": "Hund",
                        "lemma": "Hund",
                        "pos": "NOUN",
                        "ent_type": "",
                        "is_alpha": True,
                        "is_stop": False,
                    }
                ],
            }
        ]
    }
    sentences = sentences_from_payload(payload)
    assert sentences[0].text == "Der Hund bellt."
    assert sentences[0].tokens[0] == TokenRecord("Hund", "Hund", "NOUN", "", True, False)


def test_sentences_from_payload_rejects_a_payload_without_sentences():
    with pytest.raises(ValueError) as excinfo:
        sentences_from_payload({"schema_version": 1}, "work/analysis.json")
    message = str(excinfo.value)
    assert "work/analysis.json" in message
    assert "'sentences' list" in message


def test_sentences_from_payload_reports_a_malformed_record():
    payload = {"sentences": [{"text": "Der Hund bellt."}]}  # no "tokens"
    with pytest.raises(ValueError) as excinfo:
        sentences_from_payload(payload, "work/analysis.json")
    assert "malformed sentence record" in str(excinfo.value)
    assert "--force" in str(excinfo.value)


def test_real_german_pipeline_smoke():
    spacy = pytest.importorskip("spacy")
    if not spacy.util.is_package("de_core_news_lg"):
        pytest.skip("de_core_news_lg not installed")
    nlp = spacy.load("de_core_news_lg")
    doc = nlp("Der Hund läuft schnell. Anna wohnt in Berlin.")
    sentences = doc_to_sentences(doc)
    assert len(sentences) == 2
    lemmas = {t.lemma for s in sentences for t in s.tokens}
    assert "laufen" in lemmas
    berlin = [t for s in sentences for t in s.tokens if t.text == "Berlin"]
    assert berlin and berlin[0].ent_type == "LOC"
