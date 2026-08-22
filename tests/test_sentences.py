import pytest

from transcript.sentences import (
    join_segments,
    merge_ordinal_splits,
    sentences_from_payload,
    split_text,
)


def test_join_segments_strips_and_skips_empty():
    assert join_segments(["Der Hund ", "", "  ", "bellt laut."]) == "Der Hund bellt laut."


def test_split_text_basic_german():
    assert split_text("Der Hund läuft schnell. Anna wohnt in Berlin.") == [
        "Der Hund läuft schnell.",
        "Anna wohnt in Berlin.",
    ]


def test_split_text_keeps_abbreviations_together():
    assert split_text("Das ist z. B. ein Hund. Dr. Müller kommt heute.") == [
        "Das ist z. B. ein Hund.",
        "Dr. Müller kommt heute.",
    ]


def test_split_text_survives_ordinals_before_nouns():
    # syntok splits after "2." / "24."; merge_ordinal_splits folds it back.
    assert split_text("Er wohnt im 2. Stock. Das Haus ist alt.") == [
        "Er wohnt im 2. Stock.",
        "Das Haus ist alt.",
    ]
    assert split_text("Heute ist der 24. Dezember. Wir feiern.") == [
        "Heute ist der 24. Dezember.",
        "Wir feiern.",
    ]


def test_split_text_years_stay_sentence_ends():
    assert split_text("Das war 1998. Dann kam er.") == ["Das war 1998.", "Dann kam er."]


def test_split_text_normalizes_whitespace():
    assert split_text("Der  Hund\n bellt.  Er  schläft.") == [
        "Der Hund bellt.",
        "Er schläft.",
    ]


def test_merge_ordinal_splits_only_merges_short_numbers():
    assert merge_ordinal_splits(["Am 3.", "Oktober."]) == ["Am 3. Oktober."]
    assert merge_ordinal_splits(["Das war 1998.", "Dann kam er."]) == [
        "Das war 1998.",
        "Dann kam er.",
    ]
    assert merge_ordinal_splits([]) == []


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


def test_split_sentences_stage_writes_the_artifact(tmp_path):
    from transcript.artifacts import save_json
    from transcript.sentences import split_sentences

    transcript = tmp_path / "transcript.json"
    save_json(
        transcript,
        {
            "schema_version": 1,
            "segments": [
                {"text": "Der Hund läuft"},
                {"text": "schnell. Er bellt."},
            ],
        },
    )
    out = tmp_path / "sentences.json"
    payload = split_sentences(transcript, out)
    assert payload["segmenter"] == "syntok"
    assert payload["total_sentences"] == 2
    # the mid-sentence segment boundary does not split the sentence
    assert payload["sentences"] == ["Der Hund läuft schnell.", "Er bellt."]
    assert out.exists()
