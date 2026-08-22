import json

from transcript.aggregate import build_word_list, write_output
from transcript.known_words import KnownWords

NO_KNOWN = KnownWords()


def test_counts_and_example_sentences(make_sentence, make_word):
    s1 = make_sentence("Der Hund läuft schnell.", make_word(lemma="laufen", word_type="verb"))
    s2 = make_sentence("Wir laufen zum Bahnhof.", make_word(lemma="laufen", word_type="verb"))
    entries = build_word_list([s1, s2], NO_KNOWN)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.lemma == "laufen"
    assert entry.word_type == "verb"
    assert entry.count == 2
    assert entry.sentences == ["Der Hund läuft schnell.", "Wir laufen zum Bahnhof."]


def test_sentence_count_multiplies_word_count(make_sentence, make_word):
    # The LLM saw the distinct sentence once, but the film contains it 3x.
    sent = make_sentence(
        "Er fängt an.", make_word(lemma="anfangen", word_type="verb"), count=3
    )
    entries = build_word_list([sent], NO_KNOWN)
    assert entries[0].count == 3
    assert entries[0].sentences == ["Er fängt an."]


def test_homonyms_split_by_word_type(make_sentence, make_word):
    sent = make_sentence(
        "Das Laufen macht Spaß, wir laufen gern.",
        make_word(lemma="Laufen", word_type="noun", article="das"),
        make_word(lemma="laufen", word_type="verb"),
    )
    entries = build_word_list([sent], NO_KNOWN)
    assert {(e.lemma, e.word_type) for e in entries} == {
        ("Laufen", "noun"),
        ("laufen", "verb"),
    }


def test_first_seen_article_wins_and_gaps_are_filled(make_sentence, make_word):
    s1 = make_sentence("Hunde bellen.", make_word(lemma="Hund", word_type="noun"))
    s2 = make_sentence("Der Hund schläft.", make_word(lemma="Hund", word_type="noun", article="der"))
    entries = build_word_list([s1, s2], NO_KNOWN)
    assert entries[0].article == "der"


def test_known_words_are_excluded(make_sentence, make_word):
    sent = make_sentence(
        "Der Hund bellt.",
        make_word(lemma="Hund", word_type="noun", article="der"),
        make_word(lemma="bellen", word_type="verb"),
    )
    entries = build_word_list([sent], KnownWords(["der Hund"]))
    assert [e.lemma for e in entries] == ["bellen"]


def test_min_count_filter(make_sentence, make_word):
    s1 = make_sentence(
        "Der Hund bellt.",
        make_word(lemma="Hund", word_type="noun"),
        make_word(lemma="bellen", word_type="verb"),
    )
    s2 = make_sentence("Der Hund schläft.", make_word(lemma="Hund", word_type="noun"))
    entries = build_word_list([s1, s2], NO_KNOWN, min_count=2)
    assert [e.lemma for e in entries] == ["Hund"]


def test_sorted_by_count_then_lemma(make_sentence, make_word):
    s1 = make_sentence(
        "Katze und Affe.",
        make_word(lemma="Katze", word_type="noun"),
        make_word(lemma="Affe", word_type="noun"),
        make_word(lemma="Zebra", word_type="noun"),
    )
    s2 = make_sentence("Die Katze schläft.", make_word(lemma="Katze", word_type="noun"))
    entries = build_word_list([s1, s2], NO_KNOWN)
    assert [e.lemma for e in entries] == ["Katze", "Affe", "Zebra"]


def test_output_json_roundtrip_keeps_umlauts(tmp_path, make_sentence, make_word):
    sent = make_sentence(
        "Das Mädchen lächelt.", make_word(lemma="lächeln", word_type="verb")
    )
    entries = build_word_list([sent], NO_KNOWN)
    out = tmp_path / "words.json"
    payload = write_output(
        entries,
        out,
        known_words_source="https://api.example.com/words",
        llm_model="test-model",
        min_count=1,
    )

    raw = out.read_text(encoding="utf-8")
    assert "lächeln" in raw  # ensure_ascii=False: umlauts stay literal

    loaded = json.loads(raw)
    assert loaded["schema_version"] == 1
    assert loaded["total_words"] == 1
    assert loaded["llm_model"] == "test-model"
    assert loaded["words"][0] == {
        "lemma": "lächeln",
        "word_type": "verb",
        "article": None,
        "count": 1,
        "sentences": ["Das Mädchen lächelt."],
    }
    assert payload["min_count"] == 1
