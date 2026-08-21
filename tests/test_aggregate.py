import json

from transcript.aggregate import build_word_list, write_output
from transcript.config import FilterConfig
from transcript.known_words import KnownWords

CFG = FilterConfig()
NO_KNOWN = KnownWords()


def test_counts_and_example_sentences(make_sentence, make_token):
    s1 = make_sentence(
        "Der Hund läuft schnell.",
        make_token(text="läuft", lemma="laufen", pos="VERB"),
    )
    s2 = make_sentence(
        "Wir laufen zum Bahnhof.",
        make_token(text="laufen", lemma="laufen", pos="VERB"),
    )
    entries = build_word_list([s1, s2], NO_KNOWN, CFG)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.lemma == "laufen"
    assert entry.word_type == "verb"
    assert entry.count == 2
    assert entry.sentences == ["Der Hund läuft schnell.", "Wir laufen zum Bahnhof."]


def test_homonyms_split_by_pos(make_sentence, make_token):
    sent = make_sentence(
        "Das Laufen macht Spaß, wir laufen gern.",
        make_token(text="Laufen", lemma="Laufen", pos="NOUN"),
        make_token(text="laufen", lemma="laufen", pos="VERB"),
    )
    entries = build_word_list([sent], NO_KNOWN, CFG)
    assert {(e.lemma, e.pos) for e in entries} == {("Laufen", "NOUN"), ("laufen", "VERB")}


def test_sentences_deduplicated_preserving_order(make_sentence, make_token):
    tok = make_token(text="Hund", lemma="Hund", pos="NOUN")
    s1 = make_sentence("Ein Hund und noch ein Hund.", tok, tok)
    s2 = make_sentence("Der Hund schläft.", tok)
    entries = build_word_list([s1, s2], NO_KNOWN, CFG)
    assert entries[0].count == 3
    assert entries[0].sentences == ["Ein Hund und noch ein Hund.", "Der Hund schläft."]


def test_known_words_are_excluded(make_sentence, make_token):
    sent = make_sentence(
        "Der Hund bellt.",
        make_token(text="Hund", lemma="Hund", pos="NOUN"),
        make_token(text="bellt", lemma="bellen", pos="VERB"),
    )
    entries = build_word_list([sent], KnownWords(["der Hund"]), CFG)
    assert [e.lemma for e in entries] == ["bellen"]


def test_min_count_filter(make_sentence, make_token):
    s1 = make_sentence(
        "Der Hund bellt.",
        make_token(text="Hund", lemma="Hund", pos="NOUN"),
        make_token(text="bellt", lemma="bellen", pos="VERB"),
    )
    s2 = make_sentence("Der Hund schläft.", make_token(text="Hund", lemma="Hund", pos="NOUN"))
    cfg = FilterConfig(min_count=2)
    entries = build_word_list([s1, s2], NO_KNOWN, cfg)
    assert [e.lemma for e in entries] == ["Hund"]


def test_sorted_by_count_then_lemma(make_sentence, make_token):
    sent = make_sentence(
        "Katze und Affe und Katze.",
        make_token(text="Katze", lemma="Katze", pos="NOUN"),
        make_token(text="Affe", lemma="Affe", pos="NOUN"),
        make_token(text="Katze", lemma="Katze", pos="NOUN"),
        make_token(text="Zebra", lemma="Zebra", pos="NOUN"),
    )
    entries = build_word_list([sent], NO_KNOWN, CFG)
    assert [e.lemma for e in entries] == ["Katze", "Affe", "Zebra"]


def test_filtered_tokens_do_not_create_entries(make_sentence, make_token):
    sent = make_sentence(
        "Herr Müller wohnt in Berlin.",
        make_token(text="Müller", lemma="Müller", pos="PROPN", ent_type="PER"),
        make_token(text="wohnt", lemma="wohnen", pos="VERB"),
        make_token(text="Berlin", lemma="Berlin", pos="PROPN", ent_type="LOC"),
    )
    entries = build_word_list([sent], NO_KNOWN, CFG)
    assert [e.lemma for e in entries] == ["wohnen"]


def test_output_json_roundtrip_keeps_umlauts(tmp_path, make_sentence, make_token):
    sent = make_sentence(
        "Das Mädchen lächelt.",
        make_token(text="lächelt", lemma="lächeln", pos="VERB"),
    )
    entries = build_word_list([sent], NO_KNOWN, CFG)
    out = tmp_path / "words.json"
    payload = write_output(
        entries, out, known_words_source="https://api.example.com/words", cfg=CFG
    )

    raw = out.read_text(encoding="utf-8")
    assert "lächeln" in raw  # ensure_ascii=False: umlauts stay literal

    loaded = json.loads(raw)
    assert loaded["schema_version"] == 1
    assert loaded["total_words"] == 1
    assert loaded["words"][0] == {
        "lemma": "lächeln",
        "pos": "VERB",
        "word_type": "verb",
        "count": 1,
        "sentences": ["Das Mädchen lächelt."],
    }
    assert payload["filter_config"]["min_count"] == 1
