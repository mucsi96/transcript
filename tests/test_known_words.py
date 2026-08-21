import json
import sys
from types import SimpleNamespace

import pytest

from transcript.known_words import (
    ARTIFACT_NAME,
    BODY_SNIPPET_CHARS,
    TOKEN_ENV_VAR,
    URL_ENV_VAR,
    KnownWords,
    KnownWordsError,
    _snippet,
    fetch_known_words,
    is_known,
    match_keys,
    normalize_word,
    parse_known_words_response,
    save_known_words,
)


def entries_of(data):
    return parse_known_words_response(data).entries


def test_parse_array_of_strings():
    assert entries_of(["laufen", " Haus ", ""]) == ("haus", "laufen")


def test_parse_object_with_words_array():
    assert entries_of({"words": ["laufen"]}) == ("laufen",)


def test_parse_array_of_objects_word_or_lemma():
    data = [{"word": "laufen"}, {"lemma": "Haus"}]
    assert entries_of(data) == ("haus", "laufen")


def test_parse_rejects_unknown_shapes():
    with pytest.raises(ValueError):
        parse_known_words_response("laufen")
    with pytest.raises(ValueError):
        parse_known_words_response({"items": ["laufen"]})
    with pytest.raises(ValueError):
        parse_known_words_response([{"name": "laufen"}])
    with pytest.raises(ValueError):
        parse_known_words_response([42])


def test_matching_is_case_insensitive():
    known = parse_known_words_response(["haus"])
    assert is_known("Haus", known)
    assert is_known("HAUS", known)


def test_umlauts_are_not_conflated():
    known = parse_known_words_response(["Baume"])
    assert not is_known("Bäume", known)


def test_eszett_matches_ss_spelling():
    # casefold maps ß -> ss, so "Strasse" from the API matches "Straße".
    known = parse_known_words_response(["Strasse"])
    assert is_known("Straße", known)


def test_normalize_word():
    assert normalize_word("  Straße \n") == "strasse"
    assert normalize_word("der  Kaffee") == "der kaffee"


def test_a_noun_card_matches_the_bare_lemma():
    """The bug this expansion exists for: the endpoint sends flash cards, so
    a known "der Kaffee" was compared against the lemma "Kaffee" and lost."""
    known = parse_known_words_response(["der Kaffee"])
    assert is_known("Kaffee", known)
    assert is_known("kaffee", known)
    # the entry itself stays a key, and stays what the user sees
    assert known.entries == ("der kaffee",)
    assert "der kaffee" in known.keys


def test_articles_are_only_stripped_from_a_noun_phrase():
    known = parse_known_words_response(["der", "die Frau", "das Auto", "ein bisschen"])
    assert is_known("Frau", known)
    assert is_known("Auto", known)
    assert is_known("bisschen", known)
    assert is_known("der", known)  # a one-word card is left alone


def test_reflexive_marking_matches_the_verb():
    known = parse_known_words_response(["ausruhen (sich)", "sich kümmern"])
    assert is_known("ausruhen", known)
    assert is_known("kümmern", known)


def test_optional_ending_matches_both_spellings():
    known = parse_known_words_response(["gern(e)"])
    assert is_known("gern", known)
    assert is_known("gerne", known)


def test_slash_alternatives_are_expanded():
    known = parse_known_words_response(["ihr/ihm/ihn", "circa/ca."])
    assert is_known("ihm", known)
    assert is_known("ihn", known)
    assert is_known("circa", known)
    assert is_known("ca.", known)


def test_inflecting_stems_match_without_the_hyphen():
    known = parse_known_words_response(["nächst-", "lieblings-"])
    assert is_known("nächst", known)
    assert is_known("lieblings", known)


def test_a_card_listing_several_forms():
    known = parse_known_words_response(["der, die, das"])
    assert known.keys == {"der", "die", "das"}


def test_multi_word_cards_keep_their_phrase_and_do_not_leak_parts():
    """"Fall" is not known just because "auf jeden Fall" is."""
    known = parse_known_words_response(["auf jeden/keinen Fall", "Rad fahren"])
    assert known.keys == {"auf jeden fall", "auf keinen fall", "rad fahren"}
    assert not is_known("Fall", known)
    assert not is_known("fahren", known)


def test_match_keys_of_a_plain_lemma_is_just_itself():
    assert match_keys("laufen") == {"laufen"}


def test_empty_known_words_matches_nothing():
    known = KnownWords()
    assert len(known) == 0
    assert not is_known("Kaffee", known)


class FakeResponse:
    """Stands in for httpx.Response: `text` is the wire body and `json()`
    decodes it the way httpx does, so a non-JSON body fails here exactly as
    it does in production."""

    def __init__(self, text="", *, status_code=200, content_type="application/json"):
        self.text = text
        self.status_code = status_code
        self.headers = {"content-type": content_type}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise FakeStatusError(f"HTTP {self.status_code}", response=self)

    def json(self):
        return json.loads(self.text)


class FakeStatusError(Exception):
    def __init__(self, message, response):
        super().__init__(message)
        self.response = response


class FakeRequestError(Exception):
    pass


def fake_httpx(get):
    """A stub httpx module carrying the exception types fetch_known_words
    looks up."""
    return SimpleNamespace(
        get=get, RequestError=FakeRequestError, HTTPStatusError=FakeStatusError
    )


def json_response(payload, **kwargs):
    return FakeResponse(json.dumps(payload), **kwargs)


def test_fetch_sends_bearer_token(monkeypatch):
    calls = {}

    def fake_get(url, headers=None, timeout=None):
        calls["url"] = url
        calls["headers"] = headers
        return json_response(["laufen"])

    monkeypatch.setitem(sys.modules, "httpx", fake_httpx(fake_get))
    known = fetch_known_words("https://api.example.com/words", "sekrit")
    assert known.entries == ("laufen",)
    assert calls["url"] == "https://api.example.com/words"
    assert calls["headers"] == {"Authorization": "Bearer sekrit"}


def test_fetch_without_token_sends_no_auth_header(monkeypatch):
    calls = {}

    def fake_get(url, headers=None, timeout=None):
        calls["headers"] = headers
        return json_response([])

    monkeypatch.setitem(sys.modules, "httpx", fake_httpx(fake_get))
    fetch_known_words("https://api.example.com/words", None)
    assert calls["headers"] == {}


def test_fetch_reports_unauthorized_with_the_token_setting(monkeypatch):
    def fake_get(url, headers=None, timeout=None):
        return FakeResponse("Unauthorized", status_code=401, content_type="text/plain")

    monkeypatch.setitem(sys.modules, "httpx", fake_httpx(fake_get))
    with pytest.raises(KnownWordsError) as excinfo:
        fetch_known_words("https://api.example.com/words", "bad-token")
    message = str(excinfo.value)
    assert "HTTP 401" in message
    assert "https://api.example.com/words" in message
    assert "KNOWN_WORDS_API_TOKEN" in message


def test_fetch_reports_a_missing_endpoint(monkeypatch):
    def fake_get(url, headers=None, timeout=None):
        return FakeResponse("Not Found", status_code=404, content_type="text/plain")

    monkeypatch.setitem(sys.modules, "httpx", fake_httpx(fake_get))
    with pytest.raises(KnownWordsError, match="KNOWN_WORDS_API_URL"):
        fetch_known_words("https://api.example.com/words", "sekrit")


def test_fetch_reports_an_html_page_rather_than_a_json_error(monkeypatch):
    """The real-world misconfiguration: the URL is a site root, so it
    answers 200 with the app's index page."""

    def fake_get(url, headers=None, timeout=None):
        return FakeResponse(
            "<!DOCTYPE html>\n<html><head><title>Learn language</title></head></html>",
            content_type="text/html; charset=utf-8",
        )

    monkeypatch.setitem(sys.modules, "httpx", fake_httpx(fake_get))
    with pytest.raises(KnownWordsError) as excinfo:
        fetch_known_words("https://language.example.com", "sekrit")
    message = str(excinfo.value)
    assert "did not return JSON but text/html" in message
    assert "KNOWN_WORDS_API_URL" in message
    assert "<!DOCTYPE html>" in message  # a snippet of what came back


def test_fetch_reports_an_empty_body(monkeypatch):
    def fake_get(url, headers=None, timeout=None):
        return FakeResponse("")

    monkeypatch.setitem(sys.modules, "httpx", fake_httpx(fake_get))
    with pytest.raises(KnownWordsError, match="<empty body>"):
        fetch_known_words("https://api.example.com/words", "sekrit")


def test_fetch_reports_an_unreachable_host(monkeypatch):
    def fake_get(url, headers=None, timeout=None):
        raise FakeRequestError("[Errno -2] Name or service not known")

    monkeypatch.setitem(sys.modules, "httpx", fake_httpx(fake_get))
    with pytest.raises(KnownWordsError) as excinfo:
        fetch_known_words("https://nope.example.com/words", "sekrit")
    assert "cannot reach" in str(excinfo.value)
    assert "Name or service not known" in str(excinfo.value)


def test_fetch_reports_a_wrong_json_shape_with_the_url(monkeypatch):
    def fake_get(url, headers=None, timeout=None):
        return json_response({"items": ["laufen"]})

    monkeypatch.setitem(sys.modules, "httpx", fake_httpx(fake_get))
    with pytest.raises(KnownWordsError) as excinfo:
        fetch_known_words("https://api.example.com/words", "sekrit")
    assert "https://api.example.com/words" in str(excinfo.value)
    assert '"words" array' in str(excinfo.value)


def test_saved_artifact_lists_the_normalized_words_sorted(tmp_path):
    out = tmp_path / "known-words.json"
    payload = save_known_words(
        out, KnownWords(["laufen", "der Baum", "Straße"]),
        source="https://api.example.com/words",
    )
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved == payload
    assert saved["schema_version"] == 1
    assert saved["source"] == "https://api.example.com/words"
    assert saved["total_words"] == 3
    assert saved["words"] == ["der baum", "laufen", "strasse"]
    # what the lemmas are compared against, the entries included
    assert saved["match_keys"] == ["baum", "der baum", "laufen", "strasse"]
    assert saved["total_match_keys"] == 4
    assert saved["fetched_at"].endswith("Z")


def test_build_saves_the_fetched_words_next_to_the_word_list(tmp_path, monkeypatch):
    from transcript.cli import _cmd_build, build_parser

    analysis = tmp_path / "analysis.json"
    analysis.write_text(
        json.dumps({
            "schema_version": 1,
            "sentences": [{
                "text": "Der Hund bellt.",
                "tokens": [
                    {"text": "Hund", "lemma": "Hund", "pos": "NOUN",
                     "ent_type": "", "is_alpha": True, "is_stop": False},
                    {"text": "bellt", "lemma": "bellen", "pos": "VERB",
                     "ent_type": "", "is_alpha": True, "is_stop": False},
                ],
            }],
        }),
        encoding="utf-8",
    )

    def fake_get(url, headers=None, timeout=None):
        return json_response(["der Hund", "Straße"])

    monkeypatch.setitem(sys.modules, "httpx", fake_httpx(fake_get))
    monkeypatch.setenv(URL_ENV_VAR, "https://api.example.com/words")
    monkeypatch.setenv(TOKEN_ENV_VAR, "sekrit")

    words_json = tmp_path / "out" / "words.json"
    args = build_parser().parse_args(["build", str(analysis), "-o", str(words_json)])
    _cmd_build(args, analysis, words_json)

    saved = json.loads((words_json.parent / ARTIFACT_NAME).read_text(encoding="utf-8"))
    assert saved["words"] == ["der hund", "strasse"]
    # the artifact is the list the words were actually filtered against
    assert saved["match_keys"] == ["der hund", "hund", "strasse"]
    # "der Hund" is a known card, so only the verb is left to learn
    assert [w["lemma"] for w in json.loads(words_json.read_text(encoding="utf-8"))["words"]] == ["bellen"]


def test_build_writes_the_artifact_where_asked(tmp_path, monkeypatch):
    from transcript.cli import _cmd_build, build_parser

    analysis = tmp_path / "analysis.json"
    analysis.write_text(
        json.dumps({"schema_version": 1, "sentences": []}), encoding="utf-8"
    )
    monkeypatch.setitem(
        sys.modules, "httpx",
        fake_httpx(lambda url, headers=None, timeout=None: json_response(["Haus"])),
    )
    monkeypatch.setenv(URL_ENV_VAR, "https://api.example.com/words")
    monkeypatch.setenv(TOKEN_ENV_VAR, "sekrit")

    elsewhere = tmp_path / "reference" / "fetched.json"
    words_json = tmp_path / "words.json"
    args = build_parser().parse_args([
        "build", str(analysis), "-o", str(words_json),
        "--known-words-output", str(elsewhere),
    ])
    _cmd_build(args, analysis, words_json)

    assert json.loads(elsewhere.read_text(encoding="utf-8"))["words"] == ["haus"]
    assert not (tmp_path / ARTIFACT_NAME).exists()


def test_build_without_an_endpoint_writes_no_artifact(tmp_path, monkeypatch):
    from transcript.cli import _cmd_build, build_parser

    analysis = tmp_path / "analysis.json"
    analysis.write_text(
        json.dumps({"schema_version": 1, "sentences": []}), encoding="utf-8"
    )
    monkeypatch.delenv(URL_ENV_VAR, raising=False)

    words_json = tmp_path / "words.json"
    args = build_parser().parse_args(["build", str(analysis), "-o", str(words_json)])
    _cmd_build(args, analysis, words_json)

    assert not (tmp_path / ARTIFACT_NAME).exists()


def test_body_snippet_is_trimmed_to_one_line():
    assert _snippet("  a\n  b  ") == "a b"
    assert _snippet("") == "<empty body>"
    long_body = "x" * (BODY_SNIPPET_CHARS + 50)
    trimmed = _snippet(long_body)
    assert trimmed.endswith(" ...")
    assert len(trimmed) == BODY_SNIPPET_CHARS + len(" ...")
