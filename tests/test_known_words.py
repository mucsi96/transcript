import sys
from types import SimpleNamespace

import pytest

from transcript.known_words import (
    fetch_known_words,
    is_known,
    normalize_word,
    parse_known_words_response,
)


def test_parse_array_of_strings():
    assert parse_known_words_response(["laufen", " Haus ", ""]) == {"laufen", "haus"}


def test_parse_object_with_words_array():
    assert parse_known_words_response({"words": ["laufen"]}) == {"laufen"}


def test_parse_array_of_objects_word_or_lemma():
    data = [{"word": "laufen"}, {"lemma": "Haus"}]
    assert parse_known_words_response(data) == {"laufen", "haus"}


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


class FakeResponse:
    def __init__(self, payload, status_error=None):
        self._payload = payload
        self._status_error = status_error

    def raise_for_status(self):
        if self._status_error:
            raise self._status_error

    def json(self):
        return self._payload


def test_fetch_sends_bearer_token(monkeypatch):
    calls = {}

    def fake_get(url, headers=None, timeout=None):
        calls["url"] = url
        calls["headers"] = headers
        return FakeResponse(["laufen"])

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(get=fake_get))
    known = fetch_known_words("https://api.example.com/words", "sekrit")
    assert known == {"laufen"}
    assert calls["url"] == "https://api.example.com/words"
    assert calls["headers"] == {"Authorization": "Bearer sekrit"}


def test_fetch_without_token_sends_no_auth_header(monkeypatch):
    calls = {}

    def fake_get(url, headers=None, timeout=None):
        calls["headers"] = headers
        return FakeResponse([])

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(get=fake_get))
    fetch_known_words("https://api.example.com/words", None)
    assert calls["headers"] == {}


def test_fetch_raises_on_http_error(monkeypatch):
    def fake_get(url, headers=None, timeout=None):
        return FakeResponse(None, status_error=RuntimeError("401 Unauthorized"))

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(get=fake_get))
    with pytest.raises(RuntimeError, match="401"):
        fetch_known_words("https://api.example.com/words", "bad-token")
