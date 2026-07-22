"""Stage 4: fetch the user's known words from a REST API and match lemmas
against them.

The endpoint and its Bearer token come from the environment (typically a
.env file, loaded at CLI startup):

    KNOWN_WORDS_API_URL=https://example.com/api/known-words
    KNOWN_WORDS_API_TOKEN=your-secret-token

The GET endpoint may return any of:
  - a JSON array of strings:            ["laufen", "Haus"]
  - an object with a "words" array:     {"words": ["laufen", ...]}
  - an array of objects with a "word" or "lemma" field:
                                        [{"word": "laufen"}, ...]

Matching uses str.casefold(), not lower(): casefold maps German ß -> ss, so
an entry "Strasse" matches the lemma "Straße". This conflates rare pairs
like Maße/Masse, which is an acceptable trade-off for a vocabulary filter.
"""

from __future__ import annotations

URL_ENV_VAR = "KNOWN_WORDS_API_URL"
TOKEN_ENV_VAR = "KNOWN_WORDS_API_TOKEN"

REQUEST_TIMEOUT_S = 30.0


def normalize_word(word: str) -> str:
    return word.strip().casefold()


def parse_known_words_response(data) -> frozenset[str]:
    """Normalize the supported response shapes into a set of lemmas."""
    if isinstance(data, dict):
        data = data.get("words")
        if data is None:
            raise ValueError(
                'unexpected known-words response: object without a "words" array'
            )
    if not isinstance(data, list):
        raise ValueError(f"unexpected known-words response type: {type(data).__name__}")

    words = set()
    for item in data:
        if isinstance(item, str):
            word = item
        elif isinstance(item, dict):
            word = item.get("word") or item.get("lemma")
            if word is None:
                raise ValueError(
                    f'known-words entry without a "word" or "lemma" field: {item!r}'
                )
        else:
            raise ValueError(f"unexpected known-words entry: {item!r}")
        word = normalize_word(word)
        if word:
            words.add(word)
    return frozenset(words)


def fetch_known_words(url: str, token: str | None, timeout: float = REQUEST_TIMEOUT_S) -> frozenset[str]:
    """GET the known-words list, authenticating with a Bearer token."""
    import httpx

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = httpx.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return parse_known_words_response(response.json())


def is_known(lemma: str, known: frozenset[str]) -> bool:
    return normalize_word(lemma) in known
