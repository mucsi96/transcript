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

The entries are flash-card headwords, not bare lemmas, so they arrive in the
notation a vocabulary list uses: "der Kaffee" for a noun with its gender,
"ausruhen (sich)" or "sich kümmern" for a reflexive verb, "gern(e)" for an
optional ending, "auf jeden/keinen Fall" for alternatives, "all-" for an
inflecting stem. spaCy lemmatizes the text to "Kaffee", "ausruhen", "gern" —
so each entry is expanded into every form a lemma could take (see
match_keys), and lemmas are compared against those keys. Comparing against
the raw entry is what used to let a known "der Kaffee" show up in words.json
as a word to learn.

Matching uses str.casefold(), not lower(): casefold maps German ß -> ss, so
an entry "Strasse" matches the lemma "Straße". This conflates rare pairs
like Maße/Masse, which is an acceptable trade-off for a vocabulary filter.

What came back is saved to the work directory as known-words.json — both the
entries as fetched and the keys they expand to — so the list the build was
filtered against stays inspectable afterwards. It is the first thing to look
at when a word you thought you knew shows up in words.json (grep it in
match_keys: if it is missing, the entry's notation is not one the expansion
covers), or when the API silently starts answering with fewer words.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterable
from pathlib import Path

from .artifacts import SCHEMA_VERSION, save_json, utc_timestamp

URL_ENV_VAR = "KNOWN_WORDS_API_URL"
TOKEN_ENV_VAR = "KNOWN_WORDS_API_TOKEN"

# Written next to the words.json of the run that fetched it.
ARTIFACT_NAME = "known-words.json"

REQUEST_TIMEOUT_S = 30.0

# How much of an unexpected response body to quote back in an error message.
BODY_SNIPPET_CHARS = 200


class KnownWordsError(RuntimeError):
    """The known-words endpoint could not be reached, or answered with
    something that is not a word list.

    Its message is written to be shown to the user as-is: what was asked,
    what came back, and which setting to check.
    """


# Stripped from the front of a card so "der Kaffee" matches the lemma
# "Kaffee". Cards spell nouns in the nominative, which is the only case these
# forms need to cover; "den/dem/des" would also be the lemma of a determiner
# token, and are left alone.
ARTICLES = frozenset({"der", "die", "das", "ein", "eine"})

# Cards mark reflexive verbs as "ausruhen (sich)" or "sich kümmern"; spaCy
# lemmatizes the verb alone.
REFLEXIVE_PRONOUN = "sich"

# " (sich)", " (Pl)", "(e)" — a card's parenthetical aside.
_PARENTHESIZED = re.compile(r"\s*\([^)]*\)")

# Trailing card notation: the hyphen of an inflecting stem ("all-", "nächst-")
# and the period of an abbreviation ("ca.").
_TRAILING_NOTATION = "-."


def normalize_word(word: str) -> str:
    """Casefold and collapse whitespace, the form every comparison uses."""
    return " ".join(word.split()).casefold()


def _paren_variants(piece: str) -> set[str]:
    """Both readings of a parenthetical: dropped and kept.

    "gern(e)" is gern and gerne; "ausruhen (sich)" is ausruhen and the
    (never-lemmatized, but harmless) "ausruhen sich".
    """
    return {_PARENTHESIZED.sub("", piece), piece.replace("(", "").replace(")", "")}


def _slash_variants(phrase: str) -> set[str]:
    """One phrase per combination of slash alternatives.

    "auf jeden/keinen Fall" is two phrases; "ihr/ihm/ihn" is three words.
    """
    alternatives = [token.split("/") for token in phrase.split()]
    return {" ".join(combo) for combo in itertools.product(*alternatives)}


def _headword_forms(phrase: str) -> set[str]:
    """The lemma forms one fully expanded phrase stands for: as written, with
    the card's trailing notation removed, and without a leading article or a
    reflexive pronoun."""
    tokens = phrase.split()
    if not tokens:
        return set()

    forms = {" ".join(tokens)}
    forms.add(" ".join(token.rstrip(_TRAILING_NOTATION) or token for token in tokens))

    for form in list(forms):
        written = form.split()
        content = [token for token in written if token != REFLEXIVE_PRONOUN]
        if len(content) > 1 and content[0] in ARTICLES:
            content = content[1:]
        if content and content != written:
            forms.add(" ".join(content))
    return forms


def match_keys(entry: str) -> frozenset[str]:
    """Every lemma form a card entry covers.

        "der Kaffee"            -> {"der kaffee", "kaffee"}
        "ausruhen (sich)"       -> {"ausruhen", "ausruhen sich"}
        "auf jeden/keinen Fall" -> {"auf jeden fall", "auf keinen fall"}
        "all-"                  -> {"all-", "all"}

    The entry itself is always among the keys, so an endpoint that already
    sends bare lemmas keeps matching exactly as before. Extra keys can only
    add matches, never remove one, and a multi-word key simply never matches
    a single-token lemma.

    Keys carry no part of speech, so a card's gender is not used to restrict
    what it matches: "der Laden" also covers the verb lemma "laden". Same
    trade-off as the ß/ss conflation — rare, and it costs a card that the
    learner most likely has anyway.
    """
    keys: set[str] = set()
    # A card may list several forms at once: "der, die, das".
    for piece in normalize_word(entry).split(","):
        for variant in _paren_variants(piece):
            for phrase in _slash_variants(variant):
                keys.update(_headword_forms(phrase))
    keys.discard("")
    return frozenset(keys)


class KnownWords:
    """The fetched card entries plus the lemma forms they match.

    Both are kept: the entries are what the endpoint answered and what the
    user recognizes, the keys are what lemmas are actually compared against.
    len() counts entries, so "Fetched N known words" means N cards.
    """

    __slots__ = ("entries", "keys")

    def __init__(self, entries: Iterable[str] = ()) -> None:
        normalized = {word for word in map(normalize_word, entries) if word}
        self.entries: tuple[str, ...] = tuple(sorted(normalized))
        self.keys: frozenset[str] = frozenset(
            key for entry in self.entries for key in match_keys(entry)
        )

    def matches(self, lemma: str) -> bool:
        return normalize_word(lemma) in self.keys

    def __len__(self) -> int:
        return len(self.entries)

    def __eq__(self, other) -> bool:
        if isinstance(other, KnownWords):
            return self.entries == other.entries
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.entries)

    def __repr__(self) -> str:
        return f"KnownWords({list(self.entries)!r})"


def parse_known_words_response(data) -> KnownWords:
    """Normalize the supported response shapes into a KnownWords."""
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
        words.add(word)
    return KnownWords(words)


def _snippet(text: str) -> str:
    """A short, single-line excerpt of a response body for error messages."""
    text = " ".join((text or "").split())
    if not text:
        return "<empty body>"
    if len(text) > BODY_SNIPPET_CHARS:
        text = text[:BODY_SNIPPET_CHARS] + " ..."
    return text


def _status_hint(status: int) -> str:
    """The setting most likely at fault for an HTTP status."""
    if status in (401, 403):
        return f" Check {TOKEN_ENV_VAR}: the token may be missing, wrong or expired."
    if status == 404:
        return f" Check that {URL_ENV_VAR} is the full path of the known-words endpoint."
    if status >= 500:
        return " The server itself failed; try again later."
    return ""


def fetch_known_words(url: str, token: str | None, timeout: float = REQUEST_TIMEOUT_S) -> KnownWords:
    """GET the known-words list, authenticating with a Bearer token.

    Every failure — unreachable host, HTTP error status, a body that is not
    JSON, JSON of an unexpected shape — is reported as a KnownWordsError
    naming the URL, what came back, and what to check. A misconfigured URL
    typically lands on a web page that answers 200 with HTML, so the
    not-JSON case is a configuration error, not a server fault.
    """
    import httpx

    # Looked up rather than referenced directly so the tests can inject a
    # stub httpx module; `except ()` catches nothing, which is correct there.
    request_error = getattr(httpx, "RequestError", ())
    status_error = getattr(httpx, "HTTPStatusError", ())

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        response = httpx.get(url, headers=headers, timeout=timeout)
    except request_error as exc:
        raise KnownWordsError(
            f"cannot reach the known-words API at {url} "
            f"({type(exc).__name__}: {exc}). Check {URL_ENV_VAR} and your "
            f"network connection."
        ) from exc

    try:
        response.raise_for_status()
    except status_error as exc:
        status = exc.response.status_code
        raise KnownWordsError(
            f"the known-words API at {url} returned HTTP {status}."
            f"{_status_hint(status)} Response: {_snippet(exc.response.text)}"
        ) from exc

    try:
        data = response.json()
    except ValueError as exc:  # json.JSONDecodeError
        content_type = response.headers.get("content-type", "").split(";")[0].strip()
        described = f" but {content_type}" if content_type else ""
        raise KnownWordsError(
            f"the known-words API at {url} did not return JSON{described}. "
            f"{URL_ENV_VAR} may point at a web page instead of the API "
            f"endpoint. Response: {_snippet(response.text)}"
        ) from exc

    try:
        return parse_known_words_response(data)
    except ValueError as exc:
        raise KnownWordsError(f"{url}: {exc}") from exc


def save_known_words(path: Path, known: KnownWords, *, source: str) -> dict:
    """Save the fetched list as a work-directory artifact.

    Both halves are written, in normalized (casefolded) form: "words" is the
    card entries as the API sent them, "match_keys" is what the lemmas are
    actually compared against. A lemma missing from match_keys is a word that
    will not be filtered, whatever entry the endpoint shows for it.
    """
    payload = {
        "schema_version": SCHEMA_VERSION,
        "fetched_at": utc_timestamp(),
        "source": source,
        "total_words": len(known.entries),
        "words": list(known.entries),
        "total_match_keys": len(known.keys),
        "match_keys": sorted(known.keys),
    }
    save_json(path, payload)
    return payload


def is_known(lemma: str, known: KnownWords) -> bool:
    return known.matches(lemma)
