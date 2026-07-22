import pytest

from transcript.analyze import SentenceRecord, TokenRecord


@pytest.fixture
def make_token():
    def _make(
        text="Hund",
        lemma="Hund",
        pos="NOUN",
        ent_type="",
        is_alpha=True,
        is_stop=False,
    ):
        return TokenRecord(
            text=text,
            lemma=lemma,
            pos=pos,
            ent_type=ent_type,
            is_alpha=is_alpha,
            is_stop=is_stop,
        )

    return _make


@pytest.fixture
def make_sentence(make_token):
    def _make(text, *tokens):
        return SentenceRecord(text=text, tokens=tuple(tokens))

    return _make
