import pytest

from transcript.llm import SentenceWords, WordRecord


@pytest.fixture(autouse=True)
def openai_model_env(monkeypatch):
    """The LLM stages read their model from OPENAI_MODEL; pin it so the
    tests are hermetic whatever the developer's shell has."""
    monkeypatch.setenv("OPENAI_MODEL", "test-model")


@pytest.fixture
def make_word():
    def _make(lemma="Hund", word_type="noun", article=None):
        return WordRecord(lemma=lemma, word_type=word_type, article=article)

    return _make


@pytest.fixture
def make_sentence(make_word):
    def _make(text, *words, count=1):
        return SentenceWords(text=text, count=count, words=tuple(words))

    return _make
