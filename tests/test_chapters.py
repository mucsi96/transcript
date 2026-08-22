import json
from types import SimpleNamespace

import pytest

import transcript.llm as llm
from transcript.artifacts import save_json
from transcript.chapters import (
    ChapterExtractionError,
    chapter_answer_from_data,
    extract_chapter_sentences,
    load_chapter_checkpoint,
    split_html,
)
from transcript.llm import checkpoint_path


def completion(payload, finish_reason="stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(payload, ensure_ascii=False), refusal=None
                ),
                finish_reason=finish_reason,
            )
        ]
    )


class FakeClient:
    """Answers per spine href; records every user message it received."""

    def __init__(self, answers, fail_on=()):
        self.calls = []
        self._answers = answers
        self._fail_on = set(fail_on)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, *, model, messages, response_format):
        content = messages[1]["content"]
        self.calls.append(content)
        for href, answer in self._answers.items():
            if href in content:
                if href in self._fail_on:
                    raise RuntimeError(f"boom on {href}")
                return completion(answer)
        raise AssertionError(f"unexpected chapter request: {content[:80]}")


CONTENT = {
    "kind": "chapter",
    "is_content": True,
    "sentences": ["Der Hund bellt.", "Er fängt gerade an."],
}
TOC = {"kind": "toc", "is_content": False, "sentences": []}


def write_chapters(tmp_path, chapters):
    path = tmp_path / "chapters.json"
    save_json(path, {"schema_version": 1, "chapters": chapters})
    return path


def chapter(index, href, html="<p>Text</p>", title=""):
    return {"index": index, "href": href, "title": title, "html": html}


# ------------------------------------------------------------- validation


def test_chapter_answer_from_data_cleans_whitespace():
    kind, is_content, sentences = chapter_answer_from_data(
        {"kind": "chapter", "is_content": True, "sentences": [" Der  Hund. ", "  "]},
        "test",
    )
    assert (kind, is_content, sentences) == ("chapter", True, ("Der Hund.",))


def test_chapter_answer_from_data_rejects_bad_shapes():
    with pytest.raises(ValueError, match="without a kind"):
        chapter_answer_from_data({"is_content": True, "sentences": []}, "test")
    with pytest.raises(ValueError, match="is_content"):
        chapter_answer_from_data({"kind": "toc", "sentences": []}, "test")
    with pytest.raises(ValueError, match="string array"):
        chapter_answer_from_data(
            {"kind": "chapter", "is_content": True, "sentences": [1]}, "test"
        )


# -------------------------------------------------------------- splitting


def test_split_html_keeps_small_documents_whole():
    assert split_html("<p>klein</p>", max_chars=100) == ["<p>klein</p>"]


def test_split_html_splits_at_block_tags():
    html = "<html><body>" + "<p>" + "a" * 30 + "</p>" + "<p>" + "b" * 30 + "</p>"
    parts = split_html(html, max_chars=60)
    assert len(parts) > 1
    assert "".join(parts) == html
    assert parts[1].startswith("<p>")
    assert all(len(p) <= 60 for p in parts)


def test_split_html_hard_cuts_when_no_block_tags():
    html = "x" * 150
    parts = split_html(html, max_chars=60)
    assert "".join(parts) == html
    assert all(len(p) <= 60 for p in parts)


# ------------------------------------------------------------- end to end


def test_extract_chapter_sentences_skips_non_content(tmp_path, monkeypatch):
    chapters = write_chapters(
        tmp_path,
        [chapter(0, "toc.xhtml"), chapter(1, "kapitel1.xhtml", title="Kapitel 1")],
    )
    client = FakeClient({"toc.xhtml": TOC, "kapitel1.xhtml": CONTENT})
    monkeypatch.setattr(llm, "_make_client", lambda concurrency: client)

    out = tmp_path / "sentences.json"
    payload = extract_chapter_sentences(chapters, out, model="test-model")

    assert payload["segmenter"] == "llm"
    assert payload["llm_model"] == "test-model"
    assert payload["sentences"] == ["Der Hund bellt.", "Er fängt gerade an."]
    assert payload["total_sentences"] == 2
    assert payload["chapters"] == [
        {"index": 0, "href": "toc.xhtml", "title": "", "kind": "toc",
         "is_content": False, "sentences": 0},
        {"index": 1, "href": "kapitel1.xhtml", "title": "Kapitel 1",
         "kind": "chapter", "is_content": True, "sentences": 2},
    ]
    assert not checkpoint_path(out).exists()


def test_extract_chapter_sentences_resumes_from_checkpoint(tmp_path, monkeypatch):
    chapters = write_chapters(
        tmp_path, [chapter(0, "toc.xhtml"), chapter(1, "kapitel1.xhtml")]
    )
    out = tmp_path / "sentences.json"
    ckpt = checkpoint_path(out)
    ckpt.write_text(
        json.dumps({"index": 0, "href": "toc.xhtml", "title": "", **TOC}) + "\n",
        encoding="utf-8",
    )
    client = FakeClient({"kapitel1.xhtml": CONTENT})
    monkeypatch.setattr(llm, "_make_client", lambda concurrency: client)

    payload = extract_chapter_sentences(chapters, out)

    assert len(client.calls) == 1  # only the pending chapter was paid for
    assert "kapitel1.xhtml" in client.calls[0]
    assert payload["sentences"] == ["Der Hund bellt.", "Er fängt gerade an."]
    assert not ckpt.exists()


def test_extract_chapter_sentences_failure_keeps_checkpoint(tmp_path, monkeypatch):
    chapters = write_chapters(
        tmp_path, [chapter(0, "toc.xhtml"), chapter(1, "kapitel1.xhtml")]
    )
    client = FakeClient(
        {"toc.xhtml": TOC, "kapitel1.xhtml": CONTENT}, fail_on=["kapitel1.xhtml"]
    )
    monkeypatch.setattr(llm, "_make_client", lambda concurrency: client)

    out = tmp_path / "sentences.json"
    with pytest.raises(ChapterExtractionError, match="re-run the stage to resume"):
        extract_chapter_sentences(chapters, out, concurrency=1)

    assert not out.exists()
    done = load_chapter_checkpoint(checkpoint_path(out))
    assert list(done) == [(0, "toc.xhtml")]


def test_extract_chapter_sentences_rejects_all_non_content(tmp_path, monkeypatch):
    chapters = write_chapters(tmp_path, [chapter(0, "toc.xhtml")])
    client = FakeClient({"toc.xhtml": TOC})
    monkeypatch.setattr(llm, "_make_client", lambda concurrency: client)

    with pytest.raises(ValueError, match="no chapter .* to be book content"):
        extract_chapter_sentences(chapters, tmp_path / "sentences.json")


def test_extract_chapter_sentences_combines_split_parts(tmp_path, monkeypatch):
    big = "<p>" + "a" * 40 + "</p><p>" + "b" * 40 + "</p>"
    chapters = write_chapters(tmp_path, [chapter(0, "gross.xhtml", html=big)])

    class PartClient(FakeClient):
        async def _create(self, *, model, messages, response_format):
            content = messages[1]["content"]
            self.calls.append(content)
            number = 1 if "part 1" in content else 2
            return completion(
                {"kind": "chapter", "is_content": True, "sentences": [f"Satz {number}."]}
            )

    client = PartClient({})
    monkeypatch.setattr(llm, "_make_client", lambda concurrency: client)
    monkeypatch.setattr("transcript.chapters.MAX_CHAPTER_HTML_CHARS", 60)

    payload = extract_chapter_sentences(chapters, tmp_path / "sentences.json")
    assert len(client.calls) == 2
    assert payload["sentences"] == ["Satz 1.", "Satz 2."]
    assert payload["chapters"][0]["is_content"] is True


def test_truncated_answer_is_an_error(tmp_path, monkeypatch):
    chapters = write_chapters(tmp_path, [chapter(0, "kapitel1.xhtml")])

    class TruncatingClient(FakeClient):
        async def _create(self, *, model, messages, response_format):
            return completion(CONTENT, finish_reason="length")

    monkeypatch.setattr(
        llm, "_make_client", lambda concurrency: TruncatingClient({})
    )
    with pytest.raises(ChapterExtractionError, match="cut off"):
        extract_chapter_sentences(chapters, tmp_path / "sentences.json")


def test_sentences_stage_dispatches_chapters_to_the_llm(tmp_path, monkeypatch):
    from transcript.sentences import split_sentences

    chapters = write_chapters(tmp_path, [chapter(0, "kapitel1.xhtml")])
    client = FakeClient({"kapitel1.xhtml": CONTENT})
    monkeypatch.setattr(llm, "_make_client", lambda concurrency: client)

    payload = split_sentences(chapters, tmp_path / "sentences.json", model="test-model")
    assert payload["segmenter"] == "llm"
    assert payload["sentences"] == ["Der Hund bellt.", "Er fängt gerade an."]
