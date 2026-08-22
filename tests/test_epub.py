import zipfile

import pytest

from transcript.epub import (
    MAX_SEGMENT_CHARS,
    decode_markup,
    extract_text,
    format_chapters,
    opf_path,
    parse_chapter_selection,
    parse_document,
    parse_package,
    read_chapters,
    resolve_href,
    split_paragraph,
)

CONTAINER_XML = b"""<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""

OPF_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Der Prozess</dc:title>
    <dc:creator>Franz Kafka</dc:creator>
    <dc:language>de</dc:language>
  </metadata>
  <manifest>
    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
    <item id="cover" href="cover.xhtml" media-type="application/xhtml+xml"/>
    <item id="c1" href="text/kapitel%201.xhtml" media-type="application/xhtml+xml"/>
    <item id="c2" href="text/kapitel2.xhtml" media-type="application/xhtml+xml"/>
    <item id="img" href="images/cover.jpg" media-type="image/jpeg"/>
  </manifest>
  <spine toc="ncx">
    <itemref idref="nav"/>
    <itemref idref="cover"/>
    <itemref idref="c1"/>
    <itemref idref="c2"/>
  </spine>
</package>
"""

CHAPTER_1 = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
  <head><title>ignored</title><style>p { color: red; }</style></head>
  <body>
    <h1>Erstes Kapitel</h1>
    <p>Jemand musste Josef&nbsp;K. <em>verleumdet</em> haben.</p>
    <p>Der Hund   bellt<br/>laut.</p>
    <script>var x = "nicht sichtbar";</script>
    <div><p></p></div>
  </body>
</html>
"""

CHAPTER_2 = """<html><body><h2>Zweites Kapitel</h2>
<ul><li>Erster Punkt</li><li>Zweiter Punkt</li></ul>
<p>Ende &amp; Schluss.</p></body></html>
"""


def build_epub(path, *, chapters=(CHAPTER_1, CHAPTER_2), encrypted=False, container=CONTAINER_XML):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        if container is not None:
            zf.writestr("META-INF/container.xml", container)
        if encrypted:
            zf.writestr("META-INF/encryption.xml", "<encryption/>")
        zf.writestr("OEBPS/content.opf", OPF_XML)
        zf.writestr("OEBPS/nav.xhtml", "<html><body><nav><a href='#'>TOC</a></nav></body></html>")
        zf.writestr("OEBPS/cover.xhtml", "<html><body><img src='images/cover.jpg'/></body></html>")
        zf.writestr("OEBPS/text/kapitel 1.xhtml", chapters[0])
        zf.writestr("OEBPS/text/kapitel2.xhtml", chapters[1])
    return path


@pytest.fixture
def book(tmp_path):
    return build_epub(tmp_path / "buch.epub")


def test_opf_path_from_container():
    assert opf_path(CONTAINER_XML) == "OEBPS/content.opf"


def test_opf_path_without_rootfile():
    with pytest.raises(ValueError, match="no rootfile"):
        opf_path(b"<container><rootfiles/></container>")


def test_resolve_href_unquotes_and_joins():
    assert resolve_href("OEBPS", "text/kapitel%201.xhtml") == "OEBPS/text/kapitel 1.xhtml"
    assert resolve_href("OEBPS", "../images/c.jpg#frag") == "images/c.jpg"
    assert resolve_href("", "ch1.xhtml") == "ch1.xhtml"


def test_parse_package_metadata_and_spine():
    package = parse_package(OPF_XML, "OEBPS/content.opf")
    assert package.metadata == {"title": "Der Prozess", "author": "Franz Kafka", "language": "de"}
    # nav document dropped, reading order kept, hrefs resolved and unquoted
    assert [item.href for item in package.spine] == [
        "OEBPS/cover.xhtml",
        "OEBPS/text/kapitel 1.xhtml",
        "OEBPS/text/kapitel2.xhtml",
    ]
    assert [item.index for item in package.spine] == [0, 1, 2]


def test_parse_package_without_content_documents():
    opf = b"""<package><manifest><item id="i" href="a.jpg" media-type="image/jpeg"/></manifest>
    <spine><itemref idref="i"/></spine></package>"""
    with pytest.raises(ValueError, match="no XHTML content documents"):
        parse_package(opf, "content.opf")


def test_parse_document_paragraphs_and_title():
    title, paragraphs = parse_document(CHAPTER_1)
    assert title == "Erstes Kapitel"
    assert paragraphs == [
        "Erstes Kapitel",
        "Jemand musste Josef K. verleumdet haben.",
        "Der Hund bellt",
        "laut.",
    ]


def test_parse_document_skips_script_and_style():
    _, paragraphs = parse_document(CHAPTER_1)
    assert not any("nicht sichtbar" in p for p in paragraphs)
    assert not any("color: red" in p for p in paragraphs)


def test_parse_document_handles_list_items_and_entities():
    title, paragraphs = parse_document(CHAPTER_2)
    assert title == "Zweites Kapitel"
    assert paragraphs == ["Zweites Kapitel", "Erster Punkt", "Zweiter Punkt", "Ende & Schluss."]


def test_parse_document_tolerates_broken_markup():
    _, paragraphs = parse_document("<p>Kaffee & Kuchen<p>Zweiter Absatz</body>")
    assert paragraphs == ["Kaffee & Kuchen", "Zweiter Absatz"]


def test_decode_markup_honours_xml_declaration():
    data = '<?xml version="1.0" encoding="iso-8859-1"?><p>Grüße</p>'.encode("iso-8859-1")
    assert "Grüße" in decode_markup(data)


def test_decode_markup_defaults_to_utf8():
    assert decode_markup("<p>Grüße</p>".encode("utf-8")) == "<p>Grüße</p>"


def test_parse_chapter_selection():
    assert parse_chapter_selection("2") == frozenset({2})
    assert parse_chapter_selection("0, 3-5 ,8") == frozenset({0, 3, 4, 5, 8})


@pytest.mark.parametrize("spec", ["", "abc", "5-2", "1-"])
def test_parse_chapter_selection_rejects_bad_specs(spec):
    with pytest.raises(ValueError):
        parse_chapter_selection(spec)


def test_split_paragraph_keeps_short_text():
    assert split_paragraph("Der Hund bellt.") == ["Der Hund bellt."]


def test_split_paragraph_splits_at_sentence_ends():
    text = " ".join(["Satz nummer eins."] * 100)
    pieces = split_paragraph(text, max_chars=50)
    assert all(len(p) <= 50 for p in pieces)
    assert all(p.endswith(".") for p in pieces)
    assert " ".join(pieces) == text


def test_split_paragraph_falls_back_to_word_and_hard_cuts():
    pieces = split_paragraph("aaaa bbbb " + "x" * 25, max_chars=10)
    assert all(len(p) <= 10 for p in pieces)
    assert "".join(pieces).replace(" ", "") == "aaaabbbb" + "x" * 25


def test_read_chapters_reading_order(book):
    metadata, chapters = read_chapters(book)
    assert metadata["title"] == "Der Prozess"
    assert [c.index for c in chapters] == [0, 1, 2]
    # the cover page has no text but stays listed so indices are stable
    assert chapters[0].chars == 0
    assert chapters[1].title == "Erstes Kapitel"
    assert chapters[2].paragraphs[-1] == "Ende & Schluss."


def test_read_chapters_selection(book):
    _, chapters = read_chapters(book, frozenset({2}))
    assert [c.title for c in chapters] == ["Zweites Kapitel"]


def test_read_chapters_rejects_drm(tmp_path):
    path = build_epub(tmp_path / "drm.epub", encrypted=True)
    with pytest.raises(ValueError, match="DRM-protected"):
        read_chapters(path)


def test_read_chapters_rejects_non_zip(tmp_path):
    path = tmp_path / "fake.epub"
    path.write_text("not a zip")
    with pytest.raises(ValueError, match="not a valid EPUB"):
        read_chapters(path)


def test_read_chapters_missing_container(tmp_path):
    path = build_epub(tmp_path / "bare.epub", container=None)
    with pytest.raises(ValueError, match="container.xml"):
        read_chapters(path)


def test_read_chapters_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_chapters(tmp_path / "nope.epub")


def test_extract_text_payload(book, tmp_path):
    output = tmp_path / "work" / "transcript.json"
    payload = extract_text(book, output)
    assert output.exists()
    assert payload["source_type"] == "epub"
    assert payload["title"] == "Der Prozess"
    assert payload["author"] == "Franz Kafka"
    assert payload["language"] == "de"
    assert payload["chapter_count"] == 3
    # segments are paragraphs, numbered across the whole book
    assert [s["id"] for s in payload["segments"]] == list(range(len(payload["segments"])))
    assert payload["segments"][0]["chapter"] == 1
    assert payload["segments"][0]["text"] == "Erstes Kapitel"
    assert payload["char_count"] == sum(len(s["text"]) for s in payload["segments"])
    assert payload["chapters"][0] == {
        "index": 0, "href": "OEBPS/cover.xhtml", "title": "cover.xhtml",
        "segments": 0, "chars": 0,
    }


def test_extract_text_feeds_the_sentences_stage(book, tmp_path):
    from transcript.sentences import chunk_segments
    from transcript.artifacts import load_json

    payload = extract_text(book, tmp_path / "transcript.json")
    reloaded = load_json(tmp_path / "transcript.json")
    assert reloaded == payload
    chunks = chunk_segments([seg["text"] for seg in reloaded["segments"]])
    assert "Josef K." in chunks[0]


def test_extract_text_segments_stay_within_the_chunk_limit(tmp_path):
    huge = "<html><body><div>" + "Ein Satz. " * 40_000 + "</div></body></html>"
    path = build_epub(tmp_path / "huge.epub", chapters=(huge, CHAPTER_2))
    payload = extract_text(path, tmp_path / "transcript.json")
    assert max(len(s["text"]) for s in payload["segments"]) <= MAX_SEGMENT_CHARS


def test_extract_text_skips_cached_output(book, tmp_path):
    output = tmp_path / "transcript.json"
    extract_text(book, output)
    output.write_text('{"schema_version": 1, "segments": [], "cached": true}')
    assert extract_text(book, output)["cached"] is True
    assert "cached" not in extract_text(book, output, force=True)


def test_extract_text_without_any_text(tmp_path):
    empty = "<html><body></body></html>"
    path = build_epub(tmp_path / "empty.epub", chapters=(empty, empty))
    with pytest.raises(ValueError, match="No readable text"):
        extract_text(path, tmp_path / "transcript.json")


def test_format_chapters(book):
    _, chapters = read_chapters(book)
    out = format_chapters(chapters)
    assert "--chapters 1: Erstes Kapitel" in out
    assert format_chapters([]) == "  no readable chapters found"
