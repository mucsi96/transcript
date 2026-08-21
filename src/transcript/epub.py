"""Stage 1b: extract the text of an EPUB book into a text artifact.

An EPUB is a ZIP archive: ``META-INF/container.xml`` points at the OPF
package document, whose ``<manifest>`` lists the files and whose ``<spine>``
gives the reading order. Only the spine's XHTML documents are read, in that
order; the navigation document, the NCX table of contents and images are
skipped.

The output artifact has the same shape the transcribe stage produces — a
list of text segments — so `analyze` and `build` treat a book exactly like a
film. A segment is a paragraph here and a Whisper segment there: both are
small units that end on a sentence boundary, which is what the spaCy chunker
wants.

Only DRM-free EPUBs can be read; encrypted ones are reported as such.
"""

from __future__ import annotations

import logging
import posixpath
import re
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote
from xml.etree import ElementTree as ET

from .artifacts import SCHEMA_VERSION, load_json, save_json, should_skip

log = logging.getLogger(__name__)

CONTAINER_PATH = "META-INF/container.xml"
ENCRYPTION_PATH = "META-INF/encryption.xml"

XHTML_MEDIA_TYPES = frozenset({"application/xhtml+xml", "text/html"})
XHTML_SUFFIXES = (".xhtml", ".html", ".htm")

# Tags whose end (or start) closes a paragraph of text.
BLOCK_TAGS = frozenset({
    "article", "aside", "blockquote", "body", "br", "caption", "dd", "div",
    "dl", "dt", "figcaption", "footer", "h1", "h2", "h3", "h4", "h5", "h6",
    "header", "hr", "li", "nav", "ol", "p", "pre", "section", "table", "td",
    "th", "tr", "ul",
})
# Tags whose content is not prose.
SKIP_TAGS = frozenset({"head", "script", "style", "svg"})
HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})

# spaCy's memory use grows with document length; the analyze stage joins
# segments into chunks but never splits one, so oversized paragraphs are
# split here. Badly converted EPUBs sometimes put a whole chapter in a
# single <div> without any <p>.
MAX_SEGMENT_CHARS = 20_000

_XML_ENCODING = re.compile(rb"""^<\?xml[^>]*?encoding=["']([\w.-]+)["']""", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


@dataclass(frozen=True)
class SpineItem:
    """One XHTML document of the reading order. `index` is its position in
    the spine and is what --chapters selects on."""

    index: int
    idref: str
    href: str  # path inside the ZIP


@dataclass(frozen=True)
class Package:
    metadata: dict
    spine: tuple[SpineItem, ...]


@dataclass(frozen=True)
class Chapter:
    index: int
    href: str
    title: str
    paragraphs: tuple[str, ...]

    @property
    def chars(self) -> int:
        return sum(len(p) for p in self.paragraphs)


def _local(tag) -> str:
    """Local name of a possibly namespaced ElementTree tag. EPUBs in the
    wild use OPF 2 and 3 namespaces, or none at all, so every lookup goes
    by local name."""
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1].lower()


def _text_of(element) -> str:
    return " ".join("".join(element.itertext()).split())


class _DocumentParser(HTMLParser):
    """XHTML -> paragraphs. HTMLParser rather than ElementTree because EPUB
    content documents are frequently not well-formed XML (unescaped
    ampersands, HTML entities without a DTD, stray tags)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.paragraphs: list[str] = []
        self.title: str | None = None
        self._buffer: list[str] = []
        self._skipped: list[str] = []
        self._in_heading = False

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in SKIP_TAGS:
            self._skipped.append(tag)
            return
        if tag in BLOCK_TAGS:
            self._flush()
            if tag in HEADING_TAGS:
                self._in_heading = True

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._skipped and self._skipped[-1] == tag:
            self._skipped.pop()
            return
        if tag in SKIP_TAGS:
            return  # stray close tag, nothing was skipped
        if tag in BLOCK_TAGS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if not self._skipped:
            self._buffer.append(data)

    def close(self) -> None:
        super().close()
        self._flush()

    def _flush(self) -> None:
        # \xa0 and friends count as whitespace to str.split, so entity-heavy
        # markup collapses to normal spacing here.
        text = " ".join("".join(self._buffer).split())
        self._buffer.clear()
        was_heading, self._in_heading = self._in_heading, False
        if not text:
            return
        if was_heading and self.title is None:
            self.title = text
        self.paragraphs.append(text)


def parse_document(markup: str) -> tuple[str | None, list[str]]:
    """Return (title, paragraphs) for one XHTML content document. The title
    is its first heading, if any."""
    parser = _DocumentParser()
    parser.feed(markup)
    parser.close()
    return parser.title, parser.paragraphs


def decode_markup(data: bytes) -> str:
    """Decode a content document, honouring its XML declaration. UTF-8 is
    the EPUB default; legacy files converted from HTML are sometimes
    cp1252."""
    match = _XML_ENCODING.match(data.lstrip()[:200])
    encodings = ["utf-8", "cp1252"]
    if match:
        declared = match.group(1).decode("ascii", "replace")
        encodings.insert(0, declared)
    for encoding in encodings:
        try:
            return data.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return data.decode("utf-8", errors="replace")


def opf_path(container_xml: bytes) -> str:
    """Path of the OPF package document inside the ZIP, from
    META-INF/container.xml."""
    root = ET.fromstring(container_xml)
    for element in root.iter():
        if _local(element.tag) == "rootfile":
            full_path = element.get("full-path")
            if full_path:
                return posixpath.normpath(unquote(full_path))
    raise ValueError("META-INF/container.xml names no rootfile; not a readable EPUB")


def resolve_href(base_dir: str, href: str) -> str:
    """Resolve a manifest href (URL-quoted, relative to the OPF file) to a
    path inside the ZIP."""
    href = unquote(href.split("#", 1)[0])
    return posixpath.normpath(posixpath.join(base_dir, href)) if base_dir else posixpath.normpath(href)


def _is_content_document(href: str, media_type: str, properties: list[str]) -> bool:
    if "nav" in properties:  # EPUB 3 navigation document: a table of contents
        return False
    if media_type:
        return media_type in XHTML_MEDIA_TYPES
    return href.lower().endswith(XHTML_SUFFIXES)


def parse_package(opf_xml: bytes, opf_zip_path: str) -> Package:
    """Parse the OPF package document into book metadata and the spine."""
    root = ET.fromstring(opf_xml)
    base_dir = posixpath.dirname(opf_zip_path)
    metadata: dict[str, str | None] = {"title": None, "author": None, "language": None}
    manifest: dict[str, tuple[str, str, list[str]]] = {}
    idrefs: list[str] = []

    for element in root.iter():
        name = _local(element.tag)
        if name == "item":
            item_id, href = element.get("id"), element.get("href")
            if item_id and href:
                manifest[item_id] = (
                    resolve_href(base_dir, href),
                    (element.get("media-type") or "").strip().lower(),
                    (element.get("properties") or "").split(),
                )
        elif name == "itemref":
            idref = element.get("idref")
            if idref:
                idrefs.append(idref)
        elif name in ("title", "creator", "language"):
            key = {"title": "title", "creator": "author", "language": "language"}[name]
            if metadata[key] is None:
                metadata[key] = _text_of(element) or None

    spine = []
    for idref in idrefs:
        entry = manifest.get(idref)
        if entry is None:
            log.warning("spine references unknown manifest id %r, skipping", idref)
            continue
        href, media_type, properties = entry
        if not _is_content_document(href, media_type, properties):
            continue
        spine.append(SpineItem(index=len(spine), idref=idref, href=href))

    if not spine:
        raise ValueError("the EPUB spine contains no XHTML content documents")
    return Package(metadata=metadata, spine=tuple(spine))


def parse_chapter_selection(spec: str) -> frozenset[int]:
    """Parse a --chapters value such as '0,3-20,25' into spine indices."""
    selected: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        start, sep, end = part.partition("-")
        try:
            first = int(start)
            last = int(end) if sep else first
        except ValueError:
            raise ValueError(f"invalid --chapters value {part!r}; expected N or N-M") from None
        if last < first:
            raise ValueError(f"invalid --chapters range {part!r}; {last} is before {first}")
        selected.update(range(first, last + 1))
    if not selected:
        raise ValueError("--chapters selected nothing")
    return frozenset(selected)


def _split_words(text: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for word in text.split():
        while len(word) > max_chars:  # a single monster "word": hard-cut it
            if current:
                pieces.append(current)
                current = ""
            pieces.append(word[:max_chars])
            word = word[max_chars:]
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= max_chars:
            current += " " + word
        else:
            pieces.append(current)
            current = word
    if current:
        pieces.append(current)
    return pieces


def split_paragraph(text: str, max_chars: int = MAX_SEGMENT_CHARS) -> list[str]:
    """Split an oversized paragraph at sentence ends, falling back to word
    boundaries. Paragraphs within the limit are returned unchanged."""
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(text):
        for part in _split_words(sentence, max_chars):
            if not current:
                current = part
            elif len(current) + 1 + len(part) <= max_chars:
                current += " " + part
            else:
                pieces.append(current)
                current = part
    if current:
        pieces.append(current)
    return pieces


def read_chapters(source: Path, selection: frozenset[int] | None = None) -> tuple[dict, list[Chapter]]:
    """Read the spine's content documents in reading order. Chapters that
    carry no text (cover pages) are kept with zero characters so
    --list-chapters shows the full spine and its indices stay stable."""
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"EPUB file not found: {source}")

    try:
        archive = zipfile.ZipFile(source)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"{source} is not a valid EPUB (not a ZIP archive)") from exc

    with archive:
        names = set(archive.namelist())
        if ENCRYPTION_PATH in names:
            raise ValueError(
                f"{source} is DRM-protected ({ENCRYPTION_PATH} present); only "
                "DRM-free EPUBs can be read."
            )
        if CONTAINER_PATH not in names:
            raise ValueError(f"{source} has no {CONTAINER_PATH}; not a valid EPUB")

        opf_zip_path = opf_path(archive.read(CONTAINER_PATH))
        if opf_zip_path not in names:
            raise ValueError(f"{source}: package document {opf_zip_path} is missing from the archive")
        package = parse_package(archive.read(opf_zip_path), opf_zip_path)

        chapters = []
        for item in package.spine:
            if selection is not None and item.index not in selection:
                continue
            try:
                markup = archive.read(item.href)
            except KeyError:
                log.warning("spine item %s is missing from the archive, skipping", item.href)
                continue
            title, paragraphs = parse_document(decode_markup(markup))
            chapters.append(
                Chapter(
                    index=item.index,
                    href=item.href,
                    title=title or posixpath.basename(item.href),
                    paragraphs=tuple(paragraphs),
                )
            )

    return package.metadata, chapters


def format_chapters(chapters: list[Chapter]) -> str:
    lines = []
    for chapter in chapters:
        lines.append(
            f"  --chapters {chapter.index}: {chapter.title} "
            f"({chapter.chars:,} chars, {chapter.href})"
        )
    return "\n".join(lines) if lines else "  no readable chapters found"


def extract_text(
    source: Path,
    output: Path,
    *,
    chapters: frozenset[int] | None = None,
    force: bool = False,
) -> dict:
    if should_skip(output, force):
        return load_json(output)

    metadata, chapter_list = read_chapters(source, chapters)

    segments: list[dict] = []
    chapter_records: list[dict] = []
    for chapter in chapter_list:
        texts = [piece for para in chapter.paragraphs for piece in split_paragraph(para)]
        chapter_records.append(
            {
                "index": chapter.index,
                "href": chapter.href,
                "title": chapter.title,
                "segments": len(texts),
                "chars": chapter.chars,
            }
        )
        for text in texts:
            segments.append({"id": len(segments), "chapter": chapter.index, "text": text})

    if not segments:
        raise ValueError(
            f"No readable text found in {source}. Run with --list-chapters to "
            "see the spine; a --chapters selection may have excluded everything."
        )

    total_chars = sum(len(s["text"]) for s in segments)
    log.info(
        "Read %d chapter(s), %d paragraph(s), %d characters from %s",
        len(chapter_records), len(segments), total_chars, source,
    )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_type": "epub",
        "source_file": str(source),
        "title": metadata.get("title"),
        "author": metadata.get("author"),
        "language": metadata.get("language"),
        "chapter_count": len(chapter_records),
        "char_count": total_chars,
        "chapters": chapter_records,
        "segments": segments,
    }
    save_json(output, payload)
    return payload
