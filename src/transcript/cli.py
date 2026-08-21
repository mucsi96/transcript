"""Command-line interface.

Subcommands mirror the pipeline stages; each reads/writes a JSON artifact in
the work directory and skips itself if its output already exists (--force to
redo). `run` chains the text stage (transcribe for audio, epub for a book)
-> analyze -> build; it picks the first stage from the source's suffix. DVD
ripping stays a separate step since it needs the physical disc.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from .config import DEFAULT_EXCLUDE_ENT_TYPES, DEFAULT_EXCLUDE_POS, FilterConfig


def _add_extract(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("extract", help="Extract audio from a MakeMKV rip (.mkv) to FLAC with ffmpeg")
    p.add_argument("source", type=Path,
                   help="video file ripped with MakeMKV, e.g. /mnt/c/Users/you/Videos/movie.mkv")
    p.add_argument("--list-tracks", action="store_true",
                   help="list the audio tracks (language, codec, channels) and exit")
    p.add_argument("--audio-language",
                   help="pick the audio stream by language tag (DVD rips usually tag German as 'ger')")
    p.add_argument("--audio-track", type=int,
                   help="pick the audio stream by index among audio streams, 0-based (see --list-tracks)")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--channels", type=int, default=1)
    p.add_argument("--ffmpeg-binary", default="ffmpeg")
    p.add_argument("-o", "--output", type=Path, default=Path("work/movie.flac"))
    p.add_argument("--dry-run", action="store_true", help="print the ffmpeg command without running it")


def _add_epub(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("epub", help="Extract the text of an EPUB book (skips transcribe)")
    p.add_argument("source", type=Path, help="EPUB file, e.g. ~/books/buch.epub")
    p.add_argument("--list-chapters", action="store_true",
                   help="list the spine documents (index, title, size) and exit; "
                        "honours --chapters")
    p.add_argument("-o", "--output", type=Path, default=Path("work/transcript.json"))
    _add_epub_opts(p)


def _add_epub_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--chapters", metavar="SPEC",
                   help="only read these spine chapters, e.g. '3-20,25' "
                        "(0-based, see --list-chapters); default: all of them")


def _add_transcribe(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("transcribe", help="Transcribe audio with Whisper large-v3 (faster-whisper)")
    p.add_argument("audio", type=Path)
    p.add_argument("-o", "--output", type=Path, default=Path("work/transcript.json"))
    _add_transcribe_opts(p)


def _add_transcribe_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="large-v3", help="Whisper model size (default: large-v3)")
    p.add_argument("--backend", default="auto", choices=["auto", "faster-whisper", "mlx"],
                   help="transcription backend (auto: mlx on Apple Silicon, faster-whisper elsewhere)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="faster-whisper device (ignored by the mlx backend)")
    p.add_argument("--language", default="de")
    p.add_argument("--beam-size", type=int, default=5)
    p.add_argument("--no-vad", action="store_true", help="disable voice-activity-detection filtering")


def _add_analyze(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("analyze", help="Sentence-split, lemmatize and tag the transcript with spaCy")
    p.add_argument("transcript", type=Path)
    p.add_argument("-o", "--output", type=Path, default=Path("work/analysis.json"))
    _add_analyze_opts(p)


def _add_analyze_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--spacy-model", default="de_core_news_lg",
                   help="German spaCy model (default: de_core_news_lg; de_core_news_md is lighter)")


def _add_build(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("build", help="Build the words-to-learn JSON from the analysis")
    p.add_argument("analysis", type=Path)
    p.add_argument("-o", "--output", type=Path, default=Path("work/words.json"))
    _add_build_opts(p)


def _add_build_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--min-count", type=int, default=1,
                   help="drop words appearing fewer times than this (default: 1)")
    p.add_argument("--min-token-len", type=int, default=2)
    p.add_argument("--keep-pos", action="append", default=[], metavar="POS",
                   help="POS tag to keep despite the default exclusions (e.g. NUM); repeatable")
    p.add_argument("--drop-pos", action="append", default=[], metavar="POS",
                   help="additional POS tag to exclude; repeatable")
    p.add_argument("--keep-ent", action="append", default=[], metavar="ENT",
                   help="entity type to keep (default excluded: PER LOC ORG); repeatable")
    p.add_argument("--drop-ent", action="append", default=[], metavar="ENT",
                   help="additional entity type to exclude (e.g. MISC); repeatable")
    p.add_argument("--drop-stopwords", action="store_true",
                   help="also drop German stopwords (der/und/aber ...)")


def _add_run(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "run",
        help="Run transcribe/epub -> analyze -> build from an audio file or an EPUB",
    )
    p.add_argument("source", type=Path, help="audio file, or a .epub book")
    p.add_argument("--workdir", type=Path, default=Path("work"))
    _add_epub_opts(p)
    _add_transcribe_opts(p)
    _add_analyze_opts(p)
    _add_build_opts(p)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcript",
        description="Extract German vocabulary to learn from a DVD movie or an EPUB book.",
    )
    parser.add_argument("--force", action="store_true", help="re-run stages even if cached output exists")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_extract(sub)
    _add_epub(sub)
    _add_transcribe(sub)
    _add_analyze(sub)
    _add_build(sub)
    _add_run(sub)
    return parser


def is_epub(source: Path) -> bool:
    return Path(source).suffix.lower() == ".epub"


def filter_config_from_args(args: argparse.Namespace) -> FilterConfig:
    exclude_pos = (DEFAULT_EXCLUDE_POS | set(args.drop_pos)) - set(args.keep_pos)
    exclude_ent = (DEFAULT_EXCLUDE_ENT_TYPES | set(args.drop_ent)) - set(args.keep_ent)
    return FilterConfig(
        exclude_pos=frozenset(exclude_pos),
        exclude_ent_types=frozenset(exclude_ent),
        min_token_len=args.min_token_len,
        drop_stopwords=args.drop_stopwords,
        min_count=args.min_count,
    )


def _cmd_extract(args: argparse.Namespace) -> None:
    from .extract import extract_audio, format_tracks, list_audio_tracks

    if args.list_tracks:
        tracks = list_audio_tracks(args.source)
        print(f"Audio tracks in {args.source}:")
        print(format_tracks(tracks))
        return

    extract_audio(
        args.source,
        args.output,
        audio_track=args.audio_track,
        audio_language=args.audio_language,
        sample_rate=args.sample_rate,
        channels=args.channels,
        ffmpeg_binary=args.ffmpeg_binary,
        dry_run=args.dry_run,
    )


def _cmd_epub(args: argparse.Namespace, source: Path, output: Path) -> None:
    from .epub import extract_text, format_chapters, parse_chapter_selection, read_chapters

    selection = parse_chapter_selection(args.chapters) if args.chapters else None

    if getattr(args, "list_chapters", False):
        _, chapters = read_chapters(source, selection)
        print(f"Chapters in {source}:")
        print(format_chapters(chapters))
        return

    extract_text(source, output, chapters=selection, force=args.force)


def _cmd_transcribe(args: argparse.Namespace, audio: Path, output: Path) -> None:
    from .transcribe import transcribe

    transcribe(
        audio,
        output,
        model_size=args.model,
        backend=args.backend,
        device=args.device,
        language=args.language,
        beam_size=args.beam_size,
        vad=not args.no_vad,
        force=args.force,
    )


def _cmd_text(args: argparse.Namespace, source: Path, output: Path) -> None:
    """Produce the text artifact: EPUBs are read, everything else is
    transcribed."""
    if is_epub(source):
        _cmd_epub(args, source, output)
    else:
        _cmd_transcribe(args, source, output)


def _cmd_analyze(args: argparse.Namespace, transcript: Path, output: Path) -> None:
    from .analyze import analyze

    analyze(transcript, output, model_name=args.spacy_model, force=args.force)


def _cmd_build(args: argparse.Namespace, analysis: Path, output: Path) -> None:
    from .aggregate import build_word_list, write_output
    from .analyze import sentences_from_payload
    from .artifacts import load_json
    from .known_words import TOKEN_ENV_VAR, URL_ENV_VAR, fetch_known_words

    log = logging.getLogger(__name__)
    url = os.environ.get(URL_ENV_VAR)
    token = os.environ.get(TOKEN_ENV_VAR)
    if url:
        known = fetch_known_words(url, token)
        log.info("Fetched %d known words from %s", len(known), url)
    else:
        known = frozenset()
        log.warning(
            "%s is not set (configure it in .env); the list will contain "
            "every word of the source", URL_ENV_VAR,
        )

    cfg = filter_config_from_args(args)
    sentences = sentences_from_payload(load_json(analysis))
    entries = build_word_list(sentences, known, cfg, verbose=args.verbose)
    write_output(entries, output, known_words_source=url, cfg=cfg)
    log.info("Wrote %d words to learn -> %s", len(entries), output)


def main(argv: list[str] | None = None) -> int:
    from dotenv import find_dotenv, load_dotenv

    # usecwd: search for .env from the current directory upward; the default
    # searches from the installed package location, which never finds it.
    load_dotenv(find_dotenv(usecwd=True))
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    if not args.verbose:
        # huggingface-hub / httpx log every request at INFO, which buries
        # the pipeline's own progress output during the model download.
        for name in ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock"):
            logging.getLogger(name).setLevel(logging.WARNING)
    try:
        if args.command == "extract":
            _cmd_extract(args)
        elif args.command == "epub":
            _cmd_epub(args, args.source, args.output)
        elif args.command == "transcribe":
            _cmd_transcribe(args, args.audio, args.output)
        elif args.command == "analyze":
            _cmd_analyze(args, args.transcript, args.output)
        elif args.command == "build":
            _cmd_build(args, args.analysis, args.output)
        elif args.command == "run":
            workdir: Path = args.workdir
            transcript_json = workdir / "transcript.json"
            analysis_json = workdir / "analysis.json"
            words_json = workdir / "words.json"
            _cmd_text(args, args.source, transcript_json)
            _cmd_analyze(args, transcript_json, analysis_json)
            _cmd_build(args, analysis_json, words_json)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # subprocess failures, schema mismatches, ...
        if args.verbose:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
