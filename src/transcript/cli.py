"""Command-line interface.

Subcommands mirror the pipeline stages; each reads/writes a JSON artifact in
the work directory and skips itself if its output already exists (--force to
redo). `run` chains the text stage (transcribe for audio, epub for a book)
-> sentences -> words -> build; it picks the first stage from the source's
suffix. DVD ripping stays a separate step since it needs the physical disc.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


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
    p = sub.add_parser("epub", help="Unpack an EPUB book's chapters (skips transcribe)")
    p.add_argument("source", type=Path, help="EPUB file, e.g. ~/books/buch.epub")
    p.add_argument("--list-chapters", action="store_true",
                   help="list the spine documents (index, title, size) and exit; "
                        "honours --chapters")
    p.add_argument("-o", "--output", type=Path, default=Path("work/chapters.json"))
    _add_epub_opts(p)


def _add_epub_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--chapters", metavar="SPEC",
                   help="only read these spine chapters, e.g. '3-20,25' "
                        "(0-based, see --list-chapters); default: all of them — "
                        "the sentences stage skips non-content documents itself")


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


def _add_sentences(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "sentences",
        help="Produce the sentence list: syntok for a transcript, the LLM "
             "(chapter by chapter) for EPUB chapters",
    )
    p.add_argument("transcript", type=Path,
                   help="transcript.json (audio) or chapters.json (EPUB)")
    p.add_argument("-o", "--output", type=Path, default=Path("work/sentences.json"))
    _add_llm_opts(p)


def _add_words(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "words",
        help="Extract the dictionary-form words of each sentence with an OpenAI model",
    )
    p.add_argument("sentences", type=Path)
    p.add_argument("-o", "--output", type=Path, default=Path("work/sentence-words.json"))
    _add_llm_opts(p)


def _add_llm_opts(p: argparse.ArgumentParser) -> None:
    from .llm import DEFAULT_CONCURRENCY, DEFAULT_RPM

    p.add_argument("--llm-rpm", type=int, default=DEFAULT_RPM, metavar="N",
                   help=f"client-side request-per-minute cap, matched to your "
                        f"API tier's rate limit (default: {DEFAULT_RPM})")
    p.add_argument("--llm-concurrency", type=int, default=DEFAULT_CONCURRENCY, metavar="N",
                   help=f"how many LLM requests to run in parallel "
                        f"(default: {DEFAULT_CONCURRENCY})")


def _add_build(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("build", help="Build the words-to-learn JSON from the extracted words")
    p.add_argument("sentence_words", type=Path)
    p.add_argument("-o", "--output", type=Path, default=Path("work/words.json"))
    _add_build_opts(p)


def _add_build_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--min-count", type=int, default=1,
                   help="drop words appearing fewer times than this (default: 1)")
    p.add_argument("--known-words-output", type=Path, metavar="PATH",
                   help="where to save the fetched known-words list "
                        "(default: known-words.json next to the words output)")


def _add_run(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "run",
        help="Run transcribe/epub -> sentences -> words -> build from an audio file or an EPUB",
    )
    p.add_argument("source", type=Path, help="audio file, or a .epub book")
    p.add_argument("--workdir", type=Path, default=Path("work"))
    _add_epub_opts(p)
    _add_transcribe_opts(p)
    _add_llm_opts(p)
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
    _add_sentences(sub)
    _add_words(sub)
    _add_build(sub)
    _add_run(sub)
    return parser


def is_epub(source: Path) -> bool:
    return Path(source).suffix.lower() == ".epub"


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
    from .epub import extract_chapters, format_chapters, parse_chapter_selection, read_chapters

    selection = parse_chapter_selection(args.chapters) if args.chapters else None

    if getattr(args, "list_chapters", False):
        _, chapters = read_chapters(source, selection)
        print(f"Chapters in {source}:")
        print(format_chapters(chapters))
        return

    extract_chapters(source, output, chapters=selection, force=args.force)


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


def _cmd_sentences(args: argparse.Namespace, transcript: Path, output: Path) -> None:
    from .sentences import split_sentences

    split_sentences(
        transcript,
        output,
        rpm=args.llm_rpm,
        concurrency=args.llm_concurrency,
        force=args.force,
    )


def _cmd_words(args: argparse.Namespace, sentences: Path, output: Path) -> None:
    from .llm import extract_words

    extract_words(
        sentences,
        output,
        rpm=args.llm_rpm,
        concurrency=args.llm_concurrency,
        force=args.force,
    )


def _cmd_build(args: argparse.Namespace, sentence_words: Path, output: Path) -> None:
    from .aggregate import build_word_list, write_output
    from .artifacts import load_json
    from .known_words import (
        ARTIFACT_NAME,
        TOKEN_ENV_VAR,
        URL_ENV_VAR,
        KnownWords,
        fetch_known_words,
        save_known_words,
    )
    from .llm import sentence_words_from_payload

    log = logging.getLogger(__name__)
    # Read the extracted words before the network call: a missing or corrupt
    # artifact is certain and cheap to detect, and reporting it first keeps
    # an unrelated API failure from masking it.
    payload = load_json(sentence_words)
    sentences = sentence_words_from_payload(payload, sentence_words)

    url = os.environ.get(URL_ENV_VAR)
    token = os.environ.get(TOKEN_ENV_VAR)
    if url:
        if not token:
            log.warning(
                "%s is set but %s is not; the request will be sent without "
                "authentication", URL_ENV_VAR, TOKEN_ENV_VAR,
            )
        known = fetch_known_words(url, token)
        log.info("Fetched %d known words from %s", len(known), url)
        known_words_json = args.known_words_output or output.parent / ARTIFACT_NAME
        save_known_words(known_words_json, known, source=url)
        log.info("Saved the known words for reference -> %s", known_words_json)
        if not known:
            log.warning(
                "%s returned no words; every word of the source will be "
                "treated as unknown", url,
            )
    else:
        known = KnownWords()
        log.warning(
            "%s is not set (configure it in .env); the list will contain "
            "every word of the source", URL_ENV_VAR,
        )

    entries = build_word_list(
        sentences, known, min_count=args.min_count, verbose=args.verbose
    )
    write_output(
        entries,
        output,
        known_words_source=url,
        llm_model=payload.get("llm_model"),
        min_count=args.min_count,
    )
    log.info("Wrote %d words to learn -> %s", len(entries), output)


def format_error(exc: BaseException) -> str:
    """One-line, user-facing rendering of a failure.

    The stages raise their own exception types, or plain
    ValueError/RuntimeError/FileNotFoundError, with messages that are
    complete on their own; those are printed as-is. Anything escaping from a
    library gets its type name prepended, because such messages read as
    nonsense alone — json's "Expecting value: line 1 column 1 (char 0)"
    being the classic example.
    """
    message = str(exc).strip()
    if not message:
        return type(exc).__name__
    raised_here = type(exc).__module__.split(".")[0] == __package__.split(".")[0]
    if raised_here or type(exc) in (ValueError, RuntimeError, FileNotFoundError):
        return message
    return f"{type(exc).__name__}: {message}"


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
        elif args.command == "sentences":
            _cmd_sentences(args, args.transcript, args.output)
        elif args.command == "words":
            _cmd_words(args, args.sentences, args.output)
        elif args.command == "build":
            _cmd_build(args, args.sentence_words, args.output)
        elif args.command == "run":
            workdir: Path = args.workdir
            text_json = workdir / ("chapters.json" if is_epub(args.source) else "transcript.json")
            sentences_json = workdir / "sentences.json"
            sentence_words_json = workdir / "sentence-words.json"
            words_json = workdir / "words.json"
            _cmd_text(args, args.source, text_json)
            _cmd_sentences(args, text_json, sentences_json)
            _cmd_words(args, sentences_json, sentence_words_json)
            _cmd_build(args, sentence_words_json, words_json)
    except KeyboardInterrupt:
        # Stages write their artifact atomically, so a cached one is either
        # complete or absent; re-running resumes at the interrupted stage.
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # subprocess failures, schema mismatches, ...
        if args.verbose:
            raise
        print(f"error: {format_error(exc)}", file=sys.stderr)
        print("Re-run with -v for the full traceback.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
