import json
from pathlib import Path

from transcript.cli import build_parser, format_error, is_epub
from transcript.extract import build_ffmpeg_command, format_tracks, parse_ffprobe_streams
from transcript.known_words import KnownWordsError


def parse(*argv):
    return build_parser().parse_args(list(argv))


def test_extract_args():
    args = parse(
        "extract", "/mnt/c/Videos/movie.mkv",
        "--audio-language", "ger", "-o", "out.flac",
    )
    assert args.command == "extract"
    assert args.source == Path("/mnt/c/Videos/movie.mkv")
    assert args.audio_language == "ger"
    assert args.output == Path("out.flac")
    assert args.ffmpeg_binary == "ffmpeg"
    assert not args.list_tracks


def test_extract_list_tracks_flag():
    args = parse("extract", "movie.mkv", "--list-tracks")
    assert args.list_tracks


def test_epub_args():
    args = parse("epub", "buch.epub", "--chapters", "3-20", "-o", "work/text.json")
    assert args.command == "epub"
    assert args.source == Path("buch.epub")
    assert args.chapters == "3-20"
    assert args.output == Path("work/text.json")
    assert not args.list_chapters


def test_epub_defaults_to_the_chapters_artifact():
    args = parse("epub", "buch.epub")
    assert args.output == Path("work/chapters.json")
    assert args.chapters is None


def test_epub_list_chapters_flag():
    assert parse("epub", "buch.epub", "--list-chapters").list_chapters


def test_run_accepts_an_epub_source():
    args = parse("run", "buch.epub", "--chapters", "2-9")
    assert args.source == Path("buch.epub")
    assert args.chapters == "2-9"


def test_is_epub_by_suffix():
    assert is_epub(Path("Buch.EPUB"))
    assert not is_epub(Path("movie.flac"))


def test_transcribe_defaults():
    args = parse("transcribe", "movie.flac")
    assert args.model == "large-v3"
    assert args.device == "auto"
    assert args.language == "de"
    assert args.output == Path("work/transcript.json")
    assert not args.no_vad


def test_known_words_output_defaults_to_beside_the_word_list():
    assert parse("build", "sentence-words.json").known_words_output is None
    args = parse("build", "sentence-words.json", "--known-words-output", "w/fetched.json")
    assert args.known_words_output == Path("w/fetched.json")
    assert parse("run", "buch.epub").known_words_output is None


def test_sentences_defaults():
    args = parse("sentences", "work/transcript.json")
    assert args.command == "sentences"
    assert args.transcript == Path("work/transcript.json")
    assert args.output == Path("work/sentences.json")


def test_words_defaults():
    from transcript.llm import DEFAULT_CONCURRENCY, DEFAULT_RPM

    args = parse("words", "work/sentences.json")
    assert args.command == "words"
    assert args.sentences == Path("work/sentences.json")
    assert args.output == Path("work/sentence-words.json")
    assert args.llm_rpm == DEFAULT_RPM
    assert args.llm_concurrency == DEFAULT_CONCURRENCY


def test_words_rate_limit_options():
    args = parse(
        "words", "work/sentences.json",
        "--llm-rpm", "30", "--llm-concurrency", "4",
    )
    assert args.llm_rpm == 30
    assert args.llm_concurrency == 4


def test_build_min_count():
    assert parse("build", "sentence-words.json").min_count == 1
    assert parse("build", "sentence-words.json", "--min-count", "2").min_count == 2


def test_run_accepts_all_stage_options():
    args = parse(
        "run", "movie.flac", "--workdir", "w",
        "--device", "cpu",
        "--llm-rpm", "30",
        "--min-count", "2",
    )
    assert args.source == Path("movie.flac")
    assert args.workdir == Path("w")
    assert args.device == "cpu"
    assert args.llm_rpm == 30
    assert args.min_count == 2


def test_build_ffmpeg_command_by_language_exact_argv():
    cmd = build_ffmpeg_command(
        "movie.mkv", "work/movie.flac",
        audio_language="ger", sample_rate=16000, channels=1,
    )
    assert cmd == [
        "ffmpeg", "-hide_banner", "-nostdin", "-y",
        "-i", "movie.mkv",
        "-map", "0:a:m:language:ger",
        "-vn", "-sn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "flac",
        "work/movie.flac",
    ]


def test_build_ffmpeg_command_by_track_index():
    cmd = build_ffmpeg_command("movie.mkv", "out.flac", audio_track=2)
    assert cmd[cmd.index("-map") + 1] == "0:a:2"


def test_build_ffmpeg_command_defaults_to_first_audio_stream():
    cmd = build_ffmpeg_command("movie.mkv", "out.flac")
    assert cmd[cmd.index("-map") + 1] == "0:a:0"


def test_build_ffmpeg_command_language_wins_over_track():
    cmd = build_ffmpeg_command("movie.mkv", "out.flac", audio_track=1, audio_language="ger")
    assert cmd[cmd.index("-map") + 1] == "0:a:m:language:ger"


def test_parse_ffprobe_streams():
    payload = {
        "streams": [
            {"codec_name": "ac3", "channels": 6, "tags": {"language": "ger", "title": "Surround 5.1"}},
            {"codec_name": "ac3", "channels": 2, "tags": {"language": "eng"}},
            {"codec_name": "mp2", "channels": 2},
        ]
    }
    tracks = parse_ffprobe_streams(payload)
    assert tracks == [
        {"track": 0, "codec": "ac3", "channels": 6, "language": "ger", "title": "Surround 5.1"},
        {"track": 1, "codec": "ac3", "channels": 2, "language": "eng", "title": ""},
        {"track": 2, "codec": "mp2", "channels": 2, "language": "und", "title": ""},
    ]


def test_format_error_names_the_type_of_an_opaque_library_error():
    # The message this whole exercise started from: alone it says nothing.
    try:
        json.loads("<!DOCTYPE html>")
    except json.JSONDecodeError as exc:
        assert format_error(exc) == f"JSONDecodeError: {exc}"
    else:  # pragma: no cover
        raise AssertionError("expected a JSONDecodeError")


def test_format_error_passes_through_our_own_messages():
    assert format_error(ValueError("work/x.json is not valid JSON")) == (
        "work/x.json is not valid JSON"
    )
    assert format_error(KnownWordsError("the API did not return JSON")) == (
        "the API did not return JSON"
    )


def test_format_error_falls_back_to_the_type_when_there_is_no_message():
    assert format_error(KeyError()) == "KeyError"


def test_format_tracks():
    tracks = parse_ffprobe_streams(
        {"streams": [{"codec_name": "ac3", "channels": 6, "tags": {"language": "ger"}}]}
    )
    out = format_tracks(tracks)
    assert "--audio-track 0" in out
    assert "language=ger" in out
    assert format_tracks([]) == "  no audio streams found"
