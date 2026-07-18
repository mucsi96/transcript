from pathlib import Path

from transcript.cli import build_parser, filter_config_from_args
from transcript.extract import build_ffmpeg_command, format_tracks, parse_ffprobe_streams


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


def test_transcribe_defaults():
    args = parse("transcribe", "movie.flac")
    assert args.model == "large-v3"
    assert args.device == "auto"
    assert args.language == "de"
    assert args.output == Path("work/transcript.json")
    assert not args.no_vad


def test_build_filter_config_overrides():
    args = parse(
        "build", "analysis.json",
        "--keep-pos", "NUM", "--drop-ent", "MISC", "--drop-stopwords",
        "--min-count", "2",
    )
    cfg = filter_config_from_args(args)
    assert "NUM" not in cfg.exclude_pos
    assert "PROPN" in cfg.exclude_pos
    assert "MISC" in cfg.exclude_ent_types
    assert cfg.drop_stopwords
    assert cfg.min_count == 2


def test_run_accepts_all_stage_options():
    args = parse(
        "run", "movie.flac", "--workdir", "w",
        "--device", "cpu", "--spacy-model", "de_core_news_md",
        "--known-words", "known.txt",
    )
    assert args.workdir == Path("w")
    assert args.device == "cpu"
    assert args.spacy_model == "de_core_news_md"
    assert args.known_words == Path("known.txt")


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


def test_format_tracks():
    tracks = parse_ffprobe_streams(
        {"streams": [{"codec_name": "ac3", "channels": 6, "tags": {"language": "ger"}}]}
    )
    out = format_tracks(tracks)
    assert "--audio-track 0" in out
    assert "language=ger" in out
    assert format_tracks([]) == "  no audio streams found"
