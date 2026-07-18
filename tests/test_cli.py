from pathlib import Path

from transcript.cli import build_parser, filter_config_from_args
from transcript.extract import build_dvd_mrl, build_vlc_command, is_windows_binary


def parse(*argv):
    return build_parser().parse_args(list(argv))


def test_extract_args():
    args = parse(
        "extract", "--dvd", "/dev/sr0", "--title", "2",
        "--audio-language", "deu", "-o", "out.flac",
    )
    assert args.command == "extract"
    assert args.dvd == "/dev/sr0"
    assert args.title == 2
    assert args.audio_language == "deu"
    assert args.output == Path("out.flac")
    assert args.vlc_binary == "cvlc"


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


def test_build_dvd_mrl_linux_device():
    assert build_dvd_mrl("/dev/sr0", 1) == "dvdsimple:///dev/sr0#1"


def test_build_dvd_mrl_windows_drive_letter():
    assert build_dvd_mrl("D:", 1) == "dvdsimple://D:\\#1"


def test_build_dvd_mrl_with_chapter_and_iso():
    assert build_dvd_mrl("/data/movie.iso", 2, 3) == "dvdsimple:///data/movie.iso#2:3"


def test_is_windows_binary():
    assert is_windows_binary("/mnt/c/Program Files/VideoLAN/VLC/vlc.exe")
    assert not is_windows_binary("cvlc")


def test_build_vlc_command_exact_argv():
    cmd = build_vlc_command(
        "/dev/sr0", 1, "work/movie.flac",
        audio_language="deu", sample_rate=16000, channels=1,
    )
    assert cmd == [
        "cvlc",
        "--intf", "dummy",
        "--no-video",
        "dvdsimple:///dev/sr0#1",
        "--audio-language=deu",
        "--sout=#transcode{acodec=flac,channels=1,samplerate=16000}"
        ":std{access=file,mux=raw,dst=work/movie.flac}",
        "vlc://quit",
    ]


def test_build_vlc_command_audio_track_fallback():
    cmd = build_vlc_command("/dev/sr0", 1, "out.flac", audio_track=2)
    assert "--audio-track=2" in cmd
    assert not any(a.startswith("--audio-language") for a in cmd)
