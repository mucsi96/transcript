"""Stage 1: extract the audio track from a MakeMKV rip to FLAC with ffmpeg.

The DVD itself is ripped on Windows with MakeMKV, which handles DVD
decryption far more reliably than player-based ripping. Rip the main title
to an .mkv keeping the German audio track, then run this stage inside WSL
against the resulting file, e.g. /mnt/c/Users/<you>/Videos/movie.mkv.

Use --list-tracks first to see the audio streams; DVD rips usually tag
German as 'ger' (ISO 639-2/B).
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def parse_ffprobe_streams(payload: dict) -> list[dict]:
    """Flatten ffprobe's audio-stream JSON into track dicts. The 'track'
    number is the index among audio streams, matching --audio-track."""
    tracks = []
    for i, stream in enumerate(payload.get("streams", [])):
        tags = stream.get("tags") or {}
        tracks.append(
            {
                "track": i,
                "codec": stream.get("codec_name", "?"),
                "channels": stream.get("channels"),
                "language": tags.get("language", "und"),
                "title": tags.get("title", ""),
            }
        )
    return tracks


def list_audio_tracks(source: Path, ffprobe_binary: str = "ffprobe") -> list[dict]:
    result = subprocess.run(
        [
            ffprobe_binary,
            "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=codec_name,channels:stream_tags=language,title",
            "-of", "json",
            str(source),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return parse_ffprobe_streams(json.loads(result.stdout))


def format_tracks(tracks: list[dict]) -> str:
    lines = []
    for t in tracks:
        title = f" ({t['title']})" if t["title"] else ""
        lines.append(
            f"  --audio-track {t['track']}: language={t['language']}, "
            f"{t['codec']}, {t['channels']}ch{title}"
        )
    return "\n".join(lines) if lines else "  no audio streams found"


def build_ffmpeg_command(
    source: str | Path,
    dst: str | Path,
    *,
    audio_track: int | None = None,
    audio_language: str | None = None,
    sample_rate: int = 16000,
    channels: int = 1,
    ffmpeg_binary: str = "ffmpeg",
) -> list[str]:
    """Build the ffmpeg argv. 16 kHz mono FLAC matches Whisper's input
    format — much smaller than the original AC3 with no loss for speech
    recognition."""
    if audio_language is not None:
        stream_map = f"0:a:m:language:{audio_language}"
    elif audio_track is not None:
        stream_map = f"0:a:{audio_track}"
    else:
        stream_map = "0:a:0"
    return [
        ffmpeg_binary,
        "-hide_banner",
        "-nostdin",
        "-y",
        "-i", str(source),
        "-map", stream_map,
        "-vn", "-sn",
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-c:a", "flac",
        str(dst),
    ]


def extract_audio(
    source: Path,
    output: Path,
    *,
    audio_track: int | None = None,
    audio_language: str | None = None,
    sample_rate: int = 16000,
    channels: int = 1,
    ffmpeg_binary: str = "ffmpeg",
    dry_run: bool = False,
) -> Path:
    source = Path(source)
    output = Path(output)
    cmd = build_ffmpeg_command(
        source,
        output,
        audio_track=audio_track,
        audio_language=audio_language,
        sample_rate=sample_rate,
        channels=channels,
        ffmpeg_binary=ffmpeg_binary,
    )
    log.info("Running: %s", " ".join(cmd))
    if dry_run:
        return output

    if not source.exists():
        raise FileNotFoundError(f"Input file not found: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffmpeg failed (exit {exc.returncode}). If the stream mapping "
            "was not found, run with --list-tracks to see the available "
            "audio tracks and their language tags."
        ) from exc

    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg finished but {output} is missing or empty.")
    log.info("Wrote %s (%.1f MiB)", output, output.stat().st_size / 2**20)
    return output
