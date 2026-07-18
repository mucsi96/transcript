"""Stage 1: extract the audio track from a DVD to FLAC using VLC.

Works both with a native VLC (cvlc on Linux / the Nix dev shell) and with the
Windows VLC invoked from WSL2. WSL2 does not pass the optical drive through
(no /dev/sr0), so on WSL the practical route is:

    transcript extract --dvd "D:" --title 1 --audio-language deu \
        --vlc-binary "/mnt/c/Program Files/VideoLAN/VLC/vlc.exe" \
        -o work/movie.flac

When the binary ends in .exe the output path is converted with `wslpath -w`
so vlc.exe can write it. The --dvd value is passed through as-is, so it can
be a device (/dev/sr0), a Windows drive letter (D:), an ISO file, or a
VIDEO_TS directory.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def is_windows_binary(vlc_binary: str) -> bool:
    return vlc_binary.lower().endswith(".exe")


def build_dvd_mrl(dvd: str, title: int | None, chapter: int | None = None) -> str:
    """dvdsimple:// MRL: skips DVD menus and plays the given title directly."""
    source = dvd
    if re.fullmatch(r"[A-Za-z]:", source):
        source += "\\"
    mrl = f"dvdsimple://{source}"
    if title is not None:
        mrl += f"#{title}"
        if chapter is not None:
            mrl += f":{chapter}"
    return mrl


def build_vlc_command(
    dvd: str,
    title: int | None,
    dst: str,
    *,
    audio_track: int | None = None,
    audio_language: str | None = None,
    sample_rate: int = 16000,
    channels: int = 1,
    vlc_binary: str = "cvlc",
) -> list[str]:
    """Build the VLC argv. 16 kHz mono FLAC matches Whisper's input format —
    much smaller than the original AC3 with no loss for speech recognition."""
    cmd = [vlc_binary, "--intf", "dummy", "--no-video", build_dvd_mrl(dvd, title)]
    if audio_language is not None:
        cmd.append(f"--audio-language={audio_language}")
    if audio_track is not None:
        cmd.append(f"--audio-track={audio_track}")
    sout = (
        f"#transcode{{acodec=flac,channels={channels},samplerate={sample_rate}}}"
        f":std{{access=file,mux=raw,dst={dst}}}"
    )
    cmd.append(f"--sout={sout}")
    # Without vlc://quit VLC keeps running after the rip finishes.
    cmd.append("vlc://quit")
    return cmd


def to_vlc_dst(output: Path, vlc_binary: str) -> str:
    """Resolve the --sout dst for the given VLC binary. For Windows VLC run
    from WSL, translate the WSL path to a Windows path."""
    output = Path(output).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not is_windows_binary(vlc_binary):
        return str(output)
    # wslpath -w needs an existing path on some WSL versions.
    output.touch(exist_ok=True)
    result = subprocess.run(
        ["wslpath", "-w", str(output)], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def extract_audio(
    dvd: str,
    title: int | None,
    output: Path,
    *,
    audio_track: int | None = None,
    audio_language: str | None = None,
    sample_rate: int = 16000,
    channels: int = 1,
    vlc_binary: str = "cvlc",
    dry_run: bool = False,
) -> Path:
    output = Path(output)
    dst = to_vlc_dst(output, vlc_binary)
    cmd = build_vlc_command(
        dvd,
        title,
        dst,
        audio_track=audio_track,
        audio_language=audio_language,
        sample_rate=sample_rate,
        channels=channels,
        vlc_binary=vlc_binary,
    )
    log.info("Running: %s", " ".join(cmd))
    if dry_run:
        return output

    subprocess.run(cmd, check=True)

    # VLC exits 0 even when the title/track selection produced nothing.
    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError(
            f"VLC finished but {output} is missing or empty. Check the DVD "
            "title number (try `lsdvd` or open the disc in the VLC GUI; the "
            "longest title is usually the main feature) and the audio "
            "track/language selection."
        )
    log.info("Wrote %s (%.1f MiB)", output, output.stat().st_size / 2**20)
    return output
