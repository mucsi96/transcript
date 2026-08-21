"""Stage 2: transcribe FLAC audio with Whisper large-v3.

Two backends, auto-selected:

- faster-whisper (CTranslate2): Linux/WSL and Intel Macs. CUDA float16 when
  a GPU is available, CPU int8 otherwise.
- mlx-whisper (Apple MLX): Apple Silicon Macs. Runs on the M-series GPU via
  Metal — several times faster than CPU inference; CTranslate2 has no Metal
  backend.
"""

from __future__ import annotations

import logging
import platform
import sys
import time
from pathlib import Path

from .artifacts import SCHEMA_VERSION, load_json, save_json, should_skip

log = logging.getLogger(__name__)

DEFAULT_MODEL = "large-v3"
PROGRESS_INTERVAL_S = 30.0


def format_progress(done: float, total: float, elapsed: float) -> str:
    """Human-readable progress line: percent of audio transcribed, realtime
    speed factor, and ETA."""
    speed = done / elapsed if elapsed > 0 else 0.0
    pct = 100.0 * done / total if total else 0.0
    if speed > 0 and total:
        eta_min = (total - done) / speed / 60.0
        eta = f"~{eta_min:.0f} min left"
    else:
        eta = "ETA unknown"
    return f"{pct:.1f}% ({done:.0f}/{total:.0f} s of audio, {speed:.2f}x realtime, {eta})"


def pick_backend(prefer: str = "auto") -> str:
    """Resolve the transcription backend. On Apple Silicon prefer
    mlx-whisper (Metal GPU) when installed; faster-whisper elsewhere."""
    if prefer != "auto":
        return prefer
    if sys.platform == "darwin" and platform.machine() == "arm64":
        try:
            import mlx_whisper  # noqa: F401

            return "mlx"
        except ImportError:
            log.warning(
                "Apple Silicon detected but mlx-whisper is not installed; "
                'falling back to CPU. Install it with: pip install -e ".[mlx]"'
            )
    return "faster-whisper"


def pick_device(prefer: str = "auto") -> tuple[str, str]:
    """faster-whisper only — resolve (device, compute_type): CUDA ->
    float16, CPU -> int8."""
    if prefer == "cpu":
        return "cpu", "int8"
    if prefer == "cuda":
        return "cuda", "float16"
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def mlx_repo(model_size: str) -> str:
    """Map a Whisper model size to the mlx-community HF repo; a value
    containing '/' is treated as a full repo path already."""
    if "/" in model_size:
        return model_size
    return f"mlx-community/whisper-{model_size}-mlx"


def _transcribe_faster_whisper(
    audio: Path,
    *,
    model_size: str,
    device: str,
    language: str,
    beam_size: int,
    vad: bool,
) -> tuple[list[dict], float, str, dict]:
    from faster_whisper import WhisperModel

    dev, compute_type = pick_device(device)
    log.info("Loading Whisper %s on %s (%s) ...", model_size, dev, compute_type)
    try:
        model = WhisperModel(model_size, device=dev, compute_type=compute_type)
    except (RuntimeError, ValueError) as exc:
        if dev == "cpu":
            raise
        # CUDA present but unusable (missing cuDNN, driver mismatch, ...)
        log.warning("GPU load failed (%s); falling back to CPU int8", exc)
        dev, compute_type = "cpu", "int8"
        model = WhisperModel(model_size, device=dev, compute_type=compute_type)

    # vad_filter skips music/silence and condition_on_previous_text=False
    # avoids repetition loops — both curb the classic Whisper hallucinations
    # on German media ("Untertitelung der Amara.org-Community" etc.).
    segments, info = model.transcribe(
        str(audio),
        language=language,
        beam_size=beam_size,
        vad_filter=vad,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,
    )

    log.info("Transcribing %.0f s (%.1f h) of audio ...", info.duration, info.duration / 3600)
    seg_records = []
    start = time.monotonic()
    last_progress = start
    # segments is a lazy generator: transcription happens while iterating.
    for seg in segments:
        seg_records.append(
            {"id": seg.id, "start": round(seg.start, 2), "end": round(seg.end, 2), "text": seg.text.strip()}
        )
        now = time.monotonic()
        if now - last_progress >= PROGRESS_INTERVAL_S:
            last_progress = now
            log.info("%s", format_progress(seg.end, info.duration, now - start))
    meta = {"backend": "faster-whisper", "device": dev, "compute_type": compute_type}
    return seg_records, info.duration, info.language, meta


def _transcribe_mlx(
    audio: Path,
    *,
    model_size: str,
    language: str,
) -> tuple[list[dict], float, str, dict]:
    import mlx_whisper

    repo = mlx_repo(model_size)
    log.info("Transcribing with %s on the Apple GPU (Metal) ...", repo)
    # verbose=False shows a progress bar; condition_on_previous_text=False
    # curbs repetition-loop hallucinations (no VAD in this backend).
    result = mlx_whisper.transcribe(
        str(audio),
        path_or_hf_repo=repo,
        language=language,
        condition_on_previous_text=False,
        verbose=False,
    )
    seg_records = [
        {
            "id": i,
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip(),
        }
        for i, seg in enumerate(result["segments"])
    ]
    duration = seg_records[-1]["end"] if seg_records else 0.0
    meta = {"backend": "mlx", "device": "metal", "compute_type": "float16"}
    return seg_records, duration, result.get("language", language), meta


def transcribe(
    audio: Path,
    output: Path,
    *,
    model_size: str = DEFAULT_MODEL,
    backend: str = "auto",
    device: str = "auto",
    language: str = "de",
    beam_size: int = 5,
    vad: bool = True,
    force: bool = False,
) -> dict:
    if should_skip(output, force):
        return load_json(output)

    audio = Path(audio)
    if not audio.exists():
        raise FileNotFoundError(f"Audio file not found: {audio}")

    start = time.monotonic()
    resolved = pick_backend(backend)
    if resolved == "mlx":
        seg_records, duration, detected_language, meta = _transcribe_mlx(
            audio, model_size=model_size, language=language
        )
    else:
        seg_records, duration, detected_language, meta = _transcribe_faster_whisper(
            audio,
            model_size=model_size,
            device=device,
            language=language,
            beam_size=beam_size,
            vad=vad,
        )
    elapsed = time.monotonic() - start
    log.info(
        "Transcription done: %d segments, %.0f s of audio in %.0f min",
        len(seg_records), duration, elapsed / 60,
    )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_type": "audio",
        "audio_file": str(audio),
        "model": model_size,
        **meta,
        "language": detected_language,
        "duration": round(duration, 2),
        "text": " ".join(s["text"] for s in seg_records),
        "segments": seg_records,
    }
    save_json(output, payload)
    return payload
