"""Stage 2: transcribe FLAC audio with Whisper large-v3 via faster-whisper."""

from __future__ import annotations

import logging
from pathlib import Path

from .artifacts import SCHEMA_VERSION, load_json, save_json, should_skip

log = logging.getLogger(__name__)

DEFAULT_MODEL = "large-v3"


def pick_device(prefer: str = "auto") -> tuple[str, str]:
    """Resolve (device, compute_type): CUDA -> float16, CPU -> int8."""
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


def transcribe(
    audio: Path,
    output: Path,
    *,
    model_size: str = DEFAULT_MODEL,
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

    seg_records = []
    # segments is a lazy generator: transcription happens while iterating.
    for seg in segments:
        seg_records.append(
            {"id": seg.id, "start": round(seg.start, 2), "end": round(seg.end, 2), "text": seg.text.strip()}
        )
        if len(seg_records) % 50 == 0:
            log.info("... %d segments, at %.0f s / %.0f s", len(seg_records), seg.end, info.duration)
    log.info("Transcription done: %d segments, %.0f s of audio", len(seg_records), info.duration)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "audio_file": str(audio),
        "model": model_size,
        "device": dev,
        "compute_type": compute_type,
        "language": info.language,
        "duration": round(info.duration, 2),
        "text": " ".join(s["text"] for s in seg_records),
        "segments": seg_records,
    }
    save_json(output, payload)
    return payload
