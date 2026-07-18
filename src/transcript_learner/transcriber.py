"""Speechmatics real-time transcription (speechmatics-rt), run on a background thread."""

from __future__ import annotations

import asyncio
import threading
from typing import Callable

from speechmatics.rt import (
    AsyncClient,
    AudioEncoding,
    AudioFormat,
    OperatingPoint,
    ServerMessageType,
    TranscriptionConfig,
)

from .audio import AudioStream

# Event payloads pushed to the callback:
#   {"type": "partial", "text": str}
#   {"type": "final",   "text": str, "words": [str, ...]}
#   {"type": "error",   "message": str}
#   {"type": "info",    "message": str}
EventCallback = Callable[[dict], None]


def _extract_words(msg: dict) -> list[str]:
    """Pull the individual word tokens out of an AddTranscript message."""
    words: list[str] = []
    for result in msg.get("results", []):
        if result.get("type") != "word":
            continue
        alternatives = result.get("alternatives") or []
        if alternatives:
            content = alternatives[0].get("content", "").strip()
            if content:
                words.append(content)
    return words


class Transcriber:
    """Owns one Speechmatics real-time session, driven on its own event loop."""

    def __init__(
        self,
        *,
        api_key: str,
        url: str,
        language: str,
        sample_rate: int,
        stream: AudioStream,
        on_event: EventCallback,
    ):
        self._api_key = api_key
        self._url = url
        self._language = language
        self._sample_rate = sample_rate
        self._stream = stream
        self._on_event = on_event
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="speechmatics", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            asyncio.run(self._run_async())
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            self._on_event({"type": "error", "message": f"Transcription error: {exc}"})
        finally:
            self._on_event({"type": "info", "message": "Transcription stopped."})

    async def _run_async(self) -> None:
        def on_partial(msg: dict) -> None:
            text = msg.get("metadata", {}).get("transcript", "").strip()
            if text:
                self._on_event({"type": "partial", "text": text})

        def on_final(msg: dict) -> None:
            text = msg.get("metadata", {}).get("transcript", "").strip()
            words = _extract_words(msg)
            if text or words:
                self._on_event({"type": "final", "text": text, "words": words})

        def on_error(msg: dict) -> None:
            detail = msg.get("reason") or msg.get("message") or str(msg)
            self._on_event({"type": "error", "message": f"Speechmatics error: {detail}"})

        transcription_config = TranscriptionConfig(
            language=self._language,
            enable_partials=True,
            max_delay=2.0,
            operating_point=OperatingPoint.ENHANCED,
        )
        audio_format = AudioFormat(
            encoding=AudioEncoding.PCM_S16LE,
            sample_rate=self._sample_rate,
        )

        async with AsyncClient(api_key=self._api_key, url=self._url) as client:
            client.on(ServerMessageType.ADD_PARTIAL_TRANSCRIPT, on_partial)
            client.on(ServerMessageType.ADD_TRANSCRIPT, on_final)
            client.on(ServerMessageType.ERROR, on_error)

            self._on_event({"type": "info", "message": "Connected to Speechmatics."})
            # `transcribe` reads self._stream.read() (blocking) in an executor and
            # returns once the stream hits EOF — which stop() triggers by closing it.
            await client.transcribe(
                self._stream,
                transcription_config=transcription_config,
                audio_format=audio_format,
            )

    def stop(self) -> None:
        """Signal end-of-stream; `transcribe` finishes once the audio drains."""
        self._stream.close()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)
