"""Speechmatics real-time transcription, run on a background thread."""

from __future__ import annotations

import threading
from typing import Callable

import speechmatics
from speechmatics.models import (
    AudioSettings,
    ConnectionSettings,
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
    """Owns the Speechmatics websocket session for one recording."""

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
        self._client: speechmatics.client.WebsocketClient | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="speechmatics", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        settings = ConnectionSettings(url=self._url, auth_token=self._api_key)
        client = speechmatics.client.WebsocketClient(settings)
        self._client = client

        def on_partial(msg: dict) -> None:
            text = msg.get("metadata", {}).get("transcript", "").strip()
            if text:
                self._on_event({"type": "partial", "text": text})

        def on_final(msg: dict) -> None:
            text = msg.get("metadata", {}).get("transcript", "").strip()
            words = _extract_words(msg)
            if text or words:
                self._on_event({"type": "final", "text": text, "words": words})

        client.add_event_handler(ServerMessageType.AddPartialTranscript, on_partial)
        client.add_event_handler(ServerMessageType.AddTranscript, on_final)

        transcription_config = TranscriptionConfig(
            language=self._language,
            enable_partials=True,
            max_delay=2.0,
            operating_point="enhanced",
        )
        audio_settings = AudioSettings(
            encoding="pcm_s16le",
            sample_rate=self._sample_rate,
        )

        try:
            self._on_event({"type": "info", "message": "Connected to Speechmatics."})
            client.run_synchronously(self._stream, transcription_config, audio_settings)
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            self._on_event({"type": "error", "message": f"Transcription error: {exc}"})
        finally:
            self._on_event({"type": "info", "message": "Transcription stopped."})

    def stop(self) -> None:
        """Signal end-of-stream; the client finishes once audio drains."""
        self._stream.close()
        if self._client is not None:
            try:
                self._client.stop()
            except Exception:
                pass

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)
