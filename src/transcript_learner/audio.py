"""System-audio capture via PortAudio/PulseAudio (sounddevice)."""

from __future__ import annotations

import queue
import threading
from typing import Optional

import sounddevice as sd

# Speechmatics real-time expects 16-bit little-endian PCM.
DTYPE = "int16"
CHANNELS = 1


class AudioStream:
    """A blocking, `read(n)`-style byte stream fed by a sounddevice callback.

    Speechmatics' WebsocketClient pulls audio by calling ``read(n)`` until it
    returns an empty bytes object, so we buffer callback frames in a queue and
    hand them out on demand.
    """

    def __init__(self, sample_rate: int, device=None):
        self.sample_rate = sample_rate
        self.device = device
        self._q: "queue.Queue[Optional[bytes]]" = queue.Queue()
        self._buffer = b""
        self._closed = threading.Event()
        self._stream: Optional[sd.RawInputStream] = None

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        # blocksize 0 lets PortAudio pick an optimal block; ~100ms is plenty.
        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            channels=CHANNELS,
            dtype=DTYPE,
            device=self.device,
            blocksize=int(self.sample_rate * 0.1),
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, indata, frames, time_info, status):  # noqa: ANN001
        if self._closed.is_set():
            return
        # indata is a CFFI buffer for RawInputStream; copy the bytes out.
        self._q.put(bytes(indata))

    def close(self) -> None:
        self._closed.set()
        self._q.put(None)  # sentinel: wake read() so it can return b""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

    # -- stream interface used by Speechmatics -----------------------------
    def read(self, num_bytes: int) -> bytes:
        while len(self._buffer) < num_bytes:
            try:
                chunk = self._q.get(timeout=0.25)
            except queue.Empty:
                if self._closed.is_set():
                    break
                continue
            if chunk is None:  # closed sentinel
                break
            self._buffer += chunk

        data, self._buffer = self._buffer[:num_bytes], self._buffer[num_bytes:]
        return data


def list_devices() -> str:
    """Human-readable list of input-capable audio devices."""
    lines = ["Available audio input devices:", ""]
    try:
        devices = sd.query_devices()
    except Exception as exc:  # pragma: no cover - depends on host audio
        return f"Could not query audio devices: {exc}"

    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) <= 0:
            continue
        name = dev.get("name", "?")
        rate = int(dev.get("default_samplerate", 0))
        hint = "  <- likely system audio" if "monitor" in name.lower() else ""
        lines.append(f"  [{idx:>2}] {name}  ({rate} Hz){hint}")

    lines += [
        "",
        "Tip: on WSLg the '...monitor' source captures whatever is playing",
        "     (e.g. VLC). Set AUDIO_DEVICE to its index or a name substring.",
    ]
    return "\n".join(lines)
