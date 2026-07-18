"""Runtime configuration, loaded from environment / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class Config:
    speechmatics_api_key: str
    speechmatics_url: str
    language: str

    openai_api_key: str
    openai_model: str

    sample_rate: int
    audio_device: str  # substring or numeric index, empty = default

    host: str
    port: int
    output_dir: Path

    @classmethod
    def load(cls) -> "Config":
        return cls(
            speechmatics_api_key=_get("SPEECHMATICS_API_KEY"),
            speechmatics_url=_get("SPEECHMATICS_URL", "wss://eu2.rt.speechmatics.com/v2"),
            language=_get("LANGUAGE", "de"),
            openai_api_key=_get("OPENAI_API_KEY"),
            openai_model=_get("OPENAI_MODEL", "gpt-5"),
            sample_rate=int(_get("SAMPLE_RATE", "16000") or "16000"),
            audio_device=_get("AUDIO_DEVICE"),
            host=_get("HOST", "127.0.0.1"),
            port=int(_get("PORT", "8000") or "8000"),
            output_dir=Path(_get("OUTPUT_DIR", "sessions")),
        )

    def resolved_device(self):
        """Return the sounddevice device spec: int index, name substring, or None."""
        if not self.audio_device:
            return None
        if self.audio_device.isdigit():
            return int(self.audio_device)
        return self.audio_device
