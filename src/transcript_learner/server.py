"""FastAPI web app: live transcript + end-of-session word analysis."""

from __future__ import annotations

import asyncio
import json
import queue
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import analysis
from .audio import AudioStream
from .config import Config
from .session import Session
from .transcriber import Transcriber

STATIC_DIR = Path(__file__).parent / "static"


class AppState:
    """Holds the current recording session and bridges worker threads → asyncio."""

    def __init__(self, config: Config):
        self.config = config
        self.session = Session()
        self.recording = False
        self.audio: AudioStream | None = None
        self.transcriber: Transcriber | None = None

        # Worker threads push here; a background asyncio task fans out to clients.
        self.events: "queue.Queue[dict]" = queue.Queue()
        self.clients: set[WebSocket] = set()

    # -- recording control -------------------------------------------------
    def start_recording(self) -> dict:
        if self.recording:
            return {"ok": True, "message": "Already recording."}
        cfg = self.config
        if not cfg.speechmatics_api_key:
            return {"ok": False, "message": "SPEECHMATICS_API_KEY is not set."}

        self.session.reset()
        try:
            self.audio = AudioStream(cfg.sample_rate, device=cfg.resolved_device())
            self.audio.start()
        except Exception as exc:
            self.audio = None
            return {"ok": False, "message": f"Could not open audio device: {exc}"}

        self.transcriber = Transcriber(
            api_key=cfg.speechmatics_api_key,
            url=cfg.speechmatics_url,
            language=cfg.language,
            sample_rate=cfg.sample_rate,
            stream=self.audio,
            on_event=self._on_transcriber_event,
        )
        self.transcriber.start()
        self.recording = True
        return {"ok": True, "message": "Recording started."}

    def stop_recording(self) -> dict:
        if not self.recording:
            return {"ok": True, "message": "Not recording."}
        if self.transcriber is not None:
            self.transcriber.stop()
        self.recording = False
        return {"ok": True, "message": "Recording stopped."}

    def _on_transcriber_event(self, event: dict) -> None:
        """Called from the Speechmatics thread — keep it thread-safe & cheap."""
        if event.get("type") == "final":
            new_entries = self.session.add_final(event.get("text", ""), event.get("words", []))
            self.events.put(
                {
                    "type": "final",
                    "text": event.get("text", ""),
                    "new_words": [
                        {"word": e.word, "surface": e.surface, "context": e.context}
                        for e in new_entries
                    ],
                    "stats": {"unique_words": len(self.session.words)},
                }
            )
        elif event.get("type") == "partial":
            self.events.put({"type": "partial", "text": event.get("text", "")})
        else:  # info / error
            self.events.put(event)


app = FastAPI(title="German Transcript Learner")
_state: AppState | None = None


def get_state() -> AppState:
    assert _state is not None, "AppState not initialized"
    return _state


def create_app(config: Config) -> FastAPI:
    global _state
    _state = AppState(config)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


@app.on_event("startup")
async def _startup() -> None:
    asyncio.create_task(_broadcast_loop())


async def _broadcast_loop() -> None:
    """Drain the thread-safe event queue and push to all websocket clients."""
    state = get_state()
    loop = asyncio.get_running_loop()
    while True:
        try:
            event = await loop.run_in_executor(None, state.events.get, True, 0.5)
        except queue.Empty:
            continue
        except Exception:
            await asyncio.sleep(0.1)
            continue
        dead = []
        for ws in list(state.clients):
            try:
                await ws.send_text(json.dumps(event, ensure_ascii=False))
            except Exception:
                dead.append(ws)
        for ws in dead:
            state.clients.discard(ws)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/api/status")
async def status() -> JSONResponse:
    state = get_state()
    cfg = state.config
    return JSONResponse(
        {
            "recording": state.recording,
            "language": cfg.language,
            "model": cfg.openai_model,
            "has_speechmatics_key": bool(cfg.speechmatics_api_key),
            "has_openai_key": bool(cfg.openai_api_key),
            "session": state.session.snapshot(),
        }
    )


@app.post("/api/start")
async def api_start() -> JSONResponse:
    return JSONResponse(get_state().start_recording())


@app.post("/api/stop")
async def api_stop() -> JSONResponse:
    return JSONResponse(get_state().stop_recording())


@app.get("/api/words")
async def api_words() -> JSONResponse:
    return JSONResponse(get_state().session.snapshot())


@app.post("/api/analyze")
async def api_analyze(payload: dict) -> JSONResponse:
    state = get_state()
    cfg = state.config
    known_raw = str(payload.get("known_words", ""))

    entries = state.session.unique_entries()
    if not entries:
        return JSONResponse({"ok": False, "message": "No words recorded yet."}, status_code=400)

    # spaCy + OpenAI are blocking; run off the event loop.
    result = await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: analysis.analyze(
            entries,
            known_raw,
            api_key=cfg.openai_api_key,
            model=cfg.openai_model,
        ),
    )

    timestamp = datetime.now(timezone.utc)
    result["generated_at"] = timestamp.isoformat()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"session-{timestamp.strftime('%Y%m%d-%H%M%S')}.json"
    out_path = cfg.output_dir / filename
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    result["ok"] = True
    result["saved_to"] = str(out_path)
    return JSONResponse(result)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    state = get_state()
    await ws.accept()
    state.clients.add(ws)
    # Send current snapshot so a late-joining client is in sync.
    await ws.send_text(json.dumps({"type": "snapshot", "session": state.session.snapshot()}))
    try:
        while True:
            await ws.receive_text()  # we don't expect client messages; keeps it open
    except WebSocketDisconnect:
        pass
    finally:
        state.clients.discard(ws)
