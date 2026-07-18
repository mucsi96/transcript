# German Transcript Learner

A small local web app for learning German from cartoons (or any German audio).

Play a DVD in VLC, let the app transcribe the **system audio** live with
[Speechmatics](https://www.speechmatics.com/), watch the transcript and the
growing list of unique words in your browser, then — when the session ends —
hand it your list of already-known words and it produces a **JSON file of the
words actually worth learning**, each with the sentence it appeared in.

## How it works

```
 VLC (DVD) ──► system audio ──► PulseAudio monitor ──► sounddevice
                                                          │  16 kHz PCM
                                                          ▼
                                              Speechmatics real-time (de)
                                                          │  transcript + words
                                                          ▼
                              FastAPI + WebSocket  ──►  browser UI (live)
                                                          │  "finish session"
                                                          ▼
              known-word matching → lemmatization (spaCy) → GPT-5 filter
                                                          │
                                                          ▼
                                          sessions/session-*.json
```

The end-of-session filtering runs in three stages, cheapest first:

1. **Exact match** — drop recorded words that are already in your known list.
2. **Lemma match** — spaCy's German model reduces both sides to dictionary
   form, so `spielte` counts as known if you know `spielen`.
3. **AI judgement (GPT-5)** — the remainder is sent to OpenAI, which drops
   proper nouns, character names, interjections, recognition errors and
   inflected duplicates, and returns the clean base form for each keeper.

## Requirements

- WSL2 (WSLg provides the PulseAudio audio server) or any Linux desktop.
- [Nix with flakes enabled](https://nixos.org/download).
- A **Speechmatics** API key (real-time transcription).
- An **OpenAI** API key (for the GPT-5 filtering step; optional — without it
  the words are kept unfiltered).

## Setup

```bash
# 1. Enter the reproducible dev shell (creates ./.venv, installs deps and the
#    spaCy German model on first run).
nix develop

# 2. Configure keys.
cp .env.example .env
$EDITOR .env        # set SPEECHMATICS_API_KEY and OPENAI_API_KEY

# 3. Find the audio device that captures what VLC is playing.
python -m transcript_learner --list-devices
#    Look for a "...monitor" source and put its index or a name substring
#    into AUDIO_DEVICE in .env. Leave empty to use the default input.
```

### Capturing system audio on WSL

WSLg exposes audio through PulseAudio. The source that carries *what is
playing* (rather than a microphone) is the **monitor** of the output sink.
You can confirm it exists with:

```bash
pactl list short sources        # look for a name ending in ".monitor"
```

Set `AUDIO_DEVICE` in `.env` to that monitor device's index or a substring of
its name (e.g. `monitor`). Make sure VLC's audio is actually routed through
WSL's PulseAudio (play something and watch `pactl list short sink-inputs`).

## Run

```bash
nix develop            # if not already in the shell
transcript-learner     # or: python -m transcript_learner
```

Open <http://127.0.0.1:8000>, then:

1. Press **Start recording** and play your DVD in VLC.
2. Watch the live transcript and the unique-word list fill up.
3. Press **Finish session → analyze**.
4. Paste the words you already know (comma/space/newline separated) and press
   **Analyze**.
5. Review the "words worth learning" list and **Download JSON** (a copy is also
   written to `sessions/`).

## Output format

`sessions/session-YYYYMMDD-HHMMSS.json`:

```json
{
  "generated_at": "2026-07-18T20:15:00+00:00",
  "summary": {
    "total_unique": 214,
    "already_known_exact": 88,
    "already_known_lemma": 41,
    "ai_rejected": 33,
    "to_learn": 52,
    "model": "gpt-5",
    "lemmatizer": "de_core_news_sm"
  },
  "words_to_learn": [
    {
      "word": "verstecken",
      "surface": "versteckt",
      "lemma": "verstecken",
      "context": "Der Hund hat sich hinter dem Baum versteckt.",
      "count": 3,
      "reason": "common reflexive verb, useful for a learner"
    }
  ],
  "already_known_exact": ["..."],
  "already_known_lemma": ["..."],
  "ai_rejected": [{ "word": "...", "reason": "..." }]
}
```

## Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `SPEECHMATICS_API_KEY` | — | Speechmatics real-time key (required to record) |
| `SPEECHMATICS_URL` | `wss://eu2.rt.speechmatics.com/v2` | RT endpoint (EU/US) |
| `LANGUAGE` | `de` | Transcription language |
| `OPENAI_API_KEY` | — | OpenAI key for the filtering step |
| `OPENAI_MODEL` | `gpt-5` | Model used for filtering |
| `SAMPLE_RATE` | `16000` | Capture sample rate (Hz) |
| `AUDIO_DEVICE` | (default input) | Device index or name substring |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | Web server bind |
| `OUTPUT_DIR` | `sessions` | Where result JSON is written |

## Notes

- Without `nix`, you can run it manually: `pip install -e .`,
  `python -m spacy download de_core_news_sm`, then `python -m transcript_learner`.
  You'll need PortAudio and PulseAudio installed on the system.
- The AI step fails **open**: any error keeps the words rather than losing your
  session data.
