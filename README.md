# transcript

Extract the German vocabulary of a DVD movie **before** watching it, or of an
EPUB book before reading it, so you can learn the words as flash cards first.

Pipeline stages, each resumable through a JSON/FLAC artifact in `work/`:

```
DVD ──[MakeMKV on Windows]────────> movie.mkv              (rip main title, keep German audio)
    ──[extract: ffmpeg]───────────> work/movie.flac        (16 kHz mono FLAC)
    ──[transcribe: faster-whisper]─┐
                                   ├> work/transcript.json (text segments + metadata)
EPUB ─[epub: spine -> paragraphs]──┘
    ──[sentences: spaCy]───────────> work/sentences.json      (every sentence of the source)
    ──[words: OpenAI, per sentence]> work/sentence-words.json (dictionary-form words per sentence)
    ──[build: match + aggregate]───┬> work/known-words.json   (what the API answered, for reference)
                                   └> work/words.json         (words to learn + example sentences)
```

A book skips the ripping and transcribing stages: `epub` writes the same text
artifact Whisper produces, so `sentences`, `words` and `build` are shared.

Word extraction is done by an LLM that sees one whole sentence at a time,
because German separable verbs make per-token analysis wrong: in "Er fängt
gerade an" the word to learn is "anfangen", which only something that reads
the sentence can put back together from "fängt ... an". The rules for which
words to ignore (names, numbers, function words, fillers) are part of the
prompt.

Every stage skips itself if its output already exists; pass `--force` to
regenerate.

## Setup (WSL2 + Nix flakes)

```bash
nix develop
```

That is all: the dev shell provides Python 3.11 and ffmpeg, then on first
entry creates a `.venv`, runs `pip install -e ".[dev]"` (faster-whisper,
spaCy, openai, pytest), and downloads the `de_core_news_lg` German spaCy model
(~570 MB). Later entries just activate the existing venv. Nix provides the
system pieces; Python packages stay in the venv because faster-whisper and
the spaCy German models don't package cleanly in nixpkgs.

On the Windows side you need [MakeMKV](https://www.makemkv.com/) for the
DVD rip itself.

Without Nix: install ffmpeg and Python ≥ 3.10 yourself, then in a venv run
`pip install -e ".[dev]"` and `python -m spacy download de_core_news_lg`.

The first `transcribe` run downloads the Whisper large-v3 model (~3 GB) into
`~/.cache/huggingface`.

### GPU acceleration (WSL2, NVIDIA)

CUDA works in WSL2 through the regular **Windows** NVIDIA driver — do not
install a Linux driver inside WSL. If the GPU is visible (`nvidia-smi`
works in WSL, `/dev/dxg` exists), the dev shell automatically:

1. installs the cuBLAS/cuDNN 9 wheels (`pip install -e ".[gpu]"`) on first
   entry, and
2. puts `/usr/lib/wsl/lib` (the driver's `libcuda`) and the wheel library
   directories on `LD_LIBRARY_PATH`.

`transcribe` then auto-selects CUDA float16 and falls back to CPU int8 with
a warning if the GPU can't be used. Expect roughly 10–20× realtime on a
recent GPU versus ~1× realtime on CPU — a feature film drops from hours to
minutes. Force a choice with `--device cuda` or `--device cpu`.

### macOS on Apple Silicon (M1/M2/...)

The same `nix develop` works on macOS (the flake covers `aarch64-darwin`),
and [MakeMKV for Mac](https://www.makemkv.com/) handles the DVD rip — a
MacBook Air needs an external USB DVD drive. Paths are just normal ones
(`~/Movies/movie.mkv`) instead of `/mnt/c/...`.

Transcription uses the M-series **GPU via Metal**: the dev shell installs
[mlx-whisper](https://pypi.org/project/mlx-whisper/) (Apple's MLX Whisper
port) on first entry, and `transcribe` auto-selects it on Apple Silicon —
several times faster than CPU inference, since faster-whisper/CTranslate2
has no Metal backend. Force a backend with `--backend mlx` or
`--backend faster-whisper`. Notes for the mlx backend: the `--device` flag
is ignored, and there is no VAD pre-filter (use `--min-count 2` in `build`
to catch stray hallucinated words).

## Stage 1 — DVD → MKV → FLAC

WSL2 does not pass the optical drive through, and player-based ripping is
unreliable, so the disc is ripped on **Windows with MakeMKV**:

1. Open MakeMKV, insert the DVD, and rip the **main title** (usually the
   longest one). In the title's tree you can untick everything except the
   German audio track to keep the file small.
2. Note where the `.mkv` lands, e.g. `C:\Users\you\Videos\movie.mkv` — from
   WSL that is `/mnt/c/Users/you/Videos/movie.mkv`.

Then, inside the Nix shell in WSL, pull the audio out with ffmpeg:

```bash
# see which audio tracks the rip contains (language, codec, channels)
transcript extract /mnt/c/Users/you/Videos/movie.mkv --list-tracks

# extract the German track to 16 kHz mono FLAC
transcript extract /mnt/c/Users/you/Videos/movie.mkv \
    --audio-language ger -o work/movie.flac
```

Options:

- `--audio-language ger` — pick the audio stream by its language tag. DVD
  rips usually tag German as `ger` (ISO 639-2/B); check with
  `--list-tracks`. Use `--audio-track N` instead when the rip has several
  German tracks (e.g. a director's commentary) or missing language tags.
  With neither option the first audio stream is used.
- `--dry-run` — print the ffmpeg command without running it.
- Output is 16 kHz mono FLAC: exactly what Whisper consumes, no ASR quality
  loss, and far smaller than the original AC3.

Whether ripping a CSS-protected disc is legal depends on your jurisdiction —
use this only on discs you own, for personal study.

## Stage 1b — EPUB → text

For a book there is nothing to rip or transcribe; the text is already there:

```bash
# see the reading order: spine index, chapter title, size
transcript epub ~/books/buch.epub --list-chapters

# extract the text (all chapters)
transcript epub ~/books/buch.epub -o work/transcript.json
```

Only the spine's XHTML documents are read, in reading order; the navigation
document, the NCX table of contents, images, `<script>` and `<style>` are
skipped. One segment is one paragraph, so the spaCy chunker gets the same
sentence-aligned units it gets from Whisper. Chapter titles come from each
document's first heading.

Options:

- `--chapters 3-20,25` — only read these spine documents (0-based indices
  from `--list-chapters`, ranges and commas allowed). Useful for skipping
  cover pages, copyright notices, forewords and indexes; `--list-chapters`
  honours the selection, so you can preview it.
- Front matter that carries no text (a cover page that is just an image)
  stays in the `--list-chapters` output with 0 characters, so the indices
  never shift.

Only DRM-free EPUBs can be read — a file with an Adobe/FairPlay
`META-INF/encryption.xml` is reported as DRM-protected rather than parsed
into gibberish. `.mobi`/`.azw` are not supported; convert them to EPUB with
[Calibre](https://calibre-ebook.com/) first.

## Stages 2–5 — audio or text → word list

```bash
# individually
transcript transcribe work/movie.flac -o work/transcript.json
transcript sentences work/transcript.json -o work/sentences.json
transcript words work/sentences.json -o work/sentence-words.json
transcript build work/sentence-words.json -o work/words.json

# or in one go — the first stage follows the source's suffix
transcript run work/movie.flac --workdir work
transcript run ~/books/buch.epub --workdir work --chapters 3-20
```

### Sentences (`work/sentences.json`)

`sentences` splits the transcript into sentences with spaCy (only the
sentence-boundary components are loaded; tagging and lemmatization moved to
the LLM) and writes the full ordered list — duplicates included, a film
repeats its lines — as an intermediary artifact. When a word in the final
list looks wrong, this file is the place to check what the LLM was actually
given.

### LLM word extraction (`work/sentence-words.json`)

`words` sends each **distinct** sentence to an OpenAI model once and asks
for every word worth learning in its dictionary form: verbs as infinitives
(separable verbs reunited), nouns in the nominative singular with their
article, adjectives in the base form. The API key comes from `.env`:

```bash
OPENAI_API_KEY=sk-your-key
```

One call per sentence means thousands of calls per film, so the stage runs
them concurrently while respecting the provider's rate limits:

- `--llm-model NAME` — OpenAI model (default: `gpt-5-mini`).
- `--llm-rpm N` — client-side requests-per-minute cap (default: 60). Match
  it to your [API tier's rate limit](https://platform.openai.com/docs/guides/rate-limits);
  the SDK's own retry-with-backoff still catches any 429 that slips through.
- `--llm-concurrency N` — how many requests are in flight at once
  (default: 8).

Every finished sentence is appended to a `sentence-words.json.partial.jsonl`
checkpoint, so an interrupted or failed run **resumes** where it stopped
instead of paying for the finished sentences again; the checkpoint is folded
into the artifact and deleted when the stage completes. The artifact keeps
the per-sentence answers, so any entry in the final word list can be traced
back to the exact sentence and model output:

```json
{
  "schema_version": 1,
  "llm_model": "gpt-5-mini",
  "total_sentences": 1180,
  "sentences": [
    {
      "text": "Er fängt gerade an.",
      "count": 2,
      "words": [{"lemma": "anfangen", "word_type": "verb", "article": null}]
    }
  ]
}
```

### Known words

The words you already know come from a REST API. Configure it in a `.env`
file in the project root (git-ignored; see `.env.example`):

```bash
KNOWN_WORDS_API_URL=https://example.com/api/known-words
KNOWN_WORDS_API_TOKEN=your-secret-token
```

The `build` stage sends `GET $KNOWN_WORDS_API_URL` with
`Authorization: Bearer $KNOWN_WORDS_API_TOKEN` and accepts any of these
JSON response shapes:

```json
["laufen", "Haus"]
{"words": ["laufen", "Haus"]}
[{"word": "laufen"}, {"lemma": "Haus"}]
```

Entries are flash-card headwords, so the notation a vocabulary list uses is
understood and expanded to the lemma forms the LLM produces:

| Entry                    | also matches the lemma           |
| ------------------------ | -------------------------------- |
| `der Kaffee`             | `Kaffee` (any of der/die/das/ein) |
| `ausruhen (sich)`, `sich kümmern` | `ausruhen`, `kümmern`   |
| `gern(e)`                | `gern`, `gerne`                  |
| `auf jeden/keinen Fall`  | `auf jeden Fall`, `auf keinen Fall` |
| `nächst-`                | `nächst`                         |
| `der, die, das`          | `der`, `die`, `das`              |

The entry itself always stays a key, so an endpoint that sends bare lemmas
behaves exactly as before. Matching is case-insensitive and treats `ß` and
`ss` as equal; it ignores part of speech, so `der Laden` also covers the
verb `laden`. Multi-word entries match only as a whole — `auf jeden Fall`
does not make `Fall` known. If `KNOWN_WORDS_API_URL` is unset, the build
runs with a warning and no words are excluded as known.

`KNOWN_WORDS_API_URL` must be the full path of the endpoint, not the site
root — a root URL usually answers `200 text/html` with the web app's page,
which `build` reports as *"did not return JSON but text/html"*.

To check the endpoint without running the pipeline:

```bash
./scripts/check-known-words.sh                 # uses .env
./scripts/check-known-words.sh https://example.com/api/known-words
```

It prints the status, content type and body (never the token), and counts
the words when the response is a valid list.

Each `build` also saves what the API answered next to its word list, as
`work/known-words.json` (`--known-words-output PATH` to put it elsewhere):

```json
{
  "schema_version": 1,
  "fetched_at": "2026-08-21T14:03:57Z",
  "source": "https://example.com/api/known-words",
  "total_words": 1843,
  "words": ["abend", "der kaffee", "laufen"],
  "total_match_keys": 2571,
  "match_keys": ["abend", "der kaffee", "kaffee", "laufen"]
}
```

`words` is what the endpoint answered; `match_keys` is what those entries
expand to and what the lemmas are actually compared against. Both are stored
normalized — casefolded, `ß` as `ss`, whitespace collapsed. When a word you
thought you knew turns up in `words.json`, grep it in `match_keys`: if it is
missing, its entry is written in a notation the expansion does not cover.
The file is rewritten on every build, and is not written at all when
`KNOWN_WORDS_API_URL` is unset.

### Output format (`work/words.json`)

```json
{
  "schema_version": 1,
  "total_words": 412,
  "words": [
    {
      "lemma": "anfangen",
      "word_type": "verb",
      "article": null,
      "count": 12,
      "sentences": [
        "Er fängt gerade an.",
        "Wir haben gestern angefangen."
      ]
    }
  ]
}
```

One entry per (lemma, word type), sorted by how often it occurs in the film
or book, with every distinct sentence it appeared in — ready to turn into
flash cards (front: lemma + word type + article for nouns, back: example
sentences). A sentence counts every time the source contains it, even though
the LLM saw it only once.

## How "not worth learning" words are excluded

The rules live in the LLM prompt (see `SYSTEM_PROMPT` in
`src/transcript/llm.py`) — the model is asked to skip:

- **Proper names** of people, places, brands and characters.
- **Numbers, dates, punctuation, symbols.**
- **Function words** that are learned as grammar, not vocabulary: articles,
  pronouns, prepositions, conjunctions, question words, auxiliary/modal
  verbs — including the separated prefix of a separable verb, which belongs
  inside the verb's infinitive instead.
- **Interjections and fillers** ("äh", "hm", "na", "hallo").
- **Absolute beginner (A1) words** every learner meets in the first weeks
  ("gut", "machen", "gehen").
- Anything garbled or not German.

After the LLM, `build` additionally drops:

- Your known words (see above).
- `--min-count 2` — words heard only once; in a two-hour film these are
  often Whisper mis-transcriptions or too rare to matter. For a book, where
  the text is exact, `--min-count 1` (the default) is usually right.

## Known limitations

- Word extraction quality is the LLM's judgement; a rule in the prompt is a
  request, not a guarantee. The per-sentence answers stay inspectable in
  `work/sentence-words.json` when something looks off.
- One API call per distinct sentence costs real money on a large source;
  `--llm-model` picks the trade-off between price and judgement.
- Whisper can hallucinate short phrases during music or silence. The
  built-in VAD filter and disabled text conditioning suppress most of it;
  `--min-count 2` catches stragglers.
- EPUB is a loose format: chapter titles are guessed from the first heading
  of each document, and books whose chapters are split across many small
  files (or merged into one huge file) list accordingly. Check
  `--list-chapters` before selecting with `--chapters`.
- Footnotes, page headers and similar furniture are part of the text and are
  analyzed like prose.

## Development

```bash
pytest          # unit tests; no DVD, EPUB, ffmpeg, Whisper model or spaCy model needed
```

The spaCy-dependent smoke test auto-skips when `de_core_news_lg` is not
installed.
