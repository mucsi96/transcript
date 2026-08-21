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
    ──[analyze: spaCy]─────────────> work/analysis.json    (sentences, lemmas, POS, NER)
    ──[build: match + filter]──────> work/words.json       (words to learn + example sentences)
```

A book skips the ripping and transcribing stages: `epub` writes the same text
artifact Whisper produces, so `analyze` and `build` are shared.

Every stage skips itself if its output already exists; pass `--force` to
regenerate.

## Setup (WSL2 + Nix flakes)

```bash
nix develop
```

That is all: the dev shell provides Python 3.11 and ffmpeg, then on first
entry creates a `.venv`, runs `pip install -e ".[dev]"` (faster-whisper,
spaCy, pytest), and downloads the `de_core_news_lg` German spaCy model
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

## Stages 2–4 — audio or text → word list

```bash
# individually
transcript transcribe work/movie.flac -o work/transcript.json
transcript analyze work/transcript.json -o work/analysis.json
transcript build work/analysis.json -o work/words.json

# or in one go — the first stage follows the source's suffix
transcript run work/movie.flac --workdir work
transcript run ~/books/buch.epub --workdir work --chapters 3-20
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

Entries should be lemmas (dictionary forms). Matching is case-insensitive
and treats `ß` and `ss` as equal. If `KNOWN_WORDS_API_URL` is unset, the
build runs with a warning and no words are excluded as known.

### Output format (`work/words.json`)

```json
{
  "schema_version": 1,
  "total_words": 412,
  "words": [
    {
      "lemma": "laufen",
      "pos": "VERB",
      "word_type": "verb",
      "count": 12,
      "sentences": [
        "Der Hund läuft schnell.",
        "Wir sind zum Bahnhof gelaufen."
      ]
    }
  ]
}
```

One entry per (lemma, word type), sorted by how often it occurs in the film
or book, with every distinct sentence it appeared in — ready to turn into
flash cards (front: lemma + word type, back: example sentences).

## How "not worth learning" words are excluded

On by default:

- **Proper nouns** (`PROPN` POS tag) — person and place names are not
  vocabulary.
- **Named entities** tagged `PER`/`LOC`/`ORG` by spaCy's NER — catches names
  the POS tagger missed (multi-word names, surnames tagged as nouns).
  `MISC` entities are *kept*, because German NER files learnable nationality
  adjectives like "deutsch" under MISC (`--drop-ent MISC` to exclude them).
- Punctuation, symbols, whitespace, numerals (`--keep-pos NUM` if you want
  "zwei" etc.), non-alphabetic tokens ("2:30"), single characters.
- Your known words.

Optional:

- `--min-count 2` — drop words heard only once; in a two-hour film these are
  often Whisper mis-transcriptions or too rare to matter. For a book, where
  the text is exact, `--min-count 1` (the default) is usually right.
- `--drop-stopwords` — drop function words (der/und/aber). Off by default;
  putting them in your known-words list once is the cleaner fix.

Ideas for further filtering (extension points, not implemented — see
`src/transcript/filters.py`):

- **Frequency filter**: `wordfreq.zipf_frequency(lemma, "de")` to drop
  ultra-rare words (likely ASR errors) or rank cards most-frequent-first.
- **LLM review pass**: send the candidate list to GPT-5 with "which of these
  are worth a flash card for a B1 learner?" as a final polish.

## Known limitations

- spaCy's German lemmatizer is statistical; separable verbs come apart:
  "Er fängt an" yields the lemma `fangen` plus the particle `an`, not
  `anfangen`. Occasional wrong lemmas are possible.
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
