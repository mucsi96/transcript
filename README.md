# transcript

Extract the German vocabulary of a DVD movie **before** watching it, so you
can learn the words as flash cards first.

Pipeline stages, each resumable through a JSON/FLAC artifact in `work/`:

```
DVD ──[extract: VLC]──────────────> work/movie.flac        (16 kHz mono FLAC)
    ──[transcribe: faster-whisper]─> work/transcript.json  (Whisper large-v3, German)
    ──[analyze: spaCy]─────────────> work/analysis.json    (sentences, lemmas, POS, NER)
    ──[build: match + filter]──────> work/words.json       (words to learn + example sentences)
```

Every stage skips itself if its output already exists; pass `--force` to
regenerate.

## Setup (WSL2 + Nix flakes)

```bash
nix develop
```

That is all: the dev shell provides Python 3.11, VLC (with libdvdcss), and
`lsdvd`, then on first entry creates a `.venv`, runs `pip install -e ".[dev]"`
(faster-whisper, spaCy, pytest), and downloads the `de_core_news_lg` German
spaCy model (~570 MB). Later entries just activate the existing venv. Nix
provides the system pieces; Python packages stay in the venv because
faster-whisper and the spaCy German models don't package cleanly in nixpkgs.

Without Nix: install VLC and Python ≥ 3.10 yourself, then in a venv run
`pip install -e ".[dev]"` and `python -m spacy download de_core_news_lg`.

The first `transcribe` run downloads the Whisper large-v3 model (~3 GB) into
`~/.cache/huggingface`. With an NVIDIA GPU it runs in float16 (needs CUDA 12
+ cuDNN 9); otherwise it falls back to CPU int8 automatically — expect
roughly real-time speed on a modern CPU for a feature film.

## Stage 1 — DVD → FLAC

WSL2 does not pass the optical drive through (there is no `/dev/sr0`), so on
WSL use the **Windows** VLC against the Windows drive letter:

```bash
transcript extract --dvd "D:" --title 1 --audio-language deu \
    --vlc-binary "/mnt/c/Program Files/VideoLAN/VLC/vlc.exe" \
    -o work/movie.flac
```

The output path is converted with `wslpath -w` automatically so `vlc.exe`
can write it. Writing into the Linux filesystem goes through the `\\wsl$`
share; if that is slow, output to `/mnt/c/...` instead.

On native Linux (or with an ISO ripped beforehand, which also works on WSL
with the Nix-shell VLC):

```bash
transcript extract --dvd /dev/sr0 --title 1 --audio-language deu -o work/movie.flac
transcript extract --dvd /path/to/movie.iso --title 1 -o work/movie.flac
```

Options:

- `--title N` — DVD title number. Find it with `lsdvd /dev/sr0` (in the Nix
  shell) or by opening the disc in the VLC GUI; the longest title is almost
  always the main feature.
- `--audio-language deu` — pick the audio track by ISO 639-2 language code.
  Use `--audio-track N` instead when the disc has several German tracks
  (e.g. a director's commentary).
- `--dry-run` — print the VLC command without running it.
- Output is 16 kHz mono FLAC: exactly what Whisper consumes, no ASR quality
  loss, and far smaller than the original AC3.

Encrypted DVDs need `libdvdcss` (included with the Nix/Windows VLC builds).
Whether ripping a CSS-protected disc is legal depends on your jurisdiction —
use this only on discs you own, for personal study.

## Stages 2–4 — audio → word list

```bash
# individually
transcript transcribe work/movie.flac -o work/transcript.json
transcript analyze work/transcript.json -o work/analysis.json
transcript build work/analysis.json --known-words known_words.txt -o work/words.json

# or in one go
transcript run work/movie.flac --known-words known_words.txt --workdir work
```

`known_words.txt` is a plain-text list of the words you already know, one
lemma (dictionary form) per line — see `known_words.example.txt`. Matching
is case-insensitive and treats `ß` and `ss` as equal.

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

One entry per (lemma, word type), sorted by how often it occurs in the
movie, with every distinct sentence it appeared in — ready to turn into
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
  often Whisper mis-transcriptions or too rare to matter.
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

## Development

```bash
pytest          # unit tests; no DVD, VLC, Whisper model or spaCy model needed
```

The spaCy-dependent smoke test auto-skips when `de_core_news_lg` is not
installed.
