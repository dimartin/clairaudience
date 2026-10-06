# clairaudience

*Clairaudience: the gift of hearing voices nobody else can hear.*

Turn an EPUB into a full-cast audiobook, like a radio play: a narrator reads the narration, and every character speaks in their own voice. Everything runs locally on an Apple Silicon Mac. It runs on a 16 GB machine because only one model is in memory at a time.

The book's text is never changed. The pipeline only decides **who** speaks each quoted line, and **what each character sounds like**, from how the book describes them.

## How it works

`clairaudience.py` runs eight stages. Each stage is a separate process, so its model is freed when it exits, and each model does its work for the whole book before the next one loads.

| # | Stage | Model | What it does |
|---|---|---|---|
| 1 | extract | none | splits the EPUB into chapters in reading order |
| 2 | cast | LLM | reads the book chapter by chapter and builds the cast: names, aliases and disguises, plus evidence of voice and manner quoted from the text. Then it writes a whole-book persona per character |
| 3 | attribute | LLM | labels every quoted line with its speaker and an emotion. A dialogue tag ("said Danglars") always overrides the model |
| 4 | voices | LLM | writes a voice-design brief for each main character from their persona, and picks a reference line from their own dialogue. Minor characters share a few extras' voices |
| 5 | design | voice-design TTS | creates each voice from its text description and speaks the reference line |
| 6 | render | cloning TTS | voices every line by cloning the reference clips. Joins are levelled and faded |
| 7 | encode | ffmpeg | tagged audio: `.m4a` or `.mp3` per chapter, or one `.m4b` with chapter markers |
| 8 | publish | rsync | copies the book folder to a library such as Audiobookshelf (optional) |

Every stage writes its results to `out/<book>/`, and `manifest.json` records finished stages, so an interrupted run resumes where it stopped. The intermediate files are plain JSON you can edit before rendering: the cast (`cast.json`), the scripts (`NNN.script.json`), the voices (`voices.json`) and pronunciations (`pronunciations.json`).

## Requirements

- An Apple Silicon Mac (M1 or later) with **16 GB** of memory or more. Tested on an M1 Pro (16 GB) and an M5 Pro (24 GB).
- macOS with [Homebrew](https://brew.sh).
- About 15 GB of disk for the `16gb` profile's models (measured: 14 GB), plus a few GB per book for working files and audio.

Installation is in [docs/INSTALL.md](docs/INSTALL.md).

## Quick start

```bash
brew bundle                                   # uv and ffmpeg
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.lock
.venv/bin/python clairaudience.py fetch            # download the profile's models
.venv/bin/python clairaudience.py run books/mybook.epub
```

The finished audio is in `out/mybook/book/<Author>/<Title> (Full Cast)/`.

Choose the output in `[publish]` in `config.toml` or `config.local.toml`:

| `format` | Output | Plays on |
|---|---|---|
| `m4a` (default) | one AAC file per chapter | Apple devices, Windows, Android, Audiobookshelf, VLC. Some minimal Linux desktops need extra codec packages |
| `mp3` | one MP3 file per chapter | everything (about 3% larger at speech bitrates) |
| `m4b` | one file for the whole book, with chapter markers | Apple Books and most audiobook apps |

The per-chapter formats also get `<Title> (Full Cast).m3u8`, an extended-M3U playlist of the chapters in order, with durations and titles. Paths are relative, so the folder can be moved or served as it is.

Useful options:

```bash
clairaudience.py run book.epub --profile 16gb-small      # pick a profile from config.toml
clairaudience.py run book.epub --only render --chapters 4 # re-render one chapter
clairaudience.py run book.epub --from voices             # redo voices and everything after
```

## Profiles

| Profile | Text model | Voice design | Speech | Notes |
|---|---|---|---|---|
| `16gb` (default) | Qwen3.5 9B, 4-bit | Qwen3-TTS 1.7B VoiceDesign | VoxCPM2 2B | all Apache-2.0 |
| `16gb-small` | Qwen3.5 4B, 4-bit | Qwen3-TTS 1.7B VoiceDesign | Chatterbox Turbo, 4-bit | fastest: rendered about 4x faster than VoxCPM2 on an M1 Pro |
| `24gb` | Qwen3.5 9B, 4-bit | Qwen3-TTS 1.7B VoiceDesign | Higgs TTS 3, full precision | Higgs weights are non-commercial |

Put machine-specific settings in `config.local.toml`, which is git-ignored and merged over `config.toml`. Examples are a publish destination, or an LLM you already serve (`llm_backend = "server"` with an OpenAI-compatible `llm_base_url`).

### Speaker attribution accuracy

Measured on Chapter 4 of *The Count of Monte Cristo* (121 quoted lines, checked by hand), with each model given the same cast:

| Text model | Correct | Time per chapter (M5 Pro) |
|---|---|---|
| Qwen3.5 9B, MTP-optimized, served by mtplx | 119/121 (98%) | 75 s |
| Qwen3.5 9B, 4-bit, in-process (`16gb`) | 117/121 (97%) | 72 s |
| Qwen3.5 4B, 4-bit (`16gb-small`) | 112/121 (93%) | 29 s |

Score your own setup with `eval.py <profile>`.

## Pronunciation

TTS models often mangle names. `pronunciations.json` (per book in `out/<book>/`) maps a name to a respelling, for example `"Caderousse": "Cad-roose"`. The respelling applies only to what the TTS reads, never to the book text. To choose respellings, run:

```bash
.venv/bin/python names.py out/<book> examples/monte-cristo.name_options.json
```

It voices every option in the narrator's voice and transcribes each clip with Whisper, so you see what was actually said. It also writes `names/names.wav` so you can judge by ear.

## Tests

The unit tests cover the code that decides identity: cast merging and the dialogue-tag override. They need only `pytest`, and no model is loaded.

```bash
uv run --with pytest python -m pytest tests
```

## Licences

The code is released under the [MIT License](LICENSE). Each model has its own licence:

| Model | Licence |
|---|---|
| Qwen3.5 9B and 4B (text) | Apache-2.0 |
| Qwen3-TTS VoiceDesign | Apache-2.0 |
| VoxCPM2 | Apache-2.0 |
| Chatterbox Turbo (Resemble AI; 4-bit MLX build) | MIT (original), Apache-2.0 (MLX build) |
| Higgs TTS 3 (Boson AI) | research and non-commercial only |

Respect the rights to the books you convert. The examples use *The Count of Monte Cristo* (Project Gutenberg #1184), which is in the public domain.

## Roadmap

The goal is to run on the most modest hardware possible.

- **NVIDIA PCs with 6 GB of VRAM or more (planned).** The text stages already work against any OpenAI-compatible server (`llm_backend = "server"`), such as Ollama or llama.cpp with a Qwen3.5 GGUF build, on CPU or GPU. The voice stages need a PyTorch/CUDA backend: Qwen3-TTS VoiceDesign to create the voices, and Qwen3-TTS Base (0.6B or 1.7B) to clone them. Both fit on a 6 GB card.
- **A lighter Mac profile:** Qwen3-TTS Base for speech instead of VoxCPM2, so a single model family designs and speaks the voices.
