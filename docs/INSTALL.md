# Installing clairaudience

This is the procedure used for the reference install on a 16 GB **M1 Pro** (macOS 26.6.2) on 2026-10-01. Timings are from that machine.

## 1. System tools

You need Homebrew (https://brew.sh). Then, from the repository folder:

```bash
brew bundle            # installs uv (Python and packages) and ffmpeg (encoding)
```

On Apple Silicon both come as prebuilt packages. You don't need Xcode beyond the Command Line Tools Homebrew installs.

## 2. Python environment

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.lock
```

uv downloads Python 3.13 itself if it isn't already installed, so the macOS system Python (3.9) is never used. This step took **22 seconds** on the M1.

- `requirements.lock` pins every package exactly as tested.
- `requirements.txt` lists only the direct dependencies, if you'd rather resolve versions yourself.

Check the install:

```bash
.venv/bin/python -c "import mlx.core as mx, mlx_audio, mlx_lm, torch, faster_whisper; print(mx.metal.is_available())"
```

It should print `True`.

## 3. Models

```bash
.venv/bin/python clairaudience.py fetch              # the default profile, 16gb
.venv/bin/python clairaudience.py fetch --profile 24gb
```

The models go into the Hugging Face cache (`~/.cache/huggingface/hub`).

| Model | Download | Used by |
|---|---|---|
| `mlx-community/Qwen3.5-9B-MLX-4bit` | 5.6 GB | 16gb, 24gb |
| `mlx-community/Qwen3.5-4B-4bit` | 2.9 GB | 16gb-small |
| `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16` | 4.2 GB | all |
| `mlx-community/VoxCPM2-bf16` | 4.6 GB | 16gb |
| `mlx-community/chatterbox-turbo-4bit` | 0.4 GB | 16gb-small |
| `bosonai/higgs-tts-3-4b` | 9.3 GB | 24gb |

`names.py` also downloads Whisper `small.en` (~0.5 GB) the first time it runs.

## 4. Local settings (optional)

Create `config.local.toml` next to `config.toml`. It's git-ignored and merged on top. For example:

```toml
[publish]
dest = "me@nas:/srv/audiobookshelf/audiobooks"   # rsync target for finished books
```

Publishing uses `ssh` and `rsync`, so the target needs key-based SSH access.

## 5. Run

```bash
mkdir -p books && cp ~/Downloads/mybook.epub books/
.venv/bin/python clairaudience.py run books/mybook.epub
```

## Known issues, and the workarounds already in the code

- **mlx-audio 0.5.7 doesn't recognise `bosonai/higgs-tts-3-4b`.** The repository was renamed from `higgs-audio-v3-tts-4b`, and mlx-audio's loader drops its own mapping (`utils.py:293`). `render.py` passes `model_name_parts=["higgs_audio_v3"]`.
- **Higgs needs PyTorch.** Its codec loader reads `.safetensors` with `framework="pt"`. That's why `torch` is a dependency.
- **faster-whisper with the installed PyAV** fails on `metadata_errors`. `names.py` passes 16 kHz float arrays instead of file paths.
- **macOS `rsync` is openrsync** (protocol 29) and has no `--mkpath`. Publishing creates the remote folder over SSH first.
- **Don't run two large models at once on 16 GB.** The pipeline never does. If you point `llm_backend = "server"` at a model server on the same Mac, set `llm_server_stop` and `llm_server_start` in your profile, so the server is stopped for the audio stages. Running both on a 24 GB Mac pushed it 12 GB into swap.
- **The macOS Local Network permission** can stop Homebrew or uv Python from reaching other machines on your LAN, for example an LLM server elsewhere, when Python is launched from some GUI apps. Model downloads from the internet aren't affected. If a LAN server is unreachable from Python but works with `curl`, an SSH tunnel to `127.0.0.1` works around it.

## Checking disk use

The Hugging Face cache keeps each file once in `~/.cache/huggingface/hub/blobs` and links to it from each model's `snapshots/` folder. Measure with `du -sh ~/.cache/huggingface` (no `-L`). Following the links (`du -L`) counts files more than once: on the reference M1 it reported 43 GB for a cache that really uses 14 GB.
