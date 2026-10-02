"""Voice design and chapter rendering, as two separate stages (one model each).

design: the voice-design model speaks each voice's reference line from its description
        -> <book>/refs/<id>.wav
render: the TTS model clones each reference and voices every segment of each chapter
        script -> <book>/audio/NNN.wav. Narration goes to the narrator voice.

Usage: python render.py design out/<book>
       python render.py render out/<book> [chapter ...]
"""

import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np

from llm import load_config

from mlx_audio.audio_io import write as audio_write

log = logging.getLogger("render")

HIGGS_MODEL = "bosonai/higgs-tts-3-4b"
MAX_CHARS = 400          # longer segments are split at sentence ends
PAUSE_SAME_PARA = 0.25   # seconds between segments in one paragraph
PAUSE_SPEAKER = 0.4      # extra breathing room when the voice changes
PAUSE_NEW_PARA = 0.7
TARGET_RMS = 0.08        # every segment is levelled to this loudness
FADE_IN, FADE_OUT = 0.015, 0.04
PEAK = 0.89              # ~ -1 dBFS ceiling for the final mix


def tidy(audio, sr):
    """Trim silent edges, level loudness, fade the ends so segment joins don't click."""
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    env = np.abs(a)
    loud = np.where(env > 0.02)[0]
    if len(loud):
        pad = int(0.03 * sr)
        a = a[max(0, loud[0] - pad):min(len(a), loud[-1] + pad)]
    rms = float(np.sqrt(np.mean(a ** 2))) if len(a) else 0.0
    if rms > 1e-4:
        a = a * (TARGET_RMS / rms)
    fi, fo = min(int(FADE_IN * sr), len(a) // 2), min(int(FADE_OUT * sr), len(a) // 2)
    if fi:
        a[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)
    if fo:
        a[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
    return a


def limit(a):
    """Soft limiter: tanh knee above the ceiling instead of hard clipping."""
    over = np.abs(a) > PEAK
    a[over] = np.sign(a[over]) * (PEAK + (1 - PEAK) * np.tanh((np.abs(a[over]) - PEAK) / (1 - PEAK)))
    return a


def _free(model):
    import mlx.core as mx
    del model
    mx.clear_cache()


def load_voices(book):
    data = json.loads((book / "voices.json").read_text())
    return {v["id"]: v for v in data["voices"]}, data.get("voice_map", {})


def design(book):
    cfg = load_config()
    voices, _ = load_voices(book)
    refdir = book / "refs"
    refdir.mkdir(exist_ok=True)
    todo = [v for v in voices.values() if not (refdir / f"{v['id']}.wav").exists()]
    if not todo:
        log.info("all %d reference clips exist", len(voices))
        return
    from mlx_audio.tts.utils import load_model
    model = load_model(cfg["design_model"])
    for v in todo:
        t0 = time.time()
        r = list(model.generate_voice_design(text=v["text"], language="English", instruct=v["instruct"]))[0]
        audio_write(str(refdir / f"{v['id']}.wav"), r.audio, r.sample_rate)
        log.info("ref %s: %.1fs audio in %.1fs", v["id"], len(r.audio) / r.sample_rate, time.time() - t0)
    _free(model)


class TTS:
    """One cloning TTS model: Higgs (pre-encoded reference codes) or VoxCPM2 (reference wav)."""

    def __init__(self, cfg, voices, refdir):
        from mlx_audio.tts.utils import load_model
        self.kind, self.voices, self.refdir = cfg["tts"], voices, refdir
        if self.kind == "higgs":
            # mlx-audio 0.5.7 drops its own higgs_multimodal_qwen3 mapping when the repo name
            # has no matching part (utils.py:293); hint it directly.
            self.model = load_model(cfg["tts_model"], model_name_parts=["higgs_audio_v3"])
            self.codes = {vid: self.model.encode_reference_audio(str(refdir / f"{vid}.wav")) for vid in voices}
        else:
            self.model = load_model(cfg["tts_model"])

    def speak(self, text, vid):
        if self.kind == "higgs":
            r = next(self.model.generate(text=text, ref_audio_codes=self.codes[vid], ref_text=self.voices[vid]["text"],
                                         temperature=1.0, max_new_tokens=2048))
        else:
            r = next(self.model.generate(text=text, ref_audio=str(self.refdir / f"{vid}.wav")))
        return r.audio, r.sample_rate


def split(text):
    if len(text) <= MAX_CHARS:
        return [text]
    parts, cur = [], ""
    for s in re.split(r"(?<=[.!?;])\s+", text):
        if cur and len(cur) + len(s) > MAX_CHARS:
            parts.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    return parts + [cur] if cur else parts


def load_lexicon(path=Path("pronunciations.json")):
    """name -> respelling, applied only to what the TTS reads; the book text is unchanged."""
    if not path.exists():
        return []
    lex = json.loads(path.read_text())
    # longest names first so "Monte Cristo" wins over "Cristo"
    return [(re.compile(rf"\b{re.escape(k)}\b"), v) for k, v in sorted(lex.items(), key=lambda kv: -len(kv[0]))]


def speakable(text, lexicon):
    for pat, rep in lexicon:
        text = pat.sub(rep, text)
    return text


def render_chapter(tts, script, voice_map, lexicon, out_path):
    pieces, sr, prev_para, prev_vid = [], 24000, None, None
    t_start, audio_secs = time.time(), 0.0
    segs = script["segments"]
    for i, s in enumerate(segs):
        vid = voice_map.get(s["speaker"], "narrator") if s["kind"] == "quote" else "narrator"
        if vid not in tts.voices:
            vid = "narrator"
        for chunk in split(s["text"]):
            audio, sr = tts.speak(speakable(chunk, lexicon), vid)
            if prev_para is not None:
                gap = PAUSE_NEW_PARA if s["para"] != prev_para else PAUSE_SAME_PARA
                if vid != prev_vid:
                    gap = max(gap, PAUSE_SPEAKER)
                pieces.append(np.zeros(int(gap * sr), dtype=np.float32))
            audio = tidy(audio, sr)
            pieces.append(audio)
            audio_secs += len(audio) / sr
            prev_para, prev_vid = s["para"], vid
        if i % 25 == 0:
            el = time.time() - t_start
            log.info("ch%s seg %d/%d, %.0fs audio in %.0fs (RTF %.2f)", script["chapter"], i, len(segs),
                     audio_secs, el, el / max(audio_secs, 1))
    audio_write(str(out_path), limit(np.concatenate(pieces)), sr)
    el = time.time() - t_start
    log.info("wrote %s: %.0fs audio in %.0fs (RTF %.2f)", out_path, audio_secs, el, el / max(audio_secs, 1))


def render(book, chapters):
    cfg = load_config()
    voices, voice_map = load_voices(book)
    lexicon = load_lexicon(book / "pronunciations.json") or load_lexicon()
    outdir = book / "audio"
    outdir.mkdir(exist_ok=True)
    scripts = sorted(book.glob("[0-9][0-9][0-9].script.json"))
    if chapters:
        scripts = [book / f"{int(c):03}.script.json" for c in chapters]
    todo = [p for p in scripts if chapters or not (outdir / f"{p.name[:3]}.wav").exists()]
    if not todo:
        log.info("all chapters rendered")
        return
    tts = TTS(cfg, voices, book / "refs")
    for p in todo:
        render_chapter(tts, json.loads(p.read_text()), voice_map, lexicon, outdir / f"{p.name[:3]}.wav")
    _free(tts.model)


def main(cmd, book_dir, *chapters):
    book = Path(book_dir)
    if cmd == "design":
        design(book)
    elif cmd == "render":
        render(book, chapters)
    else:
        raise SystemExit(f"unknown command {cmd!r}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(*sys.argv[1:])
