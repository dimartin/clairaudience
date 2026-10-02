"""Pronunciation check: hear how the TTS says each name, and what a listener would hear.

For each name, the book's narrator voice says each respelling option in a carrier
sentence. Whisper (English) transcribes every clip, so the transcript shows what was
actually spoken; if it comes back with the real spelling, an American listener will
likely recognise the name. Put the winners in <book>/pronunciations.json.

Writes <book>/names/report.json and <book>/names/names.wav (every option, announced).
Usage: python names.py out/<book> name_options.json
"""

import json
import logging
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

from llm import load_config
from render import TTS, _free, load_voices, tidy

log = logging.getLogger("names")
WORDS = ["one", "two", "three", "four", "five", "six"]


def fold(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", "", s)


def _16k(path):
    # a float array sidesteps faster-whisper's own decoder, which trips on some PyAV versions
    from mlx_audio.audio_io import read
    a, sr = read(str(path))
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    n = int(len(a) * 16000 / sr)
    return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)


def render_clips(book, options):
    from mlx_audio.audio_io import write as audio_write
    voices, _ = load_voices(book)
    tts = TTS(load_config(), {"narrator": voices["narrator"]}, book / "refs")
    out = book / "names"
    out.mkdir(exist_ok=True)
    clips, reel, sr = [], [], 24000

    def say(text):
        audio, rate = tts.speak(text, "narrator")
        return tidy(audio, rate), rate

    for name, variants in options.items():
        intro, sr = say(f"Name: {variants[0]}.")
        reel += [intro, np.zeros(int(0.5 * sr), np.float32)]
        for i, v in enumerate(variants):
            label, _ = say(f"Option {WORDS[i]}.")
            clip, _ = say(f"{v}. I said, {v}.")
            p = out / f"{fold(name).replace(' ', '_')}.{i + 1}.wav"
            audio_write(str(p), clip, sr)
            clips.append({"name": name, "option": i + 1, "spoken_as": v, "wav": str(p)})
            reel += [label, np.zeros(int(0.3 * sr), np.float32), clip, np.zeros(int(0.8 * sr), np.float32)]
    audio_write(str(out / "names.wav"), np.concatenate(reel), sr)
    _free(tts.model)
    return clips


def score(clips):
    from faster_whisper import WhisperModel
    asr = WhisperModel("small.en", device="cpu", compute_type="int8")
    for c in clips:
        segs, _ = asr.transcribe(_16k(c["wav"]), language="en", beam_size=5)
        c["heard"] = " ".join(s.text for s in segs).strip()
        c["recognised"] = fold(c["name"]) in fold(c["heard"])
        log.info("%-14s opt %d %-18s -> %r %s", c["name"], c["option"], c["spoken_as"], c["heard"],
                 "OK" if c["recognised"] else "")


def main(book_dir, options_path):
    book = Path(book_dir)
    clips = render_clips(book, json.loads(Path(options_path).read_text()))  # TTS is freed before Whisper loads
    score(clips)
    (book / "names" / "report.json").write_text(json.dumps(clips, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(*sys.argv[1:3])
