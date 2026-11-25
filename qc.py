"""Check every rendered line with ASR and pitch, and retake the ones that fail.

Runs after render, on the line clips it keeps (<book>/lines/NNN/). One model is loaded at
a time: Whisper scores every clip and is freed, the TTS retakes the flagged lines and is
freed, then Whisper scores the takes. For each clip:

  words   Whisper's transcript against the text sent to the TTS (respellings applied):
          character error rate, the longest run of wrong words, extra words at either end,
          and missing words of 5+ letters, each confirmed by a second, shorter listen
  voice   semitones between the clip's median pitch and its voice's reference clip
  length  how far the clip's seconds per character is from its voice's median, as |ln ratio|

A clip fails a metric when it is over max(floor, k x the voice's median for that metric),
so the bar is relative to the book. Flagged lines are retaken with new seeds (several takes
for short lines) and the best take, the current one included, is kept. When every take
transcribes the same wrong way, the recognizer is blamed, not the TTS, and a misheard name
is listed as a pronunciations.json candidate.

Writes <book>/qc/NNN.json and <book>/qc/NNN.md (the flagged lines, with reasons and
before/after scores), then rebuilds audio/NNN.wav from the kept clips.

Usage: python qc.py out/<book> [chapter ...]
"""

import difflib
import gc
import json
import logging
import math
import re
import shutil
import statistics
import sys
import unicodedata
from pathlib import Path

from llm import load_config

log = logging.getLogger("qc")

DEFAULTS = {
    "asr_model": "small.en",
    "cer_floor": 0.15, "cer_k": 3.0,
    "wrong_run_floor": 3, "wrong_run_k": 3.0,
    "extra_floor": 2, "extra_k": 3.0,
    "pitch_floor": 4.0, "pitch_k": 3.0,
    "length_floor": 0.5, "length_k": 3.0,
    "min_missing_letters": 5,
    "window_pad": 0.6,
    "retakes": 3,
    "short_chars": 80,
    "short_takes": 8,
}
WORD_METRICS = ("cer", "wrong_run", "extra")
METRICS = WORD_METRICS + ("pitch", "length")


# --- words: pure ---

def fold(s):
    """Lowercase ASCII letters, digits and single spaces: 'Dantès, “Go!”' -> 'dantes go'."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", s).replace("'", "").split())


def cer(ref, hyp):
    """Character error rate: edit distance over the folded reference length."""
    a, b = fold(ref), fold(hyp)
    if not a:
        return 0.0 if not b else 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / len(a)


def word_check(ref, hyp, min_letters=5):
    """Compare a transcript with the line. Returns cer, the longest wrong run (words),
    extra words before the first and after the last match, and missing-word candidates as
    (word, hyp index where it should be) for deleted words of min_letters or more."""
    r, h = fold(ref).split(), fold(hyp).split()
    ops = difflib.SequenceMatcher(None, r, h, autojunk=False).get_opcodes()
    wrong_run, extra, missing = 0, 0, []
    for k, (op, i1, i2, j1, j2) in enumerate(ops):
        if op == "equal":
            continue
        if op == "insert" and (k == 0 or k == len(ops) - 1):
            extra += j2 - j1
            continue
        wrong_run = max(wrong_run, i2 - i1, j2 - j1)
        if op == "delete":
            missing += [(w, j1) for w in r[i1:i2] if len(w) >= min_letters]
    if not h and r:
        wrong_run = len(r)
    return {"cer": round(cer(ref, hyp), 3), "wrong_run": wrong_run, "extra": extra, "missing": missing}


def gap_window(words, j, duration, pad):
    """(start, end) seconds around hyp position j, from Whisper's word times [(w, s, e)]."""
    start = words[j - 1][2] if 0 < j <= len(words) else 0.0
    end = words[j][1] if j < len(words) else duration
    return max(0.0, start - pad), min(duration, end + pad)


def confirmed_missing(missing, heard_again):
    """Missing words the second listen didn't hear either. heard_again: word -> transcript."""
    return [w for w, _ in missing if w not in fold(heard_again.get(w, "")).split()]


# --- thresholds, flags, takes: pure ---

def semitones(f0, ref_f0):
    return abs(12 * math.log2(f0 / ref_f0)) if f0 and ref_f0 else None


def length_metrics(clips):
    """|ln(seconds per char / the voice's median)| into each clip's metrics."""
    rates = {}
    for c in clips:
        rates.setdefault(c["voice"], []).append(c["seconds"] / max(len(c["spoken"]), 1))
    med = {v: statistics.median(r) for v, r in rates.items()}
    for c in clips:
        rate = c["seconds"] / max(len(c["spoken"]), 1)
        c["metrics"]["length"] = round(abs(math.log(rate / med[c["voice"]])), 3) if rate and med[c["voice"]] else None


def thresholds(clips, cfg):
    """Per voice and metric: max(floor, k x the voice's median)."""
    out = {}
    for v in {c["voice"] for c in clips}:
        mine = [c["metrics"] for c in clips if c["voice"] == v]
        out[v] = {}
        for m in METRICS:
            vals = [x[m] for x in mine if x.get(m) is not None]
            med = statistics.median(vals) if vals else 0
            out[v][m] = round(max(cfg[f"{m}_floor"], cfg[f"{m}_k"] * med), 3)
    return out


def reasons(metrics, thr):
    """Why a clip fails: each metric over its threshold, and confirmed missing words."""
    out = [f"{m} {metrics[m]} > {thr[m]}" for m in METRICS if metrics.get(m) is not None and metrics[m] > thr[m]]
    if metrics.get("missing"):
        out.append("missing " + ", ".join(metrics["missing"]))
    return out


def badness(metrics, thr):
    """One score per take, lower is better: the worst metric/threshold ratio, plus one per
    missing word. Under 1 means the take passes."""
    ratios = [metrics[m] / thr[m] for m in METRICS if metrics.get(m) is not None and thr[m]]
    return round(max(ratios, default=0) + len(metrics.get("missing", [])), 3)


def words_ok(metrics, thr):
    return all(metrics[m] <= thr[m] for m in WORD_METRICS) and not metrics.get("missing")


def best_take(takes, thr, short):
    """The take to keep (index into takes). A short line keeps the take closest to the
    reference pitch among those whose words pass; otherwise the lowest badness."""
    if short:
        ok = [i for i, t in enumerate(takes) if words_ok(t["metrics"], thr) and t["metrics"].get("pitch") is not None]
        if ok:
            return min(ok, key=lambda i: takes[i]["metrics"]["pitch"])
    return min(range(len(takes)), key=lambda i: badness(takes[i]["metrics"], thr))


def recognizer_blamed(takes, thr, text):
    """When every take (two or more) fails on words with the same transcript, the
    recognizer is probably wrong. Returns the book's capitalised words it mishears
    (pronunciations.json candidates), or None if the takes don't agree."""
    if len(takes) < 2 or any(words_ok(t["metrics"], thr) for t in takes):
        return None
    if len({fold(t["heard"]) for t in takes}) != 1:
        return None
    heard = set(fold(takes[0]["heard"]).split())
    names = [w for w in re.findall(r"[^\W\d_]+", text) if w[0].isupper() and len(w) > 1]
    return sorted({w for w in names if fold(w) not in heard and fold(w) not in ("i", "the", "a")})


# --- report: pure ---

def report(chapter, rows):
    """Markdown table of the flagged lines, with reasons and before/after scores."""
    lines = [f"# QC chapter {chapter}", "",
             f"{len(rows)} flagged lines. Badness under 1 passes; listen to the lines still over 1.", "",
             "| clip | voice | reasons | before | after | kept | note | text |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        note = ""
        if r.get("recognizer") is not None:
            note = "all takes agree: recognizer" + (f"; candidates {', '.join(r['recognizer'])}" if r["recognizer"] else "")
        text = r["text"].replace("|", "/")
        lines.append(f'| {r["file"]} | {r["voice"]} | {"; ".join(r["reasons"])} | {r["before"]} | {r["after"]} | '
                     f'take {r["kept"]} | {note} | {text[:80]}{"..." if len(text) > 80 else ""} |')
    return "\n".join(lines) + "\n"


# --- models and audio: the seams (Apple Silicon; mocked in tests) ---

def read_audio(path):
    import numpy as np
    from mlx_audio.audio_io import read
    a, sr = read(str(path), dtype="float32")
    return np.asarray(a, dtype=np.float32).reshape(-1), sr


def median_pitch(audio, sr, lo=60.0, hi=400.0):
    """Median f0 in Hz over voiced frames (autocorrelation), or None if nothing is voiced."""
    import numpy as np
    frame, hop = int(0.04 * sr), int(0.01 * sr)
    f0s = []
    for s in range(0, len(audio) - frame, hop):
        x = audio[s:s + frame] - np.mean(audio[s:s + frame])
        if np.sqrt(np.mean(x ** 2)) < 0.01:
            continue
        ac = np.correlate(x, x, "full")[frame - 1:]
        a, b = int(sr / hi), min(int(sr / lo), frame - 1)
        lag = a + int(np.argmax(ac[a:b]))
        if ac[0] > 0 and ac[lag] / ac[0] > 0.4:
            f0s.append(sr / lag)
    return float(np.median(f0s)) if f0s else None


class ASR:
    """faster-whisper on the CPU, as names.py uses it."""

    def __init__(self, model):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model, device="cpu", compute_type="int8")

    def transcribe(self, audio, sr):
        """(text, [(word, start, end)])."""
        import numpy as np
        n = int(len(audio) * 16000 / sr)
        a16 = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)
        segs, _ = self.model.transcribe(a16, language="en", beam_size=5, word_timestamps=True)
        segs = list(segs)
        words = [(w.word.strip(), w.start, w.end) for s in segs for w in (s.words or [])]
        return " ".join(s.text.strip() for s in segs).strip(), words


# --- the passes ---

def score_clip(asr, path, spoken, ref_f0, cfg):
    """Metrics for one clip (no length yet: that needs the whole book's rates)."""
    audio, sr = read_audio(path)
    heard, words = asr.transcribe(audio, sr)
    m = word_check(spoken, heard, cfg["min_missing_letters"])
    if m["missing"]:
        again = {}
        for w, j in m["missing"]:
            s, e = gap_window(words, j, len(audio) / sr, cfg["window_pad"])
            again[w] = asr.transcribe(audio[int(s * sr):int(e * sr)], sr)[0]
        m["missing"] = confirmed_missing(m["missing"], again)
    else:
        m["missing"] = []
    m["pitch"] = semitones(median_pitch(audio, sr), ref_f0)
    m["pitch"] = round(m["pitch"], 2) if m["pitch"] is not None else None
    return {"heard": heard, "seconds": len(audio) / sr, "metrics": m}


def ref_pitches(book, voices):
    out = {}
    for v in voices:
        ref = book / "refs" / f"{v}.wav"
        out[v] = median_pitch(*read_audio(ref)) if ref.exists() else None
    return out


def score_all(asr, items, refs, cfg):
    """items: dicts with path, spoken, voice. Adds heard, seconds, metrics."""
    for it in items:
        it.update(score_clip(asr, it["path"], it["spoken"], refs.get(it["voice"]), cfg))


def take_length(t, voice_rates):
    rate = t["seconds"] / max(len(t["spoken"]), 1)
    med = voice_rates[t["voice"]]
    t["metrics"]["length"] = round(abs(math.log(rate / med)), 3) if rate and med else None


def retake(book, flagged, cfg):
    """Render new takes of each flagged clip with new seeds. Returns the take items."""
    import render
    from mlx_audio.audio_io import write as audio_write
    voices, _ = render.load_voices(book)
    tts = render.TTS(load_config(), voices, book / "refs")
    takes = []
    for c in flagged:
        count = cfg["short_takes"] if len(c["spoken"]) <= cfg["short_chars"] else cfg["retakes"]
        for k in range(1, count + 1):
            seed = render.seed_for(c["seg"], c["n"], k)
            audio, sr = tts.speak(c["spoken"], c["voice"], seed)
            path = c["path"].with_name(f'{c["seg"]}-{c["n"]}.take{k}.wav')
            audio_write(str(path), render.tidy(audio, sr), sr)
            takes.append({"clip": c["file"], "take": k, "seed": seed, "path": path, "spoken": c["spoken"],
                          "voice": c["voice"]})
    render._free(tts.model)
    return takes


def qc(book, chapters, cfg):
    dirs = [book / "lines" / f"{int(n):03}" for n in chapters] if chapters else sorted((book / "lines").glob("[0-9]*"))
    indexes = {d: json.loads((d / "index.json").read_text()) for d in dirs if (d / "index.json").exists()}
    clips = []
    for d, idx in indexes.items():
        for c in idx["clips"]:
            clips.append({**c, "chapter": idx["chapter"], "dir": d, "path": d / c["file"]})
    if not clips:
        log.info("no line clips; render first")
        return
    voices = sorted({c["voice"] for c in clips})

    asr = ASR(cfg["asr_model"])                                         # 1. score every clip
    refs = ref_pitches(book, voices)
    score_all(asr, clips, refs, cfg)
    del asr
    gc.collect()                                                        # Whisper is gone before the TTS loads
    length_metrics(clips)
    thr = thresholds(clips, cfg)
    rates = {v: statistics.median(c["seconds"] / max(len(c["spoken"]), 1) for c in clips if c["voice"] == v)
             for v in voices}
    flagged = [c for c in clips if reasons(c["metrics"], thr[c["voice"]])]
    log.info("%d of %d clips flagged; thresholds %s", len(flagged), len(clips), thr)

    takes = retake(book, flagged, cfg) if flagged else []            # 2. retake (TTS only)
    if takes:
        asr = ASR(cfg["asr_model"])                                     # 3. score the takes
        score_all(asr, takes, refs, cfg)
        del asr
        gc.collect()
        for t in takes:
            take_length(t, rates)

    rows = {}
    for c in flagged:                                                   # 4. keep the best take
        mine = [{"take": 0, "seed": c["seed"], "path": c["path"], "heard": c["heard"], "metrics": c["metrics"]}]
        mine += [t for t in takes if t["path"].parent == c["dir"] and t["clip"] == c["file"]]
        t_thr = thr[c["voice"]]
        best = best_take(mine, t_thr, len(c["spoken"]) <= cfg["short_chars"])
        blamed = recognizer_blamed(mine, t_thr, c["text"])
        if mine[best]["take"]:
            shutil.copy(mine[best]["path"], c["path"])
            for e in indexes[c["dir"]]["clips"]:
                if e["file"] == c["file"]:
                    e["seed"], e["take"] = mine[best]["seed"], mine[best]["take"]
        rows.setdefault(c["dir"], []).append({
            "file": c["file"], "voice": c["voice"], "text": c["text"], "reasons": reasons(c["metrics"], t_thr),
            "before": badness(c["metrics"], t_thr), "after": badness(mine[best]["metrics"], t_thr),
            "kept": mine[best]["take"], "takes": len(mine) - 1, "recognizer": blamed,
            "heard": mine[best]["heard"]})
        for t in mine[1:]:
            t["path"].unlink(missing_ok=True)

    qcdir = book / "qc"                                                 # 5. report and rebuild
    qcdir.mkdir(exist_ok=True)
    for d, idx in indexes.items():
        n = idx["chapter"]
        mine = rows.get(d, [])
        (d / "index.json").write_text(json.dumps(idx, indent=1, ensure_ascii=False))
        scored = [{k: c[k] for k in ("file", "voice", "heard", "metrics")} for c in clips if c["dir"] == d]
        (qcdir / f"{n:03}.json").write_text(json.dumps({"chapter": n, "thresholds": thr, "flagged": mine,
                                                        "clips": scored}, indent=1, ensure_ascii=False))
        (qcdir / f"{n:03}.md").write_text(report(n, mine))
        if any(r["kept"] for r in mine):
            import render
            render.assemble(d, book / "audio" / f"{n:03}.wav")
        log.info("chapter %d: %d flagged, %d replaced by a retake, %d blamed on the recognizer", n, len(mine),
                 sum(bool(r["kept"]) for r in mine), sum(r["recognizer"] is not None for r in mine))


def main(book_dir, *chapters):
    cfg = {**DEFAULTS, **load_config().get("qc", {})}
    qc(Path(book_dir), chapters, cfg)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(*sys.argv[1:])
