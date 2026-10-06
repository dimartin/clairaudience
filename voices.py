"""Write voice-design instructions for the cast.

Main characters (>= min_lines spoken lines across the book) get an instruction the LLM
writes from their whole-book persona, and a reference line picked from their own quotes.
Everyone else shares a small pool of extras' voices by gender, so a large cast doesn't
need hundreds of designed voices. Voices already in voices.json are kept (hand edits win).

Usage: python voices.py out/<book>
Writes out/<book>/voices.json: {"voices": [...], "voice_map": {speaker_id: voice_id}}
"""

import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from llm import chat, load_config, parse_json

log = logging.getLogger("voices")

NARRATOR = {
    "id": "narrator",
    "instruct": "A mature woman in her fifties, warm low alto voice, unhurried and measured storytelling pace, "
                "clear diction, quietly dramatic, like a seasoned stage actress reading by firelight.",
}
EXTRAS = {
    "extra_male_1": "An adult man, plain neutral mid-range voice, matter-of-fact.",
    "extra_male_2": "An older man, deep slightly gravelly voice, slow and deliberate.",
    "extra_male_3": "A young man, light clear tenor, quick and eager.",
    "extra_female_1": "An adult woman, clear mid-range voice, composed and practical.",
    "extra_female_2": "An older woman, soft slightly husky voice, gentle.",
}

PROMPT = """You are a casting director for a full-cast radio play of "{title}". Write the voice-design instruction for one character. A text-to-speech model will create the voice from your instruction alone, so describe only the SOUND.

Character: {name}
Casting brief: {persona}
Evidence from the text: {evidence}

Return ONLY JSON: {{"instruct": "one or two sentences: gender, age, pitch, timbre, pace, accent (keep accents light, the audience is American), manner of speaking", "why": "the phrases from the text this is based on"}}"""


def spoken_lines(book):
    lines = defaultdict(list)
    for f in sorted(book.glob("*.script.json")):
        for s in json.loads(f.read_text())["segments"]:
            if s["kind"] == "quote" and s["speaker"] not in ("unknown", "narrator"):
                lines[s["speaker"]].append(s["text"])
    return lines


def pick_line(texts):
    """A reference line of 60-220 characters, preferring complete sentences."""
    good = [t for t in texts if 60 <= len(t) <= 220]
    good.sort(key=lambda t: (not t.rstrip().endswith((".", "!", "?")), abs(len(t) - 140)))
    return (good or sorted(texts, key=len, reverse=True) or ["I have nothing more to say."])[0]


def narration_line(book):
    """First sentence of the first chapter's narration that makes a good reference line."""
    import re
    text = (book / "001.txt").read_text()
    sents = [x for para in text.split("\n\n") for x in re.split(r"(?<=[.!?])\s+", para)]
    for sent in sents:
        if 60 <= len(sent) <= 220 and "“" not in sent and not sent.lower().startswith(("chapter", "volume")):
            return sent.strip()
    return "It was a story told many times, and never quite the same way twice."


def gender_of(c):
    g = (c.get("persona", {}).get("gender") or "/".join(c.get("notes", {}).get("gender", []))).lower()
    return "female" if "female" in g or g.startswith("f") else "male"


ATTEMPTS = 3


def design_voice(cfg, meta, c):
    """The model's voice description for one character, or None after ATTEMPTS bad answers.
    A small model sometimes returns malformed JSON; attribution retries the same way."""
    prompt = PROMPT.format(title=meta["title"], name=c["name"],
                           persona=json.dumps(c.get("persona", c.get("notes", {})), ensure_ascii=False),
                           evidence="; ".join(c.get("evidence", [])[:12]))
    for attempt in range(ATTEMPTS):
        raw = chat(cfg, prompt)
        try:
            v = parse_json(raw)
            if isinstance(v, dict) and isinstance(v.get("instruct"), str) and v["instruct"]:
                return v
            raise ValueError("no instruct in the answer")
        except ValueError as e:
            salvaged = salvage_instruct(raw)
            if salvaged:
                log.warning("%s: bad JSON (%s); kept the instruct line", c.get("id", c["name"]), e)
                return {"instruct": salvaged, "why": ""}
            log.warning("%s attempt %d: %s", c.get("id", c["name"]), attempt + 1, e)
    return None


#: "instruct" comes first and is one JSON string; the model's trouble is the "why" that
#: follows. On the Monte Cristo run 9 of 10 failed designs broke at line 3 column 11,
#: the start of "why", where it quotes the book and sometimes produces invalid JSON.
_INSTRUCT = re.compile(r'"instruct"\s*:\s*"((?:[^"\\]|\\.)*)"', re.S)


def salvage_instruct(raw):
    """The "instruct" string from an answer whose JSON is broken elsewhere, or None."""
    m = _INSTRUCT.search(raw or "")
    if not m:
        return None
    try:
        text = json.loads(f'"{m.group(1)}"')
    except ValueError:
        return None
    return text.strip() or None


def stock_voice(c, pool):
    """Next stock (EXTRAS) voice of the character's gender, rotating through the pool."""
    g = gender_of(c)
    # startswith, not `g in k`: "male" is a substring of "female", which put both female
    # stock voices in the male pool.
    options = [k for k in EXTRAS if k.startswith(f"extra_{g}_")]
    vid = options[pool[g] % len(options)]
    pool[g] += 1
    return vid


def main(book_dir):
    cfg = load_config()
    book = Path(book_dir)
    meta = json.loads((book / "book.json").read_text())
    cast = json.loads((book / "cast.json").read_text())
    lines = spoken_lines(book)
    counts = Counter({k: len(v) for k, v in lines.items()})
    vpath = book / "voices.json"
    existing = json.loads(vpath.read_text()) if vpath.exists() else {"voices": [], "voice_map": {}}
    voices = {v["id"]: v for v in existing["voices"]}
    voices.setdefault("narrator", {**NARRATOR, "text": narration_line(book)})
    min_lines = cfg.get("min_lines", 5)
    voice_map, pool = dict(existing.get("voice_map", {})), Counter()
    for cid, n in counts.most_common():
        c = cast.get(cid)
        if not c:
            continue
        if n >= min_lines:
            voice_map[cid] = cid
            if cid in voices:
                continue
            v = design_voice(cfg, meta, c)
            if v is None:
                # Mapped to its own id with no voice, render falls back to the NARRATOR,
                # so a major character (Haydée, in Monte Cristo run3) would be read in the
                # narrator's voice. A stock voice of the right gender is the lesser harm.
                voice_map[cid] = stock_voice(c, pool)
                log.error("%s: no voice after %d attempts; using stock voice %s", cid, ATTEMPTS, voice_map[cid])
                continue
            voices[cid] = {"id": cid, "instruct": v["instruct"], "why": v.get("why", ""), "text": pick_line(lines[cid])}
            log.info("voice %s (%d lines): %s", cid, n, v["instruct"][:100])
        elif cid not in voice_map:
            voice_map[cid] = stock_voice(c, pool)
    for vid in set(voice_map.values()) & set(EXTRAS):
        voices.setdefault(vid, {"id": vid, "instruct": EXTRAS[vid], "text": "Very well. I shall see to it at once, and you may rely on me."})
    vpath.write_text(json.dumps({"voices": list(voices.values()), "voice_map": voice_map}, indent=1, ensure_ascii=False))
    log.info("%d voices, %d speakers mapped", len(voices), len(voice_map))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(sys.argv[1])
