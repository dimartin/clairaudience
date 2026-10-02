"""Attribute every quote to a speaker, using the whole-book cast as the fixed roster.

The LLM labels quotes; then a deterministic pass overrides any quote whose dialogue tag
("said Danglars", "resumed the Catalan") names a cast member. The book text is unchanged.

Usage: python attribute.py out/<book> [chapter ...]
Writes out/<book>/NNN.script.json per chapter.
"""

import json
import logging
import re
import sys
from pathlib import Path

import segment
from llm import chat, load_config, parse_json

log = logging.getLogger("attribute")

PROMPT = """You are casting an audiobook as a radio play. Below is one chapter of a novel, split into numbered segments. Quoted speech is marked [Q<n>]; narration is unmarked.

THE CAST (use only these ids):
{cast}

For EVERY [Q<n>] decide who speaks it. Use the dialogue tags ("said Fernand"), the turn-taking of the conversation, and who is present in the scene. Narration between quotes does not change the speaker unless it says so. If the speaker is not in the cast, use "unknown".

Return ONLY a JSON object, no prose:
{{"quotes": {{"<n>": {{"speaker": "cast id", "emotion": "one word, e.g. neutral, angry, sad, joyful, fearful, sarcastic, excited, whispering"}}}}}}
Every [Q<n>] number must appear.

CHAPTER:
{chapter}"""

VERBS = ("said|replied|cried|asked|muttered|remarked|exclaimed|continued|answered|returned|added|resumed|"
         "interrupted|observed|rejoined|called|whispered|shouted|murmured|stammered|persisted|repeated")
TAG = re.compile(rf"^(?:[\w’']+\s+){{0,3}}?(?:{VERBS})\s+((?:the\s+|M\.\s+|Madame\s+|Mademoiselle\s+)?[A-Z][\w’é\-]+)")


def render(segs):
    return "\n".join(f'[Q{s["id"]}] "{s["text"]}"' if s["kind"] == "quote" else s["text"] for s in segs)


def name_index(cast):
    idx = {}
    for c in cast.values():
        for n in [c["name"]] + c.get("aliases", []) + c.get("epithets", []):
            for key in {n.lower(), n.lower().removeprefix("the "), n.split()[-1].lower()}:
                idx.setdefault(key, c["id"])
    return idx


def tag_override(segs, idx):
    """A quote immediately followed (same paragraph) by 'said <name>' belongs to <name>."""
    fixed = 0
    for i, s in enumerate(segs):
        if s["kind"] != "narration" or i == 0:
            continue
        m = TAG.match(s["text"])
        prev = segs[i - 1]
        if not m or prev["kind"] != "quote" or prev["para"] != s["para"]:
            continue
        who = idx.get(m.group(1).lower()) or idx.get(m.group(1).lower().removeprefix("the "))
        if who and prev["speaker"] != who:
            log.info("tag override Q%d: %s -> %s (%r)", prev["id"], prev["speaker"], who, m.group(0))
            prev["speaker"], prev["tag_override"] = who, True
            fixed += 1
    return fixed


BATCH_QUOTES = 120   # ~30 output tokens per quote keeps a batch well under max_tokens
CONTEXT_SEGS = 12    # earlier segments shown (without [Q] markers) so a batch knows who is talking


def batches(segs):
    """Split a chapter into evenly sized windows of at most BATCH_QUOTES quotes."""
    total = sum(s["kind"] == "quote" for s in segs)
    per = -(-total // -(-total // BATCH_QUOTES)) if total else 1   # ceil(total / ceil(total / BATCH))
    out, start, count = [], 0, 0
    for i, s in enumerate(segs):
        count += s["kind"] == "quote"
        if count > per:
            out.append((start, i))
            start, count = i, 1
    out.append((start, len(segs)))
    return out


def label_batch(cfg, n, segs, lo, hi, cast_text):
    context = " ".join(s["text"] for s in segs[max(0, lo - CONTEXT_SEGS):lo])
    text = (f"(Earlier in the chapter: ...{context})\n\n" if context else "") + render(segs[lo:hi])
    for attempt in range(3):
        try:
            return parse_json(chat(cfg, PROMPT.format(cast=cast_text, chapter=text))).get("quotes", {})
        except ValueError as e:  # includes llm.Truncated
            log.warning("chapter %d segs %d-%d attempt %d: bad JSON (%s)", n, lo, hi, attempt + 1, e)
    return {}


def attribute_chapter(cfg, book, n, cast, cast_text, idx):
    path = book / f"{n:03}.txt"
    segs = segment.segments(path.read_text())
    labels = {}
    if any(s["kind"] == "quote" for s in segs):
        for lo, hi in batches(segs):
            labels.update(label_batch(cfg, n, segs, lo, hi, cast_text))
    for s in segs:
        if s["kind"] == "quote":
            a = labels.get(str(s["id"]), {})
            sp = a.get("speaker", "unknown")
            s["speaker"] = sp if sp in cast else "unknown"
            s["emotion"] = a.get("emotion", "neutral")
        else:
            s["speaker"], s["emotion"] = "narrator", "neutral"
    fixed = tag_override(segs, idx)
    quotes = [s for s in segs if s["kind"] == "quote"]
    unknown = sum(1 for s in quotes if s["speaker"] == "unknown")
    out = {"chapter": n, "segments": segs}
    (book / f"{n:03}.script.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    log.info("chapter %d: %d quotes, %d tag overrides, %d unknown", n, len(quotes), fixed, unknown)


def main(book_dir, *chapters):
    cfg = load_config()
    book = Path(book_dir)
    cast = json.loads((book / "cast.json").read_text())
    cast_text = "\n".join(f'{c["id"]} | {c["name"]} | {", ".join(c.get("aliases", []))}' for c in cast.values())
    idx = name_index(cast)
    wanted = [int(c) for c in chapters] or [c["n"] for c in json.loads((book / "index.json").read_text())]
    for n in wanted:
        if (book / f"{n:03}.script.json").exists() and not chapters:
            continue
        attribute_chapter(cfg, book, n, cast, cast_text, idx)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(*sys.argv[1:])
