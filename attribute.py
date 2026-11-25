"""Attribute every quote to a speaker, using the whole-book cast as the fixed roster.

The LLM labels quotes; then a deterministic pass overrides any quote whose dialogue tag
("said Danglars", "resumed the Catalan") names a cast member. A nameless tag ("said she",
"said his father") is asked about twice, blind; the speaker is set only when both
answers agree. The book text is unchanged.

Every quote records how its speaker was decided, in "source" (narration has none):
  tag           the dialogue tag names a cast member
  tag-resolved  a nameless tag ("said she") resolved by two agreeing reads
  llm           the model's label, with no override
  unknown       the model gave no usable speaker (none, or a name not in the cast)
  user          edited by hand; a rerun keeps the edit while the quote text is unchanged,
                and drops it with a warning when the text has changed

Usage: python attribute.py out/<book> [chapter ...]
Writes out/<book>/NNN.script.json per chapter.
"""

import json
import logging
import re
import sys
from pathlib import Path

import cast as castlib
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


def _gender(c):
    """male/female from the name's title or the cast notes; None if unknown or contradictory."""
    g = castlib._title_gender(c["name"])
    if g:
        return g
    notes = castlib._genders({"notes": c.get("notes", {})})
    return next(iter(notes)) if len(notes) == 1 else None


def _first_name(name):
    words = [w for w in name.split() if w.lower().rstrip(".") not in castlib.TITLES]
    return words[0].lower() if len(words) > 1 else None


def name_index(cast):
    idx = {}
    for c in cast.values():
        for n in [c["name"]] + c.get("aliases", []) + c.get("epithets", []):
            for key in {n.lower(), n.lower().removeprefix("the "), n.split()[-1].lower()}:
                idx.setdefault(key, c["id"])
    # weaker keys, added only where they can't point at the wrong person:
    # "M. Morrel" / "Madame Danglars" only for a character of that gender ...
    titles = {"male": ("m.",), "female": ("madame", "mademoiselle")}
    for c in cast.values():
        surname = c["name"].split()[-1].lower()
        for t in titles.get(_gender(c), ()):
            idx.setdefault(f"{t} {surname}", c["id"])
    # ... and "Fernand" for Fernand Mondego only if no one else has that first name
    firsts = {}
    for c in cast.values():
        if f := _first_name(c["name"]):
            firsts.setdefault(f, set()).add(c["id"])
    for f, ids in firsts.items():
        if len(ids) == 1:
            idx.setdefault(f, ids.pop())
    return idx


SOURCES = ("tag", "tag-resolved", "llm", "unknown", "user")


def tag_override(segs, idx):
    """A quote immediately followed (same paragraph) by 'said <name>' belongs to <name>.
    The quote's source becomes "tag" even when the model already agreed; a hand-edited
    ("user") quote is never changed. Returns how many speakers it changed."""
    fixed = 0
    for i, s in enumerate(segs):
        if s["kind"] != "narration" or i == 0:
            continue
        m = TAG.match(s["text"])
        prev = segs[i - 1]
        if not m or prev["kind"] != "quote" or prev["para"] != s["para"] or prev.get("source") == "user":
            continue
        who = idx.get(m.group(1).lower()) or idx.get(m.group(1).lower().removeprefix("the "))
        if not who:
            continue
        if prev["speaker"] != who:
            log.info("tag override Q%d: %s -> %s (%r)", prev["id"], prev["speaker"], who, m.group(0))
            fixed += 1
        prev["speaker"], prev["source"] = who, "tag"
    return fixed


# A tag with no name: "said she", "he replied", "said his father", "replied the young man".
PRONOUN = "he|she"
NAMELESS = re.compile(
    rf"^(?:(?:[\w’']+\s+){{0,2}}?(?P<pron>{PRONOUN})\s+(?:{VERBS})\b"
    rf"|(?:[\w’']+\s+){{0,3}}?(?:{VERBS})\s+(?P<who>{PRONOUN}\b|(?:his|her|their|the|a|an)\s+[a-z][\w’\-]+"
    rf"(?:\s+[a-z][\w’\-]+)?))")

RESOLVE_PROMPT = """Below is a passage from a novel. The quoted line marked >>> is followed by the dialogue tag "{tag}".

Who is "{who}" in that tag, the character who speaks the marked line? It is usually not the person the line before it speaks to.

CHARACTERS (answer with one of these ids, or "unknown" if you can't tell):
{cast}

Return ONLY a JSON object: {{"speaker": "<id>"}}

PASSAGE:
{passage}"""


def nameless_tags(segs):
    """(quote index, tag clause, the words naming the speaker) for each quote followed in
    its paragraph by a tag with no name. Quotes already decided by a tag or by hand are left out."""
    out = []
    for i, s in enumerate(segs):
        if s["kind"] != "narration" or i == 0:
            continue
        prev = segs[i - 1]
        if prev["kind"] != "quote" or prev["para"] != s["para"] or prev.get("source") in ("tag", "user"):
            continue
        if m := NAMELESS.match(s["text"]):
            clause = re.split(r"[,.;:!?—]", s["text"], maxsplit=1)[0].strip()
            out.append((i - 1, clause, m.group("pron") or m.group("who")))
    return out


def chapter_cast(segs, cast):
    """Cast ids that speak in the chapter or whose name appears in it."""
    text = " ".join(s["text"] for s in segs)
    ids = {s["speaker"] for s in segs if s["kind"] == "quote" and s["speaker"] in cast}
    for c in cast.values():
        words = [w for w in c["name"].split() if w.lower().rstrip(".") not in castlib.TITLES and w.lower() != "the"]
        names = [c["name"], *words, *c.get("aliases", []), *c.get("epithets", [])]
        if any(re.search(rf"\b{re.escape(n)}\b", text) for n in names if n):
            ids.add(c["id"])
    return sorted(ids)


def ask_speaker(cfg, prompt, allowed, seed):
    """One read: a cast id from allowed, "unknown", or None after three invalid answers."""
    schema = {"type": "object", "properties": {"speaker": {"type": "string", "enum": [*allowed, "unknown"]}},
              "required": ["speaker"]}
    for attempt in range(3):
        try:
            sp = parse_json(chat(cfg, prompt, max_tokens=100, seed=seed, schema=schema)).get("speaker")
        except ValueError as e:
            log.warning("nameless tag read attempt %d: bad JSON (%s)", attempt + 1, e)
            continue
        if sp == "unknown" or sp in allowed:
            return sp
        log.warning("nameless tag read attempt %d: %r is not a chapter cast id", attempt + 1, sp)
    return None


def resolve_nameless(cfg, segs, cast):
    """Ask twice, blind, who each nameless tag refers to: once with the profile's model and
    once with tag_check_model (or the same model with another seed). Only when both name
    the same cast member is the speaker set, as "tag-resolved". Otherwise the model's
    chapter label stays as a guess. Returns (tags found, tags anchored)."""
    tags = nameless_tags(segs)
    if not tags:
        return 0, 0
    allowed = chapter_cast(segs, cast)
    cast_text = "\n".join(f'{c} | {cast[c]["name"]} | {", ".join(cast[c].get("aliases", []))}' for c in allowed)
    second = {**cfg, "llm_model": cfg["tag_check_model"]} if cfg.get("tag_check_model") else cfg
    anchored = 0
    for qi, clause, who in tags:
        passage = "\n".join((">>> " if j == qi else "") + (f'"{s["text"]}"' if s["kind"] == "quote" else s["text"])
                             for j, s in enumerate(segs[max(0, qi - CONTEXT_SEGS):qi + 2], max(0, qi - CONTEXT_SEGS)))
        prompt = RESOLVE_PROMPT.format(tag=clause, who=who, cast=cast_text, passage=passage)
        a, b = ask_speaker(cfg, prompt, allowed, seed=1), ask_speaker(second, prompt, allowed, seed=2)
        q = segs[qi]
        if a and a == b and a != "unknown":
            log.info("nameless tag Q%d %r: %s -> %s (both reads agree)", q["id"], clause, q["speaker"], a)
            q["speaker"], q["source"] = a, "tag-resolved"
            anchored += 1
        else:
            log.info("nameless tag Q%d %r: reads %s / %s, kept %s as a guess", q["id"], clause, a, b, q["speaker"])
    return len(tags), anchored


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


def read_label(a, cast):
    """(speaker, emotion) from one model label, whatever shape it came back in.

    The prompt asks for {"speaker": "<cast id>", "emotion": "<word>"}, but a small model
    sometimes nests the speaker as an object ({"id": ..., "name": ...}) or answers with
    the bare id. A dict speaker once crashed a whole Monte Cristo run at chapter 53 with
    "unhashable type: 'dict'", 52 chapters in. Anything that isn't a cast id is
    "unknown", which the dialogue-tag override can still correct.
    """
    if isinstance(a, str):
        a = {"speaker": a}
    if not isinstance(a, dict):
        return "unknown", "neutral"
    sp = a.get("speaker", "unknown")
    if isinstance(sp, dict):
        sp = next((v for k in ("id", "speaker", "name") if isinstance(v := sp.get(k), str) and v in cast), "unknown")
    if not isinstance(sp, str) or sp not in cast:
        sp = "unknown"
    emotion = a.get("emotion", "neutral")
    return sp, emotion if isinstance(emotion, str) and emotion else "neutral"


def user_lines(path, segs):
    """Hand-edited quotes ("source": "user") from an existing script, by id, kept only
    where the quote text is still the same."""
    if not path.exists():
        return {}
    text = {s["id"]: s["text"] for s in segs}
    kept = {}
    for s in json.loads(path.read_text())["segments"]:
        if s["kind"] != "quote" or s.get("source") != "user":
            continue
        if text.get(s["id"]) == s["text"]:
            kept[s["id"]] = s
        else:
            log.warning("%s: hand edit Q%d no longer matches the text; dropped", path.name, s["id"])
    return kept


def source_counts(quotes):
    return {src: sum(q.get("source") == src for q in quotes) for src in SOURCES}


def attribute_chapter(cfg, book, n, cast, cast_text, idx):
    path = book / f"{n:03}.txt"
    out_path = book / f"{n:03}.script.json"
    segs = segment.segments(path.read_text())
    labels = {}
    if any(s["kind"] == "quote" for s in segs):
        for lo, hi in batches(segs):
            labels.update(label_batch(cfg, n, segs, lo, hi, cast_text))
    edited = user_lines(out_path, segs)
    for s in segs:
        if s["id"] in edited:
            u = edited[s["id"]]
            s["speaker"], s["emotion"], s["source"] = u["speaker"], u.get("emotion", "neutral"), "user"
        elif s["kind"] == "quote":
            s["speaker"], s["emotion"] = read_label(labels.get(str(s["id"]), {}), cast)
            s["source"] = "unknown" if s["speaker"] == "unknown" else "llm"
        else:
            s["speaker"], s["emotion"] = "narrator", "neutral"
    fixed = tag_override(segs, idx)
    nameless, anchored = resolve_nameless(cfg, segs, cast)
    quotes = [s for s in segs if s["kind"] == "quote"]
    out = {"chapter": n, "segments": segs}
    out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    counts = source_counts(quotes)
    log.info("chapter %d: %d quotes, %d tag overrides, %d/%d nameless tags anchored; by source: %s; llm share %.0f%%",
             n, len(quotes), fixed, anchored, nameless, ", ".join(f"{k} {v}" for k, v in counts.items()),
             100 * counts["llm"] / max(len(quotes), 1))


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
