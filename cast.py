"""Build a whole-book cast for voice casting, using the local LLM.

Pass 1 (collect): walk chapters in order. Each call sees the cast so far and returns
new characters, new aliases, and voice evidence. Merged in code, saved per chapter
so a rerun resumes where it stopped.
Pass 2 (personas): one call per speaking character turns the evidence into a
casting description.

Usage: python cast.py out/monte-cristo            # both passes
       python cast.py out/monte-cristo collect    # pass 1 only
       python cast.py out/monte-cristo dedupe     # merge duplicate ids
       python cast.py out/monte-cristo personas   # pass 2 only
Writes <book>/cast/NNN.json (per-chapter deltas) and <book>/cast.json.
"""

import json
import re
import logging
import sys
from pathlib import Path

from llm import chat, load_config, parse_json

log = logging.getLogger("cast")

COLLECT = """You are building the cast list for a full-cast radio play of a novel, reading it one chapter at a time.

CAST SO FAR (name | other names):
{cast}

CHAPTER {n}:
{chapter}

List every character who SPEAKS or is DESCRIBED in this chapter. If a person is already in the cast, even under a title, a nickname or a disguise, write their "name" EXACTLY as it appears in the cast list. Otherwise use the fullest proper name the text gives them, or a short description if it gives none.

Return ONLY JSON:
{{
  "characters": [
    {{"name": "...", "new_aliases": ["other proper names or titles used for them in this chapter"],
      "speaks": true, "lines": "approximate number of spoken lines in this chapter",
      "gender": "male|female|unknown", "age": "child|young|adult|middle-aged|old|unknown",
      "origin": "nationality, region or accent if stated",
      "class": "social standing if evident",
      "evidence": ["short exact phrases from the text about their voice, manner of speech, temperament or appearance"]}}
  ],
  "disguises": [{{"name": "the person's cast name", "as": "name of the disguise"}}]
}}
Limits: at most 5 entries in "new_aliases" and exactly the 3 most useful phrases in "evidence" per character. Omit characters who are only mentioned in passing.
JSON rules: inside a string never use the " character; write quoted speech with single quotes ('like this'). No comments and no text after a value."""

PERSONA = """You are casting voice actors for a full-cast radio play of "{book}".

Character: {name} (id {id})
Also known as: {aliases}
Speaks in {chapters} chapters, roughly {lines} lines, first appearing in chapter {first}.
Notes gathered while reading: gender {gender}; age {age}; origin {origin}; class {cls}.
Evidence from the text:
{evidence}
Disguises used: {disguises}

Using this evidence AND everything you know about this character across the whole novel, write the casting brief. A person keeps one voice through the book, though it may age.

Return ONLY JSON:
{{"gender": "...", "age_range": "e.g. 19 at start, about 43 at the end",
  "origin_accent": "...", "temperament": "...",
  "voice": "timbre, pitch, pace, texture: what a casting director needs",
  "delivery_notes": "how the voice shifts with mood, age or disguise",
  "tier": "lead|supporting|minor"}}"""


def chapters(book):
    return json.loads((book / "index.json").read_text())


def cast_summary(cast):
    return "\n".join(f'{c["name"]} | {", ".join(c["aliases"])}' for c in cast.values()) or "(empty)"


MAX_ALIASES = 12
FEMALE_TITLES = {"madame", "mme", "mademoiselle", "mlle", "countess", "comtesse", "baroness", "baronne",
                 "marquise", "duchess", "duchesse", "viscountess", "vicomtesse", "lady", "mrs", "miss",
                 "signora", "mother", "sister", "queen", "princess"}
MALE_TITLES = {"m", "monsieur", "count", "comte", "baron", "marquis", "duke", "duc", "viscount", "vicomte",
               "sir", "lord", "mr", "signor", "father", "brother", "king", "prince", "abbé", "abbe"}
TITLES = FEMALE_TITLES | MALE_TITLES | {"captain", "doctor", "dr", "general", "major", "citizen",
                                        "the", "a", "an", "old", "young", "de", "of", "d"}


def proper_alias(a):
    """Keep an alias only if it carries a proper name ("Baron Danglars", "the Count of Monte Cristo").
    Descriptions ("the stranger", "the young man") would link unrelated people when merging."""
    a = (a or "").strip().strip("\"'“”‘’ ").strip()
    if not a or len(a) > 60:
        return None
    words = re.findall(r"[\w'’\-]+", a)
    if any(w[0].isupper() and w.lower().rstrip(".") not in TITLES for w in words):
        return a
    return None


def _tokens(text):
    import unicodedata
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return {w.lower().rstrip(".") for w in re.findall(r"[A-Za-z'\-]+", t)
            if w[0].isupper() and w.lower().rstrip(".") not in TITLES}


def _title_gender(text):
    words = {w.lower().rstrip(".") for w in re.findall(r"[A-Za-zÀ-ÿ.]+", text or "")}
    if words & FEMALE_TITLES:
        return "female"
    if words & MALE_TITLES:
        return "male"
    return None


def _first_token(text):
    import unicodedata
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    for w in re.findall(r"[A-Za-z'\-]+", t):
        if w[0].isupper() and w.lower().rstrip(".") not in TITLES:
            return w.lower()
    return None


def _shares_name(alias, c):
    """Is this alias the same person as character c? Conservative on purpose:
    family members share surnames and spouses share titles+surnames, so
      - an alias whose names are all already the character's always fits
        ("Dantès", "Captain Edmond Dantès" for Edmond Dantès; "Baron Danglars" for M. Danglars);
      - a longer alias fits only if it LEADS with one of the character's names
        ("Valentine de Villefort" for Valentine, but never "Maximilian Morrel" for M. Morrel
        or "Albert de Morcerf" for the Comte de Morcerf);
      - a title of the other gender never fits ("Madame Danglars" is not M. Danglars)."""
    at = _tokens(alias)
    if not at:
        return False
    g_alias = _title_gender(alias)
    g_char = _title_gender(c["name"]) or next(iter(_genders(c)), None)
    if g_alias and g_char and g_alias != g_char:
        return False
    own = _tokens(c["name"])
    for a in c["aliases"]:
        own |= _tokens(a)
    if at <= own:
        return True
    name_tokens = _tokens(c["name"])
    return bool(name_tokens) and name_tokens <= at and _first_token(alias) in name_tokens


def _names(c):
    """Proper names only: a shared description is never evidence of the same person."""
    return {p.lower() for p in map(proper_alias, [c["name"]] + c["aliases"]) if p}


def _genders(c):
    return {g.split()[0].lower() for g in c["notes"].get("gender", []) if g}


def slug(name):
    import unicodedata
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_") or "someone"


def _find(cast, names, gender):
    """An existing character sharing a proper name, with no gender conflict."""
    for c in cast.values():
        if names & _names(c):
            g = _genders(c)
            if not (gender and g and gender.split()[0].lower() not in g):
                return c
    return None


def merge(cast, n, delta):
    """Fold one chapter's notes into the cast. Identity is decided here, from proper names:
    the model's own ids are not reused, because a small model reuses one id for different
    people (one 9B filed Albert, Franz and Eugénie under the same "unknown_character_2")."""
    for ch in delta.get("characters", []):
        name = (ch.get("name") or "").strip()
        if not name:
            continue
        aliases = [a for a in map(proper_alias, ch.get("new_aliases", [])) if a]
        keys = {p.lower() for p in [proper_alias(name), *aliases] if p}
        # match on the entry's own name first; its aliases are the model's guesses
        name_key = {p.lower() for p in [proper_alias(name)] if p}
        c = (_find(cast, name_key, ch.get("gender")) if name_key else cast.get(slug(name)))
        if c is None and keys - name_key:
            c = _find(cast, keys - name_key, ch.get("gender"))
            if c is not None and name_key and not _shares_name(name, c):
                c = None  # an alias match that contradicts the entry's own name: treat as a new person
        if c is None:
            cid = slug(proper_alias(name) or name)
            while cid in cast:
                cid += "_2"
            c = cast[cid] = {"id": cid, "name": name, "aliases": [], "epithets": [], "chapters": [],
                             "lines": 0, "evidence": [], "notes": {}, "disguises": []}
        elif not proper_alias(c["name"]) and proper_alias(name):
            c.setdefault("epithets", []).append(c["name"])
            c["name"] = name  # upgrade "the stranger" to a real name once one appears
            c["aliases"] = [a for a in c["aliases"] if a != name]
        for a in [a for a in [proper_alias(name), *aliases] if a]:
            if a == c["name"] or a in c["aliases"]:
                continue
            # a character known only by a description has no name to compare against, so the
            # model's proper names for it are its identity; that is what lets the upgrade above match
            nameless = not proper_alias(c["name"])
            if (nameless or _shares_name(a, c)) and len(c["aliases"]) < MAX_ALIASES:
                c["aliases"].append(a)     # identity: "Baron Danglars" for Danglars
            elif a not in c.setdefault("epithets", []) and len(c["epithets"]) < MAX_ALIASES:
                c["epithets"].append(a)    # display and dialogue tags only, never used for matching
        if ch.get("speaks"):
            c["chapters"].append(n)
            try:
                c["lines"] += int(str(ch.get("lines", 0)).split()[0])
            except ValueError:
                pass
        for k in ("gender", "age", "origin", "class"):
            v = ch.get(k)
            if v and v != "unknown":
                c["notes"].setdefault(k, [])
                if v not in c["notes"][k]:
                    c["notes"][k].append(v)
        c["evidence"] += [f"ch{n}: {e}" for e in ch.get("evidence", [])][:3]
    for d in delta.get("disguises", []):
        who = proper_alias(d.get("name") or d.get("id", "").replace("_", " ").title())
        c = _find(cast, {who.lower()}, None) if who else None
        if c and d.get("as") and d["as"] not in c["disguises"]:
            c["disguises"].append(d["as"])


def collect(book, cfg):
    outdir = book / "cast"
    outdir.mkdir(exist_ok=True)
    cast = {}
    for ch in chapters(book):
        n = ch["n"]
        f = outdir / f"{n:03}.json"
        if not f.exists():
            text = (book / f"{n:03}.txt").read_text()
            prompt = COLLECT.format(cast=cast_summary(cast), n=n, chapter=text)
            delta = None
            for attempt in range(3):
                try:
                    raw = chat(cfg, prompt, max_tokens=cfg.get("cast_max_tokens", 4096))
                    delta = parse_json(raw)
                    break
                except ValueError as e:  # includes llm.Truncated
                    raw = getattr(e, "doc", None) or str(e)
                    log.warning("chapter %d attempt %d: bad JSON (%s)", n, attempt + 1, e)
                    (outdir / f"{n:03}.raw.txt").write_text(raw)
            if delta is None:
                log.error("chapter %d: no valid JSON after 3 attempts, skipped", n)
                continue
            f.write_text(json.dumps(delta, indent=1, ensure_ascii=False))
        merge(cast, n, json.loads(f.read_text()))
        log.info("chapter %d done, cast size %d", n, len(cast))
    return cast


DEDUPE = """Below is the cast list of a novel built while reading it chapter by chapter. Some people were entered twice under different ids (a title, a description like "the man in the skiff", a disguise, a later name).

id | name | aliases | chapters | first evidence
{cast}

Find the ids that are the SAME PERSON. Use the names, aliases, evidence, and what you know of the novel. Do not merge different people who share a family name.

Return ONLY JSON: {{"merges": [{{"keep": "id to keep", "drop": ["ids that are the same person"]}}]}}
Return {{"merges": []}} if there are none."""


def rekey(cast):
    """Ids come from first sightings ("the_cousin"); remake them from final names ("fernand")."""
    out = {}
    for c in cast.values():
        cid = slug(proper_alias(c["name"]) or c["name"])
        while cid in out:
            cid += "_2"
        c["id"] = cid
        out[cid] = c
    return out


def _absorb(cast, keep, drop):
    c = cast.pop(drop["id"])
    log.info("merge %s -> %s", c["id"], keep["id"])
    keep.setdefault("epithets", []).extend(e for e in c.get("epithets", []) if e not in keep["epithets"])
    for a in map(proper_alias, [c["name"]] + c["aliases"] + c["disguises"]):
        if a and a not in keep["aliases"] and a != keep["name"] and len(keep["aliases"]) < MAX_ALIASES:
            keep["aliases"].append(a)
    keep["chapters"] = sorted(set(keep["chapters"] + c["chapters"]))
    keep["lines"] += c["lines"]
    keep["evidence"] += c["evidence"]
    for k, v in c["notes"].items():
        keep["notes"].setdefault(k, [])
        keep["notes"][k] += [x for x in v if x not in keep["notes"][k]]


def _rank(c):
    # prefer the id that speaks most, then a proper name over "The ..." descriptions
    return (len(c["chapters"]), not c["name"].lower().startswith("the "), c["lines"])


def dedupe(book, cfg, cast):
    """Drop non-speakers, auto-merge ids whose names overlap, and save the LLM's
    other merge ideas to cast.suggested-merges.json for human review (not applied)."""
    cast = {k: c for k, c in cast.items() if c["chapters"]}
    # decide every merge from each character's OWN names before anything is absorbed,
    # so aliases picked up in one merge can't chain into the next
    names = {k: _names(c) for k, c in cast.items()}
    ids = list(cast)
    pairs = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            ga, gb = _genders(cast[a]), _genders(cast[b])
            # one character's NAME must be among the other's names; two shared aliases
            # are not enough (father and son both answer to "Morrel")
            na = {p.lower() for p in [proper_alias(cast[a]["name"])] if p}
            nb = {p.lower() for p in [proper_alias(cast[b]["name"])] if p}
            if (na & names[b] or nb & names[a]) and not (ga and gb and ga.isdisjoint(gb)):
                pairs.append((a, b))
    for a, b in pairs:
        if a in cast and b in cast:
            keep, drop = sorted((cast[a], cast[b]), key=_rank, reverse=True)
            _absorb(cast, keep, drop)
    rows = [f'{c["id"]} | {c["name"]} | {", ".join(c["aliases"])} | {c["chapters"][:3]} | '
            f'{(c["evidence"] or [""])[0][:80]}' for c in cast.values()]
    try:
        merges = parse_json(chat(cfg, DEDUPE.format(cast="\n".join(rows)))).get("merges", [])
    except ValueError as e:
        log.error("dedupe: %s", e)
        merges = []
    (book / "cast.suggested-merges.json").write_text(json.dumps(merges, indent=1, ensure_ascii=False))
    log.info("%d suggested merges saved for review", len(merges))
    return rekey(cast)


def personas(book, cfg, cast, title):
    for c in sorted(cast.values(), key=lambda c: -len(c["chapters"])):
        if not c["chapters"] or "persona" in c:
            continue
        n = c["notes"]
        try:
            raw = chat(cfg, PERSONA.format(
            book=title, name=c["name"], id=c["id"], aliases=", ".join(c["aliases"]) or "none",
            chapters=len(c["chapters"]), lines=c["lines"], first=c["chapters"][0],
            gender="/".join(n.get("gender", [])) or "?", age="/".join(n.get("age", [])) or "?",
            origin="/".join(n.get("origin", [])) or "?", cls="/".join(n.get("class", [])) or "?",
            evidence="\n".join(c["evidence"][:40]) or "(none)",
            disguises=", ".join(c["disguises"]) or "none"))
            c["persona"] = parse_json(raw)
        except ValueError as e:  # includes llm.Truncated
            log.error("persona %s: %s", c["id"], e)
        save(book, cast)
        log.info("persona %s done", c["id"])


def save(book, cast):
    (book / "cast.json").write_text(json.dumps(cast, indent=1, ensure_ascii=False))


def main(book_dir, step="all"):
    cfg = load_config()
    book = Path(book_dir)
    if step in ("all", "collect"):
        cast = collect(book, cfg)
        save(book, cast)
    if step in ("all", "dedupe"):
        cast = json.loads((book / "cast.json").read_text())
        cast = dedupe(book, cfg, cast)
        save(book, cast)
    else:
        cast = json.loads((book / "cast.json").read_text())
    if step in ("all", "personas"):
        meta = book / "book.json"
        title = json.loads(meta.read_text())["title"] if meta.exists() else book.name.replace("-", " ").title()
        personas(book, cfg, cast, title)
    log.info("cast: %d characters, %d speaking", len(cast), sum(1 for c in cast.values() if c["chapters"]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(*sys.argv[1:3])
