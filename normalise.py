"""TTS-only normalisation of roman numerals and all-caps words, using the book as its own
lexicon. Like the respellings, it changes only what the TTS reads; the book text and the
scripts stay as written.

Evidence comes from the book, collected once at extract into <book>/words.json:
  known           every word the book writes with a lowercase letter somewhere
  title_numerals  roman numerals that open chapter headings, kept only when the book also
                  has the one before or after (I, II, III count; a lone DIX doesn't)

Rules:
  "Chapter XIV" -> "Chapter fourteen" (a chapter word, then a numeral); narration that
  opens a paragraph with a title numeral ("XIV. The Return", or "XIV" alone) is read as
  the number.
  Any other roman numeral stays as written ("Louis XIV", "Charles II.").
  An all-caps token the book also writes in lowercase is a word ("NO", "SECRET"), whatever
  its length or context. Any other is spelled out ("CIA" -> "C. I. A.") when it is short
  and outside an all-caps passage, or when it can't be said as a word ("HTML").
"""

import json
import logging
import re
from pathlib import Path

log = logging.getLogger("normalise")

SHORT = 3   # caps tokens up to this length are acronyms outside a caps passage (CIA, US, IT)
CHAPTER_WORDS = "chapter|book|part|volume|canto|act|scene|section"
ROMAN = re.compile(r"M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})")
ROMAN_VALUE = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
WORD = re.compile(r"[^\W\d_]+")
CAPS = re.compile(r"(?<![\w’'])([A-Z]{2,})(?![\w’'])")
AFTER_CHAPTER_WORD = re.compile(rf"\b((?i:{CHAPTER_WORDS})\s+)([IVXLCDM]+)(?![\w’'])")
# a numeral alone, or then ". Title", ": Title" or "—Title"; not "I—I—die" or "I went"
OPENS_HEADING = re.compile(r"^(?:(?i:" + CHAPTER_WORDS + r")\s+)?([IVXLCDM]+)(?=[.:]?\s*$|[.:]\s+\S|\s*[—–]\s*[A-Z][a-z])")
ONSETS = set("bl br ch cl cr dr dw fl fr gl gn gr kl kn kr ph pl pr ps sc sh sk sl sm sn sp st sw th tr tw "
             "wh wr sch scr shr spl spr squ str thr".split())
CODAS3 = set("rst nds nts rts rds lds sts nks rks mps nth rth ght tch".split())

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen " \
       "sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def roman_value(s):
    """The value of a well-formed roman numeral (XIV -> 14), or None (IIII, VX, '')."""
    if not s or not ROMAN.fullmatch(s):
        return None
    total = 0
    for a, b in zip(s, s[1:] + " "):
        v = ROMAN_VALUE[a]
        total += -v if b != " " and ROMAN_VALUE[b] > v else v
    return total


def number_words(n):
    """1..3999 in English words: 14 -> fourteen, 1815 -> one thousand eight hundred fifteen."""
    parts = []
    if n >= 1000:
        parts.append(f"{ONES[n // 1000]} thousand")
        n %= 1000
    if n >= 100:
        parts.append(f"{ONES[n // 100]} hundred")
        n %= 100
    if n >= 20:
        parts.append(TENS[n // 10] + (f"-{ONES[n % 10]}" if n % 10 else ""))
    elif n or not parts:
        parts.append(ONES[n])
    return " ".join(parts)


def known_words(texts):
    return {w.lower() for t in texts for w in WORD.findall(t) if not w.isupper()}


def title_numerals(headings):
    found = {m.group(1) for h in headings if (m := OPENS_HEADING.match(h.strip())) and roman_value(m.group(1))}
    values = {roman_value(r) for r in found}
    return sorted(r for r in found if roman_value(r) - 1 in values or roman_value(r) + 1 in values)


def book_words(texts, headings):
    return {"known": sorted(known_words(texts)), "title_numerals": title_numerals(headings)}


def load(path, texts=(), headings=()):
    """words.json as {"known": set, "title_numerals": set}. A book extracted before
    words.json existed gets it built from texts and headings (and saved) instead."""
    path = Path(path)
    if path.exists():
        data = json.loads(path.read_text())
    else:
        data = book_words(texts, headings)
        path.write_text(json.dumps(data, ensure_ascii=False))
        log.info("built %s: %d known words, title numerals %s", path, len(data["known"]), data["title_numerals"])
    return {"known": set(data["known"]), "title_numerals": set(data["title_numerals"])}


def pronounceable(token):
    """A rough English test: a vowel, a sayable onset and coda, no long consonant runs."""
    w = token.lower()
    shape = "".join("v" if c in "aeiou" or (c == "y" and (i + 1 == len(w) or w[i + 1] not in "aeiou")) else "c"
                    for i, c in enumerate(w))
    if "v" not in shape:
        return False
    first, last = shape.index("v"), shape.rindex("v")
    onset, coda = w[:first], w[last + 1:]
    if len(onset) > 1 and onset not in ONSETS:
        return False
    if len(coda) > 2 and coda not in CODAS3:
        return False
    return "cccc" not in shape[first:last + 1]


def _in_caps_passage(text, start, end, known):
    """A neighbouring word, with only spaces or a hyphen between, that is a real word in caps."""
    before = re.search(r"([^\W\d_]+)([\s\-]+)$", text[:start])
    after = re.match(r"([\s\-]+)([^\W\d_]+)", text[end:])
    for m, word in ((before, 1), (after, 2)):
        if m and len(w := m.group(word)) > 1 and w.isupper() and w.lower() in known:
            return True
    return False


def spell_out(text, known):
    def repl(m):
        tok = m.group(1)
        if roman_value(tok):
            return tok
        if tok.lower() in known:   # the book writes it in lowercase: a word, whatever its length
            return tok
        long_ = len(tok) > SHORT
        if pronounceable(tok) and (long_ or _in_caps_passage(text, m.start(), m.end(), known)):
            return tok
        spelled = ". ".join(tok)
        return spelled if text[m.end():m.end() + 1] == "." else spelled + "."
    return CAPS.sub(repl, text)


def numerals(text, title, opens_paragraph):
    def chapter(m):
        v = roman_value(m.group(2))
        return f"{m.group(1)}{number_words(v)}" if v else m.group(0)
    text = AFTER_CHAPTER_WORD.sub(chapter, text)
    m = opens_paragraph and OPENS_HEADING.match(text)
    if m and m.group(1) in title:
        text = text[:m.start(1)] + number_words(roman_value(m.group(1))) + text[m.end(1):]
    return text


def normalise(text, words, opens_paragraph=False):
    """What the TTS should read for text. words: load()'s result. opens_paragraph: text is
    narration at the start of a paragraph, where a heading can be."""
    if not words:
        return text
    out = spell_out(numerals(text, words["title_numerals"], opens_paragraph), words["known"])
    if out != text:
        log.debug("normalised %r -> %r", text, out)
    return out
