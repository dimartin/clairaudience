"""Split chapter text into narration and quote segments, deterministically.

Quotes are spans between curly double quotes (or straight ones, if the book has no curly quotes). A quote left open at the end of a
paragraph continues into the next paragraph when that one opens with a quote
(the multi-paragraph speech convention).
"""

import re

QUOTE = re.compile(r"“[^”]*(?:”|$)")
STRAIGHT = re.compile(r'"[^"]*(?:"|$)')


def segments(text):
    """Books with curly quotes use them; books with only straight quotes fall back to those."""
    pattern, strip = (QUOTE, "“”") if "“" in text else (STRAIGHT, '"')
    segs = []
    for para_no, para in enumerate(text.split("\n\n")):
        pos = 0
        for m in pattern.finditer(para):
            if m.start() > pos:
                _add(segs, "narration", para[pos:m.start()], para_no)
            _add(segs, "quote", m.group(0).strip(strip), para_no)
            pos = m.end()
        if pos < len(para):
            _add(segs, "narration", para[pos:], para_no)
    for i, s in enumerate(segs):
        s["id"] = i
    return segs


def _add(segs, kind, text, para):
    text = text.strip()
    if text:
        segs.append({"kind": kind, "text": text, "para": para})
