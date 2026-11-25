"""Split an EPUB into plain-text chapters, in spine order.

Usage: python extract.py books/monte-cristo.epub out/monte-cristo
Writes NNN.txt per spine document that has a chapter heading (an h1-h3 starting "Chapter "),
plus index.json, and words.json: the book's own lexicon for reading numerals and caps aloud
(normalise.py). Its headings are the h1-h3 headings of those kept documents only; a skipped
document's headings are not recorded.
"""

import json
import logging
import posixpath
import re
import sys
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree as ET

import normalise

log = logging.getLogger("extract")

NS = {
    "c": "urn:oasis:names:tc:opendocument:xmlns:container",
    "opf": "http://www.idpf.org/2007/opf",
}
BLOCK = {"p", "div", "h1", "h2", "h3", "h4", "br", "li", "blockquote"}
SKIP = {"script", "style", "head"}


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self.heading, self.headings, self._in_h, self._skip = [], None, [], False, 0

    def handle_starttag(self, tag, attrs):
        if tag in SKIP:
            self._skip += 1
        if tag in ("h1", "h2", "h3"):
            self._in_h, self._h = True, []
        if tag in BLOCK:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in SKIP:
            self._skip -= 1
        if tag in ("h1", "h2", "h3") and self._in_h:
            self._in_h = False
            text = " ".join("".join(self._h).split())
            if text:
                self.headings.append(text)
            if text and self.heading is None and re.match(r"chapter\s", text, re.I):
                self.heading = text
        if tag in BLOCK:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if self._skip:
            return
        if self._in_h:
            self._h.append(data)
        self.parts.append(data)

    def text(self):
        raw = "".join(self.parts)
        paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", raw)]
        return "\n\n".join(p for p in paras if p)


DC = "{http://purl.org/dc/elements/1.1/}"


def book_meta(zf):
    container = ET.fromstring(zf.read("META-INF/container.xml"))
    opf = ET.fromstring(zf.read(container.find(".//c:rootfile", NS).get("full-path")))
    md = opf.find("opf:metadata", NS)
    get = lambda tag: (md.findtext(DC + tag) or "").strip()
    return {"title": get("title"), "author": get("creator").split(" and ")[0], "language": get("language")}


def spine_docs(zf):
    container = ET.fromstring(zf.read("META-INF/container.xml"))
    opf_path = container.find(".//c:rootfile", NS).get("full-path")
    opf = ET.fromstring(zf.read(opf_path))
    base = posixpath.dirname(opf_path)
    manifest = {i.get("id"): i.get("href") for i in opf.find("opf:manifest", NS)}
    for ref in opf.find("opf:spine", NS):
        yield posixpath.join(base, manifest[ref.get("idref")])


def main(epub, outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    index, texts, headings = [], [], []
    with zipfile.ZipFile(epub) as zf:
        for path in spine_docs(zf):
            p = TextParser()
            p.feed(zf.read(path).decode("utf-8", errors="replace"))
            if not p.heading:
                log.debug("skip %s (heading=%r)", path, p.heading)
                continue
            n = len(index) + 1
            text = p.text()
            (out / f"{n:03}.txt").write_text(text)
            texts.append(text)
            headings += p.headings
            index.append({"n": n, "title": p.heading, "chars": len(text),
                          "quotes": text.count("“") + text.count('"'), "src": path})
        (out / "book.json").write_text(json.dumps(book_meta(zf), indent=1, ensure_ascii=False))
    (out / "index.json").write_text(json.dumps(index, indent=1))
    words = normalise.book_words(texts, headings)
    (out / "words.json").write_text(json.dumps(words, ensure_ascii=False))
    log.info("words.json: %d known words, title numerals %s", len(words["known"]), words["title_numerals"])
    log.info("wrote %d chapters to %s", len(index), out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main(*sys.argv[1:3])
