"""extract.py on a two-chapter EPUB built in the test: chapter texts, index and words.json."""

import json
import zipfile

import extract

CONTAINER = """<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
<rootfiles><rootfile full-path="OEBPS/book.opf"/></rootfiles></container>"""
OPF = """<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf"
 xmlns:dc="http://purl.org/dc/elements/1.1/"><metadata><dc:title>T</dc:title><dc:creator>A</dc:creator></metadata>
<manifest><item id="a" href="a.xhtml"/><item id="b" href="b.xhtml"/></manifest>
<spine><itemref idref="a"/><itemref idref="b"/></spine></package>"""


def doc(heading, body):
    return f"<html><body><h2>{heading}</h2><p>{body}</p></body></html>"


def test_extract_writes_the_book_lexicon(tmp_path):
    epub = tmp_path / "t.epub"
    with zipfile.ZipFile(epub, "w") as z:
        z.writestr("META-INF/container.xml", CONTAINER)
        z.writestr("OEBPS/book.opf", OPF)
        z.writestr("OEBPS/a.xhtml", doc("Chapter I. The Arrival", "The SECRET was out. Louis XIV knew the chapter of the arrival."))
        z.writestr("OEBPS/b.xhtml", doc("Chapter II. Father", "TOP SECRET, said the CIA to his father."))
    extract.main(str(epub), str(tmp_path / "out"))
    words = json.loads((tmp_path / "out/words.json").read_text())
    assert words["title_numerals"] == ["I", "II"]
    assert {"the", "chapter", "arrival", "father", "knew"} <= set(words["known"])
    # "Louis" is only ever capitalised, the caps words only in caps: none of them is known
    assert not {"louis", "secret", "top", "cia", "xiv"} & set(words["known"])
    assert words["headings"] == ["Chapter I. The Arrival", "Chapter II. Father"]
    # the chapter text itself is unchanged
    assert "Louis XIV knew the chapter" in (tmp_path / "out/001.txt").read_text()
