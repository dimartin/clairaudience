"""normalise.py: roman numerals and caps words as the TTS should read them. Pure Python."""

import json

import pytest

import normalise as N

# a slice of what Monte Cristo writes in lowercase somewhere (it has all of these)
KNOWN = {"volume", "one", "myself", "chapter", "secret", "top", "files", "the", "louis"}
WORDS = {"known": KNOWN, "title_numerals": {"I", "II", "III", "XIII", "XIV", "XV"}}


@pytest.mark.parametrize("roman, value", [("I", 1), ("IV", 4), ("XIV", 14), ("XVIII", 18), ("MDCCCXV", 1815),
                                          ("DIX", 509), ("IIII", None), ("VX", None), ("IC", None), ("", None)])
def test_roman_value(roman, value):
    assert N.roman_value(roman) == value


@pytest.mark.parametrize("n, words", [(1, "one"), (14, "fourteen"), (20, "twenty"), (42, "forty-two"),
                                      (100, "one hundred"), (117, "one hundred seventeen"),
                                      (1815, "one thousand eight hundred fifteen")])
def test_number_words(n, words):
    assert N.number_words(n) == words


def test_title_numerals_need_a_neighbour():
    heads = ["Chapter I. Marseilles", "II. Father and Son", "Chapter III", "DIX", "Chapter IX: Alone", "Contents"]
    assert N.title_numerals(heads) == ["I", "II", "III"]


def test_a_lone_title_I_is_not_a_number():
    assert N.title_numerals(["I", "Prelude", "Envoi"]) == []


def test_known_words_are_the_ones_written_in_lowercase():
    assert N.known_words(["VOLUME ONE\n\nThe first volume. He said NOTHING. Nothing."]) == \
        {"the", "first", "volume", "he", "said", "nothing"}


@pytest.mark.parametrize("text, said", [
    # roman numerals: converted next to a chapter word, kept as written elsewhere
    ("CHAPTER XIV", "CHAPTER fourteen"),
    ("Chapter XIV. The Return", "Chapter fourteen. The Return"),
    ("Book II, Chapter IX", "Book two, Chapter nine"),
    ("domum,” continued Louis XVIII., still annotating.", "domum,” continued Louis XVIII., still annotating."),
    ("after Cromwell, Charles II., and then James II.", "after Cromwell, Charles II., and then James II."),
    ("a lone DIX", "a lone DIX"),
    # caps words: Monte Cristo's own
    ("VOLUME ONE", "VOLUME ONE"),
    ("I will do it MYSELF!", "I will do it MYSELF!"),
    # acronyms
    ("the CIA files", "the C. I. A. files"),
    ("He joined the CIA.", "He joined the C. I. A."),
    ("HTML pages", "H. T. M. L. pages"),
    ("NATO forces", "NATO forces"),                  # unknown but sayable, and not short
    ("TOP SECRET CIA FILES", "TOP SECRET CIA FILES"),  # a caps passage
    ("TOP SECRET FBI FILES", "TOP SECRET F. B. I. FILES"),  # can't be said as a word
    ("DANTÈS", "DANTÈS"),                            # accented caps are left alone
    ("I said so.", "I said so."),
])
def test_normalise(text, said):
    assert N.normalise(text, WORDS) == said


@pytest.mark.parametrize("text, opens, said", [
    ("XIV. The Return", True, "fourteen. The Return"),
    ("XIV", True, "fourteen"),
    ("XIV. The Return", False, "XIV. The Return"),    # not a paragraph start: not a heading
    ("XIV—The Return", True, "fourteen—The Return"),
    ("I—I—die—I——", True, "I—I—die—I——"),            # Monte Cristo, ch. 17: a stammer, not a heading
    ("I went home.", True, "I went home."),
    ("IX. Alone", True, "IX. Alone"),                 # IX isn't one of the book's title numerals
])
def test_heading_numerals_only_open_a_paragraph(text, opens, said):
    assert N.normalise(text, WORDS, opens_paragraph=opens) == said


def test_no_words_means_no_change():
    assert N.normalise("the CIA", None) == "the CIA"


def test_load_builds_and_saves_words_json_when_missing(tmp_path):
    path = tmp_path / "words.json"
    words = N.load(path, ["The SECRET is out."], ["Chapter I", "Chapter II"])
    assert words == {"known": {"the", "is", "out"}, "title_numerals": {"I", "II"}}
    assert json.loads(path.read_text())["title_numerals"] == ["I", "II"]
    assert N.load(path) == words
