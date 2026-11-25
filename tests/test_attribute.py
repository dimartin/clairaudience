"""attribute.py: the deterministic dialogue-tag override, batching, and chapter assembly.
The LLM is replaced by a stub; nothing here loads a model."""

import json

import pytest

import attribute
import segment

CAST = {
    "danglars": {"id": "danglars", "name": "Danglars", "aliases": ["Baron Danglars"], "epithets": []},
    "fernand": {"id": "fernand", "name": "Fernand Mondego", "aliases": [], "epithets": ["the Catalan"]},
    "edmond_dantes": {"id": "edmond_dantes", "name": "Edmond Dantès", "aliases": [], "epithets": []},
}


def script(text, speaker="unknown"):
    segs = segment.segments(text)
    for s in segs:
        s["speaker"] = speaker if s["kind"] == "quote" else "narrator"
    return segs


# --- name_index ---

def test_name_index_keys_full_name_alias_epithet_and_surname():
    idx = attribute.name_index(CAST)
    assert idx["danglars"] == "danglars"
    assert idx["baron danglars"] == "danglars"
    assert idx["fernand mondego"] == "fernand"
    assert idx["mondego"] == "fernand"
    assert idx["the catalan"] == "fernand"
    assert idx["catalan"] == "fernand"
    assert idx["dantès"] == "edmond_dantes"


def test_name_index_first_character_keeps_a_shared_surname():
    cast = {"a": {"id": "a", "name": "Pierre Morrel", "aliases": []},
            "b": {"id": "b", "name": "Maximilian Morrel", "aliases": []}}
    idx = attribute.name_index(cast)
    assert idx["morrel"] == "a"
    assert idx["maximilian morrel"] == "b"


# --- tag_override ---

def test_tag_names_a_cast_member_and_overrides_the_model():
    segs = script("“Good day,” said Danglars.", speaker="fernand")
    assert attribute.tag_override(segs, attribute.name_index(CAST)) == 1
    assert segs[0]["speaker"] == "danglars"
    assert segs[0]["source"] == "tag"


@pytest.mark.parametrize("tag, who", [
    ("replied the Catalan.", "fernand"),
    ("resumed Mondego, angrily.", "fernand"),
    ("said Dantès.", "edmond_dantes"),
    ("then he added Danglars.", "danglars"),   # up to three words before the verb
])
def test_tag_forms_resolve_to_the_cast(tag, who):
    segs = script(f"“Yes,” {tag}")
    attribute.tag_override(segs, attribute.name_index(CAST))
    assert segs[0]["speaker"] == who


def test_first_name_tag_resolves():
    segs = script("“Yes,” said Fernand.")
    attribute.tag_override(segs, attribute.name_index(CAST))
    assert segs[0]["speaker"] == "fernand"


def test_shared_first_name_is_not_indexed():
    cast = {"a": {"id": "a", "name": "Edouard Villefort", "aliases": []},
            "b": {"id": "b", "name": "Edouard Morrel", "aliases": []}}
    assert "edouard" not in attribute.name_index(cast)


def test_first_name_never_displaces_someone_elses_surname():
    cast = {"a": {"id": "a", "name": "Albert", "aliases": []},
            "b": {"id": "b", "name": "Danglars", "aliases": []},
            "c": {"id": "c", "name": "Danglars Albert", "aliases": []}}
    assert attribute.name_index(cast)["danglars"] == "b"


def test_matching_speaker_is_not_counted_as_a_fix_but_is_a_tag():
    segs = script("“Good day,” said Danglars.", speaker="danglars")
    assert attribute.tag_override(segs, attribute.name_index(CAST)) == 0
    assert segs[0]["source"] == "tag"


def test_tag_never_changes_a_hand_edit():
    segs = script("“Good day,” said Danglars.", speaker="fernand")
    segs[0]["source"] = "user"
    assert attribute.tag_override(segs, attribute.name_index(CAST)) == 0
    assert (segs[0]["speaker"], segs[0]["source"]) == ("fernand", "user")


def test_tag_in_the_next_paragraph_does_not_apply():
    segs = script("“Good day.”\n\nsaid Danglars, later.", speaker="fernand")
    assert attribute.tag_override(segs, attribute.name_index(CAST)) == 0
    assert segs[0]["speaker"] == "fernand"


def test_name_before_the_verb_is_not_a_tag():
    # "Danglars said nothing" is narration about Danglars, not a tag for the quote
    segs = script("“Good day.” Danglars said nothing.", speaker="fernand")
    assert attribute.tag_override(segs, attribute.name_index(CAST)) == 0


def test_name_outside_the_cast_leaves_the_model_label():
    segs = script("“Good day,” said Caderousse.", speaker="fernand")
    assert attribute.tag_override(segs, attribute.name_index(CAST)) == 0
    assert segs[0]["speaker"] == "fernand"


def test_lowercase_description_is_not_a_name():
    segs = script("“Good day,” said the young man.", speaker="fernand")
    assert attribute.tag_override(segs, attribute.name_index(CAST)) == 0


def test_leading_narration_is_skipped():
    segs = script("said Danglars, to begin with.", speaker="fernand")
    assert attribute.tag_override(segs, attribute.name_index(CAST)) == 0


def test_title_prefixed_tag_resolves_by_surname():
    cast = {"morrel": {"id": "morrel", "name": "Pierre Morrel", "aliases": [],
                       "notes": {"gender": ["male"]}}}
    segs = script("“Welcome,” said M. Morrel.", speaker="unknown")
    attribute.tag_override(segs, attribute.name_index(cast))
    assert segs[0]["speaker"] == "morrel"


def test_title_of_the_other_gender_does_not_resolve():
    # "said Madame Danglars" must never fall through to her husband
    cast = {"danglars": {"id": "danglars", "name": "Danglars", "aliases": [], "notes": {"gender": ["male"]}}}
    segs = script("“Never,” said Madame Danglars.", speaker="unknown")
    attribute.tag_override(segs, attribute.name_index(cast))
    assert segs[0]["speaker"] == "unknown"


def test_title_keys_follow_gender():
    cast = {"h": {"id": "h", "name": "Danglars", "aliases": [], "notes": {"gender": ["male"]}},
            "w": {"id": "w", "name": "Hermine Danglars", "aliases": [], "notes": {"gender": ["female"]}},
            "x": {"id": "x", "name": "Caderousse", "aliases": []}}
    idx = attribute.name_index(cast)
    assert idx["m. danglars"] == "h"
    assert idx["madame danglars"] == "w"
    assert idx["mademoiselle danglars"] == "w"
    assert "m. caderousse" not in idx and "madame caderousse" not in idx   # gender unknown


# --- batches ---

def quotes(n):
    return [{"kind": "quote" if i % 2 else "narration"} for i in range(2 * n)]


@pytest.mark.parametrize("n", [0, 1, 119, 120, 121, 240, 241, 500])
def test_batches_cover_every_segment_once_and_respect_the_cap(n):
    segs = quotes(n)
    out = attribute.batches(segs)
    assert out[0][0] == 0 and out[-1][1] == len(segs)
    assert all(a[1] == b[0] for a, b in zip(out, out[1:]))
    counts = [sum(s["kind"] == "quote" for s in segs[lo:hi]) for lo, hi in out]
    assert sum(counts) == n
    assert max(counts) <= attribute.BATCH_QUOTES


def test_batches_are_even_rather_than_full_plus_remainder():
    counts = [sum(s["kind"] == "quote" for s in quotes(121)[lo:hi])
              for lo, hi in attribute.batches(quotes(121))]
    assert sorted(counts) == [60, 61]


# --- label_batch / attribute_chapter (LLM stubbed) ---

def test_label_batch_gives_up_after_three_bad_answers(monkeypatch):
    calls = []
    monkeypatch.setattr(attribute, "chat", lambda cfg, prompt: calls.append(prompt) or "not json")
    segs = script("“Hi.”")
    assert attribute.label_batch({}, 1, segs, 0, len(segs), "") == {}
    assert len(calls) == 3


def test_attribute_chapter_writes_the_script(tmp_path, monkeypatch):
    (tmp_path / "001.txt").write_text(
        "“Good day,” said Danglars.\n\n“Who goes there?”\n\n“A friend.” The wind rose.\n\n“Nobody.”")
    answer = {"quotes": {
        "0": {"speaker": "fernand", "emotion": "angry"},     # overridden by the tag
        "2": {"speaker": "edmond_dantes", "emotion": "fearful"},
        "3": {"speaker": "villefort", "emotion": "calm"},     # not in the cast
        # Q5 missing from the answer
    }}
    monkeypatch.setattr(attribute, "chat", lambda cfg, prompt: json.dumps(answer))
    attribute.attribute_chapter({}, tmp_path, 1, CAST, "", attribute.name_index(CAST))

    segs = json.loads((tmp_path / "001.script.json").read_text())["segments"]
    by_id = {s["id"]: s for s in segs}
    assert (by_id[0]["speaker"], by_id[0]["source"]) == ("danglars", "tag")
    assert by_id[0]["emotion"] == "angry"
    assert (by_id[2]["speaker"], by_id[2]["emotion"]) == ("edmond_dantes", "fearful")
    assert by_id[3]["speaker"] == "unknown"
    assert (by_id[5]["speaker"], by_id[5]["emotion"]) == ("unknown", "neutral")
    assert by_id[2]["source"] == "llm"
    assert by_id[3]["source"] == by_id[5]["source"] == "unknown"
    assert all(s["speaker"] == "narrator" and "source" not in s for s in segs if s["kind"] == "narration")
    # the book text is never changed
    assert [s["text"] for s in segs] == [s["text"] for s in segment.segments((tmp_path / "001.txt").read_text())]


def test_chapter_without_quotes_never_calls_the_model(tmp_path, monkeypatch):
    (tmp_path / "002.txt").write_text("The sea was calm.")
    monkeypatch.setattr(attribute, "chat", lambda *a: pytest.fail("LLM called"))
    attribute.attribute_chapter({}, tmp_path, 2, CAST, "", {})
    assert json.loads((tmp_path / "002.script.json").read_text())["segments"][0]["speaker"] == "narrator"


# --- read_label: whatever shape the model answers in ---


@pytest.mark.parametrize("label, want", [
    ({"speaker": "danglars", "emotion": "angry"}, ("danglars", "angry")),
    ({"speaker": {"id": "danglars", "name": "Danglars"}, "emotion": "calm"}, ("danglars", "calm")),
    ({"speaker": {"name": "fernand"}}, ("fernand", "neutral")),
    ({"speaker": {"name": "Somebody Else"}}, ("unknown", "neutral")),
    ({"speaker": ["danglars"]}, ("unknown", "neutral")),
    ({"speaker": None, "emotion": None}, ("unknown", "neutral")),
    ({"speaker": "villefort"}, ("unknown", "neutral")),
    ("edmond_dantes", ("edmond_dantes", "neutral")),
    (["danglars"], ("unknown", "neutral")),
    ({}, ("unknown", "neutral")),
])
def test_read_label_accepts_any_shape(label, want):
    assert attribute.read_label(label, CAST) == want


def test_a_dict_speaker_no_longer_crashes_the_chapter(tmp_path, monkeypatch):
    """The Monte Cristo run3 crash: chapter 53's answer nested a speaker as an object."""
    (tmp_path / "053.txt").write_text("“Who goes there?”\n\n“A friend.”")
    answer = {"quotes": {"0": {"speaker": {"id": "danglars", "name": "Danglars"}, "emotion": "wary"},
                         "1": {"speaker": {"oops": 1}}}}
    monkeypatch.setattr(attribute, "chat", lambda cfg, prompt: json.dumps(answer))
    attribute.attribute_chapter({}, tmp_path, 53, CAST, "", attribute.name_index(CAST))
    segs = json.loads((tmp_path / "053.script.json").read_text())["segments"]
    assert [(s["speaker"], s["emotion"]) for s in segs] == [("danglars", "wary"), ("unknown", "neutral")]


# --- source: hand edits survive a rerun ---


def test_rerun_keeps_hand_edited_lines(tmp_path, monkeypatch, caplog):
    (tmp_path / "001.txt").write_text("“Good day,” said Danglars.\n\n“Who goes there?”\n\n“A friend.”")
    answer = {"quotes": {"0": {"speaker": "danglars"}, "2": {"speaker": "danglars"}, "3": {"speaker": "danglars"}}}
    monkeypatch.setattr(attribute, "chat", lambda cfg, prompt: json.dumps(answer))
    attribute.attribute_chapter({}, tmp_path, 1, CAST, "", attribute.name_index(CAST))

    path = tmp_path / "001.script.json"
    data = json.loads(path.read_text())
    by_id = {s["id"]: s for s in data["segments"]}
    by_id[0].update(speaker="fernand", source="user")        # against the tag: the person wins
    by_id[2].update(speaker="edmond_dantes", emotion="fearful", source="user")
    by_id[3].update(speaker="fernand")                      # edited without marking it: not kept
    path.write_text(json.dumps(data))

    with caplog.at_level("INFO", logger="attribute"):
        attribute.attribute_chapter({}, tmp_path, 1, CAST, "", attribute.name_index(CAST))
    by_id = {s["id"]: s for s in json.loads(path.read_text())["segments"]}
    assert (by_id[0]["speaker"], by_id[0]["source"]) == ("fernand", "user")
    assert (by_id[2]["speaker"], by_id[2]["emotion"], by_id[2]["source"]) == ("edmond_dantes", "fearful", "user")
    assert (by_id[3]["speaker"], by_id[3]["source"]) == ("danglars", "llm")
    assert "tag 0, tag-resolved 0, llm 1, unknown 0, user 2; llm share 33%" in caplog.text


def test_a_hand_edit_whose_text_changed_is_dropped(tmp_path, monkeypatch):
    (tmp_path / "001.txt").write_text("“Who goes there?”")
    (tmp_path / "001.script.json").write_text(json.dumps({"segments": [
        {"kind": "quote", "id": 0, "text": "Who went there?", "speaker": "fernand", "source": "user"}]}))
    monkeypatch.setattr(attribute, "chat", lambda cfg, prompt: json.dumps({"quotes": {"0": {"speaker": "danglars"}}}))
    attribute.attribute_chapter({}, tmp_path, 1, CAST, "", attribute.name_index(CAST))
    seg = json.loads((tmp_path / "001.script.json").read_text())["segments"][0]
    assert (seg["speaker"], seg["source"]) == ("danglars", "llm")


# --- nameless tags: two blind reads, anchored only on agreement ---


@pytest.mark.parametrize("tag, who", [
    ("said she.", "she"),
    ("he replied.", "he"),
    ("then she added, softly.", "she"),
    ("said his father, smiling.", "his father"),
    ("replied the young man.", "the young man"),
])
def test_nameless_tags_are_found(tag, who):
    segs = script(f"“Yes,” {tag}", speaker="fernand")
    [(qi, clause, got)] = attribute.nameless_tags(segs)
    assert (qi, got) == (0, who)
    assert clause == tag.split(",")[0].rstrip(".")


@pytest.mark.parametrize("text", [
    "“Yes,” and he swallowed another glass.",     # narration, not a tag
    "“Yes.”\n\nshe said.",                         # tag in the next paragraph
    "“Yes,” said Danglars.",                       # a named tag is the tag rule's job
])
def test_not_nameless_tags(text):
    segs = script(text, speaker="fernand")
    attribute.tag_override(segs, attribute.name_index(CAST))
    assert attribute.nameless_tags(segs) == []


def test_a_hand_edit_is_never_asked_about():
    segs = script("“Yes,” said she.")
    segs[0]["source"] = "user"
    assert attribute.nameless_tags(segs) == []


def reads(monkeypatch, answers):
    """Stub chat: answers[seed] is the reply; records every call."""
    calls = []

    def chat(cfg, prompt, max_tokens=None, seed=None, schema=None):
        calls.append({"model": cfg.get("llm_model"), "seed": seed, "prompt": prompt, "schema": schema})
        a = answers[seed]
        return a.pop(0) if isinstance(a, list) else a

    monkeypatch.setattr(attribute, "chat", chat)
    return calls


def tagged(speaker="edmond_dantes"):
    segs = script("Danglars looked at Fernand.\n\n“You love her,” said he.", speaker=speaker)
    for s in segs:
        if s["kind"] == "quote":
            s["source"] = "llm"
    return segs


def test_agreeing_reads_anchor_the_speaker(monkeypatch):
    calls = reads(monkeypatch, {1: '{"speaker": "danglars"}', 2: '{"speaker": "danglars"}'})
    segs = tagged()
    assert attribute.resolve_nameless({"llm_model": "m"}, segs, CAST) == (1, 1)
    q = segs[1]
    assert (q["speaker"], q["source"]) == ("danglars", "tag-resolved")
    assert [c["seed"] for c in calls] == [1, 2]
    # blind: both reads get the same prompt, which carries neither the chapter label nor the other answer
    assert calls[0]["prompt"] == calls[1]["prompt"]
    assert ">>> \"You love her,\"" in calls[0]["prompt"] and '"said he"' in calls[0]["prompt"]
    # only the chapter's cast (named in it, or labelled as a speaker) is offered, as a closed set
    assert calls[0]["schema"]["properties"]["speaker"]["enum"] == ["danglars", "edmond_dantes", "fernand", "unknown"]


def test_chapter_cast_is_who_is_named_or_speaks():
    segs = script("Danglars looked at the Catalan.\n\n“Yes.”", speaker="unknown")
    assert attribute.chapter_cast(segs, CAST) == ["danglars", "fernand"]


@pytest.mark.parametrize("a, b", [
    ('{"speaker": "danglars"}', '{"speaker": "fernand"}'),
    ('{"speaker": "unknown"}', '{"speaker": "unknown"}'),
    ('{"speaker": "danglars"}', '{"speaker": "unknown"}'),
])
def test_disagreement_or_unknown_keeps_the_guess(monkeypatch, a, b):
    reads(monkeypatch, {1: a, 2: b})
    segs = tagged()
    assert attribute.resolve_nameless({}, segs, CAST) == (1, 0)
    assert (segs[1]["speaker"], segs[1]["source"]) == ("edmond_dantes", "llm")


def test_an_answer_outside_the_cast_is_retried(monkeypatch):
    calls = reads(monkeypatch, {1: ['{"speaker": "villefort"}', "no json", '{"speaker": "fernand"}'],
                                2: '{"speaker": "fernand"}'})
    segs = tagged()
    assert attribute.resolve_nameless({}, segs, CAST) == (1, 1)
    assert segs[1]["speaker"] == "fernand" and len(calls) == 4


def test_three_bad_answers_count_as_no_read(monkeypatch):
    reads(monkeypatch, {1: "no json", 2: '{"speaker": "fernand"}'})
    segs = tagged()
    assert attribute.resolve_nameless({}, segs, CAST) == (1, 0)
    assert segs[1]["source"] == "llm"


def test_second_read_uses_the_check_model(monkeypatch):
    calls = reads(monkeypatch, {1: '{"speaker": "fernand"}', 2: '{"speaker": "fernand"}'})
    attribute.resolve_nameless({"llm_model": "big", "tag_check_model": "small"}, tagged(), CAST)
    assert [c["model"] for c in calls] == ["big", "small"]


def test_no_nameless_tag_never_calls_the_model(monkeypatch):
    monkeypatch.setattr(attribute, "chat", lambda *a, **k: pytest.fail("LLM called"))
    assert attribute.resolve_nameless({}, script("“Yes,” said Danglars."), CAST) == (0, 0)
