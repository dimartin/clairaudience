"""cast.py: identity decisions made in code (alias matching, merge, dedupe).
The LLM is replaced by a stub; nothing here loads a model."""

import json

import pytest

import cast as C


def char(name, aliases=(), gender=None, chapters=(1,), lines=1):
    c = {"id": C.slug(name), "name": name, "aliases": list(aliases), "epithets": [],
         "chapters": list(chapters), "lines": lines, "evidence": [], "disguises": [],
         "notes": {"gender": [gender]} if gender else {}}
    return c


def entry(name, aliases=(), gender=None, speaks=True, lines="1", evidence=()):
    return {"name": name, "new_aliases": list(aliases), "speaks": speaks, "lines": lines,
            "gender": gender, "evidence": list(evidence)}


# --- proper_alias / slug ---

@pytest.mark.parametrize("alias", ["Baron Danglars", "the Count of Monte Cristo", "Dantès", "Abbé Busoni"])
def test_proper_alias_keeps_names(alias):
    assert C.proper_alias(alias) == alias


@pytest.mark.parametrize("alias", ["the stranger", "the young man", "Madame", "M.", "the Count", "", None,
                                   "A" * 61])
def test_proper_alias_drops_descriptions_titles_and_junk(alias):
    assert C.proper_alias(alias) is None


def test_proper_alias_strips_quote_marks():
    assert C.proper_alias("“Sinbad”") == "Sinbad"


@pytest.mark.parametrize("name, expected", [
    ("Edmond Dantès", "edmond_dantes"),
    ("M. de Villefort", "m_de_villefort"),
    ("  ", "someone"),
])
def test_slug(name, expected):
    assert C.slug(name) == expected


# --- _shares_name: the examples in its docstring ---

@pytest.mark.parametrize("alias, name, aliases, expected", [
    ("Dantès", "Edmond Dantès", [], True),
    ("Captain Edmond Dantès", "Edmond Dantès", [], True),
    ("Baron Danglars", "M. Danglars", [], True),
    ("Valentine de Villefort", "Valentine", [], True),
    ("Maximilian Morrel", "M. Morrel", [], False),
    ("Albert de Morcerf", "Comte de Morcerf", [], False),
    ("Madame Danglars", "M. Danglars", [], False),
    ("Monte Cristo", "Edmond Dantès", ["the Count of Monte Cristo"], True),
    ("the stranger", "Edmond Dantès", [], False),
])
def test_shares_name(alias, name, aliases, expected):
    assert C._shares_name(alias, char(name, aliases)) is expected


def test_shares_name_rejects_opposite_gender_from_notes():
    assert C._shares_name("Madame Danglars", char("Danglars", gender="male")) is False


# --- merge ---

def test_merge_creates_then_extends_one_character():
    cast = {}
    C.merge(cast, 1, {"characters": [entry("Edmond Dantès", gender="male", lines="12 lines",
                                           evidence=["a", "b", "c", "d"])]})
    C.merge(cast, 2, {"characters": [entry("Edmond Dantès", ["Dantès"], gender="male", lines="3")]})
    assert list(cast) == ["edmond_dantes"]
    c = cast["edmond_dantes"]
    assert c["chapters"] == [1, 2]
    assert c["lines"] == 15
    assert c["aliases"] == ["Dantès"]
    assert c["evidence"] == ["ch1: a", "ch1: b", "ch1: c"]
    assert c["notes"]["gender"] == ["male"]


def test_merge_ignores_unparseable_line_counts_and_non_speakers():
    cast = {}
    C.merge(cast, 1, {"characters": [entry("Danglars", lines="several")]})
    C.merge(cast, 2, {"characters": [entry("Danglars", speaks=False, lines="9")]})
    assert cast["danglars"]["lines"] == 0
    assert cast["danglars"]["chapters"] == [1]


def test_merge_does_not_reuse_the_models_ids():
    cast = {}
    C.merge(cast, 1, {"characters": [{**entry("Albert de Morcerf"), "id": "unknown_character_2"},
                                     {**entry("Franz d'Épinay"), "id": "unknown_character_2"}]})
    assert sorted(cast) == ["albert_de_morcerf", "franz_d_epinay"]


def test_merge_keeps_a_son_apart_from_his_father():
    cast = {}
    C.merge(cast, 1, {"characters": [entry("M. Morrel", gender="male")]})
    C.merge(cast, 2, {"characters": [entry("Maximilian Morrel", ["M. Morrel"], gender="male")]})
    assert sorted(cast) == ["m_morrel", "maximilian_morrel"]
    assert cast["m_morrel"]["aliases"] == []


def test_merge_gender_conflict_makes_a_new_character():
    cast = {}
    C.merge(cast, 1, {"characters": [entry("Danglars", gender="male")]})
    C.merge(cast, 2, {"characters": [entry("Danglars", gender="female")]})
    assert sorted(cast) == ["danglars", "danglars_2"]


def test_merge_files_unrelated_names_as_epithets_not_aliases():
    cast = {}
    C.merge(cast, 1, {"characters": [entry("Edmond Dantès", ["Sinbad the Sailor"])]})
    c = cast["edmond_dantes"]
    assert c["aliases"] == []
    assert c["epithets"] == ["Sinbad the Sailor"]


def test_merge_caps_aliases():
    cast = {}
    many = [f"Captain Edmond Dantès {'I' * i}" for i in range(1, 20)]
    C.merge(cast, 1, {"characters": [entry("Edmond Dantès", many)]})
    assert len(cast["edmond_dantes"]["aliases"]) <= C.MAX_ALIASES


def test_merge_records_disguises():
    cast = {}
    C.merge(cast, 1, {"characters": [entry("Edmond Dantès")],
                      "disguises": [{"name": "Edmond Dantès", "as": "Abbé Busoni"},
                                    {"name": "Edmond Dantès", "as": "Abbé Busoni"},
                                    {"name": "Nobody Here", "as": "Lord Wilmore"}]})
    assert cast["edmond_dantes"]["disguises"] == ["Abbé Busoni"]


def test_merge_upgrades_a_description_once_a_name_appears():
    cast = {}
    C.merge(cast, 1, {"characters": [entry("the stranger", ["Sinbad the Sailor"], lines="2")]})
    C.merge(cast, 2, {"characters": [entry("Sinbad the Sailor", ["the stranger"], lines="3")]})
    assert len(cast) == 1
    c = next(iter(cast.values()))
    assert c["name"] == "Sinbad the Sailor"
    assert c["aliases"] == []
    assert "the stranger" in c["epithets"]
    assert (c["chapters"], c["lines"]) == ([1, 2], 5)


def test_a_named_character_still_keeps_unrelated_names_as_epithets():
    # the nameless rule must not loosen matching for characters that have a name
    cast = {}
    C.merge(cast, 1, {"characters": [entry("Edmond Dantès", ["Sinbad the Sailor"])]})
    C.merge(cast, 2, {"characters": [entry("Sinbad the Sailor")]})
    assert sorted(cast) == ["edmond_dantes", "sinbad_the_sailor"]


# --- dedupe / rekey (LLM stubbed) ---

def stub_chat(monkeypatch, reply):
    calls = []
    monkeypatch.setattr(C, "chat", lambda cfg, prompt: calls.append(prompt) or reply)
    return calls


def test_dedupe_merges_on_shared_name_and_drops_non_speakers(tmp_path, monkeypatch):
    stub_chat(monkeypatch, '{"merges": []}')
    cast = {c["id"]: c for c in [
        char("Danglars", gender="male", chapters=[1, 2, 3], lines=30),
        char("Baron Danglars", ["Danglars"], gender="male", chapters=[9], lines=4),
        char("Mercédès", gender="female", chapters=[]),
    ]}
    out = C.dedupe(tmp_path, {}, cast)
    assert list(out) == ["danglars"]
    d = out["danglars"]
    assert d["chapters"] == [1, 2, 3, 9]
    assert d["lines"] == 34
    assert "Baron Danglars" in d["aliases"]


def test_dedupe_keeps_father_and_son_who_share_an_alias(tmp_path, monkeypatch):
    stub_chat(monkeypatch, '{"merges": []}')
    cast = {c["id"]: c for c in [char("M. Morrel", ["Morrel"], gender="male"),
                                 char("Maximilian Morrel", ["Morrel"], gender="male")]}
    assert sorted(C.dedupe(tmp_path, {}, cast)) == ["m_morrel", "maximilian_morrel"]


def test_dedupe_keeps_spouses_apart_by_gender(tmp_path, monkeypatch):
    stub_chat(monkeypatch, '{"merges": []}')
    cast = {c["id"]: c for c in [char("Danglars", gender="male"),
                                 char("Madame Danglars", ["Danglars"], gender="female")]}
    assert len(C.dedupe(tmp_path, {}, cast)) == 2


def test_dedupe_saves_llm_suggestions_without_applying_them(tmp_path, monkeypatch):
    stub_chat(monkeypatch, '{"merges": [{"keep": "edmond_dantes", "drop": ["sinbad"]}]}')
    cast = {c["id"]: c for c in [char("Edmond Dantès"), char("Sinbad")]}
    out = C.dedupe(tmp_path, {}, cast)
    assert sorted(out) == ["edmond_dantes", "sinbad"]
    saved = json.loads((tmp_path / "cast.suggested-merges.json").read_text())
    assert saved == [{"keep": "edmond_dantes", "drop": ["sinbad"]}]


def test_dedupe_survives_a_bad_llm_answer(tmp_path, monkeypatch):
    stub_chat(monkeypatch, "no json here")
    C.dedupe(tmp_path, {}, {"x": char("Danglars")})
    assert json.loads((tmp_path / "cast.suggested-merges.json").read_text()) == []


def test_rekey_uses_final_names_and_avoids_collisions():
    a, b = char("Fernand"), char("Fernand")
    a["id"], b["id"] = "the_cousin", "the_catalan"
    out = C.rekey({"the_cousin": a, "the_catalan": b})
    assert list(out) == ["fernand", "fernand_2"]
    assert [c["id"] for c in out.values()] == ["fernand", "fernand_2"]


def test_rank_prefers_more_chapters_then_a_proper_name():
    assert C._rank(char("Danglars", chapters=[1, 2])) > C._rank(char("the banker", chapters=[1]))
    assert C._rank(char("Danglars")) > C._rank(char("the banker"))
