"""voices.py: voice design retries, the stock-voice fallback and the stock pools.
The LLM is replaced by a stub; nothing here loads a model."""

import json
from collections import Counter

import pytest

import voices

HAYDEE = {"id": "haydee", "name": "Haydée", "persona": {"gender": "female"}, "evidence": []}
ALI = {"id": "ali", "name": "Ali", "persona": {"gender": "male"}, "evidence": []}
META = {"title": "The Count of Monte Cristo"}


def answers(*replies):
    it = iter(replies)
    return lambda cfg, prompt: next(it)


def test_design_voice_retries_a_bad_answer(monkeypatch):
    monkeypatch.setattr(voices, "chat", answers("{oops", json.dumps({"instruct": "A soft voice"})))
    assert voices.design_voice({}, META, HAYDEE)["instruct"] == "A soft voice"


def test_design_voice_gives_up_after_three_bad_answers(monkeypatch):
    monkeypatch.setattr(voices, "chat", answers("{oops", "[]", json.dumps({"why": "no instruct"})))
    assert voices.design_voice({}, META, HAYDEE) is None


@pytest.mark.parametrize("gender, want", [
    ("male", ["extra_male_1", "extra_male_2", "extra_male_3", "extra_male_1"]),
    ("female", ["extra_female_1", "extra_female_2", "extra_female_1"]),
])
def test_stock_pools_hold_only_their_gender(gender, want):
    pool = Counter()
    c = {"persona": {"gender": gender}}
    assert [voices.stock_voice(c, pool) for _ in want] == want


def test_a_character_with_no_designed_voice_gets_a_stock_voice_not_the_narrator(tmp_path, monkeypatch):
    """Monte Cristo run3: Haydée's voice JSON failed and render fell back to the narrator."""
    (tmp_path / "book.json").write_text(json.dumps(META))
    (tmp_path / "cast.json").write_text(json.dumps({"haydee": HAYDEE, "ali": ALI}))
    monkeypatch.setattr(voices, "load_config", lambda: {"min_lines": 5})
    monkeypatch.setattr(voices, "spoken_lines", lambda book: {"haydee": ["Yes."] * 9, "ali": ["Hm."] * 6})
    monkeypatch.setattr(voices, "narration_line", lambda book: "It was night.")
    monkeypatch.setattr(voices, "chat", lambda cfg, prompt:
                        "{bad" if "Haydée" in prompt else json.dumps({"instruct": "A deep calm voice"}))

    voices.main(str(tmp_path))
    out = json.loads((tmp_path / "voices.json").read_text())
    ids = {v["id"] for v in out["voices"]}
    assert out["voice_map"]["haydee"] == "extra_female_1"
    assert "extra_female_1" in ids  # the stock voice is defined, so render will find it
    assert out["voice_map"]["ali"] == "ali" and "ali" in ids
