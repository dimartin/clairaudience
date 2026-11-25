"""eval.score: overall and per-source accuracy against a gold script. No model is run."""

import eval as E


def q(i, speaker, source=None):
    s = {"kind": "quote", "id": i, "text": "", "speaker": speaker}
    if source:
        s["source"] = source
    return s


GOLD = [q(0, "a"), {"kind": "narration", "id": 1, "speaker": "narrator"}, q(2, "b"), q(3, "a"), q(4, "c"), q(5, "b")]


def test_score_counts_accuracy_per_source():
    got = [q(0, "a", "tag"), {"kind": "narration", "id": 1, "speaker": "narrator"}, q(2, "b", "llm"),
           q(3, "b", "llm"), q(4, "unknown", "unknown"), q(5, "b", "tag-resolved")]
    r = E.score(GOLD, got)
    assert (r["quotes"], r["correct"]) == (5, 3)
    assert r["by_source"] == {"tag": {"quotes": 1, "correct": 1}, "llm": {"quotes": 2, "correct": 1},
                              "unknown": {"quotes": 1, "correct": 0}, "tag-resolved": {"quotes": 1, "correct": 1}}
    assert r["wrong"] == [(3, "b", "a", "llm"), (4, "unknown", "c", "unknown")]


def test_a_script_without_sources_scores_under_none():
    r = E.score(GOLD, GOLD)
    assert r["by_source"] == {"none": {"quotes": 5, "correct": 5}}


def test_timing_lines_skip_the_chapter_summary():
    log = (
        "2026-11-25 13:00:00,001 INFO loaded mlx-community/Qwen3-8B-4bit in 4.2s\n"
        "2026-11-25 13:00:09,500 INFO llm 9.3s prompt=2100 completion=640\n"
        "2026-11-25 13:00:09,501 INFO tag override Q3: fernand -> danglars ('said Danglars')\n"
        "2026-11-25 13:00:09,502 INFO chapter 4: 52 quotes, 3 tag overrides; by source: "
        "{'tag': 20, 'llm': 32}; llm share 62%\n"
    )
    got = E.timing_lines(log)
    assert [l[24:] for l in got] == ["INFO loaded mlx-community/Qwen3-8B-4bit in 4.2s",
                                     "INFO llm 9.3s prompt=2100 completion=640"]
