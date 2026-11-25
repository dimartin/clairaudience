"""qc.py: scoring, thresholds, take selection, the recognizer rule and the report are pure;
the whole pass runs with Whisper, pitch, audio and the TTS replaced by stubs."""

import json
import sys
import types

import pytest

import qc

CFG = dict(qc.DEFAULTS)


# --- words ---

def test_fold_drops_accents_case_and_punctuation():
    assert qc.fold("Dantès, “Go!” — he’s gone.") == "dantes go hes gone"


@pytest.mark.parametrize("ref, hyp, want", [("abc", "abc", 0.0), ("abcd", "abed", 0.25), ("ab", "", 1.0),
                                            ("", "", 0.0), ("Hello, world!", "hello world", 0.0)])
def test_cer(ref, hyp, want):
    assert qc.cer(ref, hyp) == want


def test_word_check_clean_line():
    m = qc.word_check("I am going to Marseilles tomorrow.", "I am going to Marseilles tomorrow")
    assert m == {"cer": 0.0, "wrong_run": 0, "extra": 0, "missing": []}


def test_word_check_babbling_tail_and_lead_in():
    m = qc.word_check("Good day.", "uh huh good day to you all")
    assert (m["extra"], m["wrong_run"]) == (5, 0)   # 2 before, 3 after


def test_word_check_repeated_phrase_is_a_wrong_run():
    m = qc.word_check("we sail at dawn for the island", "we sail at dawn at dawn at dawn for the island")
    assert m["wrong_run"] == 4 and m["extra"] == 0


def test_word_check_missing_words_only_from_five_letters():
    m = qc.word_check("the old sailor went down to the harbour", "the sailor went down to the")
    assert m["missing"] == [("harbour", 6)]           # "old" is too short to flag
    assert qc.word_check("a b", "a", min_letters=1)["missing"] == [("b", 1)]


def test_gap_window_spans_the_neighbouring_words():
    words = [("the", 0.0, 0.2), ("sailor", 0.3, 0.8), ("went", 1.4, 1.6)]
    assert qc.gap_window(words, 2, 2.0, 0.5) == pytest.approx((0.3, 1.9))
    assert qc.gap_window(words, 3, 2.0, 0.5) == (1.1, 2.0)    # missing at the end
    assert qc.gap_window(words, 0, 2.0, 0.5) == (0.0, 0.5)    # missing at the start


def test_gap_window_folds_whispers_words_like_the_transcript():
    # "Monte-Cristo" folds to two words, so the gap for "towards" is after hyp word 3, not 2
    spoken, heard = "Monte-Cristo sailed towards the harbour.", "Monte-Cristo sailed the harbour."
    words = [("Monte-Cristo", 0.0, 0.8), ("sailed", 0.9, 1.3), ("the", 2.0, 2.1), ("harbour.", 2.2, 2.8)]
    [(w, j)] = qc.word_check(spoken, heard)["missing"]
    assert (w, j) == ("towards", 3)
    assert qc.gap_window(words, j, 3.0, 0.1) == pytest.approx((1.2, 2.1))   # between "sailed" and "the"
    assert qc.fold_words([("—", 0.0, 0.1), ("he's", 0.2, 0.4)]) == [("hes", 0.2, 0.4)]


def test_second_listen_clears_a_word_it_hears():
    missing = [("harbour", 6), ("captain", 2)]
    assert qc.confirmed_missing(missing, ["to the harbour", "the"]) == ["captain"]


def test_second_listen_is_per_occurrence():
    missing = [("harbour", 2), ("harbour", 6)]                 # missing twice from one line
    assert qc.confirmed_missing(missing, ["the harbour", "the"]) == ["harbour"]
    assert qc.confirmed_missing(missing, ["the", "the harbour"]) == ["harbour"]
    assert qc.confirmed_missing(missing, ["the harbour", "harbour"]) == []


# --- thresholds and flags ---

def clip(voice, cer=0.0, pitch=1.0, seconds=1.0, chars=20, missing=()):
    return {"voice": voice, "seconds": seconds, "spoken": "x" * chars,
            "metrics": {"cer": cer, "wrong_run": 0, "extra": 0, "pitch": pitch, "missing": list(missing)}}


def test_semitones():
    assert qc.semitones(220, 110) == pytest.approx(12)
    assert qc.semitones(None, 110) is None


def test_length_is_relative_to_the_voice_median():
    clips = [clip("a", seconds=1.0), clip("a", seconds=1.0), clip("a", seconds=2.0), clip("b", seconds=5.0)]
    qc.length_metrics(clips)
    assert [c["metrics"]["length"] for c in clips] == [0.0, 0.0, 0.693, 0.0]


def test_thresholds_are_max_of_floor_and_k_times_the_voice_median():
    clips = [clip("hard", cer=0.1), clip("hard", cer=0.1), clip("hard", cer=0.6), clip("easy", cer=0.01)]
    qc.length_metrics(clips)
    thr = qc.thresholds(clips, CFG)
    assert thr["hard"]["cer"] == pytest.approx(0.3)    # 3 x median 0.1 beats the 0.15 floor
    assert thr["easy"]["cer"] == 0.15                 # the floor
    assert thr["easy"]["pitch"] == 4.0


def test_reasons_and_badness():
    thr = {"cer": 0.2, "wrong_run": 3, "extra": 2, "pitch": 4.0, "length": 0.5}
    m = {"cer": 0.4, "wrong_run": 1, "extra": 0, "pitch": 6.0, "length": 0.1, "missing": ["harbour"]}
    assert qc.reasons(m, thr) == ["cer 0.4 > 0.2", "pitch 6.0 > 4.0", "missing harbour"]
    assert qc.badness(m, thr) == 3.0                   # worst ratio 2.0, plus one missing word
    good = {"cer": 0.1, "wrong_run": 0, "extra": 0, "pitch": 2.0, "length": 0.1, "missing": []}
    assert qc.reasons(good, thr) == [] and qc.badness(good, thr) < 1


# --- takes ---

THR = {"cer": 0.2, "wrong_run": 3, "extra": 2, "pitch": 4.0, "length": 0.5}


def take(cer=0.0, pitch=1.0, heard="", missing=(), length=0.0):
    return {"heard": heard, "metrics": {"cer": cer, "wrong_run": 0, "extra": 0, "pitch": pitch, "length": length,
                                        "missing": list(missing)}}


def test_best_take_is_the_lowest_badness_and_may_be_the_current_one():
    assert qc.best_take([take(cer=0.1), take(cer=0.5), take(cer=0.3)], THR, short=False) == 0
    assert qc.best_take([take(cer=0.5), take(cer=0.1)], THR, short=False) == 1


def test_short_line_keeps_the_closest_pitch_among_takes_whose_words_pass():
    takes = [take(pitch=7.0), take(pitch=0.5, cer=0.6), take(pitch=2.0), take(pitch=1.0, cer=0.1)]
    assert qc.best_take(takes, THR, short=True) == 3


def test_short_line_flagged_for_length_keeps_a_retake_that_passes():
    # Prudence's case: take 0 has the closest pitch but fails on length; the retakes pass
    takes = [take(pitch=0.5, length=2.0)] + [take(pitch=float(p)) for p in range(1, 9)]
    best = qc.best_take(takes, THR, short=True)
    assert best == 1                                           # the closest pitch among passing takes
    assert qc.reasons(takes[best]["metrics"], THR) == [] and qc.badness(takes[best]["metrics"], THR) == 0.25


def test_short_line_with_no_passing_take_falls_back_to_badness():
    takes = [take(pitch=0.5, length=2.0), take(pitch=3.0, length=0.8), take(pitch=1.0, length=1.5)]
    assert qc.best_take(takes, THR, short=True) == 1           # badness 1.6, against 4.0 and 3.0


def test_short_line_with_no_good_words_falls_back_to_badness():
    assert qc.best_take([take(cer=0.9), take(cer=0.4)], THR, short=True) == 1


def test_takes_that_agree_blame_the_recognizer_and_name_candidates():
    text = "“Caderousse,” said Danglars, “you are drunk.”"
    takes = [take(cer=0.3, heard="Cad Ruth said Danglars you are drunk")] * 3
    assert qc.recognizer_blamed(takes, THR, text) == ["Caderousse"]


@pytest.mark.parametrize("takes", [
    [take(cer=0.3, heard="a"), take(cer=0.3, heard="b")],     # different transcripts: the TTS varies
    [take(cer=0.3, heard="a"), take(cer=0.1, heard="a")],     # one take passes
    [take(cer=0.3, heard="a")],                               # one take proves nothing
])
def test_recognizer_not_blamed(takes):
    assert qc.recognizer_blamed(takes, THR, "Anything") is None


def test_report_lists_reasons_scores_and_the_recognizer_note():
    rows = [{"file": "12-0.wav", "voice": "danglars", "text": "Caderousse | you are drunk", "reasons": ["cer 0.3 > 0.2"],
             "before": 1.5, "after": 1.5, "kept": 0, "recognizer": ["Caderousse"]},
            {"file": "14-0.wav", "voice": "narrator", "text": "x" * 100, "reasons": ["pitch 6.0 > 4.0"],
             "before": 1.5, "after": 0.4, "kept": 3, "recognizer": None}]
    md = qc.report(4, rows)
    assert "2 flagged lines" in md
    assert "| 12-0.wav | danglars | cer 0.3 > 0.2 | 1.5 | 1.5 | take 0 | all takes agree: recognizer; " \
           "candidates Caderousse | Caderousse / you are drunk |" in md
    assert "| 14-0.wav | narrator | pitch 6.0 > 4.0 | 1.5 | 0.4 | take 3 |  | " + "x" * 80 + "... |" in md


# --- the speech model for retakes ---

def test_tts_settings_come_from_the_index():
    idx = {"chapter": 4, "profile": "16gb-small", "tts": "chatterbox", "tts_model": "mlx-community/chatterbox-turbo-4bit"}
    never = lambda: pytest.fail("the default profile must not be read when the index records the model")
    assert qc.tts_settings(idx, load=never) == {"profile": "16gb-small", "tts": "chatterbox",
                                                "tts_model": "mlx-community/chatterbox-turbo-4bit"}


def test_tts_settings_of_an_old_index_fall_back_with_a_warning(caplog):
    cfg = {"profile": "16gb", "tts": "voxcpm2", "tts_model": "mlx-community/VoxCPM2-bf16", "llm_model": "x"}
    with caplog.at_level("WARNING", logger="qc"):
        got = qc.tts_settings({"chapter": 4, "clips": []}, load=lambda: cfg)
    assert got == {"profile": "16gb", "tts": "voxcpm2", "tts_model": "mlx-community/VoxCPM2-bf16"}
    assert "lines/004/index.json does not record the speech model" in caplog.text


def test_retake_uses_the_recorded_model_one_at_a_time(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAIRAUDIENCE_PROFILE", "16gb")       # what a standalone run would have picked
    events = []

    class FakeTTS:
        def __init__(self, cfg, voices, refdir):
            events.append(("load", cfg["tts"], cfg["tts_model"]))
            self.model = cfg["tts_model"]

        def speak(self, text, vid, seed):
            return text, 1

    fake_render = types.SimpleNamespace(
        TTS=FakeTTS, load_voices=lambda book: ({"narrator": {}}, {}), tidy=lambda a, sr: a,
        seed_for=lambda seg, n, take=0: seg * 100 + n + 7919 * take, _free=lambda m: events.append(("free", m)))
    monkeypatch.setitem(sys.modules, "render", fake_render)
    audio_io = types.SimpleNamespace(write=lambda path, a, sr: None)
    monkeypatch.setitem(sys.modules, "mlx_audio", types.SimpleNamespace(audio_io=audio_io))
    monkeypatch.setitem(sys.modules, "mlx_audio.audio_io", audio_io)

    chatterbox = {"profile": "16gb-small", "tts": "chatterbox", "tts_model": "mlx-community/chatterbox-turbo-4bit"}
    voxcpm = {"profile": "16gb", "tts": "voxcpm2", "tts_model": "mlx-community/VoxCPM2-bf16"}
    flagged = [{"seg": s, "n": 0, "file": f"{s}-0.wav", "path": tmp_path / f"{s}-0.wav", "spoken": "Hello.",
                "voice": "narrator", "render": r} for s, r in ((1, chatterbox), (2, voxcpm), (3, chatterbox))]
    takes = qc.retake(tmp_path, flagged, {**CFG, "short_takes": 2})
    assert events == [("load", "chatterbox", "mlx-community/chatterbox-turbo-4bit"),
                      ("free", "mlx-community/chatterbox-turbo-4bit"),
                      ("load", "voxcpm2", "mlx-community/VoxCPM2-bf16"), ("free", "mlx-community/VoxCPM2-bf16")]
    assert sorted((t["clip"], t["take"]) for t in takes) == [(f"{s}-0.wav", k) for s in (1, 2, 3) for k in (1, 2)]


# --- the whole pass, every model stubbed ---

def test_qc_flags_retakes_keeps_the_best_and_reports(tmp_path, monkeypatch):
    book = tmp_path
    lines = book / "lines" / "004"
    lines.mkdir(parents=True)
    texts = ["The sea was calm that night.", "Good day to you, sir.", "Where is the harbour master now?"]
    clips = [{"seg": i, "n": 0, "file": f"{i}-0.wav", "kind": "narration", "para": i, "voice": "narrator",
              "text": t, "spoken": t, "seed": i * 100} for i, t in enumerate(texts)]
    recorded = {"profile": "16gb-small", "tts": "chatterbox", "tts_model": "mlx-community/chatterbox-turbo-4bit"}
    (lines / "index.json").write_text(json.dumps({"chapter": 4, **recorded, "clips": clips}))
    for c in clips:
        (lines / c["file"]).write_text(c["text"])               # a "clip" is its own transcript here
    (book / "audio").mkdir()

    # the stubs: reading a clip gives its text; Whisper hears it, except that clip 2 drops a word
    monkeypatch.setattr(qc, "read_audio", lambda p: (p.read_text() if p.exists() else "", 1))
    monkeypatch.setattr(qc, "median_pitch", lambda audio, sr: 100.0)
    loaded = []

    class FakeASR:
        def __init__(self, model):
            loaded.append(model)

        def transcribe(self, audio, sr):
            heard = audio.replace("harbour ", "") if audio.startswith("Where") and "\n" not in audio else audio
            return heard, [(w, i * 0.5, i * 0.5 + 0.4) for i, w in enumerate(heard.split())]

    monkeypatch.setattr(qc, "ASR", FakeASR)

    retaken = []

    def retake(book_, flagged, cfg):
        out = []
        for c in flagged:
            retaken.append(c["render"])
            for k in (1, 2):
                path = c["path"].with_name(f'{c["seg"]}-{c["n"]}.take{k}.wav')
                path.write_text(c["spoken"] + "\n")            # a retake: heard in full
                out.append({"clip": c["file"], "take": k, "seed": 1000 + k, "path": path, "spoken": c["spoken"],
                            "voice": c["voice"]})
        return out

    monkeypatch.setattr(qc, "retake", retake)
    assembled = []
    monkeypatch.setitem(sys.modules, "render", types.SimpleNamespace(assemble=lambda d, out: assembled.append(out)))

    qc.qc(book, ["4"], {**CFG, "retakes": 2, "short_takes": 2})

    data = json.loads((book / "qc" / "004.json").read_text())
    [row] = data["flagged"]
    assert row["file"] == "2-0.wav" and row["reasons"] == ["cer 0.258 > 0.15", "missing harbour"]
    assert row["kept"] == 1 and row["after"] < 1 <= row["before"]
    assert (lines / "2-0.wav").read_text().endswith("\n")    # the retake replaced the clip
    assert not list(lines.glob("*.take*.wav"))                 # takes are cleaned up
    idx = json.loads((lines / "index.json").read_text())["clips"]
    assert (idx[2]["seed"], idx[2]["take"]) == (1001, 1)
    assert assembled == [book / "audio" / "004.wav"]
    assert retaken == [recorded]                               # retaken with the model that rendered it
    assert json.loads((lines / "index.json").read_text())["tts_model"] == recorded["tts_model"]
    assert loaded == ["small.en", "small.en"]                   # scored, then the takes scored
    assert "| 2-0.wav | narrator | cer 0.258 > 0.15; missing harbour |" in (book / "qc" / "004.md").read_text()


def test_median_pitch_of_a_tone():
    np = pytest.importorskip("numpy")
    sr = 16000
    t = np.arange(sr) / sr
    assert qc.median_pitch((0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32), sr) == pytest.approx(150, rel=0.03)
    assert qc.median_pitch(np.zeros(sr, dtype=np.float32), sr) is None
