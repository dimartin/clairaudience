"""render.collect / long_enough. Needs mlx-audio (Apple Silicon); skipped elsewhere."""

from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
render = pytest.importorskip("render")  # imports mlx_audio at module level


def test_collect_joins_every_yielded_segment():
    gen = (SimpleNamespace(audio=np.ones(n, dtype=np.float32), sample_rate=24000) for n in (3, 4))
    audio, sr = render.collect(gen)
    assert len(audio) == 7 and sr == 24000


def test_collect_refuses_no_audio():
    with pytest.raises(RuntimeError):
        render.collect(iter(()))


def test_long_enough_loops_a_short_ref_and_keeps_a_long_one(tmp_path):
    sr = 24000
    short, long_ = tmp_path / "short.wav", tmp_path / "long.wav"
    render.audio_write(str(short), np.full(int(4.88 * sr), 0.1, dtype=np.float32), sr)
    render.audio_write(str(long_), np.full(7 * sr, 0.1, dtype=np.float32), sr)
    cache = tmp_path / "looped"
    out = render.long_enough(short, cache)
    audio, got_sr = render.audio_read(str(out), dtype="float32")
    assert out.parent == cache and abs(len(audio) / got_sr - render.CHATTERBOX_MIN_REF) < 0.05
    assert render.long_enough(long_, cache) == long_


def test_speakable_respells_then_normalises():
    lexicon = [(render.re.compile(r"\bCaderousse\b"), "Cad-roose")]
    words = {"known": {"the"}, "title_numerals": {"I", "II"}, "headings": {"II. Father", "II. Caderousse"}}
    assert render.speakable("Caderousse and the CIA", lexicon, words) == "Cad-roose and the C. I. A."
    assert render.speakable("II. Father", lexicon, words, opens_paragraph=True) == "two. Father"
    # a heading is matched before its names are respelled
    assert render.speakable("II. Caderousse", lexicon, words, opens_paragraph=True) == "two. Cad-roose"
    assert render.speakable("I. Smith", lexicon, words, opens_paragraph=True) == "I. Smith"
    assert render.speakable("the CIA", lexicon) == "the CIA"   # no words.json: respellings only


def test_assemble_joins_clips_in_index_order_with_pauses(tmp_path):
    sr = 24000
    clips = [{"file": "0-0.wav", "para": 0, "voice": "narrator"}, {"file": "1-0.wav", "para": 0, "voice": "narrator"},
             {"file": "2-0.wav", "para": 1, "voice": "danglars"}]
    for c in clips:
        render.audio_write(str(tmp_path / c["file"]), np.full(sr, 0.1, dtype=np.float32), sr)
    (tmp_path / "index.json").write_text(render.json.dumps({"chapter": 1, "clips": clips}))
    render.assemble(tmp_path, tmp_path / "001.wav")
    audio, got_sr = render.audio_read(str(tmp_path / "001.wav"), dtype="float32")
    want = 3 + render.PAUSE_SAME_PARA + max(render.PAUSE_NEW_PARA, render.PAUSE_SPEAKER)
    assert abs(len(audio) / got_sr - want) < 0.01


def test_seed_for_is_fixed_per_line_and_changes_per_take():
    assert render.seed_for(12, 0) == render.seed_for(12, 0)
    assert len({render.seed_for(12, 0, k) for k in range(8)}) == 8
