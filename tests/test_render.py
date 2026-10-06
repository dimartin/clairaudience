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
