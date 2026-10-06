"""encode.publish / copy_to_host: getting an album onto the Audiobookshelf host."""

import shlex
import subprocess

import pytest

import encode

ALBUM = "The Count of Monte Cristo (Full Cast, 16GB test)"


def test_remote_path_with_parentheses_is_quoted_for_the_remote_shell(tmp_path, monkeypatch):
    """The 2026-10-06 failure: openrsync passed this path to bash unquoted."""
    (tmp_path / "004 - Conspiracy.m4a").write_bytes(b"x")
    seen = {}

    def run(argv, **kw):
        seen["argv"] = argv
        seen["stdin"] = kw["stdin"].read()
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(encode.subprocess, "run", run)
    path = f"/var/media/audiobookshelf/audiobooks/Alexandre Dumas/{ALBUM}"
    encode.copy_to_host(tmp_path, "dimartin@mash1", path)

    host, cmd = seen["argv"][1], seen["argv"][2]
    assert seen["argv"][0] == "ssh" and host == "dimartin@mash1"
    # The remote shell splits the command exactly as intended: the path stays one word.
    words = shlex.split(cmd)
    assert words == ["mkdir", "-p", path, "&&", "tar", "-C", path, "-xf", "-"]
    assert b"004 - Conspiracy.m4a" in seen["stdin"]  # a real tar stream of src


def test_a_failed_remote_side_raises(tmp_path, monkeypatch):
    def run(argv, **kw):
        kw["stdin"].read()
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(encode.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        encode.copy_to_host(tmp_path, "h", "/x (y)")


def test_copy_round_trips_through_a_real_shell(tmp_path, monkeypatch):
    """No ssh: run the remote command in a local sh, which parses it like the host does."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "a b (c).m4a").write_bytes(b"audio")
    dest = tmp_path / f"out/{ALBUM}"
    real_run = subprocess.run

    def run(argv, **kw):
        assert argv[0] == "ssh"
        return real_run(["sh", "-c", argv[2]], stdin=kw["stdin"], check=True)

    monkeypatch.setattr(encode.subprocess, "run", run)
    encode.copy_to_host(src, "h", str(dest))
    assert (dest / "a b (c).m4a").read_bytes() == b"audio"
