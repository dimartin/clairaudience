"""Encode rendered chapters and publish the book folder.

[publish].format picks the output:
  m4a  one AAC file per chapter (default; Apple, Windows, Android, Audiobookshelf)
  mp3  one MP3 file per chapter (plays everywhere)
  m4b  one AAC file for the whole book with chapter markers (audiobook apps)
Per-chapter formats also get "<Title> (<label>).m3u8", a playlist of the chapters in order.

Usage: python encode.py encode out/<book>
       python encode.py publish out/<book>
encode writes out/<book>/book/<Author>/<Title> (<label>)/...
publish rsyncs that folder to [publish].dest (an Audiobookshelf library root, say).
"""

import json
import logging
import re
import shlex
import subprocess
import sys
from pathlib import Path

from llm import load_config

log = logging.getLogger("encode")
FFMPEG = next((p for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "ffmpeg") if Path(p).exists() or p == "ffmpeg"))


def book_dir(book):
    meta = json.loads((book / "book.json").read_text())
    label = load_config()["publish"].get("label", "Full Cast")
    album = f'{meta["title"]} ({label})'
    return meta, album, book / "book" / meta["author"] / album


CODECS = {"m4a": ["-c:a", "aac"], "m4b": ["-c:a", "aac"], "mp3": ["-c:a", "libmp3lame"]}


def tags(meta, album, title, n=None):
    t = {"title": title, "album": album, "artist": meta["author"], "album_artist": meta["author"],
         "genre": "Audiobook"}
    if n is not None:
        t["track"] = str(n)
    return [x for k, v in t.items() for x in ("-metadata", f"{k}={v}")]


def chapters(book):
    titles = {c["n"]: c["title"] for c in json.loads((book / "index.json").read_text())}
    for wav in sorted((book / "audio").glob("[0-9][0-9][0-9].wav")):
        n = int(wav.stem)
        yield n, titles.get(n, f"Chapter {n}"), wav


def ffmpeg(args):
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", *args], check=True)


def encode(book):
    cfg = load_config()
    fmt = cfg["publish"].get("format", "m4a")
    if fmt not in CODECS:
        raise SystemExit(f"unknown [publish].format {fmt!r}; use m4a, mp3 or m4b")
    meta, album, dest = book_dir(book)
    dest.mkdir(parents=True, exist_ok=True)
    audio = ["-b:a", cfg["publish"].get("bitrate", "96k"), "-ac", "1", *CODECS[fmt]]
    if fmt == "m4b":
        return encode_m4b(book, meta, album, dest, audio)
    entries = []
    for n, title, wav in chapters(book):
        short = re.sub(r"^Chapter\s+\d+\.\s*", "", title).replace("/", "-")
        out = dest / f"{n:03} - {short}.{fmt}"
        entries.append((seconds(wav), title, out.name))
        if out.exists() and out.stat().st_mtime > wav.stat().st_mtime:
            continue
        ffmpeg(["-i", str(wav), *audio, *tags(meta, album, title, n), str(out)])
        log.info("encoded %s", out.name)
    write_playlist(dest, album, meta, entries)


def seconds(wav):
    import wave
    with wave.open(str(wav)) as w:
        return w.getnframes() / w.getframerate()


def write_playlist(dest, album, meta, entries):
    """Extended M3U (UTF-8) with relative paths, so the folder can be moved or served as is."""
    lines = ["#EXTM3U", f"#PLAYLIST:{album}", f"#EXTART:{meta['author']}", f"#EXTALB:{album}"]
    for secs, title, name in entries:
        lines += [f"#EXTINF:{round(secs)},{meta['author']} - {title}", name]
    path = dest / f"{album}.m3u8"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("playlist %s: %d chapters", path.name, len(entries))


def encode_m4b(book, meta, album, dest, audio):
    """Join every rendered chapter into one .m4b with a chapter marker per chapter."""
    import wave
    chs = list(chapters(book))
    lines, meta_lines, start = [], [";FFMETADATA1"], 0
    for n, title, wav in chs:
        with wave.open(str(wav)) as w:
            ms = int(w.getnframes() * 1000 / w.getframerate())
        lines.append(f"file '{wav.resolve()}'")
        meta_lines += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={start}", f"END={start + ms}", f"title={title}"]
        start += ms
    listing, chapmeta = book / "audio" / "concat.txt", book / "audio" / "chapters.txt"
    listing.write_text("\n".join(lines) + "\n")
    chapmeta.write_text("\n".join(meta_lines) + "\n")
    out = dest / f"{album}.m4b"
    ffmpeg(["-f", "concat", "-safe", "0", "-i", str(listing), "-i", str(chapmeta), "-map_metadata", "1",
            "-map", "0:a", *audio, *tags(meta, album, album), str(out)])
    log.info("encoded %s: %d chapters, %.1f hours", out.name, len(chs), start / 3.6e6)


def publish(book):
    cfg = load_config()
    target = cfg.get("publish_dest", cfg["publish"].get("dest"))
    if not target:
        log.info("no [publish].dest configured; skipping")
        return
    meta, album, dest = book_dir(book)
    remote = f"{target}/{meta['author']}/{album}"
    if ":" in target:
        host, path = remote.split(":", 1)
        copy_to_host(dest, host, path)
    else:
        Path(remote).mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", f"{dest}/", f"{remote}/"], check=True)
    log.info("published %s to %s", album, target)


def copy_to_host(src, host, path):
    """Copy src's contents into path on host, creating it: tar piped over ssh.

    Not rsync. macOS ships openrsync (protocol 29), which hands the remote path to the
    remote shell unquoted, so an album folder like "The Count of Monte Cristo (Full
    Cast, 16GB test)" broke with "syntax error near unexpected token `('". openrsync
    has no --protect-args, and quoting the path ourselves would double-escape it under
    GNU rsync 3.2.4+, which escapes remote args itself. With ssh we own the one shell
    command, so shlex.quote is exactly right whichever rsync is installed.
    """
    q = shlex.quote(path)
    tar = subprocess.Popen(["tar", "-C", str(src), "-cf", "-", "."], stdout=subprocess.PIPE)
    try:
        subprocess.run(["ssh", host, f"mkdir -p {q} && tar -C {q} -xf -"], stdin=tar.stdout, check=True)
    finally:
        tar.stdout.close()
    if tar.wait() != 0:
        raise subprocess.CalledProcessError(tar.returncode, tar.args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    {"encode": encode, "publish": publish}[sys.argv[1]](Path(sys.argv[2]))
