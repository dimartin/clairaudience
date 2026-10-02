"""Turn an EPUB into a full-cast clairaudience audiobook, one model in memory at a time.

Each stage runs as its own process, so a stage's model is freed when it exits, and all
the work for one model is done across the whole book before the next model loads.
Progress is recorded in out/<book>/manifest.json, so a rerun resumes after the last
finished stage.

Usage:
  python clairaudience.py fetch [--profile 16gb]      # download the profile's models up front
  python clairaudience.py run books/monte-cristo.epub [--profile 16gb] [--from STAGE] [--only STAGE]
                      [--until STAGE] [--chapters 4 5] [--force]
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from llm import load_config

log = logging.getLogger("clairaudience")
HERE = Path(__file__).parent
PY = sys.executable

# (name, model kind, argv builder). Model kind decides whether an LLM server must be up.
STAGES = [
    ("extract", None, lambda b, epub, ch: ["extract.py", epub, b]),
    ("cast", "llm", lambda b, epub, ch: ["cast.py", b]),
    ("attribute", "llm", lambda b, epub, ch: ["attribute.py", b, *ch]),
    ("voices", "llm", lambda b, epub, ch: ["voices.py", b]),
    ("design", "audio", lambda b, epub, ch: ["render.py", "design", b]),
    ("render", "audio", lambda b, epub, ch: ["render.py", "render", b, *ch]),
    ("encode", None, lambda b, epub, ch: ["encode.py", "encode", b]),
    ("publish", None, lambda b, epub, ch: ["encode.py", "publish", b]),
]
NAMES = [s[0] for s in STAGES]


def shell(cmd):
    log.info("$ %s", cmd)
    subprocess.run(cmd, shell=True, check=True)


def fetch(cfg):
    """Download every model the profile uses, so a run never stalls on a download."""
    from huggingface_hub import snapshot_download
    models = [cfg["design_model"], cfg["tts_model"]]
    if cfg["llm_backend"] == "mlx":
        models.insert(0, cfg["llm_model"])
    for m in models:
        t0 = time.time()
        path = snapshot_download(m)
        log.info("fetched %s in %.0fs -> %s", m, time.time() - t0, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "fetch"])
    ap.add_argument("epub", nargs="?")
    ap.add_argument("--profile")
    ap.add_argument("--from", dest="start", choices=NAMES)
    ap.add_argument("--only", choices=NAMES)
    ap.add_argument("--until", choices=NAMES, help="stop after this stage")
    ap.add_argument("--chapters", nargs="*", default=[])
    ap.add_argument("--force", action="store_true", help="rerun stages already marked done")
    args = ap.parse_args()

    if args.profile:
        os.environ["CLAIRAUDIENCE_PROFILE"] = args.profile
    cfg = load_config()
    if args.cmd == "fetch":
        return fetch(cfg)
    if not args.epub:
        ap.error("run needs an EPUB path")
    book = str(HERE / "out" / Path(args.epub).stem)
    Path(book).mkdir(parents=True, exist_ok=True)
    mpath = Path(book) / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}

    stages = STAGES
    if args.only:
        stages = [s for s in STAGES if s[0] == args.only]
    elif args.start:
        stages = STAGES[NAMES.index(args.start):]
    if args.until and not args.only:
        stages = [s for s in stages if NAMES.index(s[0]) <= NAMES.index(args.until)]

    server_stopped = False
    try:
        for name, kind, argv in stages:
            # partial (--chapters) runs never mark a whole-book stage done
            if manifest.get(name, {}).get("done") and not (args.force or args.only or args.chapters):
                log.info("skip %s (done %s)", name, manifest[name]["at"])
                continue
            if kind == "audio" and cfg["llm_backend"] == "server" and cfg.get("llm_server_stop") and not server_stopped:
                shell(cfg["llm_server_stop"])
                server_stopped = True
            if kind == "llm" and server_stopped:
                shell(cfg["llm_server_start"])
                server_stopped = False
            t0 = time.time()
            log.info("=== %s (profile %s)", name, cfg["profile"])
            script, *rest = argv(book, args.epub, args.chapters)
            subprocess.run([PY, str(HERE / script), *rest], check=True, cwd=HERE)
            manifest[name] = {"done": not args.chapters, "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                              "seconds": round(time.time() - t0), "profile": cfg["profile"],
                              "chapters": args.chapters or "all"}
            mpath.write_text(json.dumps(manifest, indent=1))
    finally:
        if server_stopped and cfg.get("llm_server_start"):
            shell(cfg["llm_server_start"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
