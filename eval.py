"""Score speaker attribution against a verified chapter, per profile.

Gold: out/monte-cristo/004.script.json (the 9B's labels plus four human-verified fixes).
Every profile gets the same fixed cast for the chapter, so this measures attribution
alone. Reports overall accuracy and accuracy per source (how each speaker was decided:
tag, tag-resolved, llm, unknown, user; see attribute.py).

Usage: python eval.py <profile> [<profile> ...]   (each profile runs in its own process)
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
GOLD = HERE / "out/monte-cristo/004.script.json"
CAST = {
    "danglars": {"id": "danglars", "name": "Danglars", "aliases": ["M. Danglars", "the supercargo"]},
    "fernand": {"id": "fernand", "name": "Fernand Mondego", "aliases": ["Fernand", "the Catalan", "the young man"]},
    "caderousse": {"id": "caderousse", "name": "Caderousse", "aliases": ["the tailor"]},
    "waiter": {"id": "waiter", "name": "The waiter", "aliases": ["waiter"]},
    "edmond_dantes": {"id": "edmond_dantes", "name": "Edmond Dantès", "aliases": ["Dantès", "Edmond"]},
    "mercedes": {"id": "mercedes", "name": "Mercédès", "aliases": []},
}


def score(gold_segs, got_segs):
    """Overall and per-source accuracy of got_segs against gold_segs, quote by quote."""
    gold = {s["id"]: s["speaker"] for s in gold_segs if s["kind"] == "quote"}
    got = [s for s in got_segs if s["kind"] == "quote"]
    by_source = {}
    for s in got:
        b = by_source.setdefault(s.get("source", "none"), {"quotes": 0, "correct": 0})
        b["quotes"] += 1
        b["correct"] += gold[s["id"]] == s["speaker"]
    wrong = [(s["id"], s["speaker"], gold[s["id"]], s.get("source")) for s in got if gold[s["id"]] != s["speaker"]]
    return {"quotes": len(got), "correct": sum(b["correct"] for b in by_source.values()),
            "by_source": by_source, "wrong": wrong}


def run_one(profile):
    book = HERE / "out" / f"eval-{profile}"
    shutil.rmtree(book, ignore_errors=True)
    book.mkdir(parents=True)
    shutil.copy(HERE / "out/monte-cristo/004.txt", book / "004.txt")
    (book / "cast.json").write_text(json.dumps(CAST))
    (book / "index.json").write_text(json.dumps([{"n": 4}]))
    env = {**os.environ, "CLAIRAUDIENCE_PROFILE": profile}
    log = book / "attribute.log"
    with open(log, "w") as f:
        subprocess.run([sys.executable, str(HERE / "attribute.py"), str(book), "4"], env=env, cwd=HERE,
                       stdout=f, stderr=subprocess.STDOUT, check=True)
    result = score(json.loads(GOLD.read_text())["segments"],
                   json.loads((book / "004.script.json").read_text())["segments"])
    secs = [l for l in log.read_text().splitlines() if " llm " in l or "loaded" in l]
    return {"profile": profile, **result, "timing": secs}


def pct(a, b):
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "0/0"


if __name__ == "__main__":
    results = [run_one(p) for p in sys.argv[1:]]
    for r in results:
        print(f'{r["profile"]:12} overall {pct(r["correct"], r["quotes"])}')
        for src, b in r["by_source"].items():
            print(f'   {src:13} {pct(b["correct"], b["quotes"])}')
        print("   wrong (quote, got, gold, source):", r["wrong"])
        for t in r["timing"]:
            print("  ", t[24:])
    (HERE / "out" / "eval-results.json").write_text(json.dumps(results, indent=1))
