"""Score speaker attribution against a verified chapter, per profile.

Gold: out/monte-cristo/004.script.json (the 9B's labels plus four human-verified fixes).
Every profile gets the same fixed cast for the chapter, so this measures attribution
alone. Reports accuracy before and after the deterministic dialogue-tag override.

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
    gold = {s["id"]: s["speaker"] for s in json.loads(GOLD.read_text())["segments"] if s["kind"] == "quote"}
    got = [s for s in json.loads((book / "004.script.json").read_text())["segments"] if s["kind"] == "quote"]
    after = sum(gold[s["id"]] == s["speaker"] for s in got)
    before = sum(gold[s["id"]] == s["speaker"] and not s.get("tag_override") for s in got)
    wrong = [(s["id"], s["speaker"], gold[s["id"]]) for s in got if gold[s["id"]] != s["speaker"]]
    secs = [l for l in log.read_text().splitlines() if " llm " in l or "loaded" in l]
    return {"profile": profile, "quotes": len(got), "model_only": before, "with_tag_override": after,
            "wrong": wrong, "timing": secs}


if __name__ == "__main__":
    results = [run_one(p) for p in sys.argv[1:]]
    for r in results:
        n = r["quotes"]
        print(f'{r["profile"]:12} model alone {r["model_only"]}/{n} ({100 * r["model_only"] / n:.0f}%)  '
              f'with tag override {r["with_tag_override"]}/{n} ({100 * r["with_tag_override"] / n:.0f}%)')
        print("   wrong (quote, got, gold):", r["wrong"])
        for t in r["timing"]:
            print("  ", t[24:])
    (HERE / "out" / "eval-results.json").write_text(json.dumps(results, indent=1))
