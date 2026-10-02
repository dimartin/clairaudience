"""Config profiles and the LLM call, with two backends.

server: an OpenAI-compatible endpoint (e.g. mtplx). The model stays loaded elsewhere.
mlx:    mlx-lm loads the model inside this process; it is freed when the process exits,
        which is how the pipeline keeps one model in memory at a time.
"""

import json
import logging
import os
import re
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

log = logging.getLogger("llm")

CONFIG = Path(__file__).with_name("config.toml")
_mlx = {}


def _merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(profile=None):
    """config.toml holds the public profiles; config.local.toml (git-ignored) adds or
    overrides machine-specific ones (a local LLM server, a publish destination)."""
    raw = tomllib.loads(CONFIG.read_text())
    local = CONFIG.with_name("config.local.toml")
    if local.exists():
        raw = _merge(raw, tomllib.loads(local.read_text()))
    name = profile or os.environ.get("CLAIRAUDIENCE_PROFILE") or raw.get("default_profile")
    cfg = {**raw.get("llm", {}), **raw["profiles"][name], "profile": name, "publish": raw.get("publish", {})}
    if cfg.get("llm_api_key_file"):
        cfg["api_key"] = Path(cfg["llm_api_key_file"]).expanduser().read_text().strip()
    return cfg


def chat(cfg, prompt, max_tokens=None):
    if max_tokens:
        cfg = {**cfg, "max_tokens": max_tokens}
    t0 = time.time()
    if cfg["llm_backend"] == "mlx":
        text, usage = _chat_mlx(cfg, prompt)
    else:
        text, usage = _chat_server(cfg, prompt)
    log.info("llm %.1fs prompt=%s completion=%s", time.time() - t0, usage.get("prompt_tokens"),
             usage.get("completion_tokens"))
    if (usage.get("completion_tokens") or 0) >= cfg["max_tokens"] - 5:
        # small models sometimes loop on a list until the cap; the JSON is then cut off mid-string
        raise Truncated(f"output hit max_tokens={cfg['max_tokens']} (runaway, cut off mid-answer)")
    return text


class Truncated(ValueError):
    """The model's answer was cut off at max_tokens; callers retry like any bad answer."""


def _chat_server(cfg, prompt):
    body = {
        "model": cfg["llm_model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": cfg["temperature"],
        "max_tokens": cfg["max_tokens"],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        cfg["llm_base_url"] + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {cfg.get('api_key', '')}", "Content-Type": "application/json"},
    )
    retries = cfg.get("retries", 6)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=cfg["timeout_s"]) as r:
                resp = json.load(r)
            break
        except urllib.error.HTTPError as e:
            # 507 = mtplx memory refusal; it frees its caches, so waiting and retrying works
            if e.code != 507 or attempt == retries - 1:
                raise
            wait = 30 * (attempt + 1)
            log.warning("507 memory refusal, retry %d in %ds", attempt + 1, wait)
            time.sleep(wait)
    return resp["choices"][0]["message"]["content"], resp.get("usage", {})


def _chat_mlx(cfg, prompt):
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_logits_processors, make_sampler

    if "model" not in _mlx:
        t0 = time.time()
        _mlx["model"], _mlx["tok"] = load(cfg["llm_model"])
        log.info("loaded %s in %.1fs", cfg["llm_model"], time.time() - t0)
    tok = _mlx["tok"]
    messages = [{"role": "user", "content": prompt}]
    try:
        text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                       enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    # a mild repetition penalty stops small quantized models looping until max_tokens
    out = generate(_mlx["model"], tok, prompt=text, max_tokens=cfg["max_tokens"],
                   sampler=make_sampler(temp=cfg["temperature"]),
                   logits_processors=make_logits_processors(repetition_penalty=cfg.get("repetition_penalty", 1.1)))
    return out, {"prompt_tokens": len(tok.encode(text)), "completion_tokens": len(tok.encode(out))}


def parse_json(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in model output")
    return json.loads(m.group(0))
