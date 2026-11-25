"""llm._chat_server: the request body for seeds and structured output. No server is contacted."""

import io
import json

import llm


def capture(monkeypatch):
    sent = {}

    def urlopen(req, timeout):
        sent["body"] = json.loads(req.data)
        return io.StringIO(json.dumps({"choices": [{"message": {"content": "{}"}}], "usage": {}}))

    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    return sent


CFG = {"llm_model": "m", "llm_base_url": "http://x", "temperature": 0.2, "max_tokens": 10, "timeout_s": 1}
SCHEMA = {"type": "object"}


def test_seed_and_schema_are_sent_when_the_server_supports_it(monkeypatch):
    sent = capture(monkeypatch)
    llm._chat_server({**CFG, "structured_output": True}, "hi", seed=2, schema=SCHEMA)
    assert sent["body"]["seed"] == 2
    assert sent["body"]["response_format"]["json_schema"]["schema"] == SCHEMA


def test_schema_is_not_sent_by_default(monkeypatch):
    sent = capture(monkeypatch)
    llm._chat_server(CFG, "hi", schema=SCHEMA)
    assert "response_format" not in sent["body"] and "seed" not in sent["body"]
