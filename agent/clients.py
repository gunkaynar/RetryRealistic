"""
agent/clients.py — Client helpers for the Monitor Agent's LLM.

The Monitor runs on Claude (Sonnet 4.6) through _AnthropicJudgeShim, which
exposes the Anthropic SDK behind an OpenAI-style interface.

chat_json() requests strict JSON and tolerates providers/models that do not
support response_format={"type": "json_object"} by falling back to plain
completion + regex extraction.
"""

from __future__ import annotations

import json
import os
import re
import time

class _AnthropicJudgeShim:
    """OpenAI-compatible shim exposing `.chat.completions.create(...)` over the
    Anthropic SDK, so chat_json() can drive a Claude model."""

    def __init__(self, client):
        self._client = client
        self.chat = self
        self.completions = self

    def create(self, model, messages, temperature=0, max_tokens=512,
               response_format=None, **_ignore):
        system = None
        conv = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                conv.append({"role": m["role"], "content": m["content"]})
        kwargs = dict(model=model, messages=conv, max_tokens=max_tokens)
        # Opus 4.7+/Sonnet 5/Fable reject `temperature`; only send it to models
        # (Haiku 4.5, Sonnet 4.6) that still accept it.
        if any(x in model for x in ("haiku-4-5", "sonnet-4-6")):
            kwargs["temperature"] = temperature
        if system:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        text = resp.content[0].text if resp.content else ""
        msg = type("_Msg", (), {"content": text})()
        choice = type("_Choice", (), {"message": msg})()
        return type("_Resp", (), {"choices": [choice]})()


def _extract_json(text: str) -> dict:
    # Strip reasoning blocks (e.g. DeepSeek-R1) and code fences first
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```(?:json)?", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object in response: {text[:200]!r}")
    return json.loads(m.group(0))


def chat_json(client, model: str, prompt: str,
              max_tokens: int = 512, pace: float = 0.0) -> dict:
    """One-shot JSON completion, robust to providers without json_object mode."""
    if pace:
        time.sleep(pace)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return _extract_json(resp.choices[0].message.content or "")
    except Exception as ex:
        # Retry without response_format (many NIM models reject it)
        if "response_format" not in str(ex) and "json_object" not in str(ex).lower():
            # Also fall through for models that accept the param but return
            # invalid JSON — one plain retry is cheap either way.
            pass
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user",
                       "content": prompt + "\n\nRespond with a single JSON object and nothing else."}],
            temperature=0,
            max_tokens=max_tokens,
        )
        return _extract_json(resp.choices[0].message.content or "")
