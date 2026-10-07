"""
agent/providers.py — Model-specific agent implementations.

Each class wires a model's API to the shared ReactAgentMixin loop.
All providers use temperature=0.2 and max 1024 tokens per step.
"""

from __future__ import annotations

import os
import time

from agent.base import ReactAgentMixin


def _api_retry(call_fn, max_retries: int = 5, base_wait: int = 15):
    """Exponential backoff on rate limits and server errors."""
    last_err = None
    for attempt in range(max_retries):
        try:
            return call_fn()
        except Exception as ex:
            last_err = ex
            s = str(ex).lower()
            retryable = any(t in s for t in [
                "429", "rate", "quota", "resource_exhausted",
                "503", "500", "unavailable", "overloaded",
                "service_unavailable", "internal", "capacity",
            ])
            if retryable:
                wait = base_wait * (2 ** attempt)
                print(f"    [rate-limited, retry {attempt+1}/{max_retries} in {wait}s]")
                time.sleep(wait)
            else:
                raise
    raise last_err


# ---------------------------------------------------------------------------
# Anthropic  (Claude Haiku 4.5)
# ---------------------------------------------------------------------------

class AnthropicAgent(ReactAgentMixin):
    def __init__(self, model_name: str = "claude-haiku-4-5-20251001"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model_name = model_name

    def _call_api(self, messages: list) -> str:
        # Anthropic keeps system prompt separate
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        conv = [m for m in messages if m["role"] != "system"]
        resp = _api_retry(lambda: self.client.messages.create(
            model=self.model_name,
            system=system,
            messages=conv,
            temperature=0.2,
            max_tokens=1024,
            stop_sequences=["OBSERVATION:"],
        ))
        return resp.content[0].text if resp.content else ""

    def _format_messages(self, history: list, new_user: str | None,
                         system_prompt: str) -> list:
        msgs = [{"role": "system", "content": system_prompt}]
        msgs.extend(history)
        if new_user:
            msgs.append({"role": "user", "content": new_user})
        return msgs


# ---------------------------------------------------------------------------
# NVIDIA NIM  (Llama 3.1 8B / 70B — OpenAI-compatible)
# ---------------------------------------------------------------------------

class NvidiaAgent(ReactAgentMixin):
    # Reasoning models emit <think> blocks (stripped by the parser) and need a
    # larger token budget so the visible answer isn't truncated mid-thought.
    _REASONING_MARKERS = ("deepseek",)

    def __init__(self, model_name: str = "meta/llama-3.1-70b-instruct"):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.environ["NVIDIA_API_KEY"],
            base_url="https://integrate.api.nvidia.com/v1",
        )
        self.model_name = model_name
        self.is_reasoning = any(m in model_name.lower() for m in self._REASONING_MARKERS)

    def _call_api(self, messages: list) -> str:
        time.sleep(float(os.environ.get("NIM_PACE", "3")))  # NIM pacing; override via NIM_PACE
        max_tokens = 4096 if self.is_reasoning else 1024
        resp = _api_retry(lambda: self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.2,
            max_tokens=max_tokens,
            stop=["OBSERVATION:"],
        ))
        return resp.choices[0].message.content or ""

    def _format_messages(self, history: list, new_user: str | None,
                         system_prompt: str) -> list:
        msgs = [{"role": "system", "content": system_prompt}]
        msgs.extend(history)
        if new_user:
            msgs.append({"role": "user", "content": new_user})
        return msgs


# ---------------------------------------------------------------------------
# OpenRouter  (OpenAI-compatible; DeepSeek-V4 Flash)
# ---------------------------------------------------------------------------

class OpenRouterAgent(ReactAgentMixin):
    _REASONING_MARKERS = ("deepseek",)

    def __init__(self, model_name: str):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
        self.model_name = model_name
        # OpenRouter returns reasoning in a separate field, so `content` is clean
        # ReAct text; reasoning models still need a larger completion budget.
        self.is_reasoning = any(m in model_name.lower() for m in self._REASONING_MARKERS)

    def _call_api(self, messages: list) -> str:
        time.sleep(float(os.environ.get("OR_PACE", "2")))  # override via OR_PACE
        max_tokens = 4096 if self.is_reasoning else 1024
        # OpenRouter routes each call to a random provider; some return empty /
        # null content. Re-route (retry) until we get real content, and require
        # providers that honor our params + drop the separate reasoning stream.
        for attempt in range(6):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=max_tokens,
                    stop=["OBSERVATION:"],
                    timeout=25,  # abort slow/hanging providers fast, then re-route
                    extra_body={"provider": {"require_parameters": True,
                                             "sort": "throughput"},
                                "reasoning": {"exclude": True}},
                )
            except Exception as ex:
                s = str(ex).lower()
                if any(t in s for t in ["429", "rate", "quota", "503", "500",
                                        "502", "unavailable", "overloaded", "timeout"]):
                    time.sleep(min(60, 8 * (attempt + 1))); continue
                raise
            content = resp.choices[0].message.content if getattr(resp, "choices", None) else None
            if content:
                return content
            time.sleep(2)  # empty/null provider -> re-route on next attempt
        return ""

    def _format_messages(self, history: list, new_user: str | None,
                         system_prompt: str) -> list:
        msgs = [{"role": "system", "content": system_prompt}]
        msgs.extend(history)
        if new_user:
            msgs.append({"role": "user", "content": new_user})
        return msgs


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

# model_key → (required env var, factory): the four agents in the paper
_REGISTRY = {
    "llama-8b":     ("NVIDIA_API_KEY", lambda: NvidiaAgent("meta/llama-3.1-8b-instruct")),
    "llama-70b":    ("NVIDIA_API_KEY", lambda: NvidiaAgent("meta/llama-3.1-70b-instruct")),
    # DeepSeek-V4 Flash, pinned snapshot (also on NIM: nim:deepseek-ai/deepseek-v4-flash-0731)
    "deepseek-v4":  ("OPENROUTER_API_KEY", lambda: OpenRouterAgent("deepseek/deepseek-v4-flash-0731")),
    "claude-haiku": ("ANTHROPIC_API_KEY", lambda: AnthropicAgent("claude-haiku-4-5-20251001")),
}


def create_agent(model_key: str) -> ReactAgentMixin:
    # "nim:<model-id>" runs any NIM-hosted model without a registry entry
    if model_key.startswith("nim:"):
        return NvidiaAgent(model_key[4:])
    if model_key not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{model_key}'. Available: {sorted(_REGISTRY)} "
            f"or 'nim:<model-id>' for any NIM model."
        )
    return _REGISTRY[model_key][1]()


def available_models() -> list[str]:
    """Model keys whose provider API key is present in the environment."""
    return sorted(k for k, (env, _) in _REGISTRY.items() if os.environ.get(env))
