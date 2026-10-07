"""
env_loader.py — Minimal .env loader (no python-dotenv dependency).

Reads KEY=value lines from the project-root .env file into os.environ.
Existing environment variables are never overridden, and empty values
in the file are ignored.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env(env_path: str | None = None) -> None:
    path = Path(env_path) if env_path else Path(__file__).parent / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if value and key not in os.environ:
            os.environ[key] = value
