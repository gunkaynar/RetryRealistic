"""
filesystem.py — Real OS file tools scoped to the per-run sandbox directory.

All paths are resolved and checked to stay within work_dir. Real OS calls
surface real permission and not-found errors naturally when the sandbox was
set up with appropriate chmod values.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def make_filesystem_tools(work_dir: Path):
    """Return (file_read, file_write, list_dir, file_exists) bound to work_dir."""
    # Resolve once so all comparisons use the canonical (symlink-free) path.
    _root = work_dir.resolve()

    def _safe_resolve(rel_path: str) -> Path | None:
        """Resolve rel_path inside sandbox. Returns None if it escapes."""
        try:
            resolved = (_root / rel_path).resolve()
            if str(resolved).startswith(str(_root)):
                return resolved
        except Exception:
            pass
        return None

    def file_read(filename: str) -> str:
        path = _safe_resolve(filename)
        if path is None:
            return json.dumps({"status": "error",
                               "message": "Access denied: path escapes sandbox."})
        try:
            content = path.read_text(encoding="utf-8")
            return json.dumps({"status": "ok", "content": content})
        except PermissionError:
            return json.dumps({"status": "error",
                               "message": f"Permission denied: cannot read '{filename}'."})
        except FileNotFoundError:
            return json.dumps({"status": "error",
                               "message": f"File not found: '{filename}' does not exist."})
        except IsADirectoryError:
            return json.dumps({"status": "error",
                               "message": f"'{filename}' is a directory, not a file."})
        except Exception as ex:
            return json.dumps({"status": "error", "message": str(ex)})

    def file_write(filename: str, content: str) -> str:
        path = _safe_resolve(filename)
        if path is None:
            return json.dumps({"status": "error",
                               "message": "Access denied: path escapes sandbox."})
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return json.dumps({"status": "ok",
                               "message": f"Wrote {len(content)} characters to '{filename}'."})
        except PermissionError:
            return json.dumps({"status": "error",
                               "message": f"Permission denied: cannot write to '{filename}'."})
        except Exception as ex:
            return json.dumps({"status": "error", "message": str(ex)})

    def list_dir(path: str = ".") -> str:
        dir_path = _safe_resolve(path)
        if dir_path is None:
            return json.dumps({"status": "error",
                               "message": "Access denied: path escapes sandbox."})
        try:
            entries = []
            for item in sorted(dir_path.iterdir()):
                try:
                    rel = str(item.relative_to(_root))
                    entries.append({
                        "name": item.name,
                        "path": rel,
                        "type": "dir" if item.is_dir() else "file",
                        "size_bytes": item.stat().st_size if item.is_file() else None,
                    })
                except Exception:
                    pass
            return json.dumps({"status": "ok", "path": path, "entries": entries})
        except PermissionError:
            return json.dumps({"status": "error",
                               "message": f"Permission denied: cannot list '{path}'."})
        except FileNotFoundError:
            return json.dumps({"status": "error",
                               "message": f"Directory not found: '{path}'."})
        except NotADirectoryError:
            return json.dumps({"status": "error",
                               "message": f"'{path}' is not a directory."})
        except Exception as ex:
            return json.dumps({"status": "error", "message": str(ex)})

    def file_exists(filename: str) -> str:
        path = _safe_resolve(filename)
        if path is None:
            return json.dumps({"status": "error",
                               "message": "Access denied: path escapes sandbox."})
        exists = path.exists()
        kind = None
        if exists:
            kind = "dir" if path.is_dir() else "file"
        return json.dumps({"status": "ok", "exists": exists,
                           "path": filename, "type": kind})

    return file_read, file_write, list_dir, file_exists
