"""
sandbox.py — Per-run environment manager.

Reads a task's YAML spec and materializes a fresh working directory with:
  - Real files and directories (with real chmod permissions)
  - SQLite databases (schema + seed data)
  - Mock server routes registered for this task

Teardown restores permissions so the temp dir can be cleaned up.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import stat
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from environment.mock_server import MockServer

# Permission string → octal mode (owner bits only; group+other get nothing)
_PERM_MAP = {
    "r": stat.S_IRUSR,
    "w": stat.S_IWUSR,
    "x": stat.S_IXUSR,
    "-": 0,
}


def _parse_permissions(perm_str: str) -> int:
    """Convert 'rw-' style string to an octal mode integer."""
    mode = 0
    for ch in perm_str[:3]:
        mode |= _PERM_MAP.get(ch, 0)
    return mode


class SandboxManager:
    def __init__(self, task_spec: dict, mock_server: "MockServer"):
        self._spec = task_spec
        self._server = mock_server
        self.work_dir: Path | None = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, run_id: str, fault_type: str = "none",
              fault_target: dict | None = None) -> Path:
        """Create working dir and materialize the task environment.

        For permission_denied faults on file-backed tools, the target
        resource is chmod'd to 0o000 here, before the run, so the fault
        surfaces as a genuine OS error rather than an injected string.
        """
        self.work_dir = Path(tempfile.mkdtemp(prefix=f"retry-{run_id}-"))
        self.os_fault_paths: list[Path] = []

        self._create_filesystem()
        self._create_databases()
        self._server.load_task_routes(self._spec.get("web_pages", []))

        if fault_type == "permission_denied":
            self.os_fault_paths = self._apply_permission_fault(fault_target)

        return self.work_dir

    _OS_FAULT_TOOLS = {"file_read", "file_write", "list_dir", "sql_query", "shell_exec"}

    def _apply_permission_fault(self, fault_target: dict | None) -> list[Path]:
        """chmod 0o000 every sandbox path matching the fault target."""
        tool = (fault_target or {}).get("tool")
        match = ((fault_target or {}).get("match") or "").lower()
        if tool not in self._OS_FAULT_TOOLS or not match:
            return []

        candidates = [
            p for p in self.work_dir.rglob("*")
            if match in str(p.relative_to(self.work_dir)).lower()
        ]
        # sql_query targets usually name a table, not a file — deny the
        # task's database file(s) instead.
        if not candidates and tool == "sql_query":
            candidates = [
                self.work_dir / db["name"]
                for db in self._spec.get("databases", [])
                if (self.work_dir / db["name"]).exists()
            ]
        # Deepest first, so parent dirs are still traversable when we chmod children
        candidates.sort(key=lambda p: len(p.parts), reverse=True)
        denied = []
        for p in candidates:
            try:
                os.chmod(p, 0)
                denied.append(p)
            except OSError:
                pass
        return denied

    def _create_filesystem(self) -> None:
        for item in self._spec.get("filesystem", []):
            path = self.work_dir / item["path"]

            if item.get("is_dir", False) or item["path"].endswith("/"):
                path.mkdir(parents=True, exist_ok=True)
                mode = _parse_permissions(item.get("permissions", "rwx"))
                # Directories always need execute to be entered; add it unless explicitly "---"
                if mode != 0:
                    mode |= stat.S_IXUSR
                os.chmod(path, mode)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                content = item.get("content", "")
                path.write_text(content, encoding="utf-8")
                mode = _parse_permissions(item.get("permissions", "rw-"))
                os.chmod(path, mode)

    def _create_databases(self) -> None:
        for db_spec in self._spec.get("databases", []):
            db_path = self.work_dir / db_spec["name"]
            db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(db_path))
            try:
                for table in db_spec.get("tables", []):
                    conn.execute(table["schema"])
                    for row in table.get("rows", []):
                        placeholders = ", ".join("?" * len(row))
                        conn.execute(
                            f"INSERT INTO {table['name']} VALUES ({placeholders})", row
                        )
                conn.commit()
            finally:
                conn.close()

            perms = _parse_permissions(db_spec.get("permissions", "rw-"))
            os.chmod(db_path, perms)

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def teardown(self) -> None:
        """Restore permissions and delete the working directory."""
        if self.work_dir and self.work_dir.exists():
            self._restore_permissions(self.work_dir)
            shutil.rmtree(self.work_dir, ignore_errors=True)
            # rmtree(ignore_errors) silently leaks subtrees it can't traverse
            # (e.g. a permission_denied dir chmod'd to 0o000). Those leaks
            # accumulate in /tmp until the disk quota is exhausted, so surface it.
            if self.work_dir.exists():
                import sys
                print(f"WARNING: sandbox dir not removed, leaking: {self.work_dir}",
                      file=sys.stderr)
        self.work_dir = None

    def _restore_permissions(self, root: Path) -> None:
        """Make everything readable/deletable before rmtree.

        Walk TOP-DOWN and chmod each directory to 0o700 *before* descending, so
        a directory that was denied (0o000) becomes traversable in time for the
        walk to reach its children. A bare rglob/rmtree cannot enter a 0o000 dir
        and leaves the whole subtree behind.
        """
        try:
            os.chmod(root, 0o700)
        except Exception:
            pass
        for dirpath, dirnames, filenames in os.walk(root, topdown=True):
            for name in dirnames + filenames:
                try:
                    os.chmod(os.path.join(dirpath, name), 0o700)
                except Exception:
                    pass


# ------------------------------------------------------------------
# Loader helper
# ------------------------------------------------------------------

def load_task_spec(task_id: str, envs_dir: Path = None) -> dict:
    if envs_dir is None:
        envs_dir = Path(__file__).parent.parent / "tasks" / "envs"
    spec_path = envs_dir / f"{task_id}.yaml"
    with open(spec_path, encoding="utf-8") as f:
        return yaml.safe_load(f)
