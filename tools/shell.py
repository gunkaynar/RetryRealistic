"""
shell_exec — Constrained shell tool.

Allows a fixed allowlist of commands, run inside the sandbox directory
with a restricted PATH and a 30-second timeout. The agent gets stdout,
stderr, and the exit code.
"""

import json
import shlex
import subprocess
import tempfile
from pathlib import Path

ALLOWED_COMMANDS = frozenset({
    "python3", "python",
    "grep", "egrep", "fgrep",
    "sort", "uniq", "wc", "head", "tail",
    "cat", "ls", "find",
    "awk", "sed", "cut", "tr",
    "echo", "printf", "date",
    "diff", "basename", "dirname", "stat",
})

BLOCKED_PATTERNS = (
    "sudo", "rm -rf /", "mkfs", "> /dev/",
    "chmod 777 /", "chown root", ":(){:|:&};:",
)

# Self-remediation guard: without it, agents defeat permission faults by
# chmod'ing the target back (directly, or via `python3 -c "os.chmod(...)"`).
# Blocked by default to preserve fault persistence; the --allow-repair
# ablation arm lifts it to measure self-remediation rates.
REPAIR_PATTERNS = ("chmod", "chflags", "chown")

SAFE_PATH = "/usr/bin:/bin:/usr/local/bin"
TIMEOUT_SECONDS = 30


def make_shell_tool(work_dir: Path, allow_permission_repair: bool = False):
    def _repair_attempt(command: str, tokens: list) -> bool:
        """Detect permission-repair attempts, including via python3 scripts."""
        low = command.lower()
        if any(pat in low for pat in REPAIR_PATTERNS):
            return True
        # python3 running a sandbox script whose content changes permissions
        for tok in tokens[1:]:
            if tok.endswith(".py"):
                p = (work_dir / tok)
                try:
                    text = p.read_text(encoding="utf-8").lower()
                    if any(pat in text for pat in REPAIR_PATTERNS):
                        return True
                except OSError:
                    pass
        return False

    def shell_exec(command: str) -> str:
        # Block dangerous patterns
        for pat in BLOCKED_PATTERNS:
            if pat in command:
                return json.dumps({
                    "status": "error",
                    "message": f"Command blocked: contains prohibited pattern '{pat}'.",
                })

        # Parse and check the leading command against the allowlist
        try:
            tokens = shlex.split(command)
        except ValueError as ex:
            return json.dumps({"status": "error",
                               "message": f"Invalid command syntax: {ex}"})

        if not allow_permission_repair and _repair_attempt(command, tokens):
            return json.dumps({
                "status": "error",
                "message": ("Operation not permitted: changing file permissions "
                            "or ownership is blocked by security policy in this "
                            "environment."),
            })

        if not tokens:
            return json.dumps({"status": "error", "message": "Empty command."})

        cmd_name = Path(tokens[0]).name  # /usr/bin/python3 → python3
        if cmd_name not in ALLOWED_COMMANDS:
            return json.dumps({
                "status": "error",
                "message": (
                    f"Command '{tokens[0]}' is not in the allowed list. "
                    f"Allowed commands: {sorted(ALLOWED_COMMANDS)}."
                ),
            })

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                env={
                    "PATH": SAFE_PATH,
                    # Keep HOME outside the sandbox: macOS tools auto-create
                    # ~/Library/Caches, which would pollute list_dir output.
                    "HOME": tempfile.gettempdir(),
                    "PYTHONPATH": "",
                },
            )
            return json.dumps({
                "status": "ok",
                "exit_code": result.returncode,
                "stdout": result.stdout[:4000],
                "stderr": result.stderr[:1000] if result.stderr else "",
            })
        except subprocess.TimeoutExpired:
            return json.dumps({
                "status": "error",
                "message": f"Command timed out after {TIMEOUT_SECONDS} seconds.",
            })
        except Exception as ex:
            return json.dumps({"status": "error", "message": str(ex)})

    return shell_exec
