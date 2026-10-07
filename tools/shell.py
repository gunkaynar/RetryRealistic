"""
shell_exec — Constrained shell tool.

Allows a fixed allowlist of commands, run inside the sandbox directory
with a restricted PATH and a 30-second timeout. The agent gets stdout,
stderr, and the exit code. Pipelines, command sequences and command substitution
are allowed when every command in them is on the allowlist, and paths outside
the sandbox are refused. These checks keep a well-behaved
agent inside its sandbox, but a determined one can still reach the host
through python3, so run the testbed in a container when isolation matters.
"""

import json
import os
import re
import shlex
import stat
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

# tokens after which a new command starts in a shell command line
_SEPARATORS = {"|", "||", "&", "&&", ";", "(", ")", "\n"}
# heredoc start (<<EOF, <<-'EOF', << "END"), and redirections with their targets
_HEREDOC = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")
_REDIRECT = re.compile(r"\d*>&\d+|&>>?\s*[^\s'\";|&]+|\d*>>?\s*[^\s'\";|&()]+|<<-?\s*['\"]?\w+['\"]?|<(?!\()\s*[^\s'\";|&()]+")
# shell syntax after which a command name still follows
_KEYWORDS = {"while", "until", "if", "then", "else", "elif", "do", "done", "fi", "esac",
             "!", "{", "}"}
# shell builtins that are harmless here; what follows them are arguments
_BUILTINS = {"read", "test", "[", "[[", "true", "false", "for", "case", ":"}

# an absolute path quoted inside inline code, e.g. python3 -c "open('/etc/x')"
_QUOTED_ABS = re.compile(r"""['"](/[^'"\s]*)""")

SAFE_PATH = "/usr/bin:/bin:/usr/local/bin"
TIMEOUT_SECONDS = 30


def _modes(root: Path) -> dict:
    """Permission bits of everything in the sandbox."""
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            p = os.path.join(dirpath, name)
            try:
                out[p] = stat.S_IMODE(os.lstat(p).st_mode)
            except OSError:
                pass
    return out


def make_shell_tool(work_dir: Path, allow_permission_repair: bool = False):
    root = Path(work_dir).resolve()

    def _outside(path_str: str) -> bool:
        """An absolute or ..-relative path that resolves outside the sandbox."""
        path_str = path_str.rstrip(";|&()")
        if path_str in ("/dev/null", "/dev/stdin", "/dev/stdout", "/dev/stderr", ""):
            return False
        if path_str.startswith(tuple(d + "/" for d in SAFE_PATH.split(":"))):
            return False                                   # a program named by full path
        p = Path(path_str)
        if p.is_absolute():
            parts = p.parts
            # a regex such as awk '/WARN/' is not a path: only names that exist at /
            if len(parts) > 1 and not (Path("/") / parts[1]).exists():
                return False
            target = p
        elif ".." in p.parts:
            target = root / p
        else:
            return False
        try:
            target = target.resolve()
        except OSError:
            return True
        return target != root and root not in target.parents

    def _commands(command: str) -> list:
        """The command name of every simple command in a command line.

        Heredoc bodies and redirection targets are data, not commands, so they
        are removed first; shell keywords (while, do, if, ...) are syntax.
        """
        text = command
        # heredoc bodies: from the end of the line with <<WORD up to the line WORD
        for m in reversed(list(_HEREDOC.finditer(text))):
            word = m.group(2)
            nl = text.find("\n", m.end())
            if nl == -1:
                continue
            end = re.search(r"(?m)^\s*" + re.escape(word) + r"\s*$", text[nl + 1:])
            stop = nl + 1 + (end.end() if end else len(text) - nl - 1)
            text = text[:nl] + "\n" + text[stop:]
        # $(...) and `...` run commands too: check what they run
        text = _REDIRECT.sub(" ", text).replace("\n", " ; ").replace("`", " ; ").replace("$(", " ( ")
        text = text.replace("\\(", " ").replace("\\)", " ")          # find's escaped \( \)
        lex = shlex.shlex(text, posix=True, punctuation_chars=";&|()")
        lex.whitespace_split = True
        names, expect = [], True
        for tok in lex:
            if tok in _SEPARATORS or (tok and set(tok) <= set(";&|()")):
                expect = True
                continue
            if expect:
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", tok):   # VAR=value prefix
                    continue
                if tok in _KEYWORDS:
                    continue                                       # still in command position
                if tok not in _BUILTINS:
                    names.append(Path(tok).name)
                expect = False
        return names

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

        try:
            names = _commands(command)
        except ValueError as ex:
            return json.dumps({"status": "error", "message": f"Invalid command syntax: {ex}"})
        for cmd_name in names:
            if cmd_name not in ALLOWED_COMMANDS:
                return json.dumps({
                    "status": "error",
                    "message": (
                        f"Command '{cmd_name}' is not in the allowed list. "
                        f"Allowed commands: {sorted(ALLOWED_COMMANDS)}."
                    ),
                })
        # whole shell arguments, and absolute paths quoted inside inline code; a
        # "/" inside code (a division, a regex) is not a path
        pieces = [p for t in tokens if not t.startswith("-") and not re.search(r"\s", t)
                  for p in re.split(r"[=<>]", t)]                   # e.g. --out=/etc/x
        pieces += [q for q in _QUOTED_ABS.findall(command) if len(Path(q).parts) > 1]
        for piece in pieces:
            if piece and ("/" in piece) and _outside(piece):
                return json.dumps({
                    "status": "error",
                    "message": f"Path outside the sandbox is not accessible: '{piece}'.",
                })

        before = None if allow_permission_repair else _modes(root)
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
            if before is not None:
                # A permission change made by any route (os.chmod, getattr,
                # a script) is undone and reported, so the fault persists.
                changed = [p for p, m in _modes(root).items() if p in before and before[p] != m]
                for p in changed:
                    try:
                        os.chmod(p, before[p])
                    except OSError:
                        pass
                if changed:
                    return json.dumps({
                        "status": "error",
                        "message": ("Operation not permitted: changing file permissions "
                                    "or ownership is blocked by security policy in this "
                                    "environment."),
                    })
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
