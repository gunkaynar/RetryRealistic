"""
faults/injector.py — Fault injection middleware for the realistic environment.

Wraps the tool registry to intercept specific calls and replace their output
with realistic error responses. Fault persistence matches real-world semantics:
  - Transient faults (timeout, network_error, db_locked): succeed ~50% on retry
  - Persistent faults: the same tool+args always fail

Fault targeting:
  fault_target = {"tool": "sql_query", "match": "items"}
  → fires on the first sql_query call whose args contain the string "items"
  → if match is omitted, fires on the first call to that tool
  → if fault_target is None entirely, fires on the first tool call of any kind
"""

from __future__ import annotations

import json
import random
import yaml
from pathlib import Path


# ---------------------------------------------------------------------------
# Load fault config
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(_CONFIG_PATH) as _f:
    _FAULT_CONFIG = yaml.safe_load(_f)["fault_types"]

TRANSIENT_FAULTS = {
    name for name, cfg in _FAULT_CONFIG.items()
    if cfg.get("persistence") == "transient"
}
PERSISTENT_FAULTS = {
    name for name, cfg in _FAULT_CONFIG.items()
    if cfg.get("persistence") == "persistent"
}
# Content-sanitization faults (confound control): identical middleware injection
# point, target, and persistence -- they differ ONLY in the returned payload, so
# any behavioral difference isolates content from the fault label.
SANITIZE_FAULTS = {"san_partial", "san_denial", "san_empty"}
PERSISTENT_FAULTS |= SANITIZE_FAULTS


def _get_fault_message(fault_type: str, tool_name: str) -> str:
    cfg = _FAULT_CONFIG.get(fault_type, {})
    messages = cfg.get("messages", {})
    return messages.get(tool_name) or messages.get("default", f"Error: {fault_type}.")


# ---------------------------------------------------------------------------
# Partial-success corruptors (same logic as original, extended for new tools)
# ---------------------------------------------------------------------------

def _corrupt_partial(tool_name: str, result: str, rng: random.Random) -> str:
    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        return result

    if tool_name == "sql_query" and "rows" in data:
        rows = data["rows"]
        if rows:
            # Silently drop half the rows
            data["rows"] = rows[: max(1, len(rows) // 2)]
            data["count"] = len(data["rows"])
        return json.dumps(data)

    if tool_name == "http_request" and "body" in data:
        body = data["body"]
        if isinstance(body, dict):
            # Corrupt numeric values
            corrupted = {}
            for k, v in body.items():
                if isinstance(v, (int, float)):
                    corrupted[k] = round(v * rng.uniform(0.7, 1.3), 4)
                else:
                    corrupted[k] = v
            data["body"] = corrupted
        return json.dumps(data)

    if tool_name == "file_read" and "content" in data:
        lines = data["content"].split("\n")
        data["content"] = "\n".join(lines[: max(1, len(lines) // 2)])
        return json.dumps(data)

    if tool_name == "calculator" and "result" in data:
        r = data["result"]
        if isinstance(r, (int, float)) and r != 0:
            data["result"] = round(r * rng.uniform(0.7, 1.3), 4)
        return json.dumps(data)

    return result


def _corrupt_malformed(tool_name: str, result: str) -> str:
    """Truncate or garble the result to simulate malformed output."""
    if len(result) > 20:
        cut = len(result) // 2
        return result[:cut] + '", "err'
    return '{"result": null, "data": [{"incomplete": true}], "err'


# ---------------------------------------------------------------------------
# FaultInjector
# ---------------------------------------------------------------------------

OS_PERMISSION_TOOLS = {"file_read", "file_write", "list_dir", "sql_query", "shell_exec"}
SERVER_LAYER_FAULTS = {"network_error", "auth_required"}


class FaultInjector:
    """
    Wraps the tool registry, intercepting calls to inject faults.

    Injection layers:
      - OS: permission_denied on file-backed tools. The sandbox pre-chmods the
        target to 0o000; real system calls produce the error. The injector only
        does bookkeeping (fault-fired call index, retry counts).
      - Server: network_error (503) / auth_required (401) on http_request. The
        mock server's route is overridden so the HTTP client receives a real
        error response.
      - Middleware: everything else — the injector replaces the tool result.

    Args:
        tool_registry:  dict returned by build_tool_registry()
        fault_type:     one of the keys in faults/config.yaml, or "none"
        fault_target:   {"tool": str, "match": str | None} or None
                        None → fault the very first tool call (any tool)
        seed:           RNG seed for transient-fault retry outcomes
        server:         MockServer instance (enables server-layer faults)
        os_fault_paths: sandbox paths chmod'd for this fault (enables OS layer)
    """

    def __init__(
        self,
        tool_registry: dict,
        fault_type: str = "none",
        fault_target: dict | None = None,
        seed: int = 42,
        server=None,
        os_fault_paths: list | None = None,
    ):
        self._registry = tool_registry
        self.fault_type = fault_type
        self._target_tool: str | None = (fault_target or {}).get("tool")
        self._target_match: str | None = (fault_target or {}).get("match")
        self._no_target = fault_target is None

        self._rng = random.Random(seed)
        self._server = server
        self._os_fault_paths = [str(p) for p in (os_fault_paths or [])]

        self._os_level = (
            fault_type == "permission_denied" and bool(self._os_fault_paths)
        )
        self._server_level = (
            fault_type in SERVER_LAYER_FAULTS
            and server is not None
            and (self._target_tool == "http_request" or self._no_target)
        )

        # State
        self.call_count = 0
        self.fault_injected = False
        self.fault_injected_at_call = -1
        self._faulted_key: str | None = None  # "tool_name:serialised_args"
        self._faulted_route: str | None = None
        self.retry_count = 0
        self.new_errors: list[int] = []  # call counts where new (non-fault) errors appeared

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def tool_names(self) -> list[str]:
        return list(self._registry.keys())

    def call_tool(self, tool_name: str, args: dict) -> str:
        self.call_count += 1
        key = self._make_key(tool_name, args)

        if self.fault_type != "none":
            # === OS layer: real chmod'd resource; injector only bookkeeps ===
            if self._os_level and tool_name in OS_PERMISSION_TOOLS:
                return self._call_os_level(tool_name, args, key)

            # === Server layer: real HTTP error from the mock server ===
            if self._server_level and tool_name == "http_request":
                return self._call_server_level(tool_name, args, key)

            # === Middleware layer ===
            if self.fault_injected:
                if self._is_retry(tool_name, key):
                    self.retry_count += 1
                    return self._maybe_persist(tool_name, args)
            elif self._should_inject(tool_name, args):
                self.fault_injected = True
                self.fault_injected_at_call = self.call_count
                self._faulted_key = key
                return self._produce_fault(tool_name, args)

        # === Normal execution ===
        result = self._execute(tool_name, args)
        self._track_new_errors(result)
        return result

    def get_status(self) -> dict:
        return {
            "fault_type": self.fault_type,
            "fault_layer": ("os" if self._os_level
                            else "server" if self._server_level
                            else "middleware" if self.fault_type != "none"
                            else None),
            "fault_injected": self.fault_injected,
            "fault_injected_at_call": self.fault_injected_at_call,
            "total_tool_calls": self.call_count,
            "retry_count": self.retry_count,
            "new_error_calls": self.new_errors,
        }

    # ------------------------------------------------------------------
    # OS-layer path (permission_denied on file-backed tools)
    # ------------------------------------------------------------------

    def _touches_fault_target(self, args: dict) -> bool:
        """Does this call reference the chmod'd resource?"""
        hay = " ".join(str(v) for v in args.values()).lower()
        if self._target_match and self._target_match.lower() in hay:
            return True
        for p in self._os_fault_paths:
            base = p.rsplit("/", 1)[-1].lower()
            if base and base in hay:
                return True
        return False

    def _call_os_level(self, tool_name: str, args: dict, key: str) -> str:
        touches = self._touches_fault_target(args)

        if not self.fault_injected and touches:
            self.fault_injected = True
            self.fault_injected_at_call = self.call_count
            self._faulted_key = key
            return self._execute(tool_name, args)  # OS produces the error

        if self.fault_injected and (key == self._faulted_key or touches):
            self.retry_count += 1
            return self._execute(tool_name, args)  # OS still denies

        # Unrelated call: runs normally against the real substrate.
        result = self._execute(tool_name, args)
        # Don't count OS denials of the faulted resource as agent-caused errors
        # (e.g. a glob that sweeps the denied file without naming it).
        if "permission denied" not in result.lower():
            self._track_new_errors(result)
        return result

    # ------------------------------------------------------------------
    # Server-layer path (network_error / auth_required on http_request)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_route(path: str) -> str:
        return "/" + str(path).strip().lstrip("/")

    def _set_server_override(self, route: str) -> None:
        if self.fault_type == "auth_required":
            self._server.set_override(route, 401, {
                "error": "Unauthorized",
                "message": "Valid credentials are required to access this resource.",
            })
        else:  # network_error
            self._server.set_override(route, 503, {
                "error": "Service Unavailable",
                "message": "The upstream service is temporarily down. Try again later.",
            })

    def _call_server_level(self, tool_name: str, args: dict, key: str) -> str:
        route = self._normalize_route(args.get("path", ""))

        if not self.fault_injected:
            if self._should_inject(tool_name, args):
                self.fault_injected = True
                self.fault_injected_at_call = self.call_count
                self._faulted_key = key
                self._faulted_route = route
                self._set_server_override(route)
                return self._execute(tool_name, args)  # real HTTP 401/503

        elif route == self._faulted_route:
            self.retry_count += 1
            if self.fault_type in TRANSIENT_FAULTS:
                p = _FAULT_CONFIG.get(self.fault_type, {}).get("retry_success_prob", 0.5)
                if self._rng.random() < p:
                    self._server.clear_overrides()  # transient recovery
                else:
                    self._set_server_override(route)
            return self._execute(tool_name, args)

        result = self._execute(tool_name, args)
        self._track_new_errors(result)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_key(self, tool_name: str, args: dict) -> str:
        import json as _json
        return f"{tool_name}:{_json.dumps(args, sort_keys=True)}"

    def _should_inject(self, tool_name: str, args: dict) -> bool:
        if self._no_target:
            return True  # fault first call of any kind
        if self._target_tool and tool_name != self._target_tool:
            return False
        if self._target_match:
            # Check if match substring appears in any arg value
            for v in args.values():
                if self._target_match.lower() in str(v).lower():
                    return True
            return False
        return True  # tool matches, no match string required

    def _is_retry(self, tool_name: str, key: str) -> bool:
        if key == self._faulted_key:
            return True
        # permission_denied also fires on same tool with different args
        if self.fault_type == "permission_denied" and tool_name == self._target_tool:
            return True
        # auth_required same
        if self.fault_type == "auth_required" and tool_name == self._target_tool:
            return True
        return False

    def _maybe_persist(self, tool_name: str, args: dict) -> str:
        if self.fault_type in PERSISTENT_FAULTS:
            return self._produce_fault(tool_name, args)
        # Transient: succeed with probability from config
        p = _FAULT_CONFIG.get(self.fault_type, {}).get("retry_success_prob", 0.5)
        if self._rng.random() < p:
            return self._execute(tool_name, args)  # recovery!
        return self._produce_fault(tool_name, args)

    def _produce_fault(self, tool_name: str, args: dict) -> str:
        # --- content-sanitization conditions (all via this same middleware path) ---
        if self.fault_type == "san_partial":
            real = self._execute(tool_name, args)
            return _corrupt_partial(tool_name, real, self._rng)   # misleading partial data
        if self.fault_type == "san_denial":                       # explicit unavailability
            return json.dumps({"status": "error",
                               "message": "The requested resource is currently unavailable."})
        if self.fault_type == "san_empty":                        # valid but empty result
            return json.dumps({"status": "ok", "rows": [], "entries": [], "content": ""})

        if self.fault_type == "partial_success":
            real = self._execute(tool_name, args)
            return _corrupt_partial(tool_name, real, self._rng)

        if self.fault_type == "malformed_output":
            real = self._execute(tool_name, args)
            return _corrupt_malformed(tool_name, real)

        if self.fault_type == "nonzero_exit":
            msg = _FAULT_CONFIG["nonzero_exit"]["messages"].get(tool_name,
                  _FAULT_CONFIG["nonzero_exit"]["messages"]["default"])
            # nonzero_exit is structured for shell_exec
            if tool_name == "shell_exec":
                return msg
            return json.dumps({"status": "error", "message": msg})

        return _get_fault_message(self.fault_type, tool_name)

    def _execute(self, tool_name: str, args: dict) -> str:
        if tool_name not in self._registry:
            return json.dumps({"status": "error",
                               "message": f"Unknown tool: '{tool_name}'."})
        func = self._registry[tool_name]["function"]
        try:
            return func(**args)
        except TypeError as ex:
            return json.dumps({"status": "error",
                               "message": f"Invalid arguments for '{tool_name}': {ex}"})
        except Exception as ex:
            return json.dumps({"status": "error", "message": str(ex)})

    def _track_new_errors(self, result: str) -> None:
        """Record calls (after the fault) that returned errors due to agent actions."""
        if not self.fault_injected:
            return
        try:
            data = json.loads(result)
            if isinstance(data, dict) and data.get("status") == "error":
                self.new_errors.append(self.call_count)
        except Exception:
            # Non-JSON result that looks like an error message
            if result.startswith("Error:"):
                self.new_errors.append(self.call_count)
