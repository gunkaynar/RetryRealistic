"""
Offline smoke test: exercises sandbox, tools, mock server, and all three
fault-injection layers with a scripted (non-LLM) agent. No API keys needed.

Run:  python3 tests/smoke_offline.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.mock_server import MockServer
from environment.sandbox import SandboxManager, load_task_spec
from faults.injector import FaultInjector
from tools.registry import build_tool_registry

ENVS = Path(__file__).parent.parent / "tasks" / "envs"
PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def make(task_id, fault_type, server, seed=1):
    spec = load_task_spec(task_id, ENVS)
    sandbox = SandboxManager(spec, server)
    work_dir = sandbox.setup(f"smoke-{task_id}-{fault_type}", fault_type=fault_type,
                             fault_target=spec.get("fault_target"))
    registry = build_tool_registry(work_dir, server.base_url)
    injector = FaultInjector(registry, fault_type=fault_type,
                             fault_target=spec.get("fault_target"),
                             seed=seed, server=server,
                             os_fault_paths=sandbox.os_fault_paths)
    return spec, sandbox, injector


server = MockServer(port=8766)
server.start()
time.sleep(0.8)

# ---------------------------------------------------------------- baseline
print("[1] task_01 clean baseline (real fs + calculator)")
spec, sb, inj = make("task_01", "none", server)
r = json.loads(inj.call_tool("list_dir", {"path": "data"}))
check("list_dir ok", r["status"] == "ok" and any(e["name"] == "sales.csv" for e in r["entries"]))
r = json.loads(inj.call_tool("file_read", {"filename": "data/sales.csv"}))
check("file_read ok", r["status"] == "ok" and "48,750" not in r["content"] and "12500.00" in r["content"])
r = json.loads(inj.call_tool("calculator", {"expression": "12500+8300+11200+7800+5450+3500"}))
check("calculator = 48750", r.get("result") == 48750)
r = json.loads(inj.call_tool("file_write", {"filename": "reports/revenue_summary.txt",
                                            "content": "Total revenue = $48,750.00"}))
check("file_write ok", r["status"] == "ok")
check("no fault bookkeeping on baseline", inj.get_status()["fault_injected"] is False)
sb.teardown()

# ------------------------------------------------- OS-level permission fault
print("[2] task_01 permission_denied (real chmod, OS layer)")
spec, sb, inj = make("task_01", "permission_denied", server)
check("sandbox chmod'd 1 path", len(sb.os_fault_paths) == 1
      and sb.os_fault_paths[0].name == "sales.csv", str(sb.os_fault_paths))
st = inj.get_status()
check("layer = os", st["fault_layer"] == "os", st["fault_layer"])
r = json.loads(inj.call_tool("file_read", {"filename": "data/sales.csv"}))
check("real OS permission error", r["status"] == "error" and "Permission denied" in r["message"], r)
check("fault fired at call 1", inj.fault_injected and inj.fault_injected_at_call == 1)
r2 = json.loads(inj.call_tool("file_read", {"filename": "data/sales.csv"}))
check("persistent on retry", r2["status"] == "error")
check("retry counted", inj.retry_count == 1)
r3 = json.loads(inj.call_tool("file_read", {"filename": "data/readme.txt"}))
check("other file still readable", r3["status"] == "ok", r3)
check("unrelated read not a retry", inj.retry_count == 1)
# alternative route to same resource also blocked by the OS
r4 = json.loads(inj.call_tool("shell_exec", {"command": "cat data/sales.csv"}))
check("shell route also denied (exit!=0)", r4["status"] == "ok" and r4["exit_code"] != 0, r4)
check("shell attempt counted as retry, not new error",
      inj.retry_count == 2 and inj.new_errors == [], (inj.retry_count, inj.new_errors))
sb.teardown()
check("teardown removed sandbox", not sb.work_dir)

# ------------------------------------------- OS-level on sql (db fallback)
print("[3] task_13 permission_denied on sql_query (db-file fallback)")
spec, sb, inj = make("task_13", "permission_denied", server)
check("inventory.db chmod'd", len(sb.os_fault_paths) == 1
      and sb.os_fault_paths[0].name == "inventory.db", str(sb.os_fault_paths))
r = json.loads(inj.call_tool("sql_query", {"database": "inventory.db",
                                           "query": "SELECT * FROM items WHERE stock < 10"}))
check("real sqlite open error", r["status"] == "error", r)
check("fault fired", inj.fault_injected)
sb.teardown()

# ------------------------------------------- OS-level on shell script
print("[4] task_28 permission_denied on shell_exec (script chmod)")
spec, sb, inj = make("task_28", "permission_denied", server)
check("preprocess.py chmod'd", any(p.name == "preprocess.py" for p in sb.os_fault_paths),
      str(sb.os_fault_paths))
r = json.loads(inj.call_tool("shell_exec", {"command": "python3 scripts/preprocess.py"}))
check("script fails with real perm error", r["status"] == "ok" and r["exit_code"] != 0
      and "denied" in (r["stderr"] + r["stdout"]).lower(), r)
sb.teardown()

# ------------------------------------------- server-level 503 (transient)
print("[5] task_13 network_error on http_request (server layer, transient)")
spec13 = load_task_spec("task_13", ENVS)
sb = SandboxManager(spec13, server)
wd = sb.setup("smoke-net", fault_type="network_error", fault_target={"tool": "http_request", "match": "suppliers"})
reg = build_tool_registry(wd, server.base_url)
inj = FaultInjector(reg, "network_error", {"tool": "http_request", "match": "suppliers"},
                    seed=7, server=server, os_fault_paths=[])
check("layer = server", inj.get_status()["fault_layer"] == "server")
r = json.loads(inj.call_tool("http_request", {"path": "/api/suppliers", "params": {"item": "desk lamp"}}))
check("real HTTP 503 from server", r["status"] == "error" and r["status_code"] == 503, r)
codes = []
for _ in range(8):
    rr = json.loads(inj.call_tool("http_request", {"path": "/api/suppliers", "params": {"item": "desk lamp"}}))
    codes.append(rr.get("status_code"))
check("transient: some retries succeed", 200 in codes, codes)
check("transient: some retries still fail", 503 in codes, codes)
check("no new errors counted for faulted route", inj.new_errors == [], inj.new_errors)
sb.teardown()

# ------------------------------------------- server-level 401 (persistent)
print("[6] task_13 auth_required on http_request (server layer, persistent)")
sb = SandboxManager(spec13, server)
wd = sb.setup("smoke-auth", fault_type="auth_required", fault_target={"tool": "http_request", "match": "suppliers"})
reg = build_tool_registry(wd, server.base_url)
inj = FaultInjector(reg, "auth_required", {"tool": "http_request", "match": "suppliers"},
                    seed=7, server=server, os_fault_paths=[])
r = json.loads(inj.call_tool("http_request", {"path": "/api/suppliers", "params": {"item": "desk lamp"}}))
check("real HTTP 401", r["status"] == "error" and r["status_code"] == 401, r)
fails = all(
    json.loads(inj.call_tool("http_request", {"path": "/api/suppliers", "params": {"item": "webcam"}}))
    .get("status_code") == 401 for _ in range(4))
check("persistent across retries", fails)
sb.teardown()
server.clear_overrides()

# ------------------------------------------- middleware faults still work
print("[7] task_01 middleware faults (not_found, malformed, timeout, partial)")
for fault, probe in [
    ("not_found", lambda r: "not" in r.lower() and "found" in r.lower()),
    ("malformed_output", lambda r: True),
    ("timeout", lambda r: "timed out" in r.lower()),
]:
    spec, sb, inj = make("task_01", fault, server)
    out = inj.call_tool("file_read", {"filename": "data/sales.csv"})
    check(f"{fault} fires", inj.fault_injected and probe(out), out[:100])
    if fault == "malformed_output":
        try:
            json.loads(out)
            check("malformed is unparseable", False, out[:80])
        except json.JSONDecodeError:
            check("malformed is unparseable", True)
    sb.teardown()

spec, sb, inj = make("task_01", "partial_success", server)
out = json.loads(inj.call_tool("file_read", {"filename": "data/sales.csv"}))
check("partial_success returns valid-but-truncated content",
      out["status"] == "ok" and "3500.00" not in out["content"] and "12500.00" in out["content"],
      out)
sb.teardown()

# ------------------------------------------- timeout transient recovery
print("[8] timeout transient semantics")
spec, sb, inj = make("task_01", "timeout", server, seed=3)
inj.call_tool("file_read", {"filename": "data/sales.csv"})  # fires
outs = [inj.call_tool("file_read", {"filename": "data/sales.csv"}) for _ in range(8)]
succ = sum(1 for o in outs if '"status": "ok"' in o)
check("some timeout retries succeed", succ > 0, outs[:2])
sb.teardown()

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
