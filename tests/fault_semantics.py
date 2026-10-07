#!/usr/bin/env python3
"""Offline checks of exactly when and how each fault fires. No API keys.

    python tests/fault_semantics.py
"""
import sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from environment.mock_server import MockServer
from environment.sandbox import SandboxManager, load_task_spec
from faults.injector import FaultInjector
from tools.registry import build_tool_registry

server = MockServer(port=8871)
server.start(); time.sleep(0.5)
passed = failed = 0


def check(name, cond):
    global passed, failed
    print(("  ok   " if cond else "  FAIL ") + name)
    passed += bool(cond); failed += (not cond)


def setup(task, fault):
    spec = load_task_spec(task)
    sb = SandboxManager(spec, server)
    wd = sb.setup(f"sem-{task}-{fault}", fault_type=fault, fault_target=spec.get("fault_target"))
    inj = FaultInjector(build_tool_registry(wd, server.base_url), fault, spec.get("fault_target"),
                        seed=3, server=server, os_fault_paths=sb.os_fault_paths)
    return sb, inj


def call(inj, tool, args):
    return json.loads(inj.call_tool(tool, args))


def done(sb):
    server.clear_overrides(); sb.teardown()


print("[1] network_error: query strings and missing leading slash still reach the fault")
for path in ("/api/commodities?item=gold", "api/commodities"):
    sb, inj = setup("task_07", "network_error")
    r = call(inj, "http_request", {"path": path, "params": {"item": "gold"}})
    check(f"{path!r} gets a real 503", r.get("status_code") == 503 and inj.fault_injected)
    done(sb)

print("[2] network_error: a call with other arguments runs normally")
sb, inj = setup("task_07", "network_error")
call(inj, "http_request", {"path": "/api/commodities", "params": {"item": "gold"}})
codes = [call(inj, "http_request", {"path": "/api/commodities", "params": {"item": "silver"}}).get("status_code")
         for _ in range(6)]
check("item=silver always 200", codes == [200] * 6)
done(sb)

print("[3] auth_required (server): every later request to the tool is refused")
sb, inj = setup("task_07", "auth_required")
codes = [call(inj, "http_request", {"path": "/api/commodities", "params": {"item": it}}).get("status_code")
         for it in ("gold", "gold", "silver")]
check("401, 401, 401", codes == [401, 401, 401])
done(sb)

print("[4] partial_success: an identical retry returns the identical degraded data")
sb, inj = setup("task_02", "partial_success")
q = {"database": "company.db", "query": "SELECT * FROM employees"}
a, b = call(inj, "sql_query", q), call(inj, "sql_query", q)
check("fired, rows halved, retry identical", inj.fault_injected and a == b and a["count"] < 10)
done(sb)

print("[5] partial_success: a one-row result is not a fault firing")
sb, inj = setup("task_02", "partial_success")
r = call(inj, "sql_query", {"database": "company.db", "query": "SELECT COUNT(*) AS n FROM employees"})
check("aggregate passes through, fault still armed", r["rows"][0]["n"] >= 1 and not inj.fault_injected)
r = call(inj, "sql_query", q)
check("next multi-row call is degraded", inj.fault_injected)
done(sb)

print("[6] partial_success: an erroring call is not a fault firing")
sb, inj = setup("task_02", "partial_success")
call(inj, "sql_query", {"database": "company.db", "query": "SELECT * FROM employees_typo"})
check("fault still armed after an SQL error", not inj.fault_injected)
done(sb)

print("[7] permission_denied (OS): data written to a file does not count as touching the target")
sb, inj = setup("task_30", "permission_denied")
call(inj, "file_write", {"path": "reports/notes.txt", "content": "results.db was unreadable"})
check("not fired by file_write content", not inj.fault_injected)
done(sb)

print("[8] san_empty: an empty result shaped like the tool's own")
for task, tool, args, field in (("task_02", "sql_query", q, "rows"),
                                 ("task_07", "http_request", {"path": "/api/commodities", "params": {"item": "gold"}}, "body")):
    sb, inj = setup(task, "san_empty")
    r = call(inj, tool, args)
    check(f"{tool}: status ok with empty {field}", r.get("status") == "ok" and not r.get(field))
    done(sb)

print("[9] mock server: an empty or one-letter query value gets the default, not a guess")
sb, inj = setup("task_07", "none")
body = lambda it: call(inj, "http_request", {"path": "/api/commodities", "params": {"item": it}})["body"]
check("'' -> default", body("") == {"error": "commodity not found"})
check("'l' -> default", body("l") == {"error": "commodity not found"})
check("'Gold' -> gold", body("Gold").get("item") == "gold")
check("'plat' -> platinum", body("plat").get("item") == "platinum")
get = lambda **a: call(inj, "http_request", {"path": "/api/commodities", **a})
check("a lone parameter under another name is used", get(params={"commodity": "gold"})["body"].get("item") == "gold")
r = get()
check("no parameter -> 400 naming it", r.get("status_code") == 400 and "item" in r["body"]["error"])
check("params as a JSON string", get(params='{"item": "silver"}')["body"].get("item") == "silver")
done(sb)

print("[10] http_request: only the local server is reachable")
sb, inj = setup("task_07", "none")
r = call(inj, "http_request", {"path": "https://example.com/api"})
check("full URL refused", r.get("status") == "error")
done(sb)

print("[11] shell_exec: ordinary command lines run; escapes and other programs do not")
sb, inj = setup("task_01", "none")
ok = lambda cmd: call(inj, "shell_exec", {"command": cmd}).get("status") == "ok"
for cmd in ["grep -c Widget data/sales.csv 2>&1",
            "python3 << 'EOF'\nprint(sum(1 for _ in open('data/sales.csv')))\nEOF",
            "awk -F, 'NR>1 {print $3}' data/sales.csv | sort | head -2 > reports/x.txt; cat reports/x.txt",
            "find . \\( -name '*.csv' -o -name '*.txt' \\) 2>/dev/null",
            "awk -F, '{print $1, $2}' data/sales.csv | while read a b; do echo $a; done",
            "echo rows: $(wc -l < data/sales.csv)",
            "python3 -c \"print(10 / 4)\""]:
    check(f"runs: {cmd.splitlines()[0][:48]}", ok(cmd))
for cmd in ["ls /", "cat /etc/passwd", "echo hi; id -un", "cat data/sales.csv | xargs echo",
            "python3 -c \"print(open('/etc/hostname').read())\"", "echo $(whoami)"]:
    check(f"refused: {cmd[:48]}", not ok(cmd))
done(sb)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
