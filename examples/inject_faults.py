#!/usr/bin/env python3
"""Drive the testbed by hand: build a task's sandbox, inject a fault, call tools.

No model and no API key is involved. This is the layer an agent sits on, so the
same few lines plug RetryRealistic into any agent loop: call
injector.call_tool(name, args) wherever your agent would execute a tool.

Usage:
    python examples/inject_faults.py                         # task_15, partial_success
    python examples/inject_faults.py task_02 permission_denied
"""
import sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from environment.mock_server import MockServer
from environment.sandbox import SandboxManager, load_task_spec
from faults.injector import FaultInjector
from tools.registry import build_tool_registry


def main(task_id="task_15", fault="partial_success"):
    spec = load_task_spec(task_id)
    server = MockServer(port=8799)
    server.start(); time.sleep(0.5)

    sandbox = SandboxManager(spec, server)
    work_dir = sandbox.setup(f"{task_id}_demo", fault_type=fault, fault_target=spec.get("fault_target"))
    try:
        tools = build_tool_registry(work_dir, server.base_url)
        injector = FaultInjector(tools, fault_type=fault, fault_target=spec.get("fault_target"),
                                 seed=1, server=server, os_fault_paths=sandbox.os_fault_paths)
        print(f"task {task_id}: {spec['description'].strip()}")
        print(f"fault {fault} on {spec.get('fault_target')}\n")

        print("list_dir .\n ", injector.call_tool("list_dir", {"path": "."}))
        target = spec.get("fault_target") or {}
        if target.get("tool") == "sql_query":
            # the target's match string names a database file or a table in it
            m = target.get("match", "")
            db = next((d for d in spec["databases"] if m in d["name"] or any(m in t["name"] for t in d["tables"])),
                      spec["databases"][0])
            table = next((t["name"] for t in db["tables"] if m in t["name"]), db["tables"][0]["name"])
            call = ("sql_query", {"database": db["name"], "query": f"SELECT * FROM {table}"})
            print(f"\nsql_query (first call: the fault fires)\n ", injector.call_tool(*call))
            print(f"\nsql_query (identical retry)\n ", injector.call_tool(*call))
        print("\nfault status:", json.dumps(injector.get_status(), indent=2))
    finally:
        server.clear_overrides()
        sandbox.teardown()


if __name__ == "__main__":
    main(*sys.argv[1:3])
