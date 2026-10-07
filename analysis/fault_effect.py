#!/usr/bin/env python3
"""How often did a scheduled fault actually reach the agent?

Two ways a faulted run can leave the agent's view unchanged:
  never fired   the agent never made the call the fault targets
  no effect     a truncating fault fired on a call it could not degrade (a
                one-row result, an HTTP body with no top-level numbers, a call
                that had already failed); detected by replaying the faulted call
                in a fresh sandbox without the fault and comparing outputs

Sanitization is counted on its 27 analyzed tasks. Needs the extracted traces.
The released injector no longer fires on a call it cannot degrade; these counts
describe the runs as they were collected.

Usage: python analysis/fault_effect.py
"""
import os, sys, json, time, collections

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from common import REPO, ARMS, labels, tasks

TRUNCATING = ("partial_success", "san_partial")


def tool_calls(steps):
    return [s for s in steps if s.get("action") and s.get("observation") not in (None, "(no valid action parsed)")]


def main():
    from environment.mock_server import MockServer
    from environment.sandbox import SandboxManager, load_task_spec
    from faults.injector import FaultInjector
    from tools.registry import build_tool_registry

    server = MockServer(port=8879)
    server.start(); time.sleep(0.5)
    specs = {}

    def replay(task, step):
        spec = specs.setdefault(task, load_task_spec(task))
        sb = SandboxManager(spec, server)
        wd = sb.setup(f"replay-{task}", fault_type="none")
        try:
            inj = FaultInjector(build_tool_registry(wd, server.base_url), "none", None, server=server)
            return inj.call_tool(step["action"], step["action_input"])
        finally:
            server.clear_overrides(); sb.teardown()

    # sanitization is analyzed on the 27 tasks whose fault target is not a shell command
    row_tasks = {t for t, d in tasks().items() if (d.get("fault_target") or {}).get("tool") != "shell_exec"}
    print(f"{'arm':18s} {'fault':16s} {'runs':>5s} {'never fired':>12s} {'no effect':>10s}")
    for arm in ("baseline", "sanitization"):
        rows = [r for r in labels(arm) if arm != "sanitization" or r["task"] in row_tasks]
        cnt = collections.defaultdict(collections.Counter)
        for r in rows:
            t = json.load(open(os.path.join(REPO, ARMS[arm], r["task"], r["model"], f"{r['fault']}_seed{r['seed']}.json")))
            fs = t.get("fault_status") or {}
            c = cnt[r["fault"]]
            c["runs"] += 1
            if not fs.get("fault_injected"):
                c["never fired"] += 1
                continue
            if r["fault"] in TRUNCATING:
                calls = tool_calls(t.get("steps") or [])
                i = fs.get("fault_injected_at_call", -1)
                if 0 < i <= len(calls) and replay(r["task"], calls[i - 1]) == calls[i - 1]["observation"]:
                    c["no effect"] += 1
        tot = collections.Counter()
        for f in sorted(cnt):
            c = cnt[f]; tot.update(c)
            ne = str(c["no effect"]) if f in TRUNCATING else "-"
            print(f"{arm:18s} {f:16s} {c['runs']:5d} {c['never fired']:12d} {ne:>10s}")
        print(f"{arm:18s} {'all':16s} {tot['runs']:5d} {tot['never fired']:12d}")


if __name__ == "__main__":
    main()
