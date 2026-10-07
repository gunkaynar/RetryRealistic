#!/usr/bin/env python3
"""Recompute every table and in-text number of the paper.

The retry analysis (Table 5) and the run counts read the extracted traces;
everything else runs from the labels alone.

Usage: python analysis/reproduce.py
"""
import sys, os, json, glob, collections
from common import REPO, LABELS, AGENTS, AGENT_NAME, ARMS, labels, tasks

FAULT_ORDER = ("partial_success", "network_error", "malformed_output", "timeout", "permission_denied",
               "auth_required", "not_found", "db_locked", "nonzero_exit")
PERSISTENT = {"permission_denied", "auth_required", "not_found", "malformed_output",
              "partial_success", "nonzero_exit"}


def _errored(step):
    """Whether a tool call's observation reports a failure."""
    obs = step.get("observation")
    if obs is None:
        return False
    try:
        obs = json.loads(obs)
    except (TypeError, ValueError):
        low = str(obs).lower()
        return any(k in low for k in ("error", "denied", "not found", "not available",
                                      "unauthorized", "401", "forbidden"))
    if not isinstance(obs, dict):
        return False
    if obs.get("status") == "error":
        return True
    return step.get("action") == "shell_exec" and obs.get("exit_code", 0) not in (0, None)


def retries_of_failing_call(steps):
    """Repeats, beyond the first, of the most-repeated identical failing call.

    Calls count as identical when tool and arguments match after lowercasing and
    dropping whitespace and a leading ./ in paths.
    """
    cnt = collections.Counter()
    for s in steps:
        if s.get("action") and _errored(s):
            a = json.dumps(s.get("action_input"), sort_keys=True)
            cnt[(s["action"], a.lower().replace(" ", "").replace("./", ""))] += 1
    return max(cnt.values()) - 1 if cnt else 0


def fired(rows):
    return [r for r in rows if r["fault"] != "none"]


def graded(rows):
    """Runs with one of the three labels."""
    return [r for r in rows if r["label"] in LABELS]


def pct(rows):
    rows = graded(rows)
    n = len(rows) or 1
    return {l: 100.0 * sum(r["label"] == l for r in rows) / n for l in LABELS}


def by(rows, key):
    g = collections.defaultdict(list)
    for r in rows:
        g[r[key]].append(r)
    return g


class Report:
    """Collects (name, value) pairs for printing."""

    def __init__(self):
        self.items = []

    def head(self, title):
        self.items.append(("#", title))

    def put(self, name, value):
        self.items.append((name, value))


def three(rep, name, rows):
    p = pct(rows)
    rep.put(f"{name}: correct", p["correct"])
    rep.put(f"{name}: fabricated", p["fabricated"])
    rep.put(f"{name}: honest failure", p["honest_failure"])


def eta2(y, g):
    n = len(y); grand = sum(y) / n
    sst = sum((v - grand) ** 2 for v in y)
    groups = collections.defaultdict(list)
    for v, k in zip(y, g):
        groups[k].append(v)
    ssb = sum(len(vs) * (sum(vs) / len(vs) - grand) ** 2 for vs in groups.values())
    return ssb / sst if sst else 0.0


def build():
    rep = Report()
    L = {arm: labels(arm) for arm in ARMS}
    base = fired(L["baseline"])

    rep.head("Table 3: baseline recovery by agent (% of fault-fired runs)")
    for m in AGENTS:
        three(rep, AGENT_NAME[m], [r for r in base if r["model"] == m])
    three(rep, "Pooled", base)

    rep.head("Table 4: baseline recovery by fault (% , pooled over agents)")
    g = by(base, "fault")
    for f in FAULT_ORDER:
        three(rep, f, g[f])

    rep.head("Section 6.2: fault explains more of fabrication's variance than the agent")
    y = [1.0 if r["label"] == "fabricated" else 0.0 for r in base]
    ef, em = eta2(y, [r["fault"] for r in base]), eta2(y, [r["model"] for r in base])
    rep.put("eta^2 fault", ef)
    rep.put("eta^2 agent", em)
    rep.put("ratio", ef / em)
    prone = []
    for m in AGENTS:
        for f in FAULT_ORDER:
            p = pct([r for r in base if r["model"] == m and r["fault"] == f])
            if p["fabricated"] >= 30 and p["fabricated"] > p["honest_failure"]:
                prone.append((m, f))
    rep.put("fabrication-prone agent x fault cells (fab >= 30% and > honest failure)", len(prone))
    rep.put("agents for which partial_success is fabrication-prone",
            sum((m, "partial_success") in prone for m in AGENTS))

    rep.head("Table 5: outcome by retries of the same failing call, persistent faults (needs traces)")
    lab = {(r["task"], r["model"], r["fault"], r["seed"]): r["label"] for r in base}
    buckets = collections.defaultdict(list)
    split = collections.defaultdict(list)
    for p in glob.glob(os.path.join(REPO, ARMS["baseline"], "*", "*", "*.json")):
        task, model, fn = p.split(os.sep)[-3:]
        fault, seed = fn[:-5].rsplit("_seed", 1)
        if fault not in PERSISTENT:
            continue
        steps = json.load(open(p)).get("steps") or []
        y_ = lab.get((task, model, fault, int(seed)))
        if not steps or y_ not in LABELS:
            continue
        rc = retries_of_failing_call(steps)
        bk = "0-1" if rc <= 1 else "2" if rc == 2 else "3+"
        row = {"label": y_}
        buckets[bk].append(row)
        cls = "partial-data" if fault in ("partial_success", "malformed_output") else \
              "hard-denial" if fault in ("permission_denied", "auth_required") else None
        if cls:
            split[(cls, bk)].append(row)
    if not buckets:
        rep.put("(traces not extracted; skipped)", float("nan"))
    for bk in ("0-1", "2", "3+"):
        rep.put(f"retries {bk}: N", len(buckets[bk]))
        three(rep, f"retries {bk}", buckets[bk])
    if buckets:
        pd_all = sum(len(split[("partial-data", b)]) for b in ("0-1", "2", "3+"))
        rep.put("partial-data faults: runs", pd_all)
        rep.put("partial-data faults: runs at 0-1 retries", len(split[("partial-data", "0-1")]))
        rep.put("partial-data faults: fabrication at 0-1 retries", pct(split[("partial-data", "0-1")])["fabricated"])
        rep.put("partial-data faults: runs at 3+ retries", len(split[("partial-data", "3+")]))
        rep.put("partial-data faults: fabrication at 3+ retries", pct(split[("partial-data", "3+")])["fabricated"])
        hd_all = sum(len(split[("hard-denial", b)]) for b in ("0-1", "2", "3+"))
        hd2 = len(split[("hard-denial", "2")]) + len(split[("hard-denial", "3+")])
        rep.put("hard-denial faults: % of runs reaching 2+ retries", 100.0 * hd2 / (hd_all or 1))
        for b in ("0-1", "2", "3+"):
            rep.put(f"hard-denial faults: fabrication at {b} retries", pct(split[("hard-denial", b)])["fabricated"])

    rep.head("Table 6: payload sanitization, 27 data-target tasks")
    tk = tasks()
    row_tasks = {t for t, d in tk.items() if (d.get("fault_target") or {}).get("tool") != "shell_exec"}
    san = [r for r in L["sanitization"] if r["task"] in row_tasks]
    names = {"san_partial": "partial (misleading)", "san_empty": "empty (valid, none)", "san_denial": "denial (unavailable)"}
    for f in ("san_partial", "san_empty", "san_denial"):
        three(rep, names[f], [r for r in san if r["fault"] == f])
    for m in AGENTS:
        sm = [r for r in san if r["model"] == m]
        rep.put(f"{AGENT_NAME[m]}: fabrication, partial", pct([r for r in sm if r["fault"] == "san_partial"])["fabricated"])
        rep.put(f"{AGENT_NAME[m]}: fabrication, denial", pct([r for r in sm if r["fault"] == "san_denial"])["fabricated"])

    rep.head("Table 7: the Monitor (% of graded fault-fired runs)")
    full, code = fired(L["monitor_full"]), fired(L["monitor_code_only"])
    three(rep, "full cascade S1-S4", full)
    three(rep, "code-only S1-S3", code)
    for name, rows in (("baseline", base), ("full", full), ("code-only", code)):
        rep.put(f"DeepSeek-V4 correct, {name}", pct([r for r in rows if r["model"] == "deepseek-v4"])["correct"])
    for m in AGENTS:
        rep.put(f"{AGENT_NAME[m]} correct, baseline -> full: change",
                pct([r for r in full if r["model"] == m])["correct"] - pct([r for r in base if r["model"] == m])["correct"])

    rep.head("Section 6.6: no chain of thought, honesty prompt, self-remediation")
    nc, ho = fired(L["no_cot"]), fired(L["honesty"])
    three(rep, "no chain of thought", nc)
    for f in ("partial_success", "malformed_output"):
        rep.put(f"{f} fabrication, baseline", pct([r for r in base if r["fault"] == f])["fabricated"])
        rep.put(f"{f} fabrication, no chain of thought", pct([r for r in nc if r["fault"] == f])["fabricated"])
    three(rep, "honesty prompt", ho)
    for f in ("network_error", "auth_required"):
        rep.put(f"{f} fabrication, baseline", pct([r for r in base if r["fault"] == f])["fabricated"])
        rep.put(f"{f} fabrication, honesty prompt", pct([r for r in ho if r["fault"] == f])["fabricated"])
    rem = L["self_remediation"]
    rep.put("permission_denied correct, baseline, pooled", pct([r for r in base if r["fault"] == "permission_denied"])["correct"])
    for m in AGENTS:
        rep.put(f"{AGENT_NAME[m]}: permission_denied correct, baseline",
                pct([r for r in base if r["model"] == m and r["fault"] == "permission_denied"])["correct"])
        rep.put(f"{AGENT_NAME[m]}: permission_denied correct, repair allowed",
                pct([r for r in rem if r["model"] == m])["correct"])
    return rep


def counts():
    print("\n# Runs per arm (extracted traces) and task difficulty")
    for arm, d in ARMS.items():
        fs = glob.glob(os.path.join(REPO, d, "*", "*", "*.json"))
        if fs:
            clean = sum(os.path.basename(f).startswith("none_") for f in fs)
            print(f"  {arm:18s} {len(fs):5d} runs ({clean} fault-free)")
    reg = json.load(open(os.path.join(REPO, "tasks", "registry.json")))
    print("  tasks by difficulty:", dict(collections.Counter(t.get("difficulty") for t in reg)))


def show(reps, heads):
    w = max(len(n) for n, _ in reps[0].items if n != "#")
    if len(reps) > 1:
        print(f"\n  {'':{w}s}  " + "  ".join(f"{h:>9s}" for h in heads))
    for i, (name, v) in enumerate(reps[0].items):
        if name == "#":
            print(f"\n# {v}")
            continue
        vals = [r.items[i][1] for r in reps]
        fmt = lambda x: f"{x:9d}" if isinstance(x, int) else f"{x:9.3f}" if abs(x) < 1 and x != 0 else f"{x:9.1f}"
        print(f"  {name:{w}s}  " + "  ".join(fmt(x) for x in vals))


def main(argv):
    show([build()], [""])
    counts()


if __name__ == "__main__":
    main(sys.argv[1:])
