"""
run.py — Experiment runner for RetryRealistic.

Usage:
  # Full experiment (all tasks × all models × all fault types):
  python run.py

  # Single run (for testing):
  python run.py --task task_01 --model llama-8b --fault timeout

  # Specific model across all tasks:
  python run.py --model llama-8b

  # Baseline only (no faults):
  python run.py --fault none

Options:
  --task   TASK_ID     Run only this task
  --model  MODEL_KEY   Run only this model (see agent/providers.py)
  --fault  FAULT_TYPE  Run only this fault type
  --port   PORT        Mock server port (default 8765)
  --out    DIR         Output directory for traces (default traces/)
  --max-steps N        Max ReAct steps per run (default 20)
  --dry-run            Print what would run without actually running
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import yaml

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent))

from env_loader import load_env
load_env()

from agent.providers import create_agent, available_models
from environment.mock_server import MockServer
from environment.sandbox import SandboxManager, load_task_spec
from faults.injector import FaultInjector
from tools.registry import build_tool_registry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FAULT_TYPES = [
    "none",              # clean baseline
    "timeout",
    "permission_denied",
    "not_found",
    "malformed_output",
    "partial_success",
    "network_error",
    "auth_required",
    "db_locked",
    "nonzero_exit",
]

# Not all fault types make sense for every primary tool.
# These mappings skip inapplicable combinations.
TOOL_COMPATIBLE_FAULTS = {
    "file_read":    ["none", "timeout", "permission_denied", "not_found", "malformed_output", "partial_success"],
    "file_write":   ["none", "timeout", "permission_denied"],
    "sql_query":    ["none", "timeout", "permission_denied", "not_found", "malformed_output", "partial_success", "db_locked", "auth_required"],
    "http_request": ["none", "timeout", "network_error", "auth_required", "not_found", "malformed_output", "partial_success"],
    "shell_exec":   ["none", "timeout", "permission_denied", "not_found", "nonzero_exit"],
    "list_dir":     ["none", "timeout", "permission_denied", "not_found"],
    "calculator":   ["none", "malformed_output", "partial_success"],
}


def compatible_faults(primary_tool: str) -> list[str]:
    return TOOL_COMPATIBLE_FAULTS.get(primary_tool, FAULT_TYPES)


# ---------------------------------------------------------------------------
# Trace I/O
# ---------------------------------------------------------------------------

def trace_path(out_dir: Path, task_id: str, model: str, fault: str, seed: int) -> Path:
    p = out_dir / task_id / model
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{fault}_seed{seed}.json"


def already_done(out_dir: Path, task_id: str, model: str, fault: str, seed: int) -> bool:
    return trace_path(out_dir, task_id, model, fault, seed).exists()


def save_trace(trace: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(trace, f, indent=2)


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_one(
    task_meta: dict,
    task_spec: dict,
    model_key: str,
    fault_type: str,
    server: MockServer,
    out_dir: Path,
    max_steps: int,
    seed: int = 1,
    monitor=None,
    allow_repair: bool = False,
    reason: bool = True,
    honesty: bool = False,
) -> dict:
    task_id = task_meta["id"]
    run_id = f"{task_id}_{model_key}_{fault_type}_s{seed}_{int(time.time())}"

    fault_target = task_spec.get("fault_target")
    sandbox = SandboxManager(task_spec, server)
    work_dir = sandbox.setup(run_id, fault_type=fault_type, fault_target=fault_target)

    try:
        tool_registry = build_tool_registry(work_dir, server.base_url,
                                            allow_permission_repair=allow_repair)
        injector = FaultInjector(
            tool_registry=tool_registry,
            fault_type=fault_type,
            fault_target=fault_target,
            seed=seed,
            server=server,
            os_fault_paths=sandbox.os_fault_paths,
        )

        agent = create_agent(model_key)
        if monitor is not None:
            monitor.reset()
        trace = agent.run(task_spec["description"], injector,
                          max_steps=max_steps, monitor=monitor, reason=reason,
                          honesty=honesty)

        trace["meta"] = {
            "task_id": task_id,
            "model": model_key,
            "fault_type": fault_type,
            "seed": seed,
            "difficulty": task_meta.get("difficulty"),
            "primary_tool": task_meta.get("primary_tool"),
            "description": task_spec.get("description"),
            "ground_truth": task_spec.get("ground_truth"),
            "monitor_enabled": monitor is not None,
            "allow_repair": allow_repair,
        }
        return trace

    finally:
        server.clear_overrides()
        sandbox.teardown()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RetryRealistic experiment runner")
    parser.add_argument("--task",      help="Run only this task ID")
    parser.add_argument("--model",     help="Run only this model key")
    parser.add_argument("--fault",     help="Run only this fault type")
    parser.add_argument("--port",      type=int, default=8765)
    parser.add_argument("--out",       default=None,
                        help="Output dir (default: traces/, or traces_monitor/ with --monitor)")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--seeds",     type=int, default=3,
                        help="Number of seeds per condition (default 3)")
    parser.add_argument("--monitor",   action="store_true",
                        help="Enable the Monitor Agent intervention arm")
    parser.add_argument("--monitor-model", default=None,
                        help="Monitor model (default: claude-sonnet-4-6)")
    parser.add_argument("--no-monitor-drift", dest="monitor_drift", action="store_false",
                        help="Code-only cascade: stop after S1-S3, without the S4 LLM "
                             "goal-drift check.")
    parser.add_argument("--allow-repair", action="store_true",
                        help="Ablation: let agents change file permissions "
                             "(self-remediation) instead of blocking it")
    parser.add_argument("--no-reasoning", action="store_true",
                        help="Ablation: remove the THOUGHT / Chain-of-Thought step; "
                             "the agent emits actions directly with no explicit reasoning.")
    parser.add_argument("--honesty-prompt", action="store_true",
                        help="Intervention: append a 'never estimate; report unavailable' "
                             "honesty clause to the system prompt.")
    parser.add_argument("--dry-run",   action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out or
                   ("traces_monitor" if args.monitor
                    else "traces_repair" if args.allow_repair
                    else "traces_noreason" if args.no_reasoning
                    else "traces_honesty" if args.honesty_prompt
                    else "traces"))
    tasks_dir = Path(__file__).parent / "tasks"
    envs_dir = tasks_dir / "envs"

    # Load task registry
    with open(tasks_dir / "registry.json") as f:
        all_tasks = json.load(f)

    if args.task:
        all_tasks = [t for t in all_tasks if t["id"] == args.task]
        if not all_tasks:
            print(f"Unknown task: {args.task}", file=sys.stderr)
            sys.exit(1)

    models = [args.model] if args.model else available_models()
    fault_filter = args.fault

    # Start mock server
    server = MockServer(port=args.port)
    if not args.dry_run:
        server.start()
        time.sleep(0.5)  # let Flask bind
        print(f"Mock server running at {server.base_url}")

    monitor = None
    if args.monitor and not args.dry_run:
        from agent.monitor import MonitorAgent
        monitor = MonitorAgent(model_name=args.monitor_model,
                               use_drift=args.monitor_drift)

    seeds = list(range(1, args.seeds + 1))
    total = skipped = done = failed = 0

    for task_meta in all_tasks:
        task_id = task_meta["id"]
        primary_tool = task_meta.get("primary_tool", "")
        faults = [fault_filter] if fault_filter else compatible_faults(primary_tool)

        try:
            task_spec = load_task_spec(task_id, envs_dir)
        except FileNotFoundError:
            print(f"[SKIP] No env spec for {task_id}")
            continue

        for model_key in models:
            for fault_type in faults:
                for seed in seeds:
                    total += 1
                    out_path = trace_path(out_dir, task_id, model_key, fault_type, seed)

                    if already_done(out_dir, task_id, model_key, fault_type, seed):
                        skipped += 1
                        continue

                    label = f"{task_id} / {model_key} / {fault_type} / seed{seed}"
                    if args.dry_run:
                        print(f"[DRY] {label}")
                        continue

                    print(f"[RUN] {label} ...", end=" ", flush=True)
                    t0 = time.time()
                    try:
                        trace = run_one(
                            task_meta, task_spec, model_key, fault_type,
                            server, out_dir, args.max_steps,
                            seed=seed, monitor=monitor,
                            allow_repair=args.allow_repair,
                            reason=not args.no_reasoning,
                            honesty=args.honesty_prompt,
                        )
                        save_trace(trace, out_path)
                        elapsed = time.time() - t0
                        steps = len(trace.get("steps", []))
                        status = "✓" if trace.get("finished") else "×"
                        print(f"{status} {steps} steps ({elapsed:.1f}s)")
                        done += 1
                    except Exception as ex:
                        print(f"ERROR: {ex}")
                        traceback.print_exc()
                        failed += 1

    print(f"\nDone. Total={total}, ran={done}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
