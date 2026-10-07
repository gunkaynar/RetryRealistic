"""Shared paths and loaders for the analysis scripts."""
import os, json, glob

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "results")
LABELS = ("correct", "fabricated", "honest_failure")
AGENTS = ("llama-8b", "llama-70b", "deepseek-v4", "claude-haiku")
AGENT_NAME = {"llama-8b": "Llama-3.1-8B", "llama-70b": "Llama-3.1-70B",
              "deepseek-v4": "DeepSeek-V4", "claude-haiku": "Claude Haiku 4.5"}

# label file -> trace directory holding that arm's runs
ARMS = {
    "baseline":          "traces",
    "monitor_full":      "traces_monitor_v4",
    "monitor_code_only": "traces_monitor_v5",
    "no_cot":            "traces_noreason",
    "honesty":           "traces_honesty",
    "self_remediation":  "traces_repair",
    "sanitization":      "traces_sanitize",
}


def labels(arm):
    """Graded runs of one arm: dicts with task, model, fault, seed, label."""
    return json.load(open(os.path.join(RESULTS, f"{arm}.json")))


def tasks():
    """task_id -> parsed task file."""
    import yaml
    out = {}
    for fn in sorted(glob.glob(os.path.join(REPO, "tasks", "envs", "*.yaml"))):
        d = yaml.safe_load(open(fn))
        out[d["task_id"]] = d
    return out


def trace(arm, task, model, fault, seed):
    p = os.path.join(REPO, ARMS[arm], task, model, f"{fault}_seed{seed}.json")
    return json.load(open(p)) if os.path.exists(p) else None
