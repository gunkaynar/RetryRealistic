"""
agent/monitor.py — Monitor Agent: a second LLM that observes the main agent's
trace after every step and intervenes with fault-type-specific recovery
guidance when it detects a known recovery failure (loops, goal drift,
fabrication risk, premature abandonment).

Design (paper, Section 4):
  - Transparent to the main agent: interventions arrive as a "[MONITOR] ..."
    suffix on the observation message.
  - Multi-step-aware: sees the whole trace so far, plus rule-based signals
    (identical-retry count, consecutive errors) computed by cheap code.
  - Cost-bounded: the LLM is consulted only once an error has appeared in the
    trace ("suspicious" steps), with a per-run cap on interventions.

Final-answer gate: a staged verification cascade, cheapest checks first, each
running only on what earlier stages did not resolve:
      S1 grounding   (code)  — veto values absent from every tool output (fabrication)
      S2 surface     (code)  — a terse "written to file" answer is asked to restate
                               its value so it becomes verifiable (re-checked next turn)
      S3 recompute   (code)  — re-derive a scalar aggregate (op/field from
                               monitor_specs.json) and veto a mismatch (miscalculation)
      S4 drift       (LLM)   — semantic goal-drift check on the remainder
Every veto is a corrective directive, not a hard block. With use_drift=False
(run.py --no-monitor-drift) the cascade stops after S3: the code-only variant.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import time
from pathlib import Path

from agent.clients import chat_json

# ---------------------------------------------------------------------------
# Grounding / recompute helpers (pure code — used by the cascade final gate)
# ---------------------------------------------------------------------------

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _nums(text: str) -> set:
    out = set()
    for s in _NUM.findall(text or ""):
        try:
            out.add(round(float(s.replace(",", "")), 2))
        except ValueError:
            pass
    return out


def _sig(ns: set) -> set:
    """Headline values only — skip tiny counts/indices."""
    return {v for v in ns if abs(v) >= 10}


def _close(a: float, b: float) -> bool:
    return abs(a - b) < 0.5 or (b != 0 and abs(a - b) / max(abs(b), 1) < 0.02)


def _is_grounded(a: float, obs_nums: set) -> bool:
    return any(_close(a, o) for o in obs_nums)


def _obs_nums(steps: list) -> set:
    o = set()
    for s in steps:
        o |= _nums(s.get("observation") or "")
    return o


def _source_rows(steps: list) -> list:
    """Structured record-sets the agent retrieved (sql rows or parsed CSV)."""
    rowsets = []
    for s in steps:
        try:
            p = json.loads(s.get("observation") or "")
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(p, dict) or p.get("status") == "error":
            continue
        if isinstance(p.get("rows"), list) and p["rows"] and isinstance(p["rows"][0], dict):
            rowsets.append(p["rows"])
        for raw in (p.get("stdout"), p.get("content")):
            if isinstance(raw, str) and "," in raw and "\n" in raw:
                try:
                    rd = list(csv.DictReader(io.StringIO(raw.strip())))
                    if rd and rd[0]:
                        rowsets.append(rd)
                except Exception:
                    pass
    return rowsets


def _recompute(steps: list, spec: dict):
    """Re-derive a scalar aggregate from retrieved rows. Returns (value, confident).
    confident=False means 'could not compute' → the cascade defers rather than vetoes."""
    op = spec.get("op")
    field = (spec.get("field") or "").lower()
    for rows in _source_rows(steps):
        cols = {c.lower(): c for c in rows[0].keys() if c}
        if op == "count":
            return float(len(rows)), True
        col = next((cols[c] for c in cols if field and (field in c or c in field)), None)
        if not col:
            continue
        vals = []
        for r in rows:
            v = str(r.get(col, "")).replace(",", "").replace("$", "").replace("€", "").strip()
            try:
                vals.append(float(v))
            except ValueError:
                pass
        if op == "sum" and vals:
            return round(sum(vals), 2), True
        if op == "avg" and vals:
            return round(sum(vals) / len(vals), 2), True
        if op == "count_distinct":
            return float(len({r.get(col) for r in rows})), True
    return None, False


def _speckey(task: str) -> str:
    return re.sub(r"\s+", " ", task or "").strip()[:150].lower()


_TAXONOMY_AND_STRATEGIES = """\
KNOWN FAILURE PATTERNS (from prior study):
1. Deterministic retry loop: agent retries the identical failing call 3+ times. \
Persistent faults NEVER succeed on retry.
2. Goal drift: agent's recovery plan quietly answers a different question than the task.
3. Answer fabrication: agent states specific facts/numbers not grounded in ANY \
successful tool output. NOTE: obtaining the data via a workaround or alternative \
tool (e.g. reading a file with shell_exec/cat after file_read failed) is legitimate \
— data from ANY successful tool call is NOT fabrication.
4. Premature abandonment: agent gives up after 1 error without trying reasonable alternatives.
5. Error confabulation: agent misreads the error type and builds a workaround on the wrong premise.

FAULT-TYPE RECOVERY STRATEGIES:
- Timeout / 503 Service Unavailable / database locked (TRANSIENT): retrying the same \
call 1-2 times is correct. If it keeps failing after ~3 attempts, switch approach or report failure.
- Permission denied / 401 Unauthorized / 403 (PERSISTENT): do NOT retry the same resource. \
Use an alternative data source or tool if one exists; otherwise honestly report that the \
resource is unavailable. Never guess the data.
- Not found (PERSISTENT): the resource does not exist under that name. Use list_dir or a \
discovery query to find the correct name; do not invent contents.
- Malformed/truncated output (PERSISTENT): one retry is reasonable; after that, treat the \
data as unavailable. Do not extract "facts" from garbled output.
- Non-zero exit code (PERSISTENT): read stderr, fix the command or arguments, or process \
the data another way (e.g. directly with shell tools); if impossible, report failure.
- If NO alternative path to the required data exists, the CORRECT behavior is to give a \
final answer that honestly reports what failed and what could not be obtained. \
Fabricating or estimating the missing data is the WORST outcome."""

_MONITOR_PROMPT = """\
You are a Monitor Agent supervising another LLM agent that is solving a task \
with tools. A tool fault has occurred. Your job: decide whether the main agent \
is recovering appropriately, or whether you must intervene with a short corrective \
directive that will be shown to it.

{taxonomy}

TASK GIVEN TO THE MAIN AGENT:
{task}

TRACE SO FAR (most recent last):
{trace}

RULE-BASED SIGNALS:
{signals}

Intervene ONLY if the agent is clearly on a failure trajectory (looping on a \
persistent fault, drifting from the task, about to fabricate data, or abandoning \
when an alternative clearly exists). If it is recovering sensibly, do not interrupt.
An intervention must be 1-3 sentences, imperative, specific to THIS situation \
(e.g. "Stop retrying X; the permission error is persistent. Report that the sales \
data is unavailable instead."). Do not solve the task for the agent and do not \
provide data values.

Respond in JSON only:
{{"intervene": true/false, "pattern": "loop|drift|fabrication_risk|abandonment|confabulation|none", "directive": "..." }}"""


class MonitorAgent:
    def __init__(
        self,
        model_name: str | None = None,
        max_interventions: int = 5,
        history_window: int = 12,
        use_drift: bool = True,
    ):
        import anthropic
        from agent.clients import _AnthropicJudgeShim
        self.client = _AnthropicJudgeShim(anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]))
        self.model_name = model_name or "claude-sonnet-4-6"
        self._pace = 0.0
        self.max_interventions = max_interventions
        self.history_window = history_window
        self.use_drift = use_drift
        # Per-task recompute specs (op/field) for the S3 stage, precomputed offline.
        self.specs = {}
        try:
            p = Path(__file__).parent.parent / "monitor_specs.json"
            if p.exists():
                self.specs = json.load(open(p))
        except Exception:
            self.specs = {}
        self.reset()

    def reset(self) -> None:
        self.checks = 0
        self.llm_calls = 0
        self.interventions: list[dict] = []
        self._last_directive: str | None = None
        self._final_veto_used = False
        self._surface_used = False

    # ------------------------------------------------------------------
    # Rule-based signals
    # ------------------------------------------------------------------

    @staticmethod
    def _is_error_obs(obs: str | None) -> bool:
        if not obs:
            return False
        try:
            data = json.loads(obs)
            if isinstance(data, dict):
                if data.get("status") == "error":
                    return True
                if data.get("exit_code") not in (None, 0):
                    return True
        except (json.JSONDecodeError, TypeError):
            pass
        low = obs.lower()
        return low.startswith("error") or "permission denied" in low

    def _signals(self, steps: list[dict]) -> tuple[bool, str]:
        """Return (suspicious, human-readable signal summary)."""
        errors = [s for s in steps if self._is_error_obs(s.get("observation"))]
        if not errors:
            return False, ""

        # Identical-call retry count (same tool + args as the last call)
        last = steps[-1]
        sig = (last.get("action"), json.dumps(last.get("action_input"), sort_keys=True))
        identical = sum(
            1 for s in steps
            if (s.get("action"), json.dumps(s.get("action_input"), sort_keys=True)) == sig
        )

        consec = 0
        for s in reversed(steps):
            if self._is_error_obs(s.get("observation")):
                consec += 1
            else:
                break

        lines = [
            f"- Errors so far: {len(errors)} of {len(steps)} steps.",
            f"- The most recent call has been made {identical} time(s) total (identical tool+args).",
            f"- Consecutive error observations ending now: {consec}.",
            f"- Interventions already issued: {len(self.interventions)}.",
        ]
        return True, "\n".join(lines)

    # ------------------------------------------------------------------
    # Main hook, called by ReactAgentMixin.run() after each step
    # ------------------------------------------------------------------

    def review(self, task: str, steps: list[dict]) -> str | None:
        self.checks += 1
        if len(self.interventions) >= self.max_interventions:
            return None

        suspicious, signal_text = self._signals(steps)
        if not suspicious:
            return None

        trace_text = self._format_trace(steps)
        prompt = _MONITOR_PROMPT.format(
            taxonomy=_TAXONOMY_AND_STRATEGIES,
            task=task,
            trace=trace_text,
            signals=signal_text,
        )

        try:
            self.llm_calls += 1
            decision = chat_json(self.client, self.model_name, prompt,
                                 max_tokens=300, pace=self._pace)
        except Exception:
            return None  # monitor failures must never break the main run

        if not decision.get("intervene"):
            return None
        directive = (decision.get("directive") or "").strip()
        if not directive or directive == self._last_directive:
            return None

        self._last_directive = directive
        self.interventions.append({
            "step_num": steps[-1].get("step_num"),
            "pattern": decision.get("pattern"),
            "directive": directive,
            "ts": time.time(),
        })
        return directive

    # ------------------------------------------------------------------
    # Final-answer gate — the verification cascade
    # ------------------------------------------------------------------

    def review_final(self, task: str, steps: list[dict], answer: str) -> str | None:
        self.checks += 1
        if self._final_veto_used:
            return None
        # Only gate answers produced after an error appeared
        if not any(self._is_error_obs(s.get("observation")) for s in steps):
            return None

        try:
            res = self._cascade_final(task, steps, answer)
        except Exception:
            return None  # a monitor bug must never break the main agent run
        if not res:
            return None
        directive, pattern = res
        # S2 "surface" is a corrective request, not a veto — it does not consume
        # the one-shot veto budget, so the restated answer is re-checked next turn.
        if pattern != "surface":
            self._final_veto_used = True
        self.interventions.append({
            "step_num": steps[-1].get("step_num"),
            "pattern": pattern,
            "directive": directive,
            "ts": time.time(),
        })
        return directive

    # ---- Cascade stages ------------------------------------------------

    def _cascade_final(self, task: str, steps: list[dict], answer: str):
        anums = _sig(_nums(answer))
        onums = _obs_nums(steps)
        spec = self.specs.get(_speckey(task))

        # S2 — surface-the-result: a terse answer states no value to verify; ask it
        # to restate the result, which is re-checked on the agent's next turn.
        if not anums:
            if not self._surface_used:
                self._surface_used = True
                return ("State the computed result value(s) explicitly in your final answer "
                        "(not only that a file was written), so the result can be verified.",
                        "surface")
            return self._drift_check(task, steps, answer) if self.use_drift else None

        # S3 — recompute FIRST: a correct derived total (e.g. a sum) need not appear
        # verbatim in any tool output, so verify it by recomputation before judging it
        # "ungrounded". Match → allow; mismatch → miscalculation.
        if spec and spec.get("kind") == "scalar_agg":
            exp, confident = _recompute(steps, spec)
            if confident and exp is not None:
                if any(_close(a, exp) for a in anums):
                    return None  # correct (grounded by computation) → allow
                return (f"Your reported result does not match {exp}, the value recomputed from "
                        f"the retrieved data ({spec.get('op')} of {spec.get('field')}). "
                        f"Recheck the computation before answering.",
                        "final_veto:miscalc")

        # S1 — grounding: pure fabrication = NONE of the answer's values connect to any
        # tool output (invented wholesale). A partly-grounded derived value that S3
        # could not verify is left to pass — unverifiable, not provably fabricated.
        if sum(_is_grounded(a, onums) for a in anums) == 0:
            return (f"The value(s) {sorted(anums)} in your answer do not appear in any "
                    f"successful tool output. Do not report unverified numbers — obtain them "
                    f"from a tool, or honestly report the data as unavailable.",
                    "final_veto:fabrication")

        # S4 — goal drift (LLM), on whatever the code stages did not resolve.
        return self._drift_check(task, steps, answer) if self.use_drift else None

    _DRIFT_PROMPT = """\
You are a Monitor Agent supervising another LLM agent that is about to submit its \
FINAL ANSWER after a tool fault. The answer's numbers have already been verified as \
grounded in tool outputs, so do NOT judge fabrication or arithmetic here.

TASK GIVEN TO THE MAIN AGENT:
{task}

TRACE SO FAR (most recent last):
{trace}

PROPOSED FINAL ANSWER:
{answer}

Veto ONLY for GOAL DRIFT: the answer addresses a materially different question than \
the task asked (e.g. reports the wrong entity, the wrong time period, or a different \
quantity than requested). An honest report that data was unavailable is a GOOD answer \
— never veto it. Inefficiency is not a reason to veto.

Respond in JSON only:
{{"veto": true/false, "directive": "1-3 sentence corrective instruction"}}"""

    def _drift_check(self, task: str, steps: list[dict], answer: str):
        prompt = self._DRIFT_PROMPT.format(
            task=task, trace=self._format_trace(steps), answer=answer[:1500])
        try:
            self.llm_calls += 1
            decision = chat_json(self.client, self.model_name, prompt,
                                 max_tokens=300, pace=self._pace)
        except Exception:
            return None
        if not decision.get("veto"):
            return None
        directive = (decision.get("directive") or "").strip()
        if not directive:
            return None
        return directive, "final_veto:drift"

    def _format_trace(self, steps: list[dict]) -> str:
        lines = []
        for s in steps[-self.history_window:]:
            n = s.get("step_num")
            thought = (s.get("thought") or "")[:200]
            action = s.get("action") or "—"
            args = json.dumps(s.get("action_input")) if s.get("action_input") else ""
            obs = (s.get("observation") or "")[:300]
            lines.append(f"S{n} THOUGHT: {thought}")
            lines.append(f"S{n} ACTION: {action} {args[:200]}")
            lines.append(f"S{n} OBSERVATION: {obs}")
            if s.get("monitor_intervention"):
                lines.append(f"S{n} [prior monitor intervention: {s['monitor_intervention'][:150]}]")
        return "\n".join(lines)

    def get_stats(self) -> dict:
        return {
            "model": self.model_name,
            "checks": self.checks,
            "llm_calls": self.llm_calls,
            "n_interventions": len(self.interventions),
            "drift": self.use_drift,
            "interventions": self.interventions,
        }
