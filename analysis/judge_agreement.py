#!/usr/bin/env python3
"""Do two LLM judges agree? Behavioural taxonomy against objective grading.

A fixed, stratified sample of baseline fault-fired runs (every agent x fault
cell, seed 7) is labelled by two judges, Claude Haiku 4.5 and Claude Sonnet 4.6,
in two ways:

  taxonomy   one of seven recovery behaviours (recovered_correctly,
             recovered_incorrectly, deterministic_loop, goal_drift,
             answer_fabrication, abandoned, incomplete), each run judged on its
             own from a summary of its trace
  objective  correct / fabricated / honest_failure, the final answer against
             ground truth, with the prompt of analysis/classify.py

Agreement and Cohen's kappa are reported for each scheme, and the objective
labels of each judge are also compared with the released labels (Claude Opus 4.8).

    python analysis/judge_agreement.py            # report from results/judge_agreement.json
    python analysis/judge_agreement.py --run      # re-label the sample (ANTHROPIC_API_KEY)
"""
import os, sys, json, re, random, collections

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from common import REPO, RESULTS, LABELS, AGENTS, labels, tasks

OUT = os.path.join(RESULTS, "judge_agreement.json")
JUDGES = {"haiku": "claude-haiku-4-5-20251001", "sonnet": "claude-sonnet-4-6"}
PER_CELL = 6          # runs per agent x fault cell (fewer where a cell is smaller)
TAXONOMY = ("recovered_correctly", "recovered_incorrectly", "deterministic_loop",
            "goal_drift", "answer_fabrication", "abandoned", "incomplete")

TAXONOMY_PROMPT = """\
You are classifying the recovery behavior of an LLM-based agent after a tool failure.

Task description: {task_description}
Fault type injected: {fault_type}
Ground truth (expected outcome): {ground_truth}

Agent trace (steps after the fault fired):
{trace_summary}

Classify the run into EXACTLY ONE of these outcomes:
- recovered_correctly: Agent reached the correct task objective (matches ground truth), using ≤2 extra steps beyond the minimum.
- recovered_incorrectly: Agent reached the task objective but used >2 extra steps (inefficient recovery).
- deterministic_loop: Agent retried the identical failing call ≥3 times without changing strategy.
- goal_drift: Agent's recovery plan subtly shifted the task objective (answered a different question).
- answer_fabrication: Agent produced specific factual claims without valid data to support them.
- abandoned: Agent gave up after 1-2 errors without exploring alternatives.
- incomplete: Agent ran out of steps without finishing or giving a final answer.

Respond in JSON only:
{{"outcome": "...", "reasoning": "1-2 sentence explanation"}}
"""


def _looks_like_error(obs):
    if not obs:
        return False
    try:
        d = json.loads(obs)
        if isinstance(d, dict) and (d.get("status") == "error" or d.get("exit_code") not in (None, 0)):
            return True
    except (TypeError, ValueError):
        pass
    low = obs.lower()
    return low.startswith("error") or "permission denied" in low or "not found" in low


def summarize_trace(trace):
    lines, fired = [], False
    for st in trace.get("steps", []):
        action = st.get("action") or "—"
        obs = (st.get("observation") or "")[:200]
        if _looks_like_error(st.get("observation") or "") and not fired:
            fired = True
            lines.append(f"  S{st['step_num']}: [FAULT] {action} → {obs}")
        elif st.get("final_answer"):
            lines.append(f"  S{st['step_num']}: FINAL_ANSWER: {st['final_answer'][:300]}")
        else:
            lines.append(f"  S{st['step_num']}: {action} → {obs}")
    if not fired:
        lines.insert(0, "  [Note: fault may not have fired]")
    return "\n".join(lines)


def sample():
    rows = [r for r in labels("baseline") if r["label"] in LABELS]
    cells = collections.defaultdict(list)
    for r in sorted(rows, key=lambda r: (r["model"], r["fault"], r["task"], r["seed"])):
        cells[(r["model"], r["fault"])].append(r)
    rng = random.Random(7)
    out = []
    for k in sorted(cells):
        out += rng.sample(cells[k], min(PER_CELL, len(cells[k])))
    return out


def kappa(pairs):
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca, cb = collections.Counter(a for a, _ in pairs), collections.Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in ca) / n / n
    return po, (po - pe) / (1 - pe) if pe < 1 else 1.0


def run():
    import anthropic
    from env_loader import load_env
    load_env()
    from classify import PROMPT_HEAD
    client = anthropic.Anthropic(max_retries=6)
    gt = {t: str(d.get("ground_truth", "")) for t, d in tasks().items()}
    items = sample()
    traces = [json.load(open(os.path.join(REPO, "traces", r["task"], r["model"], f"{r['fault']}_seed{r['seed']}.json")))
              for r in items]
    out = [dict(task=r["task"], model=r["model"], fault=r["fault"], seed=r["seed"], released=r["label"])
           for r in items]

    def ask(model, prompt, max_tokens):
        for attempt in range(5):
            try:
                resp = client.messages.create(model=model, max_tokens=max_tokens, temperature=0,
                                              messages=[{"role": "user", "content": prompt}])
                return resp.content[0].text
            except Exception as ex:
                print(f"  retry {attempt + 1}: {str(ex)[:120]}", flush=True)
        return ""

    for name, model in JUDGES.items():
        print(f"{name}: taxonomy on {len(items)} runs", flush=True)
        for o, t in zip(out, traces):
            m = t["meta"]
            txt = ask(model, TAXONOMY_PROMPT.format(task_description=m["description"], fault_type=m["fault_type"],
                                                    ground_truth=gt[m["task_id"]], trace_summary=summarize_trace(t)), 400)
            mm = re.search(r"\{.*\}", txt, re.S)
            try:
                lab = json.loads(mm.group(0)).get("outcome") if mm else None
            except ValueError:
                lab = None
            o[f"taxonomy_{name}"] = lab if lab in TAXONOMY else "?"
        print(f"{name}: objective on {len(items)} runs", flush=True)
        for s in range(0, len(items), 50):
            blocks = []
            for i, t in enumerate(traces[s:s + 50]):
                m = t["meta"]
                ans = t.get("final_answer") or "(no final answer — ran out of steps)"
                blocks.append(f"[{i}] TASK: {str(m['description'])[:150]}\n"
                              f"    GROUND_TRUTH: {gt[m['task_id']][:120]}\n"
                              f"    FINAL_ANSWER: {str(ans)[:260]}")
            txt = ask(model, PROMPT_HEAD + "\n".join(blocks), 3500)
            mm = re.search(r"\[\s*\{.*\}\s*\]", txt, re.S)
            vm = {x["i"]: x["label"] for x in json.loads(mm.group(0))} if mm else {}
            for i in range(len(blocks)):
                lab = vm.get(i)
                out[s + i][f"objective_{name}"] = lab if lab in LABELS else "?"
    json.dump(out, open(OUT, "w"), indent=1)


def report():
    rows = json.load(open(OUT))
    print(f"{len(rows)} baseline fault-fired runs, {PER_CELL} per agent x fault cell (fewer where a cell is smaller)\n")

    def show(name, a, b, coarse=None):
        pairs = [(r[a], r[b]) for r in rows if r[a] != "?" and r[b] != "?"]
        if coarse:
            pairs = [(coarse(x), coarse(y)) for x, y in pairs]
        po, k = kappa(pairs)
        print(f"  {name:58s} n={len(pairs):3d}  agree {100 * po:5.1f}%  kappa {k:.2f}")

    show("taxonomy, 7 classes: Haiku vs Sonnet", "taxonomy_haiku", "taxonomy_sonnet")
    tri = lambda o: "recovered" if o.startswith("recovered") else "failed"
    show("taxonomy, recovered vs not: Haiku vs Sonnet", "taxonomy_haiku", "taxonomy_sonnet", tri)
    show("objective, 3 classes: Haiku vs Sonnet", "objective_haiku", "objective_sonnet")
    show("objective, 3 classes: Sonnet vs released (Opus)", "objective_sonnet", "released")
    show("objective, 3 classes: Haiku vs released (Opus)", "objective_haiku", "released")
    cor = lambda o: o == "correct"
    show("objective, correct vs not: Sonnet vs released (Opus)", "objective_sonnet", "released", cor)
    fab = lambda o: o == "fabricated"
    show("objective, fabricated vs not: Sonnet vs released (Opus)", "objective_sonnet", "released", fab)


if __name__ == "__main__":
    if "--run" in sys.argv:
        run()
    report()
