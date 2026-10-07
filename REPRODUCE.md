# Reproducing the paper

Everything here runs offline from the committed labels and traces; no API key
is needed.

```bash
pip install -r requirements.txt
cd trace_archives && for f in *.tar.gz; do tar xzf "$f" -C ..; done && cd ..
python analysis/reproduce.py        # Tables 3–7 and every number in Section 6
python analysis/make_figures.py     # Figures 2 and 3 into figs/
python analysis/judge_agreement.py  # judge agreement, Section 5
```

The retry analysis (Table 5) and the run counts read the traces; everything else
reads only the labels in `results/`.

`judge_agreement.py` reports from `results/judge_agreement.json`: 216 baseline
runs, six per agent × fault cell, labelled by Claude Haiku 4.5 and Claude Sonnet
4.6 with the seven-class behavioural taxonomy and with the three-way objective
grading. `--run` re-labels the sample (needs `ANTHROPIC_API_KEY`).

## Data

| arm | labels | traces | runs |
|---|---|---|---|
| baseline | `baseline.json` | `traces/` | 2,460 (360 fault-free) |
| Monitor, full cascade S1–S4 | `monitor_full.json` | `traces_monitor_v4/` | 2,460 |
| Monitor, code-only S1–S3 | `monitor_code_only.json` | `traces_monitor_v5/` | 2,460 |
| no chain of thought | `no_cot.json` | `traces_noreason/` | 2,460 |
| honesty prompt | `honesty.json` | `traces_honesty/` | 2,460 |
| self-remediation allowed | `self_remediation.json` | `traces_repair/` | 360 |
| payload sanitization | `sanitization.json` | `traces_sanitize/` | 1,080 |

Four agents (Llama-3.1-8B, Llama-3.1-70B, DeepSeek-V4, Claude Haiku 4.5), 30
tasks, three seeds. The label files hold one row per fault-fired run: task,
model, fault, seed and label. Sanitization is analyzed on the 27 tasks whose
fault target is not a shell command.

## Rerunning from scratch

The commands for each arm, with the four agents `llama-8b`, `llama-70b`,
`deepseek-v4` and `claude-haiku`:

| arm | command |
|---|---|
| baseline | `python run.py --model <agent>` |
| Monitor, full cascade | `python run.py --model <agent> --monitor --out traces_monitor_v4` |
| Monitor, code-only | `python run.py --model <agent> --monitor --no-monitor-drift --out traces_monitor_v5` |
| no chain of thought | `python run.py --model <agent> --no-reasoning` |
| honesty prompt | `python run.py --model <agent> --honesty-prompt` |
| self-remediation allowed | `python run.py --model <agent> --allow-repair --fault permission_denied` |
| payload sanitization | `python run.py --model <agent> --fault san_partial --out traces_sanitize` (also `san_empty`, `san_denial`) |

Then grade each directory with `python analysis/classify.py <traces_dir> <out.json>`.
