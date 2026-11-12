# Blog results

The saved answers and judge decisions behind the scores in the
[MathErrataBench blog post](https://ifm.ai/blog/matherratabench/), one folder
per model run. Scoring them makes no model calls.

## Reproduce the scores

From the repository root:

```bash
python3 scoring/reproduce_blog_results.py
```

The script checks every file against its recorded SHA-256, builds the same
paper reports as `evaluation/evaluate.py`, and scores them with
`scoring/metrics.py`. It prints the scores, and stops with an error if any of
them differs from those recorded in `blog_results_index.json`. Combining the
eleven runs, it reproduces the pooled recovery of 645 of 717 gold issues and
the 85 of 200 papers in which every model missed at least one issue. The
[results page](../scoring/results.md) lists each model's scores, defines the
metrics, and explains why some runs have fewer than 200 answers or judgments.

## Layout

```text
blog-results/
├── blog_results_index.json
└── <run-id>/
    ├── model_run_index.json
    ├── answers/<paper-id>.json
    └── judgments/<paper-id>.json
```

| File | What it contains |
|---|---|
| `<run-id>/answers/<paper-id>.json` | The model's parsed findings for one paper: description, location, proposed repair, severity, target type, and primary tag. 2,155 files across the 11 runs. |
| `<run-id>/judgments/<paper-id>.json` | The accepted judge response for one paper: for each eligible gold issue, whether it was recovered, which findings support the match, and the judge's reasoning. 2,115 files. |
| `<run-id>/model_run_index.json` | The run's settings and all 200 papers, with their arXiv versions and which answers and judgments exist. |
| `blog_results_index.json` | The 11 runs to score, the gold manifest's fingerprint, and the expected scores to compare against. |

A paper with no answer file counts as a review that found nothing. A paper
with an answer but no judgment had no findings to judge.

## Runs

| Folder | Model | Reasoning setting | Answers | Judgments |
|---|---|---|---:|---:|
| `astra-max-api-full200-2026-09-22` | GPT-6 Astra | max | 200 | 200 |
| `gpt-sol-medium-full200-2026-09-06` | GPT-5.6 Sol | medium | 200 | 200 |
| `claude-opus-5.5-max-task-budget-128k-full200-2026-10-01` | Claude Opus 5.5 | max | 199 | 196 |
| `claude-fable-5.1-medium-full200-2026-09-10` | Claude Fable 5.1 | medium | 199 | 199 |
| `glm-5.3-max-full200-2026-10-01` | GLM-5.3 | max | 196 | 196 |
| `kimi-k3-max-full200-2026-09-29` | Kimi-K3 | max | 199 | 197 |
| `mimo-v2.6-pro-rl-max-full200-2026-10-14` | MiMo-V2.6-Pro-RL | max | 198 | 194 |
| `k2-horizon-high-full200-2026-09-29` | K2-Horizon-375B-A23B | high | 195 | 195 |
| `deepseek-v4.1-flash-max-full200-2026-10-02` | DeepSeek-V4.1-Flash | max | 198 | 196 |
| `glm-5.3-flash-max-std-full200-2026-10-10` | GLM-5.3-Flash | max | 176 | 174 |
| `gemini-3.8-flash-medium-full200-2026-09-09` | Gemini-3.8-Flash | medium | 195 | 168 |
