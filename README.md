# MathErrataBench

MathErrataBench measures how well AI systems find substantive mathematical
errors in research papers. An agent reads the full LaTeX source of an arXiv
paper and lists the non-typo issues it finds, giving each a severity, a target
type, and a location. A judging model then checks which annotated (gold) issues
the agent recovered.

This repository accompanies our [MathErrataBench blog post](https://ifm.ai/blog/matherratabench/).
Browse the 200 papers in **[PAPERS.md](paper-list-info/PAPERS.md)**, or filter
**[papers.csv](paper-list-info/papers.csv)** in a spreadsheet. Both list titles,
authors, input versions, arXiv links, and issue counts.

## Overview

| Step | Folder | What happens | Output |
|---|---|---|---|
| 1. Prepare | [`data-preparation/`](data-preparation/) | Download the exact arXiv sources and pair them with the gold annotations. | `generated/gold/` |
| 2. Run and parse | [`run-agent/`](run-agent/) | Give each prompt to your agent, save its raw answer, and parse it into predictions. | `generated/parsed/<run>/` |
| 3. Evaluate | [`evaluation/`](evaluation/) | A judging model matches predictions to gold issues and writes one report per paper. | `generated/evaluation/<run>/` |
| 4. Score | [`scoring/`](scoring/) | Compute issue-level and paper-level metrics from the reports. | `generated/scores/` |

Step 1 alone reconstructs the benchmark. The [`blog-results/`](blog-results/)
folder holds the saved answers and judgments behind the blog post. See
[Reproducing the blog post](#reproducing-the-blog-post). The scripts create
`generated/` on first use.

The examples below prepare all 200 papers, then run one paper (`1803.09446`)
under the run name `first-paper`. A one-paper run tests the pipeline; it is not
a benchmark result. To run every prepared paper, omit `--paper-id` in step 2.

## Before you start

Use Python 3.10 or later and run all commands from the repository root. Paper
preparation needs only Python; parsing and the example runner also need
`jsonschema`:

```bash
python3 -m pip install "jsonschema>=4.18,<5"
```

You bring your own agent. Judging (step 3) uses an authenticated Codex CLI by
default, plus an `OPENAI_API_KEY` for escalation attempts. The optional example
runner needs a Chat Completions-compatible endpoint. Parsing and scoring make
no model calls.

`prepare.py paper` accepts versioned IDs such as `2502.02843v1` and uses the
latest version when you omit the suffix. The `--paper-id` options in steps 2
and 4 and the raw-answer and prediction filenames use bare IDs such as
`2502.02843`. Preparation records the version in its metadata.

## The dataset

The benchmark contains **200 papers and 1,038 annotated issues, 717 of them
non-typo**. We identified issues through arXiv revision comparisons,
model-assisted review and adjudication, and targeted expert input. See
[Reporting gold errors](#reporting-gold-errors) for the limits of this process
and how to suggest a correction.

### Severity

Severity reflects the least severe *supported* complete repair of an issue and
the claims it affects, weighing both the work required and what is lost. A
repair is supported if a defensible argument from the paper's mathematics
justifies it; a speculative possibility is not enough. Later revisions and
expert analysis helped us find repairs, but a later edit alone does not
determine severity. Severity is an annotation judgment, not a machine-checked
certificate.

| Label | Meaning |
|---|---|
| `typo` | A reader with the mathematical background expected for the paper can identify a uniquely forced correction from the source and continue without repairing any mathematical inference, argument, or construction. The correction does not withdraw a substantive mathematical promise. |
| `minor_issue` | A genuine mathematical defect with a short, routine local repair or bypass that resolves the issue and its consequences while preserving the substantive claims and guarantees, with no meaningful mathematical loss. |
| `moderate_issue` | A supported complete repair goes beyond a routine local fix but avoids both global proof reconstruction and major mathematical loss. It may require a contained nontrivial argument or a limited weakening that leaves recognizably comparable claims and guarantees. |
| `serious_issue` | No supported complete repair avoids both essentially new proof architecture and major mathematical loss. This includes cases where no complete repair is supported. |

A one-symbol correction is not automatically a typo. Severity describes the
issue and its affected claims, not the whole paper, and does not depend on
whether the affected statement is a main theorem.

### Target types and primary tags

Each issue records the kind of claim it affects (its **target type**). The
**primary tag** is the LaTeX label of the claim or procedure that owns the
defect, when available. Prose claims can have no primary tag.

| Target type | Meaning |
|---|---|
| `formal_result` | A theorem, lemma, proposition, or other formal claim whose statement or proof fails. |
| `algorithm` | A procedure or construction whose operation or guarantee fails. |
| `prose_claim` | An independently unsupported or misleading mathematical claim in the text. |
| `other` | A labelled standalone mathematical assertion that fits none of the above. |

Tags are the literal LaTeX labels in the prepared source, including labels added
during preparation, so tag names differ from paper to paper. Evidence locations
and downstream consequences are recorded in separate fields.

### What the scores mean

The gold set is **nonexhaustive and may contain errors**: a finding missing from
gold is not necessarily wrong, and a claim without a gold issue is not
certified correct. We therefore measure issue discovery by **recall, not
precision**, and never penalize extra predictions. The other metrics check
labels on recovered issues, or require recovery and correct labels together.
The task asks for concrete mathematical evidence and sets no cap on the number
of predictions. Report prediction counts and generation budgets alongside
scores.

## 1. Prepare the papers

| File in `data-preparation/` | Role |
|---|---|
| [prepare.py](data-preparation/prepare.py) | Downloads each arXiv archive and combines and normalizes the LaTeX, adding labels where needed. For gold inputs, it checks the result against the manifest. |
| [gold-manifest.json](data-preparation/gold-manifest.json) | Pins the exact paper versions, preparation settings, and checksums. |
| [annotations/](data-preparation/annotations/) | Gold issues, one `<paper_id>.json` file per paper. |
| [vendor/](data-preparation/vendor/) | Source-preparation code from the benchmark project. It combines LaTeX files, identifies claims and algorithms, and normalizes labels. |

```bash
# Reconstruct all gold inputs and pair them with their annotations.
python3 data-preparation/prepare.py gold

# Prepare one paper, several papers, or a file with one ID per line.
python3 data-preparation/prepare.py paper 2502.02843v1
python3 data-preparation/prepare.py paper 2012.03503v4 2502.02843v1
python3 data-preparation/prepare.py paper --ids-file papers.txt
```

Gold uses v1 for 199 papers and v4 for `2012.03503`, which entered the
benchmark through an expert-guided review of versions v4–v6; its nine issues
concern v4, with v6 providing revision context. Gold papers prepared at their
gold versions match the benchmark sources byte-for-byte. You can prepare and
run other papers or versions, but gold evaluation covers only the pinned
inputs.

`prepare.py gold` writes `generated/gold/papers/<paper_id>/` (`source.tex` and
`issues.json`) and records its settings in a run manifest. To continue an
interrupted build, rerun it with `--resume`; changed settings or damaged
outputs are rejected. Use `--output` to start a fresh build. `prepare.py paper`
writes `generated/papers/<paper_id>/vN/source.tex` and skips papers that are
already prepared and verified, so you can add IDs to a list without redoing
earlier ones.

## 2. Run your agent

| File in `run-agent/` | Role |
|---|---|
| [question.txt](run-agent/question.txt) | Instructions for the agent, including the answer format. |
| [output_schema.json](run-agent/output_schema.json) | Defines the answer's required fields and allowed values. |
| [parse.py](run-agent/parse.py) | `prepare` builds prompts from prepared sources; `parse` turns raw answers into predictions. Neither calls a model or reads gold. |
| [optional_runner.py](run-agent/optional_runner.py) | Optional example that sends prompts to a Chat Completions-compatible API. |

### Run conditions

To compare with the blog post's source-only results, give the agent nothing but
its prompt: no web browsing, search, or external retrieval, and no access to
gold annotations, later revisions, or benchmark answers. The parser sees only
final answers, so enforce these conditions in your agent setup. Report runs
with web access, or with access to gold or later revisions, as a separate
condition.

A run has one answer per paper. Keep repeated runs separate; choosing the best
of several attempts per paper is a different protocol and must be disclosed.
Report the model and version, sampling settings, token budget, tools, number of
repeated runs, and prediction counts.

### Prepare the prompts

```bash
python3 run-agent/parse.py prepare \
  --papers generated/gold --paper-id 1803.09446 \
  --run-name first-paper --output generated/agent-inputs/first-paper
```

Each prompt combines `question.txt` with the full paper source and the paper's
identifiers. Repeat `--paper-id` for several papers, or omit it to include every
paper under `--papers`. `--papers` can also point to `generated/papers` or to a
single prepared paper, but a run may contain only one version of each paper.

### Run the agent and save its answer

```bash
mkdir -p generated/raw-answers/first-paper
```

Give `generated/agent-inputs/first-paper/prompts/1803.09446.txt` to your agent
in whatever way suits it (paste it, use a CLI, or call it from code). Save the
raw answer, Markdown fences included, as
`generated/raw-answers/first-paper/1803.09446.txt`, even if it is broken or
incomplete; the parser can often recover it.

The answer is a single JSON object defined by `output_schema.json`. A
completed review with no findings has `status: "ok"`, `issues: []`,
`issues_found: 0`, and `error: null`; never report a failed or incomplete
review this way.

Agents are told not to report pure typos, and `typo` is not an allowed
classification, so the parser may drop entries classified as typos and keep the
rest. A finding that merely mentions the word "typo" is judged on its actual
diagnosis.

<details>
<summary>Optional: use the example API runner</summary>

[optional_runner.py](run-agent/optional_runner.py) sends each prompt as one user
message to a Chat Completions-compatible API
([reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)).
It makes one attempt per paper, uses no tools, and does not mimic any
particular agent system. Use an endpoint whose context window fits a whole
paper.

Put the key in `MODEL_API_KEY` (or name another variable with `--api-key-env`);
unauthenticated local endpoints need none. Start with a dry run, which sends no
requests and writes no files:

```bash
export MODEL_BASE_URL="https://api.openai.com/v1"   # the runner appends /chat/completions
export MODEL_ID="<model name>"

python3 run-agent/optional_runner.py \
  --inputs generated/agent-inputs/first-paper \
  --base-url "$MODEL_BASE_URL" --model "$MODEL_ID" \
  --output generated/example-runs/first-paper --dry-run
```

Remove `--dry-run` to call the model; provider charges may apply. The runner
uses the endpoint's default sampling and token limits unless you pass
`--max-output-tokens` (sent as `max_completion_tokens`).

Use a new `--output` directory for each run. It receives raw answers (`raw/`),
API responses (`responses/`), settings (`run.json`), and strict schema-check
results (`validation.json`); answers that pass the strict check are also copied
to `answers/`. Truncated responses are recorded, and API keys are never saved.
To parse the run, set the parser's `--raw` to
`generated/example-runs/first-paper/raw`.

</details>

### Parse the answers

```bash
python3 run-agent/parse.py parse \
  --inputs generated/agent-inputs/first-paper \
  --raw generated/raw-answers/first-paper \
  --output generated/parsed/first-paper
```

The parser reads one `<paper_id>.txt` or `<paper_id>.json` file per paper,
verifies the prepared inputs, and sets each answer's identifiers from
`inputs.json`, recording any replacements. It then applies the parsing rules we
used for the blog post: it repairs formatting problems, marks omitted fields as
`NA`, and tolerates a narrow set of schema warnings. If the example runner
recorded a truncated response but the visible text is a complete, scoreable
answer, the answer is kept and the truncation is noted in the audit. Otherwise
the parser keeps only the complete issue objects that pass the scoreability
rules and discards the rest, including any unfinished tail. It never invents
mathematical content or turns a failed answer into an empty review.

The output folder holds `parsed_predictions/`, unchanged raw copies, a
per-paper `parse_audit/`, and `parse_summary.json`. Each kept answer has the
audit status `valid`, `scoreable_schema_warning`, or
`scoreable_partial_issue_salvage`. All three are evaluated, and `NA` labels earn
no label credit. Gold issues missed by a partial answer count as ordinary
misses. Missing answers, and answers with nothing scoreable, score zero
recovery. If any paper is unscoreable, the command exits with a nonzero status
but still saves everything else. Existing output folders are never overwritten.

<details>
<summary>What the parser repairs</summary>

The parser takes the first complete answer object (recording any others in the
audit), strips Markdown wrappers, and fixes formatting problems such as invalid
JSON backslashes and trailing commas. It then sets identifiers from the
prepared input, recalculates the issue count, numbers issues in order,
deduplicates dependent tags, and sets failure mechanisms for non-formal issues
to `not_applicable`. Unlike the runner's strict check against
`output_schema.json`, it accepts `NA` labels, empty prose evidence, and missing
primary tags. Every change is audited, no model is involved, and the raw answer
is preserved.

</details>

## 3. Evaluate predictions

You run only `evaluate.py`; it loads the other files itself.

| File in `evaluation/` | Role |
|---|---|
| [evaluate.py](evaluation/evaluate.py) | Runs the judge and writes one report per paper for the scorers. |
| [judge_prompt.txt](evaluation/judge_prompt.txt) | Tells the judge what counts as a match between a prediction and a gold issue. |
| [judge_schema.json](evaluation/judge_schema.json) | Defines the fields and allowed values in the judge's response. |
| [helpers/judge_contract.py](evaluation/helpers/judge_contract.py) | Builds the evidence sent to the judge and checks that its response covers every eligible gold issue and refers to valid predictions. |
| [helpers/run_judge_luna.py](evaluation/helpers/run_judge_luna.py) | Sends judge requests through the Codex CLI or the Responses API and saves accepted judgments. |
| [helpers/escalation.py](evaluation/helpers/escalation.py) | Runs the extra attempts when the first round gives no valid judgment. |
| [helpers/score_q1.py](evaluation/helpers/score_q1.py) | Loads and verifies gold and prediction files and normalizes tags. Also used by `score_tags.py`. |

The judge sees structured evidence for the gold and predicted issues
(quotations, explanations, effects, proposed repairs, locations, and candidate
rankings). It receives these issue records without the paper source, severity
and target-type fields, or severity rationales. Evaluation therefore measures whether the agent recovered
the gold diagnoses, not whether its findings are independently correct.

Start with a dry run, which checks the inputs and judge settings without
calling a model or writing output:

```bash
python3 evaluation/evaluate.py \
  --predictions generated/parsed/first-paper \
  --gold generated/gold \
  --output generated/evaluation/first-paper --dry-run
```

Remove `--dry-run` to run the judge. First-stage attempts go through the Codex
CLI by default, or through the OpenAI Responses API with `--judge-runner api`.
Escalation attempts (stages 2 and 3 below) always use the Responses API, so set
`OPENAI_API_KEY` either way. API output is capped at 16,384 tokens by default.
You need access to the selected models, and charges may apply.

Only papers in the step 2 run are evaluated, including those whose answers
failed parsing, and their versions and source checksums must match gold.
Papers with an empty completed review or no scoreable answer need no judge
call: the first count as known misses, the second as zero recovery.

### Judge configuration

The default judge is `gpt-5.6-luna` at `medium` reasoning effort, as in the
blog post. If an attempt gives no valid decision, judging escalates:

1. Up to three attempts with `gpt-5.6-luna` (`--retries`, default 2 retries after the first attempt).
2. One extra `gpt-5.6-luna` attempt (`--extra-luna-attempts`).
3. Up to three attempts with `gpt-6.1-sol` (`--fallback-model`, `--fallback-attempts`).

The first complete, valid judgment for a paper, including one with no matches,
ends the sequence; `--no-escalation` stops after stage 1. The evaluation index
and score files count accepted judgments, unresolved judgments, and papers that
needed no judgment, across the selected papers before gold-issue filters.

The whole sequence is one judge policy. Changing the model, reasoning effort,
or any of these settings changes the measurement, so record your configuration
and report such scores separately from the blog post's results.

### Output

```text
generated/evaluation/first-paper/
    evaluation_manifest.json     # Frozen inputs, versions, configuration, code hashes
    judge-inputs/                # Evidence prepared for the judge
    judge-run/                   # Judge decisions and attempt records for resuming
    reports/0001/
        index.json               # Evaluated paper list and per-report checksums
        papers/<paper_id>.json   # One report per paper, one entry per gold issue
```

Each paper report lists the gold and predicted issues, matched prediction IDs,
judge explanations, and label comparisons (correct, incorrect, or not
assessed). Each gold issue is `matched`, `unmatched`, `unresolved`, or
`not_evaluated` (gold typos, which the task excludes). An issue is `unresolved`
when its prediction or a required judge decision is unavailable; the `reason`
field says which. Parsing failures are marked explicitly, never shown as empty
reviews.

Normally one prediction must cover a gold issue on its own. Two or three
predictions may match jointly only if the gold issue bundles distinct defects
that can each be diagnosed separately, and each prediction diagnoses its part.
The judge then picks the most direct one as the **focus** prediction, without
seeing severity labels; the focus supplies the target type and primary tag
compared with gold. The compared severity is the highest among the predictions
needed for the match, or `NA` if any of them lacks a recognized severity. Each
supporting prediction also gets its own diagnostic comparison, but scoring
never picks whichever prediction's labels agree best with gold.

### Resuming and unresolved issues

To continue an interrupted evaluation, rerun it with `--resume` and the same
paths, inputs, and configuration. Validated decisions are reused, and the code
must match the saved manifest. A resume may change `--jobs`,
`--timeout-seconds` and `--codex-bin`, raise `--retries`, or use an updated
Codex CLI. The saved manifests keep the original values, and
`judge-run/invocations/` records the settings of each run. Ctrl-C stops Codex
attempts at once; with the API runner it waits for requests already sent to
return or time out. Either way, an interrupted attempt counts toward its
paper's attempt budget. Each completed run writes a new report
generation (`reports/0001`, `0002`, …), leaving earlier ones intact.

If any eligible gold issue remains unresolved because a prediction or required
judgment is unavailable, the command saves the reports and exits with a nonzero
status. For judge failures, check the records in `judge-run/`. Resuming uses the
remaining attempt budget; to give papers whose attempts ran out more, resume
with a larger `--retries`. Like any change to the judge settings, this changes
the measurement for those papers. Use
`--unresolved-policy bounds` (step 4) to report a range when judge decisions are
missing. Missing predictions already count as zero recovery under the default
scoring policy.

### Time and cost

A full 200-paper run needs at most 200 initial judge calls (none for papers
without scoreable findings), and a hard case can take up to seven attempts.
Cost and runtime depend on token counts, which models end up deciding, your
provider's rates, and parallelism (`--jobs`, default 1). Judge a few
representative papers first to budget a full run.

## 4. Calculate scores

| File in `scoring/` | Role |
|---|---|
| [score_issues.py](scoring/score_issues.py) | Calculates the five issue-level metrics from the evaluation reports. |
| [score_papers.py](scoring/score_papers.py) | Calculates the fraction of papers in which every eligible gold issue was recovered. |
| [score_tags.py](scoring/score_tags.py) | Optional. Measures primary-tag coverage directly from parsed predictions and gold, without the judge. |
| [metrics.py](scoring/metrics.py) | Shared scoring functions used by the issue scorer, the paper scorer, and the reproduction script. |
| [reproduce_blog_results.py](scoring/reproduce_blog_results.py) | Recomputes the blog post's results from [`blog-results/`](blog-results/). |
| [results.md](scoring/results.md) | The blog post's score table, metric definitions, and answer and judgment counts. |

The scorers read the saved reports and make no model calls, so you can change
options and rescore without judging again.

```bash
# The five issue-level metrics.
python3 scoring/score_issues.py \
  --evaluations generated/evaluation/first-paper \
  --output generated/scores/first-paper-issues.json

# The fraction of papers in which every eligible issue was recovered.
python3 scoring/score_papers.py \
  --evaluations generated/evaluation/first-paper \
  --output generated/scores/first-paper-papers.json
```

Each command verifies the report checksums and writes a new JSON file with the
metrics, counts, denominators, per-issue or per-paper details, settings, and the
report generation it read. Existing score files are never overwritten, so
choose a new filename when you change options.

### Options

Filters select **gold issues**; the non-typo gold issues that pass them are
*eligible*.

| Option | Meaning |
|---|---|
| `--evaluations PATH` | Required. The step 3 output folder. The latest report generation is used unless you point to one, e.g. `generated/evaluation/first-paper/reports/0001` or its `index.json`. |
| `--output FILE` | Required. A new JSON filename. |
| `--paper-id ID` | Score only this paper (bare ID). Repeatable. Default: all evaluated papers. |
| `--target-type TYPE` | Keep gold issues of type `formal_result`, `algorithm`, `prose_claim`, or `other`. Repeatable. Default: all. |
| `--gold-severity LEVEL` | Keep gold issues graded `minor_issue`, `moderate_issue`, or `serious_issue`. Repeatable. Default: all three. |
| `--unresolved-policy POLICY` | `error` (default) refuses an incomplete evaluation; `bounds` reports lower and upper bounds; `exclude` drops unresolved issues from the denominator, which changes the reporting scope. |
| `--formal-weights MINOR MODERATE SERIOUS` | `score_issues.py` only. Weights for strict formal-issue recovery. Default: `0.5 1 1`. |
| `--zero-eligible POLICY` | `score_papers.py` only. `exclude` (default) skips papers with no eligible issues; `count-as-satisfied` counts them as successes. |

For example, to score serious issues only:

```bash
python3 scoring/score_issues.py \
  --evaluations generated/evaluation/first-paper \
  --gold-severity serious_issue \
  --output generated/scores/first-paper-serious-issues.json
```

With the same filter, `score_papers.py` counts a paper as a success only if all
its serious gold issues were recovered, and skips papers with none by default.
Run either script with `--help` for all options.

### Metrics

`score_issues.py` reports five metrics on a 0–1 scale, with their counts,
denominators, and weights:

| Metric | Definition |
|---|---|
| Issue recovery | Matched eligible gold issues divided by all eligible gold issues. |
| Serious recovery | Matched serious gold issues divided by all eligible serious gold issues. |
| Severity score | Among matched issues, average credit of 1 for exact severity, 0.5 for an adjacent level, and 0 for a two-level error or ungraded severity. |
| Strict issue recovery | Match plus exact severity and target type; formal results also need the exact, nonmissing primary tag. All eligible gold issues stay in the denominator. |
| Strict formal-issue recovery | Among formal-result gold issues, match plus exact severity and nonmissing primary tag, weighted minor 0.5, moderate 1, serious 1. The predicted target type is not checked. |

On the full gold set, the denominators are 717 non-typo issues, 47 serious
issues, and 476 formal issues worth 347 weighted points. Subsets use their own
denominators.

`score_papers.py` reports **complete recovery**: the number of papers in which
every eligible gold issue was recovered, divided by the number of eligible
papers. It uses the same match decisions with no extra criteria. It is not an
average of per-paper recovery fractions, though those are included as
diagnostics.

The two kinds of failure are handled differently:

| What happened | Scoring behavior |
|---|---|
| The agent gave no answer, or its answer could not be parsed. | Zero recovery. Its eligible gold issues stay in the denominator, as in the blog post. |
| The judge never reached a valid decision. | Unknown, handled by `--unresolved-policy`. Not counted as a miss. |

Under `--unresolved-policy bounds`, a metric's `value` is `null` when
unresolved judgments leave it undetermined, and `lower_bound` and `upper_bound`
give the range. A metric with a zero denominator also has `value: null`. A
known miss makes a paper's complete recovery false even if another of its
issues is unresolved.

### Optional: primary-tag coverage

`score_tags.py` works directly from parsed predictions and gold, so it does not
need step 3:

```bash
python3 scoring/score_tags.py \
  --predictions generated/parsed/first-paper \
  --gold generated/gold \
  --output generated/scores/first-paper-tags.json
```

It reports the fraction of distinct primary tags on non-typo formal-result gold
issues that the agent also names as a primary tag for the same paper. Gold
issues that share a tag count once. Surrounding whitespace is ignored, but
aliases and spelling variants are not resolved. Predicted severity and target
type don't affect the result, extra tags are not penalized, and missing or
failed answers cover no tags. The output lists covered and missed tags. Options
are `--paper-id` and `--gold-severity`, both repeatable. Because this measures
tag overlap, not whether the agent found the same defect, report it as
**primary-tag coverage**, separately from the five metrics.

## Reproducing the blog post

We produced the blog post's results with an earlier, private research
repository. This release includes the parsing rules, judge prompt, and judge
policy from a pinned revision. The [results page](scoring/results.md) lists
the five scores for each of the thirteen models, defines the metrics, and
explains the answer and judgment counts.

The answers and accepted judgments used for the post are in
[`blog-results/`](blog-results/), one folder per model run; its
[README](blog-results/README.md) describes the files. The parser in step 2
takes one supplied answer per paper. The original campaign selected those
answers using its attempt and eligibility records.

To recompute the scores, run this from the repository root:

```bash
python3 scoring/reproduce_blog_results.py
```

See the [blog-results README](blog-results/README.md) for what the script
checks and prints.

## Reporting gold errors

Despite careful checks informed by mathematical expertise, and grounding
reported issues in authors' publicly available arXiv revisions, our gold
annotations were ultimately prepared by models and may contain errors. We
offer this benchmark as a scalable resource for evaluating mathematical
review, with judgments that remain open to correction. Authors and experts in
the relevant topics are best positioned to assess these judgments. If you
believe an annotation is incorrect or missing important context, please
contact us at [Elizaveta.Rebrova@gmail.com](mailto:Elizaveta.Rebrova@gmail.com). We welcome community input to improve the
gold data.

## License

The code, gold annotations and saved results in this repository are released
under the Apache License, Version 2.0; see [LICENSE](LICENSE). The papers are
not included: `prepare.py` downloads their sources from arXiv. The annotations
and saved model answers quote short excerpts from the papers as evidence. Those
excerpts remain the work of the papers' authors and are not covered by the
Apache license; each paper's license is recorded as `licence_url` in
[gold-manifest.json](data-preparation/gold-manifest.json).

The model outputs in [`blog-results/`](blog-results/) were generated by
third-party models and are included for evaluation and reproducibility. They
may be subject to each provider's terms of use or model license, some of which
restrict using outputs to develop competing models.

## Citation

If you use MathErrataBench, please cite the blog post:

```bibtex
@article{rebrova2026matherratabench,
  author  = {Rebrova, Elizaveta and Sreenivasan, Kartik and Yurochkin, Mikhail},
  title   = {{MathErrataBench}: Can {LLMs} find the issues in numerical
             analysis preprints?},
  journal = {IFM Blog},
  year    = {2026},
  note    = {https://ifm.ai/blog/matherratabench/}
}
```
