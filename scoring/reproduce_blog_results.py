#!/usr/bin/env python3
"""Replay the blog results using frozen answers, judge decisions, and release gold."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import metrics

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'evaluation'))
import evaluate as evaluation


def read(path):
    return json.loads(path.read_bytes())


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def checked_json(path, sha256):
    raw = path.read_bytes()
    if digest(raw) != sha256:
        raise ValueError(f'Checksum mismatch: {path}')
    return json.loads(raw)


def load_gold(bundle):
    gold_root = ROOT / 'data-preparation'
    index = checked_json(gold_root / 'gold-manifest.json', bundle['gold_manifest_sha256'])
    papers = {}
    for row in index['papers']:
        pid = row['paper_id']
        issues = checked_json(gold_root / 'annotations' / f'{pid}.json', row['issues_sha256'])
        papers[pid] = (row, issues['issues'])
    return papers


def score_run(run_path, gold):
    run = read(run_path / 'model_run_index.json')
    papers = run['papers']
    if [row['paper_id'] for row in papers] != sorted(gold):
        raise ValueError(f'Unexpected paper cohort: {run_path}')
    reports = []
    for row in papers:
        pid = row['paper_id']
        authority, gold_issues = gold[pid]
        if row['source_sha256'] != authority['source_sha256']:
            raise ValueError(f'Wrong benchmark source: {run_path.name}/{pid}')
        answer_ref, judge_ref = row['answer'], row['judgment']
        answer = None
        if answer_ref:
            answer = checked_json(run_path / 'answers' / f'{pid}.json', answer_ref['sha256'])
            if (answer['paper_id'], answer['source_sha256']) != (pid, row['source_sha256']):
                raise ValueError(f'Wrong answer provenance: {run_path.name}/{pid}')
        predictions = answer['issues'] if answer is not None else []
        eligible_gold = [issue for issue in gold_issues if issue['classification'] != 'typo']
        needs_judgment = bool(predictions and eligible_gold)
        decisions = None
        if judge_ref:
            judgment = checked_json(run_path / 'judgments' / f'{pid}.json', judge_ref['sha256'])
            if not needs_judgment:
                raise ValueError(f'Unexpected judgment: {run_path.name}/{pid}')
            evidence = evaluation.semantic.build_input_document(
                judge_ref['prediction_set_name'], pid, row['source_sha256'], eligible_gold, predictions)
            raw_evidence = (json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode()
            evaluation.semantic.validate_judgment(
                judgment, input_document=evidence, input_sha256=digest(raw_evidence),
                judge_run_name=judge_ref['judge_run_name'], prediction_set_name=judge_ref['prediction_set_name'])
            decisions = judgment['gold_decisions']
        elif needs_judgment:
            raise ValueError(f'Missing judgment: {run_path.name}/{pid}')
        reports.append(evaluation.build_paper_report(
            authority, gold_issues, answer, decisions, run_name=run['run_id'],
            input_status=row['parse_status'], needs_judgment=needs_judgment,
            input_response_incomplete=None,
            provenance={'parse_audit_sha256': None,
                        'prediction_sha256': answer_ref['sha256'] if answer_ref else None,
                        'gold_issues_sha256': authority['issues_sha256'],
                        'judgment_sha256': judge_ref['sha256'] if judge_ref else None},
        ))
    rows = [issue for report in reports for issue in metrics.assess(report)]
    scores = metrics.issue_scores(rows, evaluation.semantic.GOLD_WEIGHTS)
    paper_rows, _ = metrics.paper_scores(reports, rows, 'exclude')
    accepted = sum(report['judgment_status'] == 'complete' for report in reports)
    recovered = {row['gold_issue_id'] for row in rows if row['recovered']}
    complete = {row['paper_id'] for row in paper_rows if row['all_recovered']}
    return scores, accepted, recovered, complete


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--bundle', type=Path, required=True, help='Extracted directory containing blog_results_index.json')
    args = cli.parse_args()
    bundle_dir = args.bundle.expanduser().resolve()
    bundle = read(bundle_dir / 'blog_results_index.json')
    gold = load_gold(bundle)
    scores_by_run, pooled, fully_recovered = {}, set(), set()
    accepted_total = 0
    for run in bundle['runs']:
        scores, accepted, recovered, complete = score_run(bundle_dir / 'results' / run, gold)
        scores_by_run[run] = scores
        pooled.update(recovered)
        fully_recovered.update(complete)
        accepted_total += accepted
    result = {'runs': scores_by_run, 'accepted_judgments': accepted_total,
              'pooled_issues_recovered': len(pooled), 'gold_issues': sum(issue['classification'] != 'typo' for _, issues in gold.values() for issue in issues),
              'papers_every_model_missed_at_least_one_issue': len(set(gold) - fully_recovered)}
    if result != bundle['recomputed_results']:
        raise ValueError('Recomputed scores differ from the scores recorded in blog_results_index.json')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
