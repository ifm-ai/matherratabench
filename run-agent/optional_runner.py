#!/usr/bin/env python3
"""Optional single-request-per-paper example for a Chat Completions-compatible API.

Uses parse.py for input verification and strict answer checks. The parser does
not depend on this runner. API keys are read from the environment and not saved.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request

sys.dont_write_bytecode = True
import parse as contract


class NoRedirect(urllib.request.HTTPRedirectHandler):

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise contract.InputError('API redirected the request; provide its final base URL')


def example(args):
    (bundle, schema) = contract.load_inputs(args.inputs)
    validator = contract.validator_for(schema)
    parsed_url = urllib.parse.urlsplit(args.base_url)
    if parsed_url.scheme not in ('http', 'https') or not parsed_url.hostname or parsed_url.username or parsed_url.password or parsed_url.query or parsed_url.fragment:
        raise contract.InputError('Use an HTTP(S) API base URL without credentials, query parameters or fragments')
    if args.output.exists():
        raise contract.InputError('Output already exists; choose a new example-run directory')
    if args.max_output_tokens is not None and args.max_output_tokens <= 0:
        raise contract.InputError('--max-output-tokens must be positive')
    endpoint = args.base_url.rstrip('/') + '/chat/completions'
    if args.dry_run:
        print(f"Would send {bundle['paper_count']} independent requests to {endpoint} using {args.model}")
        print('Dry run: no API requests and no output files')
        return 0
    api_key = os.environ.get(args.api_key_env, '')
    if api_key and parsed_url.scheme != 'https' and (parsed_url.hostname not in ('127.0.0.1', 'localhost', '::1')):
        raise contract.InputError('Use HTTPS when sending an API key to a remote endpoint')
    args.output.mkdir(parents=True)
    answers = args.output / 'answers'
    answers.mkdir()
    settings = {'model': args.model, 'stream': False}
    if args.max_output_tokens is not None:
        settings['max_completion_tokens'] = args.max_output_tokens
    contract.write_json(args.output / 'run.json', {'runner': 'single-user-message-chat-completions-example-v1', 'runner_sha256': contract.sha256(Path(__file__).read_bytes()), 'inputs_sha256': contract.sha256((args.inputs / 'inputs.json').read_bytes()), 'run_name': bundle['run_name'], 'endpoint': endpoint, 'settings': settings})
    opener = urllib.request.build_opener(NoRedirect())
    failed = 0
    for row in bundle['papers']:
        paper = row['paper_id']
        prompt = (args.inputs / row['prompt_file']).read_bytes().decode('utf-8')
        payload = {**settings, 'messages': [{'role': 'user', 'content': prompt}]}
        headers = {'Content-Type': 'application/json'}
        if api_key:
            headers['Authorization'] = 'Bearer ' + api_key
        request = urllib.request.Request(endpoint, json.dumps(payload).encode('utf-8'), headers=headers)
        try:
            with opener.open(request, timeout=args.timeout) as response:
                raw = response.read()
            response_dir = args.output / 'responses'
            response_dir.mkdir(exist_ok=True)
            (response_dir / f'{paper}.json').write_bytes(raw)
            response_data = contract.read_json(raw)
            choice = response_data['choices'][0]
            content = choice['message'].get('content')
            if not isinstance(content, str):
                raise contract.InputError('API did not return a text answer')
            raw_dir = args.output / 'raw'
            raw_dir.mkdir(exist_ok=True)
            (raw_dir / f'{paper}.txt').write_text(content, encoding='utf-8')
            if choice.get('finish_reason') != 'stop':
                raise contract.InputError('API response was incomplete or did not finish with a text answer')
            answer = contract.read_json(content)
            contract.validate_answer(answer, row, bundle['run_name'], validator)
            contract.write_json(answers / f'{paper}.json', answer)
            print(f"{paper}: {answer['status']}")
        except urllib.error.HTTPError as exc:
            failed += 1
            print(f'{paper}: API HTTP {exc.code}', file=sys.stderr)
        except (ValueError, OSError, KeyError, IndexError, TypeError):
            failed += 1
            print(f'{paper}: request or answer failed; inspect saved response files', file=sys.stderr)
    report = contract.check_answers(args.inputs, answers)
    contract.write_json(args.output / 'validation.json', report)
    print(f'Saved answers in {answers}; failed requests/answers: {failed}')
    return 0 if report['valid'] and (not failed) else 1

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="MODEL_API_KEY")
    parser.add_argument("--max-output-tokens", type=int, help="Optional max_completion_tokens parameter")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        return example(args)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
