#!/usr/bin/env python3
"""Bounded synthetic OaiLLMModel probe. No network without --execute."""

import argparse
import asyncio
import copy
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ("F", "pov\u011bst", "EN", "J", "ot\u00e1zka", "Dobr\u00fd den.")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_new(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, ensure_ascii=True, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


async def trial(model, source, target, examples, timeout, artifact=None, offline=False):
    import httpx
    from app.models.oai_llm_model import LLMTranslationSegment

    expected = []
    for example in examples:
        expected.extend([
            {'role': 'user', 'content': model._format_prompt(
                example['source'], 0, 'cs', target, None, None)},
            {'role': 'assistant', 'content': example['target']},
        ])
    expected.append({'role': 'user', 'content': model._format_prompt(
        source, 0, 'cs', target, None, None)})
    calls = 0
    observed: dict[str, object] = {
        'correctness': 'pending_human_review', 'termination': 'not_observed'}
    baseline = model._request_body('probe')
    baseline.pop('messages')

    async def request_hook(request):
        nonlocal calls
        body = json.loads(request.content)
        messages = body.pop('messages')
        if calls or messages != expected or body != baseline:
            raise RuntimeError('Request contract mismatch or extra request blocked')
        calls += 1
        observed['request_id'] = request.headers.get('X-Request-ID')
        observed['generation'] = {key: body.get(key) for key in (
            'model', 'temperature', 'stop', 'max_tokens', 'max_completion_tokens')}

    async def response_hook(response):
        await response.aread()
        observed['http_status'] = response.status_code
        observed['response_request_id'] = response.headers.get('x-request-id')
        if not response.is_success:
            observed['termination'] = 'http_error'
            return  # Never persist backend error bodies.
        try:
            data = response.json()
            choice = data['choices'][0]
            reason = choice.get('finish_reason')
            observed.update({
                'completion_id': data.get('id'), 'response_model': data.get('model'),
                'finish_reason': reason,
                'termination': 'truncated' if reason == 'length' else (
                    'stopped' if reason == 'stop' else 'other_or_unknown'),
                'usage': {key: data.get('usage', {}).get(key) for key in (
                    'prompt_tokens', 'completion_tokens', 'total_tokens')},
                'output': choice['message']['content'],
            })
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            observed['termination'] = 'invalid_response'

    if offline:
        client = httpx.AsyncClient(base_url='https://offline.invalid/v1',
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={
                'choices': [{'finish_reason': 'stop', 'message': {'content': 'mock'}}]
            })))
    else:
        client = model._create_client(model.server)
    client.event_hooks['request'].append(request_hook)
    client.event_hooks['response'].append(response_hook)
    started = time.monotonic()
    try:
        async with client:
            result = await asyncio.wait_for(model._translate_segment(
                client, asyncio.Semaphore(1), LLMTranslationSegment(source, 0, '0'),
                str(uuid.uuid4()), 'cs', target, None, None), timeout)
            observed['app_output'] = result.output
            observed['app_translated'] = all(record.translated for record in result.records)
            if result.resplit_count:
                raise RuntimeError('Unexpected resplit')
        if calls != 1:
            raise RuntimeError('Expected one HTTP request')
    except Exception as error:
        observed['error_type'] = type(error).__name__
        if offline or isinstance(error, RuntimeError):
            raise
    finally:
        observed['duration_seconds'] = round(time.monotonic() - started, 3)
        observed['http_requests'] = calls
        if artifact is not None:
            write_new(artifact, observed)
    return observed


async def experiment(args, output):
    from flask import Flask
    from app.models.oai_llm_model import OaiLLMModel

    configs = json.loads(Path(args.models_config).read_text())
    fixture = json.loads(Path(args.examples).read_text())['few_shot_attempts']
    selected = {}
    for target in ('de', 'uk'):
        name = getattr(args, 'model_' + target)
        candidates = [cfg for cfg in configs if cfg.get('model_framework') == 'oai_llm'
                      and 'cs' in cfg['source'] and target in cfg['target']
                      and (name is None or cfg['model'] == name)]
        if len(candidates) != 1:
            raise ValueError('Select one unambiguous model per pair with --model-de/--model-uk')
        selected[target] = candidates[0]
        stages = fixture['cs-' + target]
        if len(stages) != 1 or len(stages[0]) != 5:
            raise ValueError('Probe fixture must contain one direct five-example stage')
        for example in stages[0]:
            if set(example) != {'source', 'target'} or not all(
                    isinstance(value, str) and value for value in example.values()):
                raise ValueError('Invalid example')
            if example['source'] in CORPUS:
                raise ValueError('Example overlaps held-out corpus')

    plan = []
    for target in ('de', 'uk'):
        for index, source in enumerate(CORPUS):
            for arm in ('zero', 'five'):
                examples = fixture['cs-' + target][0] if arm == 'five' else []
                cfg = copy.deepcopy(selected[target])
                cfg.update(few_shot_attempts={'cs-' + target: [examples]},
                           retry_attempts=1, max_resplit_depth=0, batch_size=1,
                           max_input_tokens=args.input_tokens,
                           max_completion_tokens=args.output_tokens,
                           overall_timeout=args.call_timeout,
                           connect_timeout=args.call_timeout, read_timeout=args.call_timeout,
                           write_timeout=args.call_timeout, pool_timeout=args.call_timeout)
                model = OaiLLMModel(cfg)
                if model._estimated_text_tokens(source) > args.input_tokens:
                    raise ValueError('Input budget is smaller than a held-out corpus estimate')
                plan.append((f'cs-{target}-{index}-{arm}', model, source,
                             target, examples))

    app = Flask('few-shot-probe')
    app.config.from_object('app.settings')
    app.config.from_envvar('LOCAL_SETTINGS', silent=True)
    with app.app_context():
        endpoint_fingerprints = {target: fingerprint(model.server)
                                 for _, model, _, target, _ in plan}

    manifest = {
        'schema': 1, 'corpus': CORPUS, 'examples': fixture,
        'example_fingerprints': {pair: fingerprint(stages) for pair, stages in fixture.items()},
        'config_fingerprints': {target: fingerprint(cfg) for target, cfg in selected.items()},
        'endpoint_fingerprints': endpoint_fingerprints,
        'model_ids': {target: cfg['model'] for target, cfg in selected.items()},
        'code_hashes': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in [Path(__file__).resolve(), *sorted((ROOT / 'app/models').glob('*.py'))]},
        'args': {key: value for key, value in vars(args).items()
                 if key not in ('execute', 'resume', 'output_dir', 'models_config', 'examples')},
        'plan': [row[0] for row in plan],
        'correctness': 'pending_human_review',
    }
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != json.loads(json.dumps(manifest)):
            raise ValueError('Resume manifest mismatch; do not reuse this experiment directory')
    else:
        if args.resume:
            raise ValueError('Cannot resume without a complete manifest')
        write_new(manifest_path, manifest)

    # Validate every rendered exchange without contacting any server, including
    # implementations that silently ignore an as-yet unsupported config key.
    for _, model, source, target, examples in plan:
        await trial(model, source, target, examples, args.call_timeout, offline=True)

    deadline = time.monotonic() + args.overall_timeout
    with app.app_context():
        for key, model, source, target, examples in plan[:args.max_requests]:
            claim = output / (key + '.claim.json')
            if claim.exists():
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            # A durable claim precedes transport. An uncertain interrupted call
            # is deliberately never replayed, even if no result was saved.
            write_new(claim, {'case': key, 'source': source, 'model': model.model,
                              'claimed_at_unix': time.time()})
            await trial(model, source, target, examples,
                        min(args.call_timeout, remaining), output / (key + '.result.json'))

    rows = []
    for key, *_ in plan:
        result = output / (key + '.result.json')
        if result.exists():
            data = json.loads(result.read_text())
            rows.append({'case': key, **{field: data.get(field) for field in (
                'termination', 'finish_reason', 'http_status', 'duration_seconds',
                'correctness', 'error_type')}})
        elif (output / (key + '.claim.json')).exists():
            rows.append({'case': key, 'termination': 'interrupted_or_unknown',
                         'correctness': 'pending_human_review'})
    write_new(output / ('report-' + uuid.uuid4().hex + '.json'), {
        'results': rows, 'correctness': 'pending_human_review',
        'note': 'Termination is not translation correctness. No automatic quality verdict.'})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output-dir')
    parser.add_argument('--models-config', default=str(ROOT / 'app/models.json'))
    parser.add_argument('--examples', default=str(ROOT / 'examples/few_shot_attempts.json'))
    parser.add_argument('--model-de')
    parser.add_argument('--model-uk')
    parser.add_argument('--max-requests', type=int, default=24)
    parser.add_argument('--call-timeout', type=int, default=120)
    parser.add_argument('--overall-timeout', type=int, default=2880)
    parser.add_argument('--input-tokens', type=int, default=512)
    parser.add_argument('--output-tokens', type=int, default=512)
    args = parser.parse_args(argv)
    if not (1 <= args.max_requests <= 24 and 1 <= args.call_timeout <= 120
            and 1 <= args.overall_timeout <= 2880
            and 1 <= args.input_tokens <= 512 and 1 <= args.output_tokens <= 512):
        parser.error('Bounds: requests 1..24, call 1..120s, overall 1..2880s, tokens 1..512')
    if not args.execute:
        print('NOT EXECUTED. Plan: 2 pairs x 6 held-out inputs x 2 arms; at most 24 requests.')
        return 0
    if args.resume and not args.output_dir:
        parser.error('--resume requires --output-dir')
    output = Path(args.output_dir or ('/tmp/opencode/few-shot-' + uuid.uuid4().hex))
    logging.disable(logging.CRITICAL)  # App/HTTP logs must not expose config or prompts.
    sys.path.insert(0, str(ROOT))
    try:
        if not args.resume:
            output.mkdir(mode=0o700, parents=False, exist_ok=False)
        stat = output.lstat()
        if output.is_symlink() or not output.is_dir() or stat.st_uid != os.getuid() or stat.st_mode & 0o077:
            raise ValueError('Output must be an owned, private directory (0700)')
        print('Artifact directory: ' + str(output), flush=True)
        with open(output / '.lock', 'a') as lock:
            os.chmod(output / '.lock', 0o600)
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            asyncio.run(experiment(args, output))
        print('Probe artifacts written; correctness requires human review.')
        return 0
    except Exception as error:
        # Exceptions can include private prompts, URLs, tokens or backend bodies.
        print('Probe stopped safely (' + type(error).__name__ + '). See operator guide.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
