#!/usr/bin/env python3
"""Offline-first FLORES 0/5/10-shot benchmark over user-approved local data.

Execution requires both terms and training-data attestations. No gate is accepted
and no data is downloaded by this command. Only the approved direction-specific
EDUKATE model IDs are permitted, including in preview. Model revisions are declarations,
not server-verified pins; 'unknown' cannot establish training-data cleanliness.
The cleanliness flag is the operator's independent attestation, not our verdict.
Use a dedicated private run directory, never a batch output directory. Resume
never replays a claimed job, even after a crash or transport uncertainty. Budgets
apply to total durable claims across invocations. Incomplete runs are not
eligible for scoring; consumers must require a terminal record for every job.
The companion scorer treats terminal failures as blank hypotheses, not omissions.
"""

import argparse
import copy
import fcntl
import hashlib
import importlib
import json
import logging
import math
import os
from pathlib import Path
import stat
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
VERSION = '1.0'
CONDITIONS = [0, 5, 10]
APPROVED_MODELS = {'en': 'edukate_csen_v1', 'de': 'edukate_csde_v1', 'uk': 'edukate_csuk_v1'}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def assignments(values, targets):
    result = {}
    for value in values:
        target, separator, name = value.partition('=')
        if not separator or not name.strip() or target in result or target not in targets:
            raise ValueError('Invalid or duplicate target assignment')
        result[target] = name
    if set(result) != set(targets):
        raise ValueError('Every target requires an explicit assignment')
    return result


def check_private(path, directory=False):
    info = path.lstat()
    if (info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != (0o700 if directory else 0o600)
            or not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))
            or (not directory and info.st_nlink != 1)):
        raise ValueError('Artifacts must be private, owned, and not links')


def write_new(path, value):
    """Publish a complete fsynced file without ever replacing an existing claim."""
    temporary = path.parent / ('.pending-' + uuid.uuid4().hex)
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=True, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
    finally:
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def execute(args, app, selected, manifest):
    from app.models.async_llm_runner import close_async_llm_runner
    from app.models.llm_errors import LLMBackendTimeout
    from app.models.oai_llm_model import OaiLLMModel
    from flask import has_request_context

    if has_request_context():
        raise RuntimeError('Benchmark must run outside an HTTP request context')

    root = args.run_dir.absolute()
    # Reject symlinks in the entire path, including ancestors, before any write.
    for path in [*reversed(root.parents), root]:
        if path.is_symlink():
            raise ValueError('Symlink in run path')
    if not args.resume:
        root.mkdir(mode=0o700)
    check_private(root, True)
    for path in root.rglob('*'):
        check_private(path, path.is_dir())
    lock_fd = os.open(root / '.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        check_private(root / '.lock')
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume:
            check_private(root / 'manifest.json')
            if json.loads((root / 'manifest.json').read_text()) != manifest:
                raise ValueError('Resume manifest mismatch')
        else:
            write_new(root / 'manifest.json', manifest)
            (root / 'claims').mkdir(mode=0o700)
            (root / 'records').mkdir(mode=0o700)
            directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        for name in ['claims', 'records']:
            check_private(root / name, True)
        jobs = manifest['jobs']
        known = {job['job_id']: job for job in jobs}
        for name in ['claims', 'records']:
            for path in (root / name).iterdir():
                if path.name.startswith('.pending-'):
                    continue  # Never transport before publication of the claim.
                if path.suffix != '.json' or path.stem not in known:
                    raise ValueError('Unexpected artifact')
                row = json.loads(path.read_text())
                if any(row.get(key) != value for key, value in known[path.stem].items()):
                    raise ValueError('Artifact identity mismatch')
                if row.get('resume_fingerprint') != manifest['resume_fingerprint']:
                    raise ValueError('Artifact fingerprint mismatch')
                if name == 'records' and not (root / 'claims' / path.name).exists():
                    raise ValueError('Record without durable claim')
                if name == 'records':
                    if row.get('status') not in {
                            'success', 'truncated', 'http_error', 'timeout', 'invalid_response', 'error'}:
                        raise ValueError('Invalid terminal status')
                    if row['status'] == 'success':
                        if (not isinstance(row.get('output'), str) or not row['output'].strip()
                                or row.get('finish_reason') == 'length' or row.get('http_requests') != 1):
                            raise ValueError('Invalid successful record')
                    elif row.get('output') is not None:
                        raise ValueError('Failed records cannot contain output')
        claimed = sum((root / 'claims' / (job['job_id'] + '.json')).exists() for job in jobs)
        deadline = time.monotonic() + args.overall_timeout
        evaluation = {row['id']: row for row in manifest['dataset']['evaluation']}
        active: dict = {}

        async def request_hook(request):
            body = json.loads(request.content)
            messages = body.pop('messages')
            if active['calls'] or messages != active['messages'] or body != active['body']:
                raise RuntimeError('Extra request or altered benchmark input blocked')
            active['calls'] += 1
            active['record']['request_id'] = request.headers.get('x-request-id')

        async def response_hook(response):
            record = active['record']
            record['http_status'] = response.status_code
            record['response_request_id'] = response.headers.get('x-request-id')
            await response.aread()
            if not response.is_success:
                record['status'] = 'http_error'
                return
            try:
                data = response.json()
                choice = data['choices'][0]
                reason = choice.get('finish_reason')
                record['finish_reason'] = reason if isinstance(reason, str) else None
                usage = data.get('usage') or {}
                record['usage'] = {key: usage[key] for key in (
                    'prompt_tokens', 'completion_tokens', 'total_tokens')
                    if isinstance(usage.get(key), int) and not isinstance(usage[key], bool)}
                record['status'] = 'truncated' if reason == 'length' else 'invalid_response'
            except (ValueError, KeyError, IndexError, TypeError, AttributeError):
                record['status'] = 'invalid_response'

        class ObservedModel(OaiLLMModel):
            def _create_client(self, server):
                client = super()._create_client(server)
                client.event_hooks['request'].append(request_hook)
                client.event_hooks['response'].append(response_hook)
                return client

        current_target = None
        arms = {}
        with app.app_context():
            for job in jobs:
                claim = root / 'claims' / (job['job_id'] + '.json')
                if claim.exists():
                    continue
                remaining = deadline - time.monotonic()
                if claimed >= args.max_requests or remaining <= 0:
                    break
                target, shots = job['target'], job['shots']
                if current_target != target:
                    close_async_llm_runner()
                    arms = {}
                    current_target = target
                if shots not in arms:
                    cfg = copy.deepcopy(selected[target])
                    examples = [{key: example[key] for key in ('source', 'target')}
                                for example in manifest['dataset']['demonstrations'][target][:shots]]
                    if len(examples) != shots:
                        raise ValueError('Insufficient demonstrations')
                    cfg.update(few_shot_attempts={'cs-' + target: [examples]},
                               retry_attempts=1, max_resplit_depth=0, batch_size=1,
                               overall_timeout=args.call_timeout, connect_timeout=args.call_timeout,
                               read_timeout=args.call_timeout, write_timeout=args.call_timeout,
                               pool_timeout=args.call_timeout)
                    arms[shots] = ObservedModel(cfg)
                model = arms[shots]
                model.overall_timeout = min(args.call_timeout, remaining)
                source = evaluation[job['eval_id']]['source']
                messages = []
                for example in model.few_shot_attempts['cs-' + target][0]:
                    messages.extend([
                        {'role': 'user', 'content': model._format_prompt(
                            example['source'], 0, 'cs', target, None, None)},
                        {'role': 'assistant', 'content': example['target']}])
                messages.append({'role': 'user', 'content': model._format_prompt(
                    source, 0, 'cs', target, None, None)})
                if len(messages) != 2 * shots + 1:
                    raise ValueError('Message count mismatch')
                body = model._request_body(messages)
                body.pop('messages')
                record: dict = dict(job, status='error', output=None, finish_reason=None,
                               resume_fingerprint=manifest['resume_fingerprint'],
                               http_status=None, usage={}, request_id=None,
                              response_request_id=None)
                active.update(calls=0, messages=messages, body=body, record=record)
                write_new(claim, dict(job, claimed_at_unix=time.time(),
                    resume_fingerprint=manifest['resume_fingerprint'],
                    acknowledge_flores_terms=True, confirm_training_data_clean=True))
                claimed += 1
                started = time.monotonic()
                try:
                    outputs = model.send_sentences_to_backend([source], src='cs', tgt=target)
                    if active['calls'] != 1 or len(outputs) != 1 or not isinstance(outputs[0], str):
                        raise RuntimeError('Invalid public model result')
                    record.update(status='success', output=outputs[0])
                except Exception as error:
                    record['error_type'] = type(error).__name__
                    if isinstance(error, LLMBackendTimeout):
                        record['status'] = 'timeout'
                finally:
                    record['duration_seconds'] = time.monotonic() - started
                    record['http_requests'] = active['calls']
                    write_new(root / 'records' / (job['job_id'] + '.json'), record)
        successful = completed = 0
        for job in jobs:
            path = root / 'records' / (job['job_id'] + '.json')
            if path.exists():
                completed += 1
                if json.loads(path.read_text()).get('status') == 'success':
                    successful += 1
        complete = completed == len(jobs)
        print(f'{successful}/{len(jobs)} successful; {completed} terminal records; '
              f'scoring_eligible={str(complete).lower()}. '
              'Claimed jobs are never retried.')
        return 0 if successful == len(jobs) else 2
    finally:
        close_async_llm_runner()
        os.close(lock_fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--targets', nargs='+', choices=['en', 'de', 'uk'], default=['en', 'de', 'uk'])
    parser.add_argument('--model', action='append', default=[], metavar='TARGET=PUBLIC_ID')
    parser.add_argument('--model-revision', action='append', default=[], metavar='TARGET=REVISION')
    parser.add_argument('--models-config', type=Path, default=ROOT / 'app/models.json')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--acknowledge-flores-terms', action='store_true')
    parser.add_argument('--confirm-training-data-clean', action='store_true')
    parser.add_argument('--max-requests', type=int)
    parser.add_argument('--call-timeout', type=float, default=120)
    parser.add_argument('--overall-timeout', type=float, default=86400)
    args = parser.parse_args(argv)
    previous_logging = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        if len(set(args.targets)) != len(args.targets):
            raise ValueError('Duplicate target')
        if args.limit is not None and args.limit < 1:
            raise ValueError('Limit must be positive')
        if not (math.isfinite(args.call_timeout) and 0 < args.call_timeout <= 120
                and math.isfinite(args.overall_timeout) and 0 < args.overall_timeout <= 86400):
            raise ValueError('Timeout bounds exceeded')
        if args.max_requests is not None and args.max_requests < 1:
            raise ValueError('Budget must be positive')
        if args.execute and not (args.acknowledge_flores_terms
                and args.confirm_training_data_clean and args.max_requests and args.run_dir):
            print('Execution requires --run-dir, --max-requests, '
                  '--acknowledge-flores-terms and --confirm-training-data-clean.', file=sys.stderr)
            return 1
        names = assignments(args.model, args.targets)
        if any(name != APPROVED_MODELS[target] for target, name in names.items()):
            raise ValueError('Only the approved EDUKATE model for each target is permitted')
        revisions = assignments(args.model_revision, args.targets)
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from flask import Flask
        from app.models.oai_llm_model import OaiLLMModel
        data_module = importlib.import_module('scripts.flores_data')
        dataset = data_module.load_flores(args.data_dir, args.targets, args.revision,
                                        limit=args.limit, seed=args.seed)
        configs = json.loads(args.models_config.read_text())
        selected = {}
        models = {}
        app = Flask('flores-benchmark')
        app.config.from_object('app.settings')
        app.config.from_envvar('LOCAL_SETTINGS', silent=True)
        for target in args.targets:
            candidates = [cfg for cfg in configs if cfg.get('model') == names[target]
                          and cfg.get('model_framework') == 'oai_llm'
                          and 'cs' in cfg.get('source', []) and target in cfg.get('target', [])]
            if len(candidates) != 1:
                raise ValueError('Model must match exactly one configuration and direction')
            cfg = copy.deepcopy(candidates[0])
            selected[target] = cfg
            model = OaiLLMModel(cfg)
            with app.app_context():
                endpoint_hash = fingerprint(model.server)
            models[target] = dict(model_id=names[target], revision=revisions[target],
                revision_verification='declared_not_verified',
                config_fingerprint=fingerprint(cfg), effective_endpoint_hash=endpoint_hash,
                generation=dict(temperature=model.temperature,
                    max_completion_tokens=model.max_completion_tokens, stop=['###'],
                    prompt_hash=fingerprint(model.prompt), provider=model.provider))
        jobs = [dict(job_id=f'{target}-{index}-{shots}', target=target,
                     shots=shots, eval_id=row['id'])
                for target in args.targets for index, row in enumerate(dataset['evaluation'])
                for shots in CONDITIONS]
        if data_module.__file__ is None:
            raise ValueError('Dataset module must have a hashable source file')
        code_paths = [Path(__file__).resolve(), Path(data_module.__file__),
                      *sorted((ROOT / 'app/models').glob('*.py'))]
        manifest = dict(schema_version=1, program_version=VERSION, dataset=dataset,
            models=models, conditions=CONDITIONS, jobs=jobs,
            generation=dict(retry_attempts=1, max_resplit_depth=0, batch_size=1),
            timeouts=dict(call_seconds=args.call_timeout, overall_seconds=args.overall_timeout),
            code_hashes={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else 'flores_data':
                         hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths})
        manifest['resume_fingerprint'] = fingerprint(manifest)
        print(f"{'EXECUTE' if args.execute else 'DRY RUN'}: {len(jobs)} requests; "
              + ', '.join(f'cs-{t}={names[t]}' for t in args.targets))
        print('Model revisions are declared, not verified; unknown does not establish clean training data.')
        if not args.execute:
            return 0
        return execute(args, app, selected, manifest)
    except Exception as error:
        print('Benchmark stopped safely: ' + type(error).__name__, file=sys.stderr)
        return 1
    finally:
        logging.disable(previous_logging)


if __name__ == '__main__':
    sys.exit(main())
