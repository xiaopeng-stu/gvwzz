"""Path-only recording, selective or streaming replay, and verification."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from .core import Signature, decode_integer, exact_integer, validate_monomial
from .orders import ModuleOrder
from .polynomial import IntegerPolynomialRing
from .reconstruction import (PathReconstruction, SyzygyReplay, check_syzygy,
                             verify_generation_paths)


def add_path_commands(commands):
    p = commands.add_parser('reconstruct', help='Replay saved generation paths; no re-reduction or GVW rerun')
    p.add_argument('file')
    p.add_argument('--output', required=True)
    p.add_argument('--format', choices=['json', 'jsonl'], default='json',
                   help='jsonl streams one S vector at a time; interrupted output stays as .partial')
    p.add_argument('--indices', help='One-based S indices, e.g. 1,3,5; default: all final S')
    p.add_argument('--verify', action='store_true', help='Check expanded u*f^T=0 as well as full signatures')
    p.add_argument('--progress', type=float, default=5.0)
    p.add_argument('--show-syzygies', action='store_true')
    p.add_argument('--syzygies-only', action='store_true', help='Compatibility alias; this is always the only mode')

    p = commands.add_parser('verify-paths', help='Verify local path identities and final criteria without any full P u-vectors')
    p.add_argument('file')
    p.add_argument('--strong', action='store_true')
    p.add_argument('--paths-only', action='store_true', help='Check path identities, not final completeness criteria')
    p.add_argument('--timeout', type=float, default=60.0, help='0 disables the verification time limit')
    p.add_argument('--max-steps', type=int, default=200000)

    p = commands.add_parser('verify-syzygies', help='Check all explicit zero relations and complete signatures in JSON/JSONL')
    p.add_argument('file')


def _progress(interval):
    if not math.isfinite(interval) or interval < 0:
        raise ValueError('Progress interval must be finite and nonnegative')
    if interval == 0:
        return None
    last = [-float('inf')]
    def report(stats):
        now = perf_counter()
        if now - last[0] < interval:
            return
        last[0] = now
        i = stats.get('current_syzygy_index')
        label = f'S[{i+1}]' if i is not None else 'done'
        print(f'[path replay] {label}; completed={stats["syzygies_reconstructed"]}; '
              f'visited={stats.get("ancestor_nodes_visited", 0)}; '
              f'P-u-expanded=0', file=sys.stderr, flush=True)
    return report


def replay_to_file(result, output, *, indices=None, verify=False,
                   progress_interval=5.0, show=False, stream=False):
    from .cli import write_json
    from .printing import format_vector
    replay = SyzygyReplay(result, indices=indices, verify=verify, progress=_progress(progress_interval))
    output = Path(output)
    if stream:
        # Never replace a valid output by an interrupted stream.
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_name(output.name + '.partial')
        header = PathReconstruction(result, [], {}).to_dict()
        header.pop('syzygies'); header.pop('statistics'); header.pop('status')
        header.update(format='gvwzz-path-syzygies-jsonl-v1', type='header',
                      expected_count=len(replay.indices), selected_indices=[i+1 for i in replay.indices])
        with partial.open('w', encoding='utf-8') as sink:
            sink.write(json.dumps(header, ensure_ascii=False) + '\n')
            for record in replay:
                data = dict(type='syzygy', index=record.index+1,
                            signature=record.signature.to_dict(),
                            vector=[result.backend.serialize(f) for f in record.vector])
                sink.write(json.dumps(data, ensure_ascii=False) + '\n')
                sink.flush()
                if show:
                    print(f'z{record.index+1} = {format_vector(record.vector, result.backend)}')
            sink.write(json.dumps(dict(type='footer', status='completed', statistics=replay.statistics),
                                  ensure_ascii=False) + '\n')
        partial.replace(output)
    else:
        records = []
        for record in replay:
            records.append(record)
            if show:
                print(f'z{record.index+1} = {format_vector(record.vector, result.backend)}')
        data = PathReconstruction(result, records, dict(replay.statistics)).to_dict()
        write_json(output, data)
    stats = replay.statistics
    print(f'Reconstructed {stats["syzygies_reconstructed"]} syzygies; '
          f'P u-vectors expanded: 0; re-reductions: 0; '
          f'replay: {stats["replay_seconds"]:.6f}s; checks: {stats["verification_seconds"]:.6f}s')
    print('Syzygies:', output.resolve())
    return stats


def _context(data):
    if data.get('coefficient_ring') != 'ZZ':
        raise ValueError('Syzygy file must use ZZ')
    b = IntegerPolynomialRing(data['variables'], data['polynomial_order'])
    inputs = [b.deserialize(f, allow_expression=False) for f in data['input_polynomials']]
    pos = data.get('positions_low_to_high', list(range(1, len(inputs)+1)))
    order = ModuleOrder(b, inputs, data['module_order'],
                        [exact_integer(i, 'position') - 1 for i in pos],
                        block_sizes=data.get('signature_blocks'))
    return SimpleNamespace(backend=b, inputs=inputs, module_order=order)


def _check_record(record, context, seen, source_count):
    i = exact_integer(record['index'], 'S index')
    if not 1 <= i <= source_count or i in seen:
        raise ValueError('Duplicate or out-of-range S index')
    seen.add(i)
    item = record['signature']
    sig = Signature(decode_integer(item['coefficient']),
                    validate_monomial(item['exponents'], len(context.backend.names)),
                    exact_integer(item['index'], 'signature index') - 1)
    if not 0 <= sig.index < len(context.inputs):
        raise ValueError('Signature index outside original inputs')
    if not isinstance(record['vector'], list):
        raise ValueError('Syzygy vector must be a list')
    vector = tuple(context.backend.deserialize(f, allow_expression=False) for f in record['vector'])
    check_syzygy(vector, sig, context, identity=True)


def _nonnegative_count(value, name):
    count = exact_integer(value, name)
    if count < 0:
        raise ValueError(f'{name} must be nonnegative')
    return count


def verify_syzygy_file(path):
    start = perf_counter()
    report = dict(passed=False, status='invalid_syzygies', vectors_checked=0, errors=[],
                  signature_basis_certified=False,
                  scope='Zero identities and complete signatures only; not completeness of S')
    try:
        with Path(path).open(encoding='utf-8') as source:
            first = source.readline()
            try:
                header = json.loads(first, parse_int=decode_integer)
            except (ValueError, TypeError):
                header = None
            if isinstance(header, dict) and header.get('format') == 'gvwzz-path-syzygies-jsonl-v1':
                if header.get('type') != 'header':
                    raise ValueError('Missing stream header')
                context = _context(header)
                source_count = _nonnegative_count(header['source_syzygy_count'], 'source S count')
                expected_count = _nonnegative_count(header['expected_count'], 'expected S count')
                selected = header['selected_indices']
                if not isinstance(selected, list):
                    raise ValueError('Stream selected indices must be a list')
                selected = [exact_integer(i, 'selected S index') for i in selected]
                if (len(selected) != expected_count or len(set(selected)) != len(selected)
                        or any(not 1 <= i <= source_count for i in selected)):
                    raise ValueError('Stream selected indices must be distinct, in range, and match expected count')
                seen, footer = set(), None
                for line in source:
                    record = json.loads(line, parse_int=decode_integer)
                    if footer is not None:
                        raise ValueError('Unexpected record after stream footer')
                    if record.get('type') == 'footer':
                        footer = record
                    elif record.get('type') == 'syzygy':
                        _check_record(record, context, seen, source_count)
                        report['vectors_checked'] += 1
                    else:
                        raise ValueError('Invalid stream record')
                if footer is None or footer.get('status') != 'completed':
                    raise ValueError('Incomplete stream: completion footer missing')
                if report['vectors_checked'] != expected_count:
                    raise ValueError('Stream vector count mismatch')
                if seen != set(selected):
                    raise ValueError('Stream selected index mismatch')
                footer_count = _nonnegative_count(
                    footer['statistics']['syzygies_reconstructed'], 'stream footer count')
                if footer_count != report['vectors_checked']:
                    raise ValueError('Stream footer count mismatch')
            else:
                source.seek(0)
                data = json.load(source, parse_int=decode_integer)
                if data.get('format') != 'gvwzz-path-syzygies-v1' or data.get('status') != 'completed':
                    raise ValueError('Unsupported syzygy JSON format')
                context = _context(data)
                source_count = _nonnegative_count(data['source_syzygy_count'], 'source S count')
                seen = set()
                for record in data['syzygies']:
                    _check_record(record, context, seen, source_count)
                    report['vectors_checked'] += 1
                reconstructed = _nonnegative_count(
                    data['statistics']['syzygies_reconstructed'], 'reconstructed S count')
                if reconstructed != report['vectors_checked']:
                    raise ValueError('Syzygy vector count mismatch')
        report.update(passed=True, status='verified')
    except (ValueError, TypeError, OSError, KeyError, IndexError, AttributeError) as exc:
        report['errors'].append(str(exc))
    report['seconds'] = perf_counter() - start
    return report


def dispatch_path_command(args):
    from .cli import ensure_distinct_paths, load_result
    if args.command == 'verify-syzygies':
        report = verify_syzygy_file(args.file)
    elif args.command == 'verify-paths':
        result = load_result(args.file)
        report = verify_generation_paths(result, criteria=not args.paths_only,
                    strong=args.strong, timeout=args.timeout, max_reduction_steps=args.max_steps)
    else:
        output = Path(args.output)
        paths = {'input': Path(args.file), 'output': output}
        if args.format == 'jsonl':
            paths['partial stream'] = output.with_name(output.name + '.partial')
        ensure_distinct_paths(paths)
        result = load_result(args.file)
        indices = None
        if args.indices:
            indices = [int(s.strip())-1 for s in args.indices.split(',')]
        replay_to_file(result, args.output, indices=indices, verify=args.verify,
                       progress_interval=args.progress, show=args.show_syzygies,
                       stream=args.format == 'jsonl')
        return 0
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['passed'] else 3
