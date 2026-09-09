"""Export existing final selections/scores without distributing corpus text.

Standard library only. No summarizer, evaluator or protected-data loader is run.
Use --output with a new ZIP filename; existing artifacts are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]
POPULATIONS = {'govreport': 973, 'multinews': 5621}
SCORE_FIELDS = {'rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'macro_rouge'}


def export_selection(path: Path, expected_rows: int):
    digest = hashlib.sha256()
    rows, ids = [], []
    with path.open('rb') as handle:
        for line in handle:
            digest.update(line)
            item = json.loads(line)
            indices = item['selected_indices']
            if (not isinstance(indices, list) or
                    any(type(i) is not int or i < 0 for i in indices) or
                    len(set(indices)) != len(indices)):
                raise ValueError(f'invalid selected indices: {path}')
            ids.append(item['id'])
            rows.append({
                'id': item['id'], 'selected_indices': indices,
                'summary_words': len(item['summary'].split()),
                'summary_sentences': len(indices),
                'feasible': item['feasible'],
                'infeasible_code': item.get('infeasible_code'),
                'source_record_sha256': hashlib.sha256(line).hexdigest(),
            })
    if len(rows) != expected_rows or len(set(ids)) != expected_rows:
        raise ValueError(f'unexpected population or duplicate IDs: {path}')
    return rows, ids, digest.hexdigest()


def validate_scores(path: Path, expected_ids: list[str]):
    payload = path.read_bytes()
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    if [row['id'] for row in rows] != expected_ids:
        raise ValueError(f'score ID/order mismatch: {path}')
    for row in rows:
        if set(row) - (SCORE_FIELDS | {'id'}):
            raise ValueError(f'unexpected score fields (possible text): {path}')
        for key, value in row.items():
            if key != 'id' and (type(value) not in (int, float) or not 0 <= value <= 1):
                raise ValueError(f'invalid score: {path}')
    return payload


def build(output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {'schema_version': 1, 'release_tag': 'v1.0.0-ieee-access',
                'scope': 'Final official test selections and official/internal per-document scores; no source, reference or summary text.',
                'datasets': {}, 'files': {}}
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        def add(name, payload, **metadata):
            archive.writestr(name, payload)
            manifest['files'][name] = {'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload), **metadata}

        for dataset, count in POPULATIONS.items():
            run = ROOT / f'runs_v2/{dataset}_final_test_v1'
            paths = sorted((run / 'predictions').glob('*/predictions.jsonl'))
            if len(paths) != 18:
                raise ValueError(f'expected 8 deterministic systems and 10 random seeds: {dataset}')
            reference_ids = None
            for path in paths:
                system = path.parent.name
                selections, ids, source_sha = export_selection(path, count)
                if reference_ids is None: reference_ids = ids
                if ids != reference_ids: raise ValueError(f'prediction ID/order mismatch: {path}')
                payload = ''.join(json.dumps(row, ensure_ascii=True, separators=(',', ':'))+'\n' for row in selections).encode()
                add(f'{dataset}/selections/{system}.jsonl', payload, rows=count,
                    original_source=path.relative_to(ROOT).as_posix(), original_source_sha256=source_sha)
                for protocol in ['official', 'internal']:
                    scores = run / protocol / system / 'per_example.jsonl'
                    payload = validate_scores(scores, ids)
                    add(f'{dataset}/{protocol}/{system}.jsonl', payload, rows=count,
                        original_source=scores.relative_to(ROOT).as_posix(), original_source_sha256=hashlib.sha256(payload).hexdigest())
            # The random aggregate has scores but no single selected-index sequence.
            for protocol in ['official', 'internal']:
                scores = run / protocol / 'random_mean_10_seeds/per_example.jsonl'
                if scores.exists():
                    payload = validate_scores(scores, reference_ids)
                    add(f'{dataset}/{protocol}/random_mean_10_seeds.jsonl',payload,rows=count,
                        original_source=scores.relative_to(ROOT).as_posix(),original_source_sha256=hashlib.sha256(payload).hexdigest())
            manifest['datasets'][dataset] = {'documents': count, 'selection_runs': len(paths),
                                           'ordered_ids_sha256': hashlib.sha256(('\n'.join(reference_ids)+'\n').encode()).hexdigest()}
            print(f'{dataset}: verified and exported {len(paths)} selection runs and matching scores',flush=True)
        instructions = '''PAMR-ES numerical reproduction artifact

Release: v1.0.0-ieee-access
Code: https://github.com/Caspar15/metaheuristic-summarization/tree/v1.0.0-ieee-access

Each dataset contains 8 deterministic systems and 10 fixed Random seeds.
selections/ records use zero-based indices in canonical source-sentence order.
They preserve the exported selected-index order and contain no source, reference
or generated summary text. Reconstruct summaries only against the original
canonical inputs and the frozen output-ordering contract. summary_words counts
whitespace-delimited words in the archived prediction, not regenerated text.

official/ contains the Stanza/Perl score scale; internal/ contains Python
ROUGE-Lsum. Do not mix these protocols. Random aggregate scores, when available,
are separate from the ten individual seeds and have no single selection sequence.

Run from the unpacked artifact directory:
  python verify_artifact.py

The manifest contains SHA-256 of every exported file and the original files.
Each selection row also records the SHA-256 of its original prediction line.
Original predictions are not distributed here; their hashes provide identity
when those files are available, not proof obtainable from the derived row alone.

This package reproduces inspection/aggregation of archived results. Full output
regeneration additionally requires providers' datasets/model/runtime resources.
The project MIT license does not relicense underlying datasets or model weights.
'''
        add('README.txt',instructions.encode())
        for filename in ['LICENSE', 'THIRD_PARTY_NOTICES.md']:
            add(filename,(ROOT/filename).read_bytes())
        verifier = '''import hashlib, json
from pathlib import Path
root = Path(__file__).resolve().parent
m = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
assert m["files"], "empty manifest"
for name, pin in m["files"].items():
    target = (root / name).resolve()
    assert target.is_relative_to(root), name
    payload = target.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == pin["sha256"], name
    assert len(payload) == pin["bytes"], name
    if "rows" in pin:
        assert len(payload.splitlines()) == pin["rows"], name
print(json.dumps({"status": "pass", "files": len(m["files"]), "datasets": m["datasets"]}, indent=2))
'''
        add('verify_artifact.py',verifier.encode())
        archive.writestr('MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(output) as archive:
        if archive.testzip(): raise ValueError('ZIP CRC failure')
        for name,pin in manifest['files'].items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=pin['sha256']:
                raise ValueError(f'ZIP hash mismatch: {name}')
    print(json.dumps({'output':str(output),'files':len(manifest['files']),
                      'bytes':output.stat().st_size,'sha256':hashlib.sha256(output.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    build(parser.parse_args().output)
