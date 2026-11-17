"""Read-only helpers for the reviewer-requested v4.1 evidence workflow."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT.parent / 'ACCESS_latex_template_20260513/review/20261111_v4_1'
OUT = ROOT / 'runs_v2/access_resubmission_20261110_v4_1'
CONDITIONS = ('C01', 'A01_no_semantic', 'A02_no_graph', 'A03_lexical_only_capacity_80', 'A06_no_route_reservation')
TARGETS = ('rouge1', 'rouge2', 'rougeLsum')
DATASETS = ('govreport', 'multinews')


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def readj(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def writej(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def write_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def write_tsv(path, values):
    values = list(values)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='') as f:
        if values:
            fields = list(dict.fromkeys(k for r in values for k in r))
            writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
            writer.writeheader()
            writer.writerows(values)


def run_dir(ds, condition):
    if condition == 'C01':
        return ROOT / f'runs_v2/d3b_cross_profile_combination_v1/{ds}/dev/C01_combined_salience_route_weight'
    if condition == 'A06_no_route_reservation':
        return ROOT / f'runs_v2/postfreeze_no_reservation_v1/{ds}/dev/{condition}'
    return ROOT / f'runs_v2/{ds}_e3_route_provenance_v1/dev/{condition}'


def partition_path(ds):
    return ROOT / f'configs/validation_partitions/{ds}_validation_dev_v1.json'


def canonical_path(ds, split='validation'):
    name = 'multi_news' if ds == 'multinews' else ds
    return ROOT / f'data/processed/{name}_{split}_canonical.jsonl'


def flatten(doc):
    return [dict(s, document_id=d['document_id'], section_id=sec['section_id'])
            for d in sorted(doc['documents'], key=lambda x: x['source_order'])
            for sec in d['sections'] for s in sec['sentences']]


def unique_map(values):
    result = {}
    for value in values:
        key = value['id']
        if key in result:
            raise ValueError(f'duplicate ID: {key}')
        result[key] = value
    return result


def structural_group(left, right):
    """Reference-blind strata; do not mistake size for set containment."""
    left, right = set(left), set(right)
    m, n = len(left), len(right)
    group = 'C01_smaller' if m < n else 'C01_larger' if m > n else 'equal_size'
    child = None if m != n else 'both_at_80' if m == 80 else 'both_empty' if m == 0 else 'equal_positive_below_80'
    relation = ('identical' if left == right else 'C01_strict_subset' if left < right
                else 'control_strict_subset' if right < left else 'neither_contains_other')
    return group, child, relation


def sample_manifest(documents, ds, n_per_stratum, phase):
    items = sorted((sum(len(s['text'].split()) for s in flatten(d)), d['id']) for d in documents)
    strata = [[] for _ in range(4)]
    for position, (words, key) in enumerate(items):
        h = min(3, 4 * position // len(items))
        literal = f'access-2026-46834|{phase}-v1|20261109|{ds}|{key}'
        strata[h].append({'dataset': ds, 'id': key, 'stratum': h,
                          'source_words': words, 'sampling_sha256': hashlib.sha256(literal.encode()).hexdigest()})
    if any(len(group) < n_per_stratum for group in strata):
        raise ValueError('insufficient stratum size; replacement forbidden')
    return [dict(x, stratum_population_n=len(group), population_n=len(items))
            for group in strata for x in sorted(group, key=lambda x: (x['sampling_sha256'], x['id']))[:n_per_stratum]]


def config_diff(left, right, prefix=''):
    result = {}
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            path = f'{prefix}.{key}' if prefix else key
            if prefix == '' and key == 'study':
                continue
            result.update(config_diff(left.get(key), right.get(key), path))
    elif left != right:
        result[prefix] = {'C01': left, 'control': right}
    return result


def validate_mapping(pred, doc):
    source = flatten(doc)
    pool = pred['candidate_records']
    indices = [r['original_index'] for r in pool]
    selected = pred['selected_indices']
    if len(indices) != len(set(indices)) or len(selected) != len(set(selected)):
        raise ValueError(f'duplicate indices: {pred["id"]}')
    if not set(selected) <= set(indices):
        raise ValueError(f'selected outside pool: {pred["id"]}')
    cap = pred['output_budget']['max_words']
    for row in pool:
        idx = row['original_index']
        if not isinstance(idx, int) or not 0 <= idx < len(source):
            raise ValueError(f'invalid original index: {pred["id"]}')
        expected = source[idx]
        if any(row[k] != expected[k] for k in ['sentence_id', 'text', 'document_id', 'section_id']):
            raise ValueError(f'candidate source mismatch: {pred["id"]}:{idx}')
        words = len(expected['text'].split())
        if words != row['word_count'] or not 0 < words <= cap:
            raise ValueError(f'candidate eligibility mismatch: {pred["id"]}:{idx}')
    texts = [source[i]['text'] for i in selected]
    if pred['summary_sentences'] != texts or pred['summary'] != '\n'.join(texts):
        raise ValueError(f'summary reconstruction mismatch: {pred["id"]}')
    if pred['candidate_pool']['actual_size'] != len(indices):
        raise ValueError(f'pool size mismatch: {pred["id"]}')
    return {'id': pred['id'], 'pool_indices': indices, 'selected_indices': selected,
            'pool_size': len(indices), 'pool_words': sum(r['word_count'] for r in pool),
            'selected_words': sum(len(t.split()) for t in texts), 'selected_size': len(selected),
            'source_words': sum(len(s['text'].split()) for s in source), 'source_size': len(source),
            'eligible_size': sum(0 < len(s['text'].split()) <= cap for s in source),
            'eligible_words': sum(len(s['text'].split()) for s in source if 0 < len(s['text'].split()) <= cap),
            'source_document_count': len(doc['documents']),
            'pool_document_count': len({r['document_id'] for r in pool}),
            'feasible': pred['feasible'], 'infeasible_code': pred['infeasible_code']}


def verify_frozen_inputs():
    freeze = readj(OUT/'protocol/execution_freeze.json')
    for record in freeze['inputs']:
        if sha(ROOT/record['path']) != record['sha256']:
            raise ValueError(f'frozen input changed: {record["path"]}')
    return freeze
