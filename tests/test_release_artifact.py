import json

import pytest

from scripts.release.build_numerical_artifact import export_selection, validate_scores


def test_selection_export_omits_corpus_and_summary_text(tmp_path):
    path = tmp_path / 'predictions.jsonl'
    path.write_text(json.dumps({'id': 'a', 'selected_indices': [3, 1],
                               'summary': 'Example source text.', 'selected_sentences': ['private corpus text'],
                               'feasible': True}) + '\n', encoding='utf-8')
    rows, ids, digest = export_selection(path, 1)
    assert ids == ['a']
    assert rows[0]['selected_indices'] == [3, 1]
    assert rows[0]['summary_words'] == 3
    assert 'Example' not in json.dumps(rows) and 'private corpus' not in json.dumps(rows)
    assert len(digest) == 64


def test_selection_export_rejects_duplicate_population(tmp_path):
    path = tmp_path / 'predictions.jsonl'
    row = json.dumps({'id': 'a', 'selected_indices': [0], 'summary': 'Text', 'feasible': True}) + '\n'
    path.write_text(row * 2, encoding='utf-8')
    with pytest.raises(ValueError, match='duplicate IDs'):
        export_selection(path, 2)


def test_score_export_rejects_text_and_id_misalignment(tmp_path):
    path = tmp_path / 'per_example.jsonl'
    path.write_text(json.dumps({'id':'a','rouge1':0.5})+'\n', encoding='utf-8')
    assert validate_scores(path, ['a']) == path.read_bytes()
    with pytest.raises(ValueError, match='ID/order'):
        validate_scores(path, ['b'])
    path.write_text(json.dumps({'id':'a','rouge1':0.5,'reference':'Corpus text'})+'\n', encoding='utf-8')
    with pytest.raises(ValueError, match='possible text'):
        validate_scores(path, ['a'])
