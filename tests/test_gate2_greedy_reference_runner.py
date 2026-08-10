"""Correctness and governance tests for the Gate 2 greedy-reference runner."""

import pytest

from scripts.audit.run_gate2_greedy_reference import (
    _aggregate,
    _bounded_ordered_results,
    _checkpoint_prefix_length,
    _evaluate_row,
    _exclusive_run_lock,
    _log_interruption,
    _load_protocol,
    _validate_checkpoint_prefix,
    build_parser,
)
from src.data.schemas import build_document_example
from src.eval.oracle import greedy_reference_run


def _row(row_id, sentences, reference):
    return build_document_example(
        example_id=row_id,
        split="validation",
        documents=[sentences],
        references=[reference],
        input_mode="single_document",
        output_mode="multi_sentence",
        dataset_name="Toy",
    )


def test_preregistered_runner_has_no_split_argument():
    protocol = _load_protocol()
    assert protocol["partition"] == "dev"
    assert protocol["test_split_prohibited"] is True
    assert "split" not in {action.dest for action in build_parser()._actions}


def test_document_parallel_unit_is_exactly_equivalent_to_corpus_api():
    docs = [
        _row("a", ["alpha beta", "gamma delta"], "alpha beta gamma"),
        _row("b", ["one two", "two three"], "one two three"),
    ]
    target = "rouge2"
    expected = greedy_reference_run(docs, max_words=5, target_metric=target)
    actual_rows = [
        _evaluate_row((index, row, target, 5)) for index, row in enumerate(docs)
    ]
    actual = _aggregate(actual_rows)
    assert [row["selected_indices"] for row in actual_rows] == [
        row["selected_indices"] for row in expected["selections"]
    ]
    assert actual["rouge"] == pytest.approx(expected["scores"], abs=0.0)


def test_document_parallel_worker_is_spawn_safe_and_ordered():
    docs = [
        _row("a", ["alpha beta", "gamma delta"], "alpha beta gamma"),
        _row("b", ["one two", "two three"], "one two three"),
    ]
    tasks = [(index, row, "rouge1", 5) for index, row in enumerate(docs)]
    rows = list(_bounded_ordered_results(tasks, workers=2))
    assert [row["id"] for row in rows] == ["a", "b"]
    assert [row["position"] for row in rows] == [0, 1]


def test_runner_defaults_to_two_resource_safe_workers():
    args = build_parser().parse_args(["--dataset", "govreport", "--target", "rougeLsum"])
    assert args.workers == 2


def test_interruption_logging_is_idempotent_for_same_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scripts.audit.run_length_contract_study.REPO_ROOT", tmp_path
    )
    rows_path = tmp_path / "run" / "rows.jsonl"
    rows_path.parent.mkdir(parents=True)
    rows_path.write_text('{"position": 0}\n', encoding="utf-8")
    logged = []
    monkeypatch.setattr(
        "scripts.audit.run_gate2_greedy_reference._append_search_log", logged.append
    )
    kwargs = {
        "dataset_label": "Toy",
        "target": "rougeLsum",
        "config_hash": "abc",
        "rows_path": rows_path,
        "rows": 1,
    }
    _log_interruption(**kwargs)
    _log_interruption(**kwargs)
    evidence = list((rows_path.parent / "attempts").glob("*/interruption_evidence.json"))
    assert len(evidence) == 1
    assert len(logged) == 1


def test_checkpoint_requires_exact_frozen_prefix_and_target():
    checkpoint = [
        {
            "position": 0,
            "id": "b",
            "optimization_target": "rouge1",
            "scores": {"rouge1": 1.0, "rouge2": 1.0, "rougeLsum": 1.0},
        }
    ]
    with pytest.raises(ValueError, match="exact frozen ID prefix"):
        _validate_checkpoint_prefix(checkpoint, ["a", "b"], "rouge1")


def test_checkpoint_prefix_length_stops_before_duplicate_writer_row():
    checkpoint = [
        {
            "position": 0,
            "id": "a",
            "optimization_target": "rouge1",
            "scores": {"rouge1": 1.0, "rouge2": 1.0, "rougeLsum": 1.0},
        },
        {
            "position": 0,
            "id": "a",
            "optimization_target": "rouge1",
            "scores": {"rouge1": 1.0, "rouge2": 1.0, "rougeLsum": 1.0},
        },
    ]
    assert _checkpoint_prefix_length(checkpoint, ["a", "b"], "rouge1") == 1


def test_exclusive_writer_lock_rejects_second_holder(tmp_path):
    lock_path = tmp_path / "greedy.lock"
    with _exclusive_run_lock(lock_path):
        with pytest.raises(RuntimeError, match="another greedy-reference writer"):
            with _exclusive_run_lock(lock_path):
                pass
