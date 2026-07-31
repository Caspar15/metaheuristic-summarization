"""Provenance test for the baseline CLI.

Pins down PR #10 review item 4: before this, two Lead runs against the same
config but different ``--ordering`` produced run directories with no way to
tell them apart. ``baseline_run.json`` must record ``--baseline``,
``--ordering``, ``--first_k``, and (for Random) ``--seed`` at the run
level -- the same shape of problem, just recurring for a different flag.
"""

import json
import sys

import pytest

from src.baselines import cli as baseline_cli
from src.data.schemas import build_document_example
from src.utils.io import read_jsonl, write_jsonl_atomic


def _toy_doc():
    return build_document_example(
        example_id="cli_doc1",
        split="validation",
        documents=[["one two three", "four five six seven"]],
        references=["a reference"],
        input_mode="single_document",
        output_mode="multi_sentence",
        dataset_name="toy",
    )


def test_main_writes_baseline_run_provenance(tmp_path, monkeypatch):
    input_path = tmp_path / "toy.jsonl"
    write_jsonl_atomic(str(input_path), [_toy_doc()])

    config_path = tmp_path / "toy_config.yaml"
    config_path.write_text(
        "length_control:\n"
        "  unit: words\n"
        "  max_words: 50\n"
        "  min_words: 0\n"
        "  require_nonempty: true\n",
        encoding="utf-8",
    )

    run_dir = tmp_path / "runs"
    argv = [
        "baselines-cli",
        "--baseline", "lead",
        "--config", str(config_path),
        "--split", "validation",
        "--input", str(input_path),
        "--run_dir", str(run_dir),
        "--stamp", "test-stamp",
        "--ordering", "round_robin",
        "--first_k", "2",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    baseline_cli.main()

    provenance_path = run_dir / "test-stamp" / "baseline_run.json"
    assert provenance_path.exists()
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["baseline"] == "lead"
    assert provenance["ordering"] == "round_robin"
    assert provenance["first_k"] == 2
    assert provenance["split"] == "validation"
    assert provenance["input"] == str(input_path)


def test_main_provenance_distinguishes_orderings_for_the_same_config(tmp_path, monkeypatch):
    input_path = tmp_path / "toy.jsonl"
    write_jsonl_atomic(str(input_path), [_toy_doc()])

    config_path = tmp_path / "toy_config.yaml"
    config_path.write_text(
        "length_control:\n"
        "  unit: words\n"
        "  max_words: 50\n"
        "  min_words: 0\n"
        "  require_nonempty: true\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "runs"

    def run(stamp, ordering):
        argv = [
            "baselines-cli",
            "--baseline", "lead",
            "--config", str(config_path),
            "--split", "validation",
            "--input", str(input_path),
            "--run_dir", str(run_dir),
            "--stamp", stamp,
            "--ordering", ordering,
        ]
        monkeypatch.setattr(sys, "argv", argv)
        baseline_cli.main()

    run("stamp-a", "document_order")
    run("stamp-b", "round_robin")

    provenance_a = json.loads((run_dir / "stamp-a" / "baseline_run.json").read_text(encoding="utf-8"))
    provenance_b = json.loads((run_dir / "stamp-b" / "baseline_run.json").read_text(encoding="utf-8"))
    assert provenance_a["ordering"] == "document_order"
    assert provenance_b["ordering"] == "round_robin"


def _random_cfg_path(tmp_path):
    config_path = tmp_path / "random_config.yaml"
    config_path.write_text(
        "length_control:\n"
        "  unit: words\n"
        "  max_words: 50\n"
        "  min_words: 0\n"
        "  require_nonempty: true\n",
        encoding="utf-8",
    )
    return config_path


def _run_cli(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", argv)
    baseline_cli.main()


def test_random_without_seed_fails_loud(tmp_path, monkeypatch):
    input_path = tmp_path / "toy.jsonl"
    write_jsonl_atomic(str(input_path), [_toy_doc()])
    config_path = _random_cfg_path(tmp_path)
    run_dir = tmp_path / "runs"

    argv = [
        "baselines-cli",
        "--baseline", "random",
        "--config", str(config_path),
        "--split", "validation",
        "--input", str(input_path),
        "--run_dir", str(run_dir),
        "--stamp", "no-seed",
    ]
    with pytest.raises(ValueError, match="seed"):
        _run_cli(monkeypatch, argv)

    # Fails before touching the filesystem: no run directory left behind.
    assert not (run_dir / "no-seed").exists()


def test_lead_with_seed_fails_loud(tmp_path, monkeypatch):
    input_path = tmp_path / "toy.jsonl"
    write_jsonl_atomic(str(input_path), [_toy_doc()])
    config_path = _random_cfg_path(tmp_path)
    run_dir = tmp_path / "runs"

    argv = [
        "baselines-cli",
        "--baseline", "lead",
        "--config", str(config_path),
        "--split", "validation",
        "--input", str(input_path),
        "--run_dir", str(run_dir),
        "--stamp", "lead-with-seed",
        "--seed", "7",
    ]
    with pytest.raises(ValueError, match="seed"):
        _run_cli(monkeypatch, argv)

    assert not (run_dir / "lead-with-seed").exists()


def test_random_baseline_run_records_seed_matching_every_row(tmp_path, monkeypatch):
    input_path = tmp_path / "toy.jsonl"
    write_jsonl_atomic(str(input_path), [_toy_doc()])
    config_path = _random_cfg_path(tmp_path)
    run_dir = tmp_path / "runs"

    argv = [
        "baselines-cli",
        "--baseline", "random",
        "--config", str(config_path),
        "--split", "validation",
        "--input", str(input_path),
        "--run_dir", str(run_dir),
        "--stamp", "random-run",
        "--seed", "42",
    ]
    _run_cli(monkeypatch, argv)

    provenance = json.loads((run_dir / "random-run" / "baseline_run.json").read_text(encoding="utf-8"))
    assert provenance["baseline"] == "random"
    assert provenance["seed"] == 42

    rows = list(read_jsonl(str(run_dir / "random-run" / "predictions.jsonl")))
    assert len(rows) == 1
    assert rows[0]["seed"] == 42
    assert isinstance(rows[0]["row_seed"], int)


def test_random_same_config_same_seed_reruns_produce_byte_identical_predictions(tmp_path, monkeypatch):
    """CLI-level check of Gate 1's "same seed reruns to the same indices":
    two independent invocations, same config and --seed, must write
    byte-identical predictions.jsonl files -- not just equal indices."""

    input_path = tmp_path / "toy.jsonl"
    write_jsonl_atomic(str(input_path), [_toy_doc()])
    config_path = _random_cfg_path(tmp_path)
    run_dir = tmp_path / "runs"

    for stamp in ("rerun-a", "rerun-b"):
        argv = [
            "baselines-cli",
            "--baseline", "random",
            "--config", str(config_path),
            "--split", "validation",
            "--input", str(input_path),
            "--run_dir", str(run_dir),
            "--stamp", stamp,
            "--seed", "1234",
        ]
        _run_cli(monkeypatch, argv)

    bytes_a = (run_dir / "rerun-a" / "predictions.jsonl").read_bytes()
    bytes_b = (run_dir / "rerun-b" / "predictions.jsonl").read_bytes()
    assert bytes_a == bytes_b
