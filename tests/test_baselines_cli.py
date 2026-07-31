"""Provenance test for the baseline CLI.

Pins down PR #10 review item 4: before this, two Lead runs against the same
config but different ``--ordering`` produced run directories with no way to
tell them apart. ``baseline_run.json`` must record ``--baseline``,
``--ordering``, and ``--first_k`` at the run level.
"""

import json
import sys

from src.baselines import cli as baseline_cli
from src.data.schemas import build_document_example
from src.utils.io import write_jsonl_atomic


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
