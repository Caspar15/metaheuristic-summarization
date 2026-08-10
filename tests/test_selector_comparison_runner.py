import hashlib
import json

import pytest

from scripts.audit.freeze_selector_pilot import freeze_manifest
from scripts.audit.aggregate_nsga_seed_stability import selection_stability
from scripts.audit.run_selector_comparison import load_frozen_rows
from src.eval.paired import holm_adjust, paired_bootstrap_difference


def _write_rows(path, ids):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row_id in ids:
            handle.write(json.dumps({"id": row_id}) + "\n")


def test_manifest_is_reference_blind_deterministic_hash_sample(tmp_path):
    input_path = tmp_path / "rows.jsonl"
    _write_rows(input_path, ["a", "b", "c", "d"])
    first = freeze_manifest(input_path, sample_size=2, salt="frozen")
    second = freeze_manifest(input_path, sample_size=2, salt="frozen")
    assert first == second
    expected = sorted(
        ["a", "b", "c", "d"],
        key=lambda row_id: hashlib.sha256(
            f"frozen\0{row_id}".encode("utf-8")
        ).hexdigest(),
    )[:2]
    assert set(first["selected_ids"]) == set(expected)


def test_load_frozen_rows_rejects_tampered_manifest(tmp_path):
    input_path = tmp_path / "rows.jsonl"
    _write_rows(input_path, ["a", "b", "c"])
    manifest = freeze_manifest(input_path, sample_size=2, salt="frozen")
    manifest["selected_ids"][0] = "tampered"
    with pytest.raises(ValueError, match="digest"):
        load_frozen_rows(input_path, manifest)


def test_paired_bootstrap_is_aligned_and_directional():
    result = paired_bootstrap_difference(
        [0.5, 0.6, 0.7],
        [0.4, 0.5, 0.6],
        n_resamples=1000,
        seed=7,
    )
    assert result["mean_difference"] == pytest.approx(0.1)
    assert result["ci_lower"] == pytest.approx(0.1)
    assert result["ci_upper"] == pytest.approx(0.1)
    assert result["p_value_two_sided"] < 0.01


def test_holm_adjust_is_monotone_in_sorted_order():
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.02})
    assert adjusted == pytest.approx({"a": 0.03, "c": 0.04, "b": 0.04})


def test_selection_stability_reports_pairwise_jaccard_and_exact_rate():
    result = selection_stability(
        {
            "seed1": {"a": [1, 2], "b": [3]},
            "seed2": {"a": [1, 2], "b": [3, 4]},
            "seed3": {"a": [1, 2], "b": [4]},
        }
    )
    assert result["all_seeds_identical_rate"] == pytest.approx(0.5)
    # row a contributes three 1.0 values; row b contributes 0.5, 0.0, 0.5.
    assert result["mean_pairwise_jaccard"] == pytest.approx(2 / 3)
    assert result["max_unique_selection_sets_per_row"] == 3
