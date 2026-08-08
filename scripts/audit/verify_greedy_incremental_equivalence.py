"""Post-F-30 real-pipeline selected-index equivalence audit.

The audit is intentionally fixed to GovReport frozen validation-dev L00 and
the pre-F-30 artifact whose digest was frozen before this script existed.  It
runs selection only: references are not evaluated and there is no CLI path to
dev-test or a dataset test split.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

from scripts.audit.run_length_contract_study import (
    REPO_ROOT,
    _git_commit,
    _selected_indices_digest,
    _utc_now,
)
from src.data.policy import sha256_file
from src.pipeline.select_sentences import summarize_jsonl
from src.utils.io import load_yaml, read_jsonl, set_global_seed


INPUT = REPO_ROOT / "data/processed/govreport_validation_canonical.jsonl"
CONFIG = (
    REPO_ROOT
    / "runs_v2/d1_greedy_sensitivity/govreport/dev/lexical_objective/"
    "L00_base/resolved_config.yaml"
)
EXPECTED = (
    REPO_ROOT
    / "runs_v2/d1_greedy_sensitivity/govreport/dev/lexical_objective/"
    "L00_base/greedy/run/predictions.jsonl"
)
OUTPUT_ROOT = REPO_ROOT / "runs_v2/f30_greedy_equivalence/govreport_l00_post_f30"
EXPECTED_SELECTED_INDICES_SHA256 = (
    "8273f16296008019ccd566240ae868e46d841dcb1e04cfa4435b8fc72b4d7982"
)


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise ValueError(f"refusing to overwrite equivalence audit: {OUTPUT_ROOT}")
    if "test" in INPUT.name.lower() or "test" in CONFIG.name.lower():
        raise ValueError("equivalence audit refuses filenames containing 'test'")
    config = load_yaml(str(CONFIG))
    partition = config.get("experiment_partition", {})
    if partition.get("name") != "dev":
        raise ValueError("equivalence audit requires frozen dev partition")
    if config.get("experiment", {}).get("dataset") != "GovReport":
        raise ValueError("equivalence audit is fixed to GovReport")

    expected_rows = list(read_jsonl(str(EXPECTED)))
    expected_digest = _selected_indices_digest(expected_rows)
    if expected_digest != EXPECTED_SELECTED_INDICES_SHA256:
        raise ValueError("frozen pre-F-30 selected-indices digest drifted")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    predictions = OUTPUT_ROOT / "predictions.jsonl"
    set_global_seed(config.get("seed"))
    started = _utc_now()
    before = time.perf_counter()
    processed = summarize_jsonl(str(INPUT), str(predictions), config, "validation")
    elapsed = time.perf_counter() - before
    actual_rows = list(read_jsonl(str(predictions)))
    actual_digest = _selected_indices_digest(actual_rows)

    expected_by_id = {
        row["id"]: list(row.get("selected_indices", [])) for row in expected_rows
    }
    actual_by_id = {
        row["id"]: list(row.get("selected_indices", [])) for row in actual_rows
    }
    differing_ids = sorted(
        identifier
        for identifier in set(expected_by_id) | set(actual_by_id)
        if expected_by_id.get(identifier) != actual_by_id.get(identifier)
    )
    equivalent = (
        processed == 681
        and actual_digest == EXPECTED_SELECTED_INDICES_SHA256
        and not differing_ids
    )
    evidence = {
        "evidence_schema_version": "1.0",
        "measured_at_utc": _utc_now(),
        "started_at_utc": started,
        "status": "passed" if equivalent else "failed",
        "audit": "post_f30_real_pipeline_selected_indices_equivalence",
        "implementation_commit": _git_commit(),
        "dataset": "GovReport",
        "base_split": "validation",
        "partition": "dev",
        "rows": processed,
        "references_evaluated": False,
        "dev_test_accessed": False,
        "test_split_accessed": False,
        "config_path": CONFIG.relative_to(REPO_ROOT).as_posix(),
        "config_sha256": sha256_file(str(CONFIG)),
        "input_path": INPUT.relative_to(REPO_ROOT).as_posix(),
        "input_sha256": sha256_file(str(INPUT)),
        "expected_predictions_path": EXPECTED.relative_to(REPO_ROOT).as_posix(),
        "expected_selected_indices_sha256": expected_digest,
        "actual_predictions_path": predictions.relative_to(REPO_ROOT).as_posix(),
        "actual_predictions_sha256": sha256_file(str(predictions)),
        "actual_selected_indices_sha256": actual_digest,
        "differing_row_count": len(differing_ids),
        "differing_ids": differing_ids,
        "selection_wall_seconds": elapsed,
        "equivalent": equivalent,
    }
    (OUTPUT_ROOT / "evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    if not equivalent:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
