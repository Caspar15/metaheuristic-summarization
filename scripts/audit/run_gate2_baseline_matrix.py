"""Run the preregistered Gate 2 baseline matrix on frozen dev only.

There is intentionally no split argument and no code path for dev-test or
test.  ``non_plm`` and ``plm`` are separate resumable stages so expensive
encoder work can be scheduled independently.  Every attempted candidate is
appended to ``runs_v2/search_log.jsonl``, including failures.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import yaml

from scripts.audit.run_length_contract_study import (
    REPO_ROOT,
    _append_search_log,
    _canonical_sha256,
    _git_commit,
    _load_gold,
    _load_search_log,
    _relative,
    _run_method,
    _utc_now,
    _write_json,
)
from src.data.partitions import selected_ids_sha256
from src.data.policy import sha256_file
from src.utils.io import load_yaml


PREREGISTRATION = "configs/preregistrations/gate2_baseline_matrix_v1.json"
PREREGISTRATION_SHA256 = "b6a38e53d2f25f6340d2ed7b9552d56e6b159748093f77922faa42c0b41078bd"
STUDIES: dict[str, dict[str, str]] = {
    "multinews": {
        "dataset_label": "Multi-News",
        "input": "data/processed/multi_news_validation_canonical.jsonl",
        "base_config": "configs/studies/d1/multinews_base.yaml",
        "manifest": "configs/validation_partitions/multinews_validation_dev_v1.json",
        "manifest_sha256": "e61405482cda203c0bd50dda3e958986b11a51b48124129e68617f97ce9e42ee",
    },
    "govreport": {
        "dataset_label": "GovReport",
        "input": "data/processed/govreport_validation_canonical.jsonl",
        "base_config": "configs/studies/d1/govreport_base.yaml",
        "manifest": "configs/validation_partitions/govreport_validation_dev_v1.json",
        "manifest_sha256": "7a15ffbb87abe690fe4e72a1e0daf27bf34b3a3293371983ae8e362d06e2717e",
    },
}


def _pacsum_variants(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    method = str(spec["method"])
    default = dict(spec["default"])
    variants = [
        {"id": f"{method}_B00_default", "method": method, "pacsum": default}
    ]
    beta_values = [float(value) for value in spec["ofat"]["beta"]]
    previous_values = [float(value) for value in spec["ofat"]["lambda_previous"]]
    if beta_values[0] != 0.0 or previous_values[0] != 0.0:
        raise ValueError("PacSum OFAT grids must start at the frozen default")
    for index, beta in enumerate(beta_values[1:], start=1):
        values = copy.deepcopy(default)
        values["beta"] = beta
        variants.append(
            {"id": f"{method}_B{index:02d}_beta_{beta:.1f}", "method": method, "pacsum": values}
        )
    for index, previous in enumerate(previous_values[1:], start=1):
        values = copy.deepcopy(default)
        values["lambda_previous"] = previous
        values["lambda_following"] = 1.0 + previous
        variants.append(
            {"id": f"{method}_P{index:02d}_previous_{previous:.1f}", "method": method, "pacsum": values}
        )
    if len(variants) != 21:
        raise ValueError(f"PacSum preregistration must expand to 21 candidates, got {len(variants)}")
    return variants


def expand_variants(preregistration: Mapping[str, Any], family: str) -> list[dict[str, Any]]:
    spec = preregistration["families"][family]
    variants = [{"id": method, "method": method} for method in spec.get("fixed", [])]
    if family == "plm":
        for value in spec["sbert_mmr"]["lambda_relevance"]:
            variants.append(
                {
                    "id": f"sbert_mmr_lambda_{float(value):.1f}",
                    "method": "sbert_mmr",
                    "sbert_mmr": {"lambda_relevance": float(value)},
                }
            )
    variants.extend(_pacsum_variants(spec["pacsum"]))
    expected = int(preregistration["expected_candidate_counts_per_dataset"][family])
    if len(variants) != expected:
        raise ValueError(f"{family} must expand to {expected} candidates, got {len(variants)}")
    ids = [variant["id"] for variant in variants]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate Gate 2 candidate ID in {family}")
    return variants


def load_protocol() -> dict[str, Any]:
    path = REPO_ROOT / PREREGISTRATION
    actual = sha256_file(str(path))
    if actual != PREREGISTRATION_SHA256:
        raise ValueError(f"frozen Gate 2 preregistration drifted: {actual}")
    prereg = json.loads(path.read_text(encoding="utf-8"))
    if prereg.get("status") != "frozen_before_gate2_baseline_scores":
        raise ValueError("Gate 2 preregistration is not frozen before scores")
    if prereg.get("frozen_partition") != "dev":
        raise ValueError("Gate 2 baseline search may read only frozen dev")
    if prereg.get("dev_test_access") != "none_in_baseline_search":
        raise ValueError("Gate 2 baseline search must prohibit dev-test")
    if prereg.get("test_split_prohibited") is not True:
        raise ValueError("Gate 2 baseline search must prohibit test")
    expand_variants(prereg, "non_plm")
    expand_variants(prereg, "plm")
    return prereg


def _resolved_config(
    base: Mapping[str, Any],
    *,
    spec: Mapping[str, str],
    family: str,
    variant: Mapping[str, Any],
    candidate_hash: str,
) -> dict[str, Any]:
    config = copy.deepcopy(dict(base))
    config["experiment_partition"] = {
        "manifest_path": spec["manifest"],
        "manifest_sha256": spec["manifest_sha256"],
        "name": "dev",
    }
    config.setdefault("baselines", {})
    config["baselines"].setdefault("pacsum", {})
    config["baselines"]["pacsum"].update(
        {
            "beta": 0.0,
            "lambda_previous": 0.0,
            "lambda_following": 1.0,
            "tfidf": {
                "lowercase": True,
                "sublinear_tf": False,
                "ngram_range": [1, 1],
                "stop_words": None,
            },
        }
    )
    config["baselines"].setdefault("sbert_mmr", {"lambda_relevance": 0.7})
    if "pacsum" in variant:
        config["baselines"]["pacsum"].update(copy.deepcopy(variant["pacsum"]))
    if "sbert_mmr" in variant:
        config["baselines"]["sbert_mmr"].update(copy.deepcopy(variant["sbert_mmr"]))
    config["study"] = {
        "study_id": "gate2-baseline-matrix-v1",
        "family": family,
        "candidate": variant["id"],
        "candidate_hash": candidate_hash,
        "partition": "dev",
        "preregistration_path": PREREGISTRATION,
        "preregistration_sha256": PREREGISTRATION_SHA256,
    }
    return config


def _archive_partial(candidate_root: Path, method: str, context: Mapping[str, Any]) -> None:
    run_path = candidate_root / method / "run"
    if not run_path.exists():
        return
    attempts = candidate_root / method / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    number = 1
    while (attempts / f"attempt_{number:02d}_interrupted").exists():
        number += 1
    destination = attempts / f"attempt_{number:02d}_interrupted"
    shutil.move(str(run_path), str(destination))
    evidence = {
        "evidence_schema_version": "1.0",
        "measured_at_utc": _utc_now(),
        "status": "failed",
        "failure_type": "external_runner_interruption",
        "failure": "Incomplete run preserved before an explicit --resume retry.",
        "dev_test_accessed": False,
        "test_split_accessed": False,
        "archived_run_path": _relative(destination),
        **dict(context),
    }
    _write_json(destination / "interruption_evidence.json", evidence)


def _append_result_log(
    *,
    spec: Mapping[str, str],
    family: str,
    variant: Mapping[str, Any],
    candidate_hash: str,
    config_path: Path,
    config_sha256: str,
    result: Mapping[str, Any],
) -> None:
    if any(
        row.get("study_id") == "gate2-baseline-matrix-v1"
        and row.get("dataset") == spec["dataset_label"]
        and row.get("candidate_hash") == candidate_hash
        and row.get("status") == result["status"]
        for row in _load_search_log()
    ):
        return
    _append_search_log(
        {
            "logged_at_utc": _utc_now(),
            "study_id": "gate2-baseline-matrix-v1",
            "dataset": spec["dataset_label"],
            "partition": "dev",
            "family": family,
            "candidate": variant["id"],
            "method": variant["method"],
            "candidate_hash": candidate_hash,
            "config_path": _relative(config_path),
            "config_hash": config_sha256,
            "run_attempt": "final",
            "dev_score": (
                result.get("metrics", {}).get("macro_rouge")
                if result["status"] == "completed"
                else None
            ),
            "dev_test_score": None,
            "status": result["status"],
            "promoted": False,
            "reason": (
                "pending complete Gate 2 baseline selection"
                if result["status"] == "completed"
                else result.get("failure")
            ),
            "comparison_family_size": 50,
            "dev_test_accessed": False,
            "test_split_accessed": False,
        }
    )


def run_family(dataset: str, family: str, *, resume: bool = False) -> dict[str, Any]:
    prereg = load_protocol()
    variants = expand_variants(prereg, family)
    spec = copy.deepcopy(STUDIES[dataset])
    manifest_path = REPO_ROOT / spec["manifest"]
    if sha256_file(str(manifest_path)) != spec["manifest_sha256"]:
        raise ValueError("frozen Gate 2 manifest SHA-256 drifted")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    partition = manifest["partitions"]["dev"]
    ordered_ids = partition["selected_ids"]
    if selected_ids_sha256(ordered_ids) != partition["selected_ids_sha256"]:
        raise ValueError("frozen Gate 2 dev selected-ID digest is inconsistent")
    input_path = REPO_ROOT / spec["input"]
    gold = _load_gold(input_path, ordered_ids)
    base = load_yaml(str(REPO_ROOT / spec["base_config"]))
    output_root = REPO_ROOT / "runs_v2" / "gate2_baseline_matrix_v1" / dataset / "dev" / family
    if output_root.exists() and not resume:
        raise ValueError(f"refusing to overwrite existing Gate 2 family: {output_root}")
    output_root.mkdir(parents=True, exist_ok=resume)

    results: dict[str, Any] = {}
    for variant in variants:
        candidate_hash = _canonical_sha256(
            {
                "study_id": "gate2-baseline-matrix-v1",
                "dataset": dataset,
                "family": family,
                "candidate": variant,
                "base_config": base,
                "partition": "dev",
                "evaluation": prereg["evaluation"],
            }
        )
        candidate_root = output_root / variant["id"]
        summary_path = candidate_root / "candidate_summary.json"
        if summary_path.exists():
            if not resume:
                raise ValueError(f"refusing to overwrite candidate: {candidate_root}")
            results[variant["id"]] = json.loads(summary_path.read_text(encoding="utf-8"))
            continue
        candidate_root.mkdir(parents=True, exist_ok=resume)
        config = _resolved_config(
            base,
            spec=spec,
            family=family,
            variant=variant,
            candidate_hash=candidate_hash,
        )
        config_path = candidate_root / "resolved_config.yaml"
        expected = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
        if config_path.exists():
            if config_path.read_text(encoding="utf-8") != expected:
                raise ValueError(f"resume config drift for {variant['id']}")
        else:
            config_path.write_text(expected, encoding="utf-8", newline="\n")
        config_sha256 = sha256_file(str(config_path))
        context = {
            "study_id": "gate2-baseline-matrix-v1",
            "dataset": spec["dataset_label"],
            "partition": "dev",
            "partition_rows": len(ordered_ids),
            "partition_manifest_path": spec["manifest"],
            "partition_manifest_sha256": spec["manifest_sha256"],
            "partition_selected_ids_sha256": partition["selected_ids_sha256"],
            "preregistration_path": PREREGISTRATION,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "family": family,
            "candidate": variant["id"],
            "candidate_hash": candidate_hash,
            "declared_variant": variant,
            "dev_test_accessed": False,
        }
        if resume:
            _archive_partial(candidate_root, variant["method"], context)
        try:
            metrics, _ = _run_method(
                variant["method"],
                config_path=config_path,
                config_sha256=config_sha256,
                input_path=input_path,
                candidate_root=candidate_root,
                ordered_ids=ordered_ids,
                gold=gold,
                study_context=context,
            )
            result = {
                "status": "completed",
                "candidate": variant["id"],
                "method": variant["method"],
                "candidate_hash": candidate_hash,
                "config_path": _relative(config_path),
                "config_sha256": config_sha256,
                "metrics": metrics,
                "dev_test_accessed": False,
                "test_split_accessed": False,
            }
        except Exception as error:
            result = {
                "status": "failed",
                "candidate": variant["id"],
                "method": variant["method"],
                "candidate_hash": candidate_hash,
                "config_path": _relative(config_path),
                "config_sha256": config_sha256,
                "failure": f"{type(error).__name__}: {error}",
                "dev_test_accessed": False,
                "test_split_accessed": False,
            }
        _write_json(summary_path, result)
        _append_result_log(
            spec=spec,
            family=family,
            variant=variant,
            candidate_hash=candidate_hash,
            config_path=config_path,
            config_sha256=config_sha256,
            result=result,
        )
        results[variant["id"]] = result

    summary = {
        "evidence_schema_version": "1.0",
        "measured_at_utc": _utc_now(),
        "status": "completed_with_failures" if any(v["status"] == "failed" for v in results.values()) else "completed",
        "study_id": "gate2-baseline-matrix-v1",
        "implementation_commit": _git_commit(),
        "dataset": spec["dataset_label"],
        "partition": "dev",
        "family": family,
        "preregistration_path": PREREGISTRATION,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "candidate_count": len(variants),
        "completed_count": sum(v["status"] == "completed" for v in results.values()),
        "failed_count": sum(v["status"] == "failed" for v in results.values()),
        "results": results,
        "dev_test_accessed": False,
        "test_split_accessed": False,
    }
    _write_json(output_root / "family_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(STUDIES))
    parser.add_argument("--family", required=True, choices=("non_plm", "plm"))
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_family(args.dataset, args.family, resume=args.resume)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
