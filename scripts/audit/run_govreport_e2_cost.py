"""Measure preregistered GovReport E2 cold/warm runtime, memory, and scaling."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import psutil
import yaml

from scripts.audit.run_greedy_sensitivity import REPO_ROOT
from src.baselines.centrality import summarize_one_lexrank
from src.baselines.lead import summarize_one_lead
from src.baselines.pacsum import summarize_one_pacsum_sbert, summarize_one_pacsum_tfidf
from src.baselines.semantic import summarize_one_sbert_centroid, summarize_one_sbert_mmr
from src.data.schemas import flatten_sentence_records
from src.pipeline.select_sentences import summarize_one
from src.utils.io import read_jsonl, set_global_seed


ADDENDUM = REPO_ROOT / "configs/preregistrations/govreport_centered_cost_addendum_v1.json"
CACHE_ERRATUM = REPO_ROOT / "configs/preregistrations/govreport_e2_cache_classification_erratum_v1.json"
SAMPLE = REPO_ROOT / "configs/pilot_manifests/govreport_cost_scaling_sample_v1.json"
CANONICAL = REPO_ROOT / "data/processed/govreport_validation_canonical.jsonl"
OUTPUT_ROOT = REPO_ROOT / "runs_v2/govreport_cost_scaling_v1"
SEARCH_LOG = REPO_ROOT / "runs_v2/search_log.jsonl"
PLM_SYSTEMS = {
    "frozen_C01_proposed",
    "full_source_sbert_mmr_lambda_0.9",
    # These selector labels are TF-IDF, but their frozen S02b candidate
    # generators still execute the enabled semantic route and write embeddings.
    "D2_matched_greedy_tfidf_anchor",
    "D2_matched_nsga2_tfidf",
    "pacsum_sbert_beta_0.5",
    "sbert_centroid",
}
SYSTEM_ORDER = (
    "frozen_C01_proposed",
    "full_source_sbert_mmr_lambda_0.9",
    "D2_matched_greedy_tfidf_anchor",
    "D2_matched_nsga2_tfidf",
    "lead",
    "lexrank",
    "pacsum_tfidf_P07",
    "pacsum_sbert_beta_0.5",
    "sbert_centroid",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if not path.exists():
        return digest.hexdigest()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(item)))
    return digest.hexdigest()


def _tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _append_search_log(evidence: Mapping[str, Any]) -> None:
    """Record every newly executed E2 attempt without treating timing as quality search."""
    entry = {
        "logged_at_utc": _utc_now(),
        "study_id": "govreport-cost-scaling-v1",
        "dataset": "GovReport",
        "partition": "dev_reference_blind_30_document_cost_sample",
        "family": "cost_scaling_evidence_only",
        "candidate": (
            f"{evidence['system']}:{evidence['state']}:"
            f"{evidence['role']}:{evidence['repetition']}"
        ),
        "config_path": evidence["config_path"],
        "config_hash": evidence["config_sha256"],
        "dev_score": None,
        "dev_test_score": None,
        "status": evidence["status"],
        "promoted": False,
        "reason": "E2 cost evidence only; timing cannot tune or promote a method",
        "test_split_accessed": False,
    }
    SEARCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    with SEARCH_LOG.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def _ids_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        payload = json.dumps(
            {"id": row["id"], "selected_indices": row["selected_indices"]},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update(payload.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _specs() -> dict[str, dict[str, str]]:
    addendum = json.loads(ADDENDUM.read_text(encoding="utf-8"))
    specs = {
        **addendum["core_systems"],
        **addendum["descriptive_baseline_systems"],
    }
    return specs


def _validate_cache_classification(specs: Mapping[str, Mapping[str, str]]) -> None:
    erratum = json.loads(CACHE_ERRATUM.read_text(encoding="utf-8"))
    affected = erratum["static_config_finding"]["affected_systems"]
    for system in affected:
        if system not in PLM_SYSTEMS:
            raise ValueError(f"semantic-route E2 system is not cache-aware: {system}")
        config = _load_config(REPO_ROOT / specs[system]["config_path"])
        if (
            "semantic" not in config["compute_budget"]["enabled_routes"]
            or not isinstance(config["routes"].get("semantic"), dict)
        ):
            raise ValueError(f"E2 cache erratum no longer matches frozen config: {system}")


def _selected_rows() -> tuple[list[dict[str, Any]], dict[str, str]]:
    manifest = json.loads(SAMPLE.read_text(encoding="utf-8"))
    ordered_ids = list(manifest["selected_ids"])
    strata = {row["id"]: row["stratum"] for row in manifest["rows"]}
    wanted = set(ordered_ids)
    by_id: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(str(CANONICAL)):
        if row.get("id") in wanted:
            by_id[str(row["id"])] = row
    if set(by_id) != wanted:
        raise ValueError("E2 sample IDs are missing from canonical GovReport")
    return [by_id[row_id] for row_id in ordered_ids], strata


def _runner(system: str) -> Callable[[Mapping[str, Any], Mapping[str, Any]], dict]:
    if system in {
        "frozen_C01_proposed",
        "D2_matched_greedy_tfidf_anchor",
        "D2_matched_nsga2_tfidf",
    }:
        return summarize_one
    if system == "full_source_sbert_mmr_lambda_0.9":
        return summarize_one_sbert_mmr
    if system == "lead":
        return lambda doc, cfg: summarize_one_lead(
            doc, cfg, ordering="document_order", first_k=3
        )
    if system == "lexrank":
        return summarize_one_lexrank
    if system == "pacsum_tfidf_P07":
        return summarize_one_pacsum_tfidf
    if system == "pacsum_sbert_beta_0.5":
        return summarize_one_pacsum_sbert
    if system == "sbert_centroid":
        return summarize_one_sbert_centroid
    raise ValueError(f"unknown E2 system {system!r}")


def _embedding_cache_records(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        cache = value.get("embedding_cache")
        if isinstance(cache, Mapping):
            found.append(dict(cache))
        for child in value.values():
            found.extend(_embedding_cache_records(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_embedding_cache_records(child))
    return found


def worker(system: str, output: Path, cache_dir: Path) -> None:
    specs = _specs()
    _validate_cache_classification(specs)
    spec = specs[system]
    config_path = REPO_ROOT / spec["config_path"]
    if _sha256(config_path) != spec["config_sha256"]:
        raise ValueError(f"E2 config SHA drifted for {system}")
    config = _load_config(config_path)
    set_global_seed(config.get("seed"))
    os.environ["META_SUM_EMBEDDING_CACHE_DIR"] = str(cache_dir.resolve())
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    rows, strata = _selected_rows()
    summarize = _runner(system)
    measured_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for doc in rows:
        sentence_records = flatten_sentence_records(doc)
        doc_started = time.perf_counter()
        result = summarize(doc, config)
        elapsed = time.perf_counter() - doc_started
        cache_records = _embedding_cache_records(result)
        statuses = sorted({str(record.get("status")) for record in cache_records})
        sections = {
            (record.get("document_id"), record.get("section_id"))
            for record in sentence_records
            if record.get("section_id") is not None
        }
        measured_rows.append(
            {
                "id": doc["id"],
                "stratum": strata[doc["id"]],
                "source_words": sum(len(record["text"].split()) for record in sentence_records),
                "source_sentences": len(sentence_records),
                "source_sections": len(sections),
                "candidate_count": len(result.get("candidate_records") or []),
                "selected_count": len(result["selected_indices"]),
                "selected_indices": result["selected_indices"],
                "document_wall_seconds": elapsed,
                "embedding_cache_statuses": statuses,
            }
        )
    payload = {
        "worker_schema_version": "1.0",
        "completed_at_utc": _utc_now(),
        "system": system,
        "rows": len(measured_rows),
        "worker_wall_seconds": time.perf_counter() - started,
        "selected_indices_sha256": _ids_digest(measured_rows),
        "documents": measured_rows,
        "dev_test_accessed": False,
        "test_split_accessed": False,
    }
    _write_json(output, payload)


def _monitor(command: Sequence[str], run_root: Path, environment: Mapping[str, str]) -> dict[str, Any]:
    run_root.mkdir(parents=True, exist_ok=True)
    stdout_path = run_root / "command.stdout.log"
    stderr_path = run_root / "command.stderr.log"
    trace_path = run_root / "rss_trace.jsonl"
    started_at = _utc_now()
    wall_started = time.perf_counter()
    max_cpu_by_pid: dict[int, float] = {}
    peak_rss = 0
    samples = 0
    with stdout_path.open("w", encoding="utf-8", newline="\n") as stdout, stderr_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as stderr, trace_path.open("w", encoding="utf-8", newline="\n") as trace:
        process = subprocess.Popen(
            list(command), cwd=REPO_ROOT, stdout=stdout, stderr=stderr,
            text=True, encoding="utf-8", errors="replace", env=dict(environment),
        )
        root = psutil.Process(process.pid)
        while process.poll() is None:
            current = []
            try:
                current = [root, *root.children(recursive=True)]
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            rss = 0
            pids = []
            for observed in current:
                try:
                    rss += observed.memory_info().rss
                    cpu = observed.cpu_times()
                    max_cpu_by_pid[observed.pid] = max(
                        max_cpu_by_pid.get(observed.pid, 0.0), cpu.user + cpu.system
                    )
                    pids.append(observed.pid)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            peak_rss = max(peak_rss, rss)
            trace.write(json.dumps({
                "elapsed_seconds": time.perf_counter() - wall_started,
                "process_tree_rss_bytes": rss,
                "pids": pids,
            }) + "\n")
            samples += 1
            time.sleep(0.05)
        return_code = process.wait()
    return {
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "return_code": return_code,
        "wall_seconds": time.perf_counter() - wall_started,
        "cpu_process_tree_seconds": sum(max_cpu_by_pid.values()),
        "peak_process_tree_rss_bytes": peak_rss,
        "rss_sample_interval_ms": 50,
        "rss_trace_samples": samples,
        "rss_trace_path": trace_path.relative_to(REPO_ROOT).as_posix(),
        "rss_trace_sha256": _sha256(trace_path),
        "stdout_path": stdout_path.relative_to(REPO_ROOT).as_posix(),
        "stderr_path": stderr_path.relative_to(REPO_ROOT).as_posix(),
        "command": list(command),
    }


def _run_once(
    system: str,
    state: str,
    role: str,
    repetition: int,
    cache_dir: Path,
    *,
    expected_warm_cache_sha: str | None,
) -> dict[str, Any]:
    run_root = OUTPUT_ROOT / system / state / f"{role}_{repetition:02d}"
    evidence_path = run_root / "evidence.json"
    if evidence_path.is_file():
        value = json.loads(evidence_path.read_text(encoding="utf-8"))
        if value.get("status") != "completed":
            raise ValueError(f"preserved failed E2 attempt exists: {evidence_path}")
        return value
    cache_dir.mkdir(parents=True, exist_ok=True)
    before_sha = _tree_sha256(cache_dir)
    before_bytes = _tree_bytes(cache_dir)
    if state == "cold" and before_bytes != 0:
        raise ValueError(f"cold E2 cache is not empty: {cache_dir}")
    if state == "warm_cache" and system in PLM_SYSTEMS:
        if before_bytes == 0 or before_sha != expected_warm_cache_sha:
            raise ValueError(f"warm E2 cache is not byte-verified for {system}")
    worker_output = run_root / "worker_output.json"
    command = [
        sys.executable,
        "-m",
        "scripts.audit.run_govreport_e2_cost",
        "--worker",
        "--system",
        system,
        "--output",
        str(worker_output),
        "--cache-dir",
        str(cache_dir),
    ]
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    measurement = _monitor(command, run_root, environment)
    after_sha = _tree_sha256(cache_dir)
    after_bytes = _tree_bytes(cache_dir)
    status = "completed" if measurement["return_code"] == 0 and worker_output.is_file() else "failed"
    worker_payload = json.loads(worker_output.read_text(encoding="utf-8")) if worker_output.is_file() else None
    cache_valid = True
    if state == "warm_cache" and system in PLM_SYSTEMS:
        cache_valid = before_sha == after_sha == expected_warm_cache_sha
    evidence = {
        "evidence_schema_version": "1.0",
        "measured_at_utc": _utc_now(),
        "status": status if cache_valid else "failed",
        "study_id": "govreport-cost-scaling-v1",
        "system": system,
        "state": state,
        "role": role,
        "repetition": repetition,
        "sample_manifest_path": SAMPLE.relative_to(REPO_ROOT).as_posix(),
        "sample_manifest_sha256": _sha256(SAMPLE),
        "implementation_commit": _git_commit(),
        "config_path": _specs()[system]["config_path"],
        "config_sha256": _specs()[system]["config_sha256"],
        "rows": 30,
        "measurement": measurement,
        "cache": {
            "applicable": system in PLM_SYSTEMS,
            "path": cache_dir.relative_to(REPO_ROOT).as_posix(),
            "before_sha256": before_sha,
            "after_sha256": after_sha,
            "before_bytes": before_bytes,
            "after_bytes": after_bytes,
            "warm_byte_identity_preserved": cache_valid,
        },
        "worker_output_path": worker_output.relative_to(REPO_ROOT).as_posix(),
        "worker_output_sha256": _sha256(worker_output) if worker_output.is_file() else None,
        "selected_indices_sha256": worker_payload.get("selected_indices_sha256") if worker_payload else None,
        "discarded_from_statistics": role != "measured",
        "dev_test_accessed": False,
        "test_split_accessed": False,
    }
    _write_json(evidence_path, evidence)
    _append_search_log(evidence)
    if evidence["status"] != "completed":
        raise RuntimeError(f"E2 attempt failed; preserved at {evidence_path}")
    return evidence


def _prime_warm_cache(system: str, cache_dir: Path) -> str | None:
    if system not in PLM_SYSTEMS:
        return None
    prime = _run_once(system, "cold", "warm_prime", 0, cache_dir, expected_warm_cache_sha=None)
    cache_sha = prime["cache"]["after_sha256"]
    if prime["cache"]["after_bytes"] <= 0:
        raise ValueError(f"warm cache prime wrote no bytes for {system}")
    return str(cache_sha)


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
    }


def analyze(evidence: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]) -> dict[str, Any]:
    systems: dict[str, Any] = {}
    for system, states in evidence.items():
        systems[system] = {}
        for state, runs in states.items():
            measured = [run for run in runs if run["role"] == "measured"]
            if len(measured) != 3:
                raise ValueError(f"E2 requires three measured repetitions: {system}/{state}")
            digests = {run["selected_indices_sha256"] for run in measured}
            if len(digests) != 1:
                raise ValueError(f"E2 selected indices changed across repetitions: {system}/{state}")
            totals = {
                field: _quantiles([float(run["measurement"][field]) for run in measured])
                for field in ("wall_seconds", "cpu_process_tree_seconds", "peak_process_tree_rss_bytes")
            }
            documents: dict[str, list[dict[str, Any]]] = {"q10": [], "q50": [], "q90": []}
            all_docs: list[dict[str, Any]] = []
            for run in measured:
                worker = json.loads((REPO_ROOT / run["worker_output_path"]).read_text(encoding="utf-8"))
                for row in worker["documents"]:
                    documents[row["stratum"]].append(row)
                    all_docs.append(row)
            strata = {
                label: {
                    "document_wall_seconds": _quantiles([float(row["document_wall_seconds"]) for row in rows]),
                    "source_sentences": _quantiles([float(row["source_sentences"]) for row in rows]),
                    "source_words": _quantiles([float(row["source_words"]) for row in rows]),
                }
                for label, rows in documents.items()
            }
            x_sentences = np.log([max(1, row["source_sentences"]) for row in all_docs])
            x_words = np.log([max(1, row["source_words"]) for row in all_docs])
            y = np.log([max(1e-9, row["document_wall_seconds"]) for row in all_docs])
            systems[system][state] = {
                "measured_repetitions": 3,
                "selected_indices_sha256": next(iter(digests)),
                "total_run": totals,
                "strata": strata,
                "descriptive_log_log_slopes": {
                    "wall_vs_source_sentences": float(np.polyfit(x_sentences, y, 1)[0]),
                    "wall_vs_source_words": float(np.polyfit(x_words, y, 1)[0]),
                },
            }
    return {
        "evidence_schema_version": "1.0",
        "measured_at_utc": _utc_now(),
        "status": "completed",
        "study_id": "govreport-cost-scaling-v1",
        "systems": systems,
        "interpretation": "Cost evidence only; no timing outcome can promote or tune a method.",
        "dev_test_accessed": False,
        "test_split_accessed": False,
    }


def run(selected_systems: Sequence[str], *, resume: bool) -> dict[str, Any] | None:
    addendum = json.loads(ADDENDUM.read_text(encoding="utf-8"))
    if addendum.get("timings_observed_before_registration") is not False:
        raise ValueError("E2 addendum is not timing-blind")
    if _sha256(SAMPLE) != addendum["sample_manifest"]["sha256"]:
        raise ValueError("E2 sample manifest SHA drifted")
    specs = _specs()
    unknown = set(selected_systems) - set(specs)
    if unknown:
        raise ValueError(f"unknown E2 systems: {sorted(unknown)}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    collected: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for system in selected_systems:
        collected[system] = {"cold": [], "warm_cache": []}
        warm_cache = OUTPUT_ROOT / system / "warm_cache" / "_embedding_cache"
        warm_sha = _prime_warm_cache(system, warm_cache)
        for state in ("cold", "warm_cache"):
            for role, repetition in [("smoke", 0), *[("measured", i) for i in range(1, 4)]]:
                cache_dir = (
                    OUTPUT_ROOT / system / state / f"{role}_{repetition:02d}" / "_embedding_cache"
                    if state == "cold"
                    else warm_cache
                )
                evidence = _run_once(
                    system, state, role, repetition, cache_dir,
                    expected_warm_cache_sha=warm_sha,
                )
                collected[system][state].append(evidence)
                print(json.dumps({
                    "system": system,
                    "state": state,
                    "role": role,
                    "repetition": repetition,
                    "wall_seconds": evidence["measurement"]["wall_seconds"],
                }))
    if set(selected_systems) == set(SYSTEM_ORDER):
        analysis = analyze(collected)
        _write_json(OUTPUT_ROOT / "analysis.json", analysis)
        _write_json(OUTPUT_ROOT / "environment.json", {
            "measured_at_utc": _utc_now(),
            "platform": platform.platform(),
            "python": sys.version,
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "total_memory_bytes": psutil.virtual_memory().total,
            "no_overlapping_timed_jobs": True,
            "dev_test_accessed": False,
            "test_split_accessed": False,
        })
        return analysis
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--system", action="append", default=[])
    parser.add_argument("--output")
    parser.add_argument("--cache-dir")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.worker:
        if len(args.system) != 1 or not args.output or not args.cache_dir:
            parser.error("worker mode requires exactly one --system, --output, --cache-dir")
        worker(args.system[0], Path(args.output), Path(args.cache_dir))
        return
    selected = args.system or list(SYSTEM_ORDER)
    run(selected, resume=args.resume)


if __name__ == "__main__":
    main()
