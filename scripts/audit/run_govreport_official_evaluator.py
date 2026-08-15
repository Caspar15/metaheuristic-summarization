"""Run the GovReport authors' released Stanza + Perl ROUGE protocol.

This adapter intentionally evaluates immutable prediction artifacts.  It does
not import, instantiate, or rerun a summarizer, and it refuses partial ID
alignment.  ``-d`` asks Perl ROUGE to emit per-evaluation rows for paired
inference; a smoke check proves it does not change corpus Average_F values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

import stanza
from pyrouge import Rouge155
from pyrouge.utils import log as pyrouge_log
from stanza.pipeline.core import DownloadMethod

from src.data.schemas import extract_references
from src.eval.paired import holm_adjust, paired_bootstrap_difference
from src.utils.io import read_jsonl


REPO_ROOT = Path(__file__).resolve().parents[2]
PREREG = REPO_ROOT / "configs/preregistrations/govreport_centered_evidence_completion_v1.json"
EXECUTION_ADDENDUM = REPO_ROOT / "configs/preregistrations/govreport_official_evaluator_execution_addendum_v1.json"
WINDOWS_RUNTIME_ADDENDUM = REPO_ROOT / "configs/preregistrations/govreport_official_evaluator_windows_runtime_addendum_v1.json"
PARTITION = REPO_ROOT / "configs/validation_partitions/govreport_validation_dev_v1.json"
CANONICAL = REPO_ROOT / "data/processed/govreport_validation_canonical.jsonl"
OUTPUT_ROOT = REPO_ROOT / "runs_v2/govreport_official_evaluator_v1"
METRIC_NAMES = ("rouge1", "rouge2", "rougeL")
ROUGE_LABELS = {"ROUGE-1": "rouge1", "ROUGE-2": "rouge2", "ROUGE-L": "rougeL"}


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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def parse_rouge_output(text: str) -> tuple[dict[str, float], dict[int, dict[str, float]]]:
    averages: dict[str, float] = {}
    per_eval: dict[int, dict[str, float]] = {}
    average_pattern = re.compile(
        r"^\S+\s+(ROUGE-(?:1|2|L))\s+Average_F:\s+([0-9.eE+-]+)"
    )
    eval_pattern = re.compile(
        r"^\S+\s+(ROUGE-(?:1|2|L))\s+Eval\s+(\d+)\.\S+\s+"
        r"R:[0-9.eE+-]+\s+P:[0-9.eE+-]+\s+F:([0-9.eE+-]+)"
    )
    for raw_line in text.splitlines():
        line = raw_line.strip()
        average_match = average_pattern.match(line)
        if average_match:
            averages[ROUGE_LABELS[average_match.group(1)]] = float(average_match.group(2))
            continue
        eval_match = eval_pattern.match(line)
        if eval_match:
            metric = ROUGE_LABELS[eval_match.group(1)]
            per_eval.setdefault(int(eval_match.group(2)), {})[metric] = float(
                eval_match.group(3)
            )
    if set(averages) != set(METRIC_NAMES):
        raise ValueError(f"could not parse all corpus ROUGE metrics: {averages}")
    return averages, per_eval


def _tokenize(nlp: stanza.Pipeline, text: str) -> str:
    document = nlp(text.strip())
    return "\n".join(
        " ".join(token.text for token in sentence.tokens)
        for sentence in document.sentences
    )


def _load_frozen_rows() -> tuple[list[str], dict[str, str]]:
    manifest = json.loads(PARTITION.read_text(encoding="utf-8"))
    ordered_ids = list(manifest["partitions"]["dev"]["selected_ids"])
    selected = set(ordered_ids)
    references: dict[str, str] = {}
    for row in read_jsonl(str(CANONICAL)):
        row_id = row.get("id")
        if row_id not in selected:
            continue
        refs = extract_references(row)
        if len(refs) != 1:
            raise ValueError(f"GovReport E1 requires exactly one reference for {row_id!r}")
        references[str(row_id)] = refs[0]
    if set(references) != selected:
        raise ValueError("canonical GovReport rows do not match the frozen dev manifest")
    return ordered_ids, references


def _load_predictions(path: Path, ordered_ids: Sequence[str]) -> list[str]:
    by_id: dict[str, str] = {}
    for row in read_jsonl(str(path)):
        row_id = row.get("id")
        summary = row.get("summary")
        if not isinstance(row_id, str) or not isinstance(summary, str):
            raise ValueError(f"invalid prediction row in {path}")
        if row_id in by_id:
            raise ValueError(f"duplicate prediction ID {row_id!r}")
        by_id[row_id] = summary
    if set(by_id) != set(ordered_ids):
        raise ValueError(f"prediction IDs do not exactly match frozen GovReport dev: {path}")
    return [by_id[row_id] for row_id in ordered_ids]


def _tokenize_files(
    nlp: stanza.Pipeline,
    texts: Sequence[str],
    output_dir: Path,
    extension: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob(f"*.{extension}"))
    if existing:
        if len(existing) != len(texts):
            raise ValueError(f"partial tokenization cache at {output_dir}")
        return
    for index, text in enumerate(texts):
        (output_dir / f"{index}.{extension}").write_text(
            _tokenize(nlp, text), encoding="utf-8", newline="\n"
        )


def _prepare_rouge_directory(plain_dir: Path, rouge_dir: Path, rows: int) -> None:
    existing = list(rouge_dir.glob("*")) if rouge_dir.exists() else []
    if existing:
        if len(existing) != rows:
            raise ValueError(f"partial pyrouge conversion cache at {rouge_dir}")
        return
    Rouge155.convert_summaries_to_rouge_format(str(plain_dir), str(rouge_dir))


def _run_perl(
    *,
    perl: Path,
    rouge_home: Path,
    system_rouge: Path,
    reference_rouge: Path,
    config_path: Path,
    detailed: bool,
) -> tuple[str, str, float, list[str]]:
    Rouge155.write_config_static(
        str(system_rouge),
        r"(\d+).dec",
        str(reference_rouge),
        "#ID#.ref",
        str(config_path),
        1,
    )
    command = [
        str(perl),
        str(rouge_home / "ROUGE-1.5.5.pl"),
        "-e",
        str(rouge_home / "data"),
        "-c",
        "95",
        "-r",
        "1000",
        "-n",
        "2",
        "-m",
    ]
    if detailed:
        command.append("-d")
    command.extend(["-a", str(config_path)])
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"Perl ROUGE failed ({completed.returncode}): {completed.stderr[-2000:]}"
        )
    return completed.stdout, completed.stderr, elapsed, command


def _system_specs() -> dict[str, dict[str, str]]:
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    specs: dict[str, dict[str, str]] = {}
    for label, pair in prereg["work_packages"]["E1_published_evaluator"]["systems"].items():
        specs[label] = {"path": pair[0], "sha256": pair[1]}
    return specs


def _evaluate_system(
    label: str,
    spec: Mapping[str, str],
    *,
    nlp: stanza.Pipeline,
    ordered_ids: Sequence[str],
    reference_rouge: Path,
    rouge_home: Path,
    perl: Path,
    perl_repo_root: Path | None,
    smoke_parity: bool,
) -> dict[str, Any]:
    root = OUTPUT_ROOT / label
    evidence_path = root / "evidence.json"
    if evidence_path.is_file():
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if evidence.get("status") != "completed":
            raise ValueError(f"preserved failed E1 evidence exists for {label}")
        return evidence
    prediction_path = REPO_ROOT / spec["path"]
    if _sha256(prediction_path) != spec["sha256"]:
        raise ValueError(f"frozen prediction SHA drifted for {label}")
    predictions = _load_predictions(prediction_path, ordered_ids)
    plain_dir = root / "tokenized_plain"
    rouge_dir = root / "rouge_workspace" / "system"
    _tokenize_files(nlp, predictions, plain_dir, "dec")
    _prepare_rouge_directory(plain_dir, rouge_dir, len(ordered_ids))
    config = root / "rouge_workspace" / "settings.xml"
    if perl_repo_root is not None:
        perl_rouge_home = perl_repo_root / rouge_home.relative_to(REPO_ROOT)
        perl_system_rouge = perl_repo_root / rouge_dir.relative_to(REPO_ROOT)
        perl_reference_rouge = perl_repo_root / reference_rouge.relative_to(REPO_ROOT)
        perl_config = perl_repo_root / config.relative_to(REPO_ROOT)
    else:
        perl_rouge_home = rouge_home
        perl_system_rouge = rouge_dir
        perl_reference_rouge = reference_rouge
        perl_config = config
    raw, stderr, elapsed, command = _run_perl(
        perl=perl,
        rouge_home=perl_rouge_home,
        system_rouge=perl_system_rouge,
        reference_rouge=perl_reference_rouge,
        config_path=perl_config,
        detailed=True,
    )
    averages, per_eval = parse_rouge_output(raw)
    if len(per_eval) != len(ordered_ids) or any(
        set(values) != set(METRIC_NAMES) for values in per_eval.values()
    ):
        raise ValueError(f"Perl -d output did not contain 3 metrics x all rows for {label}")

    # pyrouge lexicographically sorts numeric filenames.  Preserve that exact
    # official behavior when mapping task IDs back to canonical document IDs.
    lexical_file_order = sorted(f"{index}.dec" for index in range(len(ordered_ids)))
    task_to_id = {
        task_id: ordered_ids[int(filename.removesuffix(".dec"))]
        for task_id, filename in enumerate(lexical_file_order, start=1)
    }
    per_rows = []
    for task_id in range(1, len(ordered_ids) + 1):
        scores = per_eval[task_id]
        per_rows.append(
            {
                "id": task_to_id[task_id],
                **scores,
                "macro_rouge": fmean(scores[name] for name in METRIC_NAMES),
            }
        )
    per_rows.sort(key=lambda row: ordered_ids.index(row["id"]))
    per_path = root / "per_example.jsonl"
    _write_jsonl(per_path, per_rows)
    raw_path = root / "rouge_raw.txt"
    raw_path.write_text(raw, encoding="utf-8", newline="\n")
    (root / "rouge_stderr.txt").write_text(stderr, encoding="utf-8", newline="\n")

    smoke = None
    if smoke_parity:
        aggregate_raw, _, _, aggregate_command = _run_perl(
            perl=perl,
            rouge_home=perl_rouge_home,
            system_rouge=perl_system_rouge,
            reference_rouge=perl_reference_rouge,
            config_path=(
                perl_repo_root / (root / "rouge_workspace" / "settings_no_d.xml").relative_to(REPO_ROOT)
                if perl_repo_root is not None
                else root / "rouge_workspace" / "settings_no_d.xml"
            ),
            detailed=False,
        )
        aggregate_averages, _ = parse_rouge_output(aggregate_raw)
        if aggregate_averages != averages:
            raise ValueError("ROUGE -d changed corpus Average_F values")
        smoke = {"passed": True, "without_d_command": aggregate_command}

    evidence = {
        "evidence_schema_version": "1.0",
        "measured_at_utc": _utc_now(),
        "status": "completed",
        "study_id": "govreport-official-evaluator-v1",
        "system": label,
        "partition": "GovReport frozen dev",
        "rows": len(ordered_ids),
        "prediction_path": spec["path"],
        "prediction_sha256": spec["sha256"],
        "metrics": {**averages, "macro_rouge": fmean(averages.values())},
        "per_example_path": per_path.relative_to(REPO_ROOT).as_posix(),
        "per_example_sha256": _sha256(per_path),
        "raw_output_path": raw_path.relative_to(REPO_ROOT).as_posix(),
        "raw_output_sha256": _sha256(raw_path),
        "execution_seconds": elapsed,
        "command": command,
        "d_flag_corpus_parity_smoke": smoke,
        "dev_test_accessed": False,
        "test_split_accessed": False,
    }
    _write_json(evidence_path, evidence)
    return evidence


def _analyze(results: Mapping[str, Mapping[str, Any]], ordered_ids: Sequence[str]) -> dict:
    rows_by_system: dict[str, dict[str, list[float]]] = {}
    for label, evidence in results.items():
        rows = list(read_jsonl(str(REPO_ROOT / evidence["per_example_path"])))
        if [row["id"] for row in rows] != list(ordered_ids):
            raise ValueError(f"official per-row order drifted for {label}")
        rows_by_system[label] = {
            metric: [float(row[metric]) for row in rows]
            for metric in (*METRIC_NAMES, "macro_rouge")
        }

    proposed = rows_by_system["proposed"]
    comparator = rows_by_system["sbert_mmr_lambda_0.9"]
    primary = paired_bootstrap_difference(
        proposed["macro_rouge"], comparator["macro_rouge"],
        n_resamples=100_000, seed=20260815,
    )
    components: dict[str, dict[str, Any]] = {}
    raw_component_p: dict[str, float] = {}
    for offset, metric in enumerate(METRIC_NAMES, start=1):
        result = paired_bootstrap_difference(
            proposed[metric], comparator[metric],
            n_resamples=100_000, seed=20260815 + offset,
        )
        components[metric] = result
        raw_component_p[metric] = float(result["p_value_two_sided"])
    adjusted_components = holm_adjust(raw_component_p)
    for metric, value in adjusted_components.items():
        components[metric]["p_value_holm_3"] = value

    secondary: dict[str, dict[str, Any]] = {}
    raw_secondary_p: dict[str, float] = {}
    endpoint = 0
    for label in sorted(rows_by_system):
        if label == "proposed":
            continue
        secondary[label] = {}
        for metric in (*METRIC_NAMES, "macro_rouge"):
            result = paired_bootstrap_difference(
                proposed[metric], rows_by_system[label][metric],
                n_resamples=100_000, seed=20261815 + endpoint,
            )
            key = f"{label}:{metric}"
            secondary[label][metric] = result
            raw_secondary_p[key] = float(result["p_value_two_sided"])
            endpoint += 1
    adjusted = holm_adjust(raw_secondary_p)
    for label, metrics in secondary.items():
        for metric, result in metrics.items():
            result["p_value_holm_32"] = adjusted[f"{label}:{metric}"]

    macro_pass = (
        float(primary["mean_difference"]) > 0
        and float(primary["ci_lower"]) > 0
        and float(primary["p_value_two_sided"]) <= 0.05
    )
    component_guard = all(float(result["ci_upper"]) >= 0 for result in components.values())
    return {
        "evidence_schema_version": "1.0",
        "measured_at_utc": _utc_now(),
        "status": "completed",
        "study_id": "govreport-official-evaluator-v1",
        "partition": "GovReport frozen dev",
        "rows": len(ordered_ids),
        "corpus_ranking": sorted(
            ({"system": label, **dict(evidence["metrics"])} for label, evidence in results.items()),
            key=lambda row: (-float(row["macro_rouge"]), row["system"]),
        ),
        "primary_proposed_vs_sbert_mmr_lambda_0.9": {
            "macro": primary,
            "components": components,
            "macro_pass": macro_pass,
            "component_guard_pass": component_guard,
            "official_dev_comparison_survives": macro_pass and component_guard,
        },
        "secondary_holm_32": secondary,
        "decision": (
            "retain GovReport superiority claim for the later untouched-test gate"
            if macro_pass and component_guard
            else "revoke or downgrade GovReport superiority claim; no tuning"
        ),
        "dev_test_accessed": False,
        "test_split_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stanza-resources", required=True)
    parser.add_argument("--rouge-home", required=True)
    parser.add_argument("--perl", default="perl")
    parser.add_argument(
        "--perl-repo-root",
        default=None,
        help="optional ASCII-only alias of the repository root (for Windows Perl)",
    )
    parser.add_argument("--only", action="append", default=[])
    args = parser.parse_args()

    # The official LongDocSum wrapper silences pyrouge's per-file INFO log.
    pyrouge_log.get_global_console_logger().setLevel("WARNING")

    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg.get("test_split_accessed") is not False:
        raise ValueError("E1 preregistration lost its protected-test guard")
    addendum = json.loads(EXECUTION_ADDENDUM.read_text(encoding="utf-8"))
    if addendum.get("scores_observed_before_registration") is not False:
        raise ValueError("E1 execution addendum is not score-blind")
    runtime_addendum = json.loads(
        WINDOWS_RUNTIME_ADDENDUM.read_text(encoding="utf-8")
    )
    if runtime_addendum.get("official_scores_observed_before_registration") is not False:
        raise ValueError("E1 Windows runtime addendum is not score-blind")
    rouge_home = Path(args.rouge_home)
    if not rouge_home.is_absolute():
        rouge_home = REPO_ROOT / rouge_home
    perl_repo_root = Path(args.perl_repo_root) if args.perl_repo_root else None
    if perl_repo_root is not None:
        if not perl_repo_root.is_absolute() or not perl_repo_root.exists():
            raise ValueError("--perl-repo-root must be an existing absolute path")
        if _sha256(
            perl_repo_root / EXECUTION_ADDENDUM.relative_to(REPO_ROOT)
        ) != _sha256(EXECUTION_ADDENDUM):
            raise ValueError("ASCII Perl repo alias does not resolve to this frozen repository")
    perl = Path(args.perl) if os.path.sep in args.perl else Path(
        subprocess.check_output(["where", args.perl], text=True).splitlines()[0]
    )
    expected_script_sha = addendum["execution_dependencies"]["rouge_1_5_5_pl_sha256"]
    if _sha256(rouge_home / "ROUGE-1.5.5.pl") != expected_script_sha:
        raise ValueError("ROUGE-1.5.5.pl SHA does not match frozen E1 addendum")
    expected_db_sha = runtime_addendum["wordnet_runtime_db"][
        "platform_native_db_sha256"
    ]
    if _sha256(rouge_home / "data/WordNet-2.0.exc.db") != expected_db_sha:
        raise ValueError("platform-native WordNet DB SHA does not match frozen addendum")

    ordered_ids, references = _load_frozen_rows()
    nlp = stanza.Pipeline(
        lang="en", processors="tokenize,mwt", use_gpu=False,
        model_dir=args.stanza_resources, download_method=DownloadMethod.NONE,
        verbose=False,
    )
    shared_plain = OUTPUT_ROOT / "_shared" / "reference_plain"
    shared_rouge = OUTPUT_ROOT / "_shared" / "reference_rouge"
    _tokenize_files(nlp, [references[row_id] for row_id in ordered_ids], shared_plain, "ref")
    _prepare_rouge_directory(shared_plain, shared_rouge, len(ordered_ids))

    specs = _system_specs()
    selected = args.only or list(specs)
    unknown = set(selected) - set(specs)
    if unknown:
        raise ValueError(f"unknown E1 systems: {sorted(unknown)}")
    results: dict[str, dict[str, Any]] = {}
    for index, label in enumerate(selected):
        results[label] = _evaluate_system(
            label, specs[label], nlp=nlp, ordered_ids=ordered_ids,
            reference_rouge=shared_rouge, rouge_home=rouge_home, perl=perl,
            perl_repo_root=perl_repo_root,
            smoke_parity=index == 0,
        )
        print(json.dumps({"system": label, "status": results[label]["status"]}))
    if set(results) == set(specs):
        analysis = _analyze(results, ordered_ids)
        _write_json(OUTPUT_ROOT / "analysis.json", analysis)
        _write_json(
            OUTPUT_ROOT / "environment.json",
            {
                "measured_at_utc": _utc_now(),
                "platform": platform.platform(),
                "python": sys.version,
                "implementation_commit": _git_commit(),
                "stanza": version("stanza"),
                "pyrouge": version("pyrouge"),
                "perl": str(perl),
                "rouge_home": str(rouge_home),
                "preregistration_sha256": _sha256(PREREG),
                "execution_addendum_sha256": _sha256(EXECUTION_ADDENDUM),
                "windows_runtime_addendum_sha256": _sha256(WINDOWS_RUNTIME_ADDENDUM),
                "partition_manifest_sha256": _sha256(PARTITION),
                "canonical_input_sha256": _sha256(CANONICAL),
                "dev_test_accessed": False,
                "test_split_accessed": False,
            },
        )
        print(json.dumps({"decision": analysis["decision"]}))


if __name__ == "__main__":
    main()
