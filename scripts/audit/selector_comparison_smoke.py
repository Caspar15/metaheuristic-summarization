"""Run a bounded, non-ROUGE matched-selector correctness/cost smoke."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from copy import deepcopy
from pathlib import Path

from src.pipeline.select_sentences import summarize_one, validate_requested_split
from src.utils.io import load_yaml, read_jsonl, set_global_seed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_smoke(input_path: Path, config_path: Path, methods, limit: int) -> dict:
    if limit < 1:
        raise ValueError("limit must be positive")
    cfg = load_yaml(str(config_path))
    rows = []
    for index, doc in enumerate(read_jsonl(str(input_path))):
        if index >= limit:
            break
        validate_requested_split(doc, "validation")
        rows.append(doc)
    if not rows:
        raise ValueError("input contains no rows")

    results = {method: [] for method in methods}
    timing = {}
    for method in methods:
        method_cfg = deepcopy(cfg)
        method_cfg.setdefault("optimizer", {})["method"] = method
        set_global_seed(method_cfg.get("seed"))
        started = time.perf_counter()
        for doc in rows:
            prediction = summarize_one(doc, method_cfg)
            results[method].append(
                {
                    "id": prediction["id"],
                    "selected_indices": prediction["selected_indices"],
                    "feasible": prediction["feasible"],
                    "violations": prediction["violations"],
                    "selection_evaluation": prediction["selection_evaluation"],
                    "selector_inputs": prediction["selector_inputs"],
                    "optimizer_diagnostics": prediction["optimizer_diagnostics"],
                }
            )
        timing[method] = time.perf_counter() - started

    matched = []
    reference_method = methods[0]
    for row_index, doc in enumerate(rows):
        reference = results[reference_method][row_index]["selector_inputs"]
        equality = {
            method: results[method][row_index]["selector_inputs"] == reference
            for method in methods
        }
        if not all(equality.values()):
            raise RuntimeError(
                f"selector inputs differ for row {doc.get('id')!r}: {equality}"
            )
        matched.append({"id": doc.get("id"), "all_inputs_equal": True})

    return {
        "status": "diagnostic_no_rouge",
        "purpose": "matched selector correctness and cost smoke only",
        "input": str(input_path),
        "input_sha256": _sha256(input_path),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "methods": list(methods),
        "rows": len(rows),
        "row_ids": [row.get("id") for row in rows],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "pymoo": importlib.metadata.version("pymoo"),
        },
        "timing_scope": (
            "ordered smoke only: the first method includes encoder cold load; "
            "later methods reuse the process-global model cache, so these "
            "numbers are not a fair cross-method runtime comparison"
        ),
        "method_order": list(methods),
        "model_cache_shared_across_methods": True,
        "time_seconds": timing,
        "matched_input_checks": matched,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument(
        "--methods", nargs="+", default=["greedy", "mmr", "nsga2"]
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = run_smoke(
        Path(args.input), Path(args.config), args.methods, args.limit
    )
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
