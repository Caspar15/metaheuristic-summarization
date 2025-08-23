from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.utils.io import read_jsonl
from src.utils.logging import setup_logging
from src.eval.rouge import compute_rouge
from src.eval.report import write_report


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate system outputs using ROUGE")
    parser.add_argument("--config", type=str, required=True, help="Path to eval config YAML")
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="configs/dataset_cnn_dm.yaml",
        help="Path to dataset config YAML (to locate split)",
    )
    args = parser.parse_args()

    setup_logging()
    log = logging.getLogger("evaluate_run")

    root = Path(__file__).resolve().parents[2]
    ecfg = load_yaml(root / args.config)
    dcfg = load_yaml(root / args.dataset_config)

    eval_types = list(ecfg.get("eval", {}).get("rouge_types", ["rouge1", "rouge2", "rougeL"]))
    use_stemmer = bool(ecfg.get("eval", {}).get("use_stemmer", True))
    out_dir = str(ecfg.get("eval", {}).get("output", {}).get("dir", "runs"))

    split = dcfg["dataset"].get("split", "sample")
    in_path = root / f"data/interim/{split}_system.jsonl"

    records = list(read_jsonl(in_path))
    log.info("Evaluating %d docs", len(records))
    items: List[Dict[str, Any]] = []
    for r in records:
        sys = r.get("system", "")
        ref = r.get("reference", "")
        metrics = compute_rouge(system=sys, reference=ref, types=eval_types, use_stemmer=use_stemmer)
        items.append({"id": r["id"], "system": sys, "reference": ref, "metrics": metrics})

    paths = write_report(items, out_dir=out_dir)
    log.info("Wrote predictions to %s and metrics to %s", paths["predictions"], paths["metrics"])


if __name__ == "__main__":
    main()
