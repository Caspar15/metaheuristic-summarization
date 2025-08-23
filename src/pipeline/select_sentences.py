from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import yaml

from src.utils.io import read_jsonl, write_jsonl, ensure_dir
from src.utils.logging import setup_logging
from src.models.extractive.greedy import select_greedy


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select sentences to form system summaries")
    parser.add_argument("--config", type=str, required=True, help="Path to features config YAML")
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="configs/dataset_cnn_dm.yaml",
        help="Path to dataset config YAML (to locate split and interim paths)",
    )
    parser.add_argument("--method", type=str, default="greedy", choices=["greedy"], help="Selection method")
    args = parser.parse_args()

    setup_logging()
    log = logging.getLogger("select_sentences")

    root = Path(__file__).resolve().parents[2]
    fcfg = load_yaml(root / args.config)
    dcfg = load_yaml(root / args.dataset_config)

    split = dcfg["dataset"].get("split", "sample")
    in_path = root / f"data/interim/{split}_with_feats.jsonl"
    out_path = root / f"data/interim/{split}_system.jsonl"
    ensure_dir(out_path.parent)

    sel = fcfg.get("features", {}).get("selection", {})
    max_tokens = int(sel.get("max_tokens", 100))
    redundancy_ngram = int(sel.get("redundancy_ngram", 3))
    redundancy_threshold = float(sel.get("redundancy_threshold", 0.6))

    records = list(read_jsonl(in_path))
    log.info("Selecting summaries for %d docs", len(records))
    out_items = []
    for r in records:
        sents = r.get("sentences", [])
        scores = [float(s.get("score", 0.0)) for s in sents]
        idxs, summary = select_greedy(
            sentences=sents,
            scores=scores,
            max_tokens=max_tokens,
            redundancy_ngram=redundancy_ngram,
            redundancy_threshold=redundancy_threshold,
        )
        out_items.append({
            "id": r["id"],
            "system": summary,
            "reference": r.get("reference", ""),
        })
    write_jsonl(out_path, out_items)
    log.info("Wrote system outputs to %s", out_path)


if __name__ == "__main__":
    main()
