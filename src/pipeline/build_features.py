from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import yaml

from src.utils.io import read_jsonl, write_jsonl, ensure_dir
from src.utils.logging import setup_logging
from src.features.feature_bank import build_features_for_doc


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-sentence features for processed data")
    parser.add_argument("--config", type=str, required=True, help="Path to features config YAML")
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="configs/dataset_cnn_dm.yaml",
        help="Path to dataset config YAML (to locate processed split)",
    )
    args = parser.parse_args()

    setup_logging()
    log = logging.getLogger("build_features")

    root = Path(__file__).resolve().parents[2]
    fcfg = load_yaml(root / args.config)
    dcfg = load_yaml(root / args.dataset_config)

    processed_path = root / dcfg["dataset"]["output_path"]
    split = dcfg["dataset"].get("split", "sample")
    out_path = root / f"data/interim/{split}_with_feats.jsonl"
    ensure_dir(out_path.parent)

    records = list(read_jsonl(processed_path))
    log.info("Computing features for %d docs", len(records))
    out_items = []
    for r in records:
        sents = r.get("sentences", [])
        feats, scores = build_features_for_doc(sents, fcfg)
        out_items.append({
            "id": r["id"],
            "sentences": [
                {
                    "text": sents[i]["text"],
                    "tokens": sents[i]["tokens"],
                    "features": feats[i]["features"],
                    "score": feats[i]["score"],
                }
                for i in range(len(sents))
            ],
            "reference": r.get("reference", ""),
        })
    write_jsonl(out_path, out_items)
    log.info("Wrote features to %s", out_path)


if __name__ == "__main__":
    main()
