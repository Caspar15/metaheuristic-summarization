import argparse
import json
from pathlib import Path

from src.data.preprocess import preprocess_csv
from src.pipeline.build_features import build_features
from src.selection.candidate_pool import top_k_candidates
from src.selection.length_controller import enforce_length
from src.models.extractive.greedy import greedy_select
from src.eval.rouge_eval import evaluate_rouge


def run(csv_path, config_path, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    processed = out / "processed.jsonl"
    preprocess_csv(csv_path, processed)

    cfg = (
        json.loads(Path(config_path).read_text(encoding="utf-8").replace("'", '"'))
        if config_path.endswith(".json")
        else _load_yaml(config_path)
    )

    items = build_features(processed, cfg)
    preds_path = out / "predictions.jsonl"
    with preds_path.open("w", encoding="utf-8") as fw:
        for item in items:
            cand = top_k_candidates(item["sentences"], item["scores"], cfg)
            chosen = greedy_select(cand, item, cfg)
            chosen = enforce_length(chosen, cfg)
            fw.write(
                json.dumps(
                    {"id": item["id"], "prediction": " ".join(chosen)}, ensure_ascii=False
                )
                + "\n"
            )

    evaluate_rouge(preds_path, out / "metrics.csv")


def _load_yaml(path):
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        import json

        text = Path(path).read_text(encoding="utf-8")
        return json.loads(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    run(**vars(ap.parse_args()))

