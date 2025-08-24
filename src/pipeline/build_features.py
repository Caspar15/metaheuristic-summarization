import argparse
import os
from typing import Dict, List

import numpy as np

from src.utils.io import load_yaml, ensure_dir, now_stamp, read_jsonl, write_jsonl
from src.features.tf_isf import sentence_tf_isf_scores
from src.features.length import length_scores
from src.features.position import position_scores
from src.features.compose import combine_scores
from src.representations.sent_vectors import SentenceVectors
from src.representations.similarity import cosine_similarity_matrix


def build_features_for_doc(doc: Dict, cfg: Dict) -> Dict:
    sentences: List[str] = doc.get("sentences", [])
    f_imp = sentence_tf_isf_scores(sentences)
    f_len = length_scores(sentences)
    f_pos = position_scores(sentences)
    feats = {"importance": f_imp, "length": f_len, "position": f_pos}
    weights = {
        "importance": float(cfg.get("objectives", {}).get("lambda_importance", 1.0)),
        "length": 0.3,
        "position": 0.3,
    }
    base = combine_scores(feats, weights)
    rep_cfg = cfg.get("representations", {})
    method = rep_cfg.get("method", "tfidf")
    vec = SentenceVectors(method=method)
    X = vec.fit_transform(sentences)
    sim = cosine_similarity_matrix(X)
    return {
        "id": doc.get("id"),
        "base_scores": base,
        "n_sentences": len(sentences),
        # 不直接序列化 sim 矩陣避免檔案過大
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--input", required=True, help="processed jsonl path")
    ap.add_argument("--out_dir", default="runs", help="output root")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    stamp = now_stamp()
    out_dir = os.path.join(args.out_dir, f"features-{stamp}")
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"{args.split}.jsonl")

    rows = []
    for doc in read_jsonl(args.input):
        rows.append(build_features_for_doc(doc, cfg))
    write_jsonl(out_path, rows)
    print(f"Wrote features summary to {out_path}")


if __name__ == "__main__":
    main()

