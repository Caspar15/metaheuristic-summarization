import argparse
import os
import yaml


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def build_base_cfg(k: int) -> dict:
    return {
        "objectives": {
            "lambda_importance": 1.0,
            "lambda_coverage": 0.8,
            "lambda_redundancy": 0.7,
        },
        "features": {
            "weights": {"importance": 1.0, "length": 0.3, "position": 0.4}
        },
        "representations": {"use": True, "method": "tfidf", "cache": False},
        "candidates": {
            "use": True,
            "k": 25,
            "mode": "hard",
            "sources": ["score", "centrality"],
            "soft_boost": 1.05,
            "recall_target": None,
        },
        "redundancy": {"method": "mmr", "lambda": 0.7, "sim_metric": "cosine"},
        "length_control": {"unit": "sentences", "max_sentences": int(k)},
        # 默認給 greedy；建議以 CLI --optimizer 覆蓋
        "optimizer": {"method": "greedy"},
        "seed": 2024,
    }


def build_llm_cfg(k: int, model: str) -> dict:
    return {
        "representations": {"use": False},
        "candidates": {"use": False},
        "redundancy": {"method": "mmr", "lambda": 0.7, "sim_metric": "cosine"},
        "length_control": {"unit": "sentences", "max_sentences": int(k)},
        "optimizer": {"method": model},
        "seed": 2024,
        "bert": {"model_name": f"{model}-base-cased" if model == "xlnet" else f"{model}-base" if model == "roberta" else "bert-base-uncased"},
    }


def main():
    ap = argparse.ArgumentParser(description="Generate Stage1 configs into a structured folder")
    ap.add_argument("--type", required=True, choices=["base", "llm"], help="stage1 route type")
    ap.add_argument("--k", type=int, required=True, help="max sentences K")
    ap.add_argument("--model", default=None, help="llm model: bert|roberta|xlnet (required when --type llm)")
    ap.add_argument("--out_root", default="configs/stage1", help="output root directory")
    args = ap.parse_args()

    if args.type == "base":
        out_dir = os.path.join(args.out_root, "base")
        ensure_dir(out_dir)
        path = os.path.join(out_dir, f"k{args.k}.yaml")
        cfg = build_base_cfg(args.k)
    else:
        if not args.model or args.model not in {"bert", "roberta", "xlnet"}:
            raise SystemExit("--model must be one of: bert|roberta|xlnet for --type llm")
        out_dir = os.path.join(args.out_root, "llm", args.model)
        ensure_dir(out_dir)
        path = os.path.join(out_dir, f"k{args.k}.yaml")
        cfg = build_llm_cfg(args.k, args.model)

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print("Wrote:", path)


if __name__ == "__main__":
    main()

