import argparse
import os
import sys
import csv
from datetime import datetime
import subprocess as sp


def run(cmd):
    print("$", " ".join(cmd))
    sp.run(cmd, check=True)


def read_metrics_csv(path):
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) >= 2:
                out[row[0]] = float(row[1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--train_csv", default=None, help="override path to train.csv")
    ap.add_argument("--valid_csv", default=None, help="override path to validation.csv (or dev/val)")
    ap.add_argument("--test_csv", default=None, help="override path to test.csv")
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--run_dir", default="runs")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--train_n", type=int, default=200)
    ap.add_argument("--valid_n", type=int, default=50)
    ap.add_argument("--test_n", type=int, default=50)
    ap.add_argument("--max_sentences", type=int, default=25)
    ap.add_argument("--optimizers", default="greedy,grasp,nsga2")
    args = ap.parse_args()

    py = sys.executable
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    opts = [o.strip() for o in args.optimizers.split(",") if o.strip()]
    splits = [
        ("train", args.train_n),
        ("validation", args.valid_n),
        ("test", args.test_n),
    ]

    # 1) Preprocess sampled subsets per split
    def pick_csv(split: str) -> str:
        if split == "train" and args.train_csv:
            return args.train_csv
        if split == "validation" and args.valid_csv:
            return args.valid_csv
        if split == "test" and args.test_csv:
            return args.test_csv
        # primary guess
        primary = os.path.join(args.raw_dir, f"{split}.csv")
        if os.path.exists(primary):
            return primary
        # aliases for validation
        if split == "validation":
            for name in ["validation.csv", "valid.csv", "val.csv", "dev.csv"]:
                p = os.path.join(args.raw_dir, name)
                if os.path.exists(p):
                    return p
        # try common nested folder name
        nested = os.path.join(args.raw_dir, "cnn_dailymail", f"{split}.csv")
        if os.path.exists(nested):
            return nested
        if split == "validation":
            for name in ["validation.csv", "valid.csv", "val.csv", "dev.csv"]:
                p = os.path.join(args.raw_dir, "cnn_dailymail", name)
                if os.path.exists(p):
                    return p
        return primary

    for split, n in splits:
        in_csv = pick_csv(split)
        if not os.path.exists(in_csv):
            raise FileNotFoundError(f"CSV not found for split '{split}': {in_csv}. Use --{split if split!='validation' else 'valid'}_csv to override, or check --raw_dir.")
        out_jsonl = os.path.join(args.processed_dir, f"{split}.jsonl")
        cmd = [
            py, "-m", "src.data.preprocess",
            "--input", in_csv,
            "--split", split,
            "--out", out_jsonl,
            "--seed", str(args.seed),
            "--sample_n", str(n),
            "--max_sentences", str(args.max_sentences),
        ]
        run(cmd)

    # Supervised training removed

    # 2) For each optimizer, run summarize + evaluate for each split
    summary_rows = []
    for opt in opts:
        for split, _ in splits:
            in_jsonl = os.path.join(args.processed_dir, f"{split}.jsonl")
            out_stamp = f"{stamp}-{opt}-{split}"
            cmd = [
                py, "-m", "src.pipeline.select_sentences",
                "--config", args.config,
                "--split", split,
                "--input", in_jsonl,
                "--run_dir", args.run_dir,
                "--stamp", out_stamp,
                "--optimizer", opt,
            ]
            # no supervised model
            run(cmd)
            pred = os.path.join(args.run_dir, out_stamp, "predictions.jsonl")
            metrics = os.path.join(args.run_dir, out_stamp, "metrics.csv")
            run([py, "-m", "src.pipeline.evaluate", "--pred", pred, "--out", metrics])
            m = read_metrics_csv(metrics)
            summary_rows.append({
                "optimizer": opt,
                "split": split,
                **m,
            })

    # 3) Write combined summary
    os.makedirs(args.run_dir, exist_ok=True)
    summary_path = os.path.join(args.run_dir, f"{stamp}-summary.csv")
    fields = ["optimizer", "split", "rouge1", "rouge2", "rougeL"]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in summary_rows:
            w.writerow({k: row.get(k, "") for k in fields})
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
