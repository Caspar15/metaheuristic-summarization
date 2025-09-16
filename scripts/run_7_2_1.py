import argparse
import os
import sys
import subprocess as sp
from datetime import datetime


def run(cmd):
    print("$", " ".join(cmd))
    sp.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/stage1/base/k20.yaml")
    ap.add_argument("--input_csv", default=None, help="single CSV to split (optional)")
    ap.add_argument("--raw_dir", default="data/raw", help="directory containing train/validation/test.csv or a subfolder cnn_dailymail/")
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--run_dir", default="runs")
    ap.add_argument("--optimizer", default=None, help="override optimizer.method")
    ap.add_argument("--seed", type=int, default=2024)
    # sampling controls passed to preprocess for each split
    ap.add_argument("--sample_n", type=int, default=None, help="randomly sample N rows per split")
    ap.add_argument("--sample_frac", type=float, default=None, help="randomly sample a fraction per split (0,1]")
    ap.add_argument("--limit", type=int, default=None, help="take first N rows after sampling")
    ap.add_argument("--max_sentences", type=int, default=None, help="cap sentences per document after filtering")
    args = ap.parse_args()

    py = sys.executable
    if args.input_csv:
        run([py, "scripts/split_dataset.py", "--input", args.input_csv, "--out_dir", args.raw_dir, "--seed", str(args.seed)])

    # Prepare stamp for all outputs
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    splits = ["train", "validation", "test"]
    # Preprocess
    for sp_name in splits:
        # resolve csv path with fallbacks
        candidates = [
            os.path.join(args.raw_dir, f"{sp_name}.csv"),
        ]
        if sp_name == "validation":
            candidates += [
                os.path.join(args.raw_dir, n) for n in ["validation.csv", "valid.csv", "val.csv", "dev.csv"]
            ]
        candidates += [
            os.path.join(args.raw_dir, "cnn_dailymail", f"{sp_name}.csv"),
        ]
        if sp_name == "validation":
            candidates += [
                os.path.join(args.raw_dir, "cnn_dailymail", n) for n in ["validation.csv", "valid.csv", "val.csv", "dev.csv"]
            ]
        in_csv = next((p for p in candidates if os.path.exists(p)), os.path.join(args.raw_dir, f"{sp_name}.csv"))
        out_jsonl = os.path.join(args.processed_dir, f"{sp_name}.jsonl")
        cmd = [py, "-m", "src.data.preprocess", "--input", in_csv, "--split", sp_name, "--out", out_jsonl, "--seed", str(args.seed)]
        if args.sample_n is not None:
            cmd += ["--sample_n", str(args.sample_n)]
        if args.sample_frac is not None:
            cmd += ["--sample_frac", str(args.sample_frac)]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        if args.max_sentences is not None:
            cmd += ["--max_sentences", str(args.max_sentences)]
        run(cmd)

    # Summarize + Evaluate
    for sp_name in splits:
        in_jsonl = os.path.join(args.processed_dir, f"{sp_name}.jsonl")
        out_stamp = f"{stamp}-{sp_name}"
        cmd = [py, "-m", "src.pipeline.select_sentences", "--config", args.config, "--split", sp_name, "--input", in_jsonl, "--run_dir", args.run_dir, "--stamp", out_stamp]
        if args.optimizer:
            cmd += ["--optimizer", args.optimizer]
        run(cmd)
        pred = os.path.join(args.run_dir, out_stamp, "predictions.jsonl")
        metrics = os.path.join(args.run_dir, out_stamp, "metrics.csv")
        run([py, "-m", "src.pipeline.evaluate", "--pred", pred, "--out", metrics])

    print("Done. See:")
    for sp_name in splits:
        out_stamp = f"{stamp}-{sp_name}"
        print(f"  {sp_name}: {os.path.join(args.run_dir, out_stamp)}")


if __name__ == "__main__":
    main()

