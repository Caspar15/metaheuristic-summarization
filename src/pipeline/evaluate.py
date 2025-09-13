import argparse
import csv
import os
import time
from typing import List, Dict

from src.utils.io import read_jsonl
from src.eval.rouge import rouge_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="predictions.jsonl path")
    ap.add_argument("--out", required=True, help="metrics.csv output path")
    args = ap.parse_args()

    preds: List[str] = []
    refs: List[str] = []
    for row in read_jsonl(args.pred):
        preds.append(row.get("summary", ""))
        refs.append(row.get("reference", ""))

    t0 = time.perf_counter()
    m = rouge_scores(preds, refs)
    t1 = time.perf_counter()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        # write rouge metrics
        for k, v in m.items():
            w.writerow([k, f"{v:.6f}"])
        # append time statistics
        # selection time (if produced by select_sentences in the same directory)
        try:
            sel_time_file = os.path.join(os.path.dirname(args.out), "time_select_seconds.txt")
            if os.path.exists(sel_time_file):
                with open(sel_time_file, "r", encoding="utf-8") as fr:
                    val = float((fr.read() or "0").strip())
                    w.writerow(["time_select_seconds", f"{val:.6f}"])
        except Exception:
            pass
        # evaluation time
        w.writerow(["time_eval_seconds", f"{(t1 - t0):.6f}"])
    print(f"ROUGE written to {args.out}")


if __name__ == "__main__":
    main()

