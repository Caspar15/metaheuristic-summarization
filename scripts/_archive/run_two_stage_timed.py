import argparse
import csv
import os
import subprocess as sp
from datetime import datetime
import time


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    sp.run(cmd, check=True)


def read_time(path: str) -> float | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return float((f.read() or "0").strip())
    except Exception:
        return None


def read_metrics_csv(path: str) -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                if len(row) >= 2:
                    try:
                        out[row[0]] = float(row[1])
                    except Exception:
                        pass
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser(description="Run two-stage pipeline with per-stage timing summary")
    ap.add_argument("--input", required=True, help="processed jsonl path (preprocessed)")
    ap.add_argument("--run_dir", default="runs")
    # stage1 base
    ap.add_argument("--base_cfg", required=True, help="config for Stage1 Base (e.g., configs/stage1/base/k20.yaml)")
    ap.add_argument("--opt1", required=True, help="Stage1 Base optimizer (greedy|grasp|nsga2)")
    # stage1 llm
    ap.add_argument("--llm_cfg", required=True, help="config for Stage1 LLM (e.g., configs/stage1/llm/bert/k20.yaml)")
    ap.add_argument("--opt2", required=True, help="Stage1 LLM optimizer (bert|roberta|xlnet)")
    # union
    ap.add_argument("--cap", type=int, default=15)
    # stage2
    ap.add_argument("--stage2_cfg", required=True, help="config for Stage2 (e.g., configs/stage2/fast/3sent.yaml)")
    ap.add_argument("--opt3", required=True, help="Stage2 optimizer (fast|fast_grasp|fast_nsga2)")
    # misc
    ap.add_argument("--stamp_prefix", default=None)
    args = ap.parse_args()

    py = os.environ.get("PYTHON", os.sys.executable)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    prefix = args.stamp_prefix or f"timed-{ts}"

    # Stage1 Base
    stamp1 = f"{prefix}-stage1-{args.opt1}"
    out1_dir = os.path.join(args.run_dir, stamp1)
    run([py, "-m", "src.pipeline.select_sentences", "--config", args.base_cfg, "--split", "validation", "--input", args.input, "--run_dir", args.run_dir, "--optimizer", args.opt1, "--stamp", stamp1])
    t1 = read_time(os.path.join(out1_dir, "time_select_seconds.txt"))

    # Stage1 LLM
    stamp2 = f"{prefix}-stage1-{args.opt2}"
    out2_dir = os.path.join(args.run_dir, stamp2)
    run([py, "-m", "src.pipeline.select_sentences", "--config", args.llm_cfg, "--split", "validation", "--input", args.input, "--run_dir", args.run_dir, "--optimizer", args.opt2, "--stamp", stamp2])
    t2 = read_time(os.path.join(out2_dir, "time_select_seconds.txt"))

    # Union (time the union building step)
    union_out = os.path.join(os.path.dirname(args.input), f"{os.path.basename(args.input).replace('.jsonl','')}.stage2.union.jsonl")
    tU0 = time.perf_counter()
    run([py, "scripts/build_union_stage2.py", "--input", args.input, "--base_pred", os.path.join(out1_dir, "predictions.jsonl"), "--bert_pred", os.path.join(out2_dir, "predictions.jsonl"), "--out", union_out, "--cap", str(int(args.cap))])
    tU1 = time.perf_counter()
    tU = tU1 - tU0

    # Stage2
    stamp3 = f"{prefix}-stage2-{args.opt3}"
    out3_dir = os.path.join(args.run_dir, stamp3)
    run([py, "-m", "src.pipeline.select_sentences", "--config", args.stage2_cfg, "--split", "validation", "--input", union_out, "--run_dir", args.run_dir, "--optimizer", args.opt3, "--stamp", stamp3])
    t3 = read_time(os.path.join(out3_dir, "time_select_seconds.txt"))

    # Evaluate
    metrics_path = os.path.join(out3_dir, "metrics.csv")
    run([py, "-m", "src.pipeline.evaluate", "--pred", os.path.join(out3_dir, "predictions.jsonl"), "--out", metrics_path])
    m = read_metrics_csv(metrics_path)

    # Summary CSV
    os.makedirs(args.run_dir, exist_ok=True)
    summary = os.path.join(args.run_dir, f"timed_summary_{ts}.csv")
    fields = [
        "stage1_base", "stage1_llm", "stage2_method",
        "time_stage1_base", "time_stage1_llm", "time_union", "time_stage2", "time_eval",
        "rouge1", "rouge2", "rougeL",
        "stage1_base_run", "stage1_llm_run", "stage2_run", "union_out"
    ]
    with open(summary, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({
            "stage1_base": args.opt1,
            "stage1_llm": args.opt2,
            "stage2_method": args.opt3,
            "time_stage1_base": f"{t1 if t1 is not None else ''}",
            "time_stage1_llm": f"{t2 if t2 is not None else ''}",
            "time_union": f"{tU:.6f}",
            "time_stage2": f"{t3 if t3 is not None else ''}",
            "time_eval": m.get("time_eval_seconds", ""),
            "rouge1": m.get("rouge1", ""),
            "rouge2": m.get("rouge2", ""),
            "rougeL": m.get("rougeL", ""),
            "stage1_base_run": out1_dir,
            "stage1_llm_run": out2_dir,
            "stage2_run": out3_dir,
            "union_out": union_out,
        })
    print("Summary written:", summary)


if __name__ == "__main__":
    main()
