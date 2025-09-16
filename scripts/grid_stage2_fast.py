import argparse
import csv
import os
import subprocess as sp
from datetime import datetime


def run(cmd: list[str]):
    print("$", " ".join(cmd), flush=True)
    sp.run(cmd, check=True)


def load_yaml(path: str):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(obj, path: str):
    import yaml
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def make_cfg(template: str, out_path: str, method: str, w_sem: float, alpha: float) -> str:
    cfg = load_yaml(template)
    cfg.setdefault("length_control", {})["unit"] = "sentences"
    cfg["length_control"]["max_sentences"] = 3
    cfg.setdefault("optimizer", {})["method"] = method
    cfg.setdefault("fusion", {})
    cfg["fusion"]["w_bert"] = float(w_sem)
    cfg["fusion"]["w_base"] = float(1.0 - float(w_sem))
    cfg.setdefault("redundancy", {})["lambda"] = float(alpha)
    dump_yaml(cfg, out_path)
    return out_path


def read_metrics_csv(path: str) -> dict[str, float]:
    out: dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) >= 2:
                try:
                    out[row[0]] = float(row[1])
                except Exception:
                    pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="processed jsonl (preprocessed subset, e.g., validation.1k.jsonl)")
    ap.add_argument("--base_pred", required=True, help="Stage1 base predictions.jsonl (K1)")
    ap.add_argument("--bert_pred", required=True, help="Stage1 bert predictions.jsonl (K2)")
    ap.add_argument("--run_dir", default="runs")
    ap.add_argument("--caps", nargs="*", type=int, default=[20, 25, 30])
    ap.add_argument("--w_sem", nargs="*", type=float, default=[0.6, 0.7, 0.8])
    ap.add_argument("--alpha", nargs="*", type=float, default=[0.6, 0.7, 0.8])
    ap.add_argument("--methods", nargs="*", default=["fast", "fast_grasp"], help="stage2 methods (fast,fast_grasp)")
    args = ap.parse_args()

    py = os.environ.get("PYTHON", os.sys.executable)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg_dir = os.path.join("configs", "_generated", f"tune-{stamp}")
    os.makedirs(cfg_dir, exist_ok=True)

    # Build union per cap
    unions: dict[int, str] = {}
    for cap in args.caps:
        out_union = os.path.join(
            os.path.dirname(args.input), f"validation.stage2.union.k1_20.k2_20.cap_{cap}.{stamp}.jsonl"
        )
        run([
            py,
            "scripts/build_union_stage2.py",
            "--input",
            args.input,
            "--base_pred",
            args.base_pred,
            "--bert_pred",
            args.bert_pred,
            "--out",
            out_union,
            "--cap",
            str(cap),
            "--dedup_threshold",
            "0.95",
        ])
        unions[cap] = out_union

    # Run stage2 grid
    rows = []
    template = "configs/stage2/fast/3sent.yaml"
    for method in args.methods:
        for cap, union_path in unions.items():
            for w in args.w_sem:
                for a in args.alpha:
                    cfg_path = os.path.join(cfg_dir, f"stage2_{method}_w{w}_a{a}.yaml")
                    make_cfg(template, cfg_path, method, w_sem=w, alpha=a)
                    stamp2 = f"fast2-{method}-k1_20-k2_20-cap{cap}-w{w}-a{a}-{stamp}"
                    out_dir = os.path.join(args.run_dir, stamp2)
                    run([
                        py,
                        "-m",
                        "src.pipeline.select_sentences",
                        "--config",
                        cfg_path,
                        "--split",
                        "validation",
                        "--input",
                        union_path,
                        "--run_dir",
                        args.run_dir,
                        "--optimizer",
                        method,
                        "--stamp",
                        stamp2,
                    ])
                    metrics_path = os.path.join(out_dir, "metrics.csv")
                    run([
                        py,
                        "-m",
                        "src.pipeline.evaluate",
                        "--pred",
                        os.path.join(out_dir, "predictions.jsonl"),
                        "--out",
                        metrics_path,
                    ])
                    m = read_metrics_csv(metrics_path)
                    rows.append({
                        "method": method,
                        "cap": cap,
                        "w_sem": w,
                        "alpha": a,
                        **m,
                        "pred": os.path.join(out_dir, "predictions.jsonl"),
                    })

    # Write summary
    os.makedirs(args.run_dir, exist_ok=True)
    summary_path = os.path.join(args.run_dir, f"tune_summary_stage2_fast_{stamp}.csv")
    fields = [
        "method",
        "cap",
        "w_sem",
        "alpha",
        "rouge1",
        "rouge2",
        "rougeL",
        "time_select_seconds",
        "time_eval_seconds",
        "pred",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
