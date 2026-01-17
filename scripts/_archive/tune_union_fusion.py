import argparse
import csv
import json
import os
import subprocess as sp
from datetime import datetime
from typing import List, Dict, Any


def run(cmd: List[str]):
    print("$", " ".join(cmd), flush=True)
    sp.run(cmd, check=True)


def load_yaml(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(obj: Dict[str, Any], path: str):
    import yaml

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def ensure_stage1_base_cfg(base_cfg_path: str, out_cfg_path: str, k: int, optimizer: str = "greedy") -> str:
    cfg = load_yaml(base_cfg_path)
    cfg.setdefault("length_control", {})["unit"] = "sentences"
    cfg["length_control"]["max_sentences"] = int(k)
    cfg.setdefault("optimizer", {})["method"] = optimizer
    dump_yaml(cfg, out_cfg_path)
    return out_cfg_path


def ensure_stage1_bert_cfg(bert_cfg_path: str, out_cfg_path: str, k: int) -> str:
    cfg = load_yaml(bert_cfg_path)
    cfg.setdefault("length_control", {})["unit"] = "sentences"
    cfg["length_control"]["max_sentences"] = int(k)
    cfg.setdefault("optimizer", {})["method"] = "bert"
    dump_yaml(cfg, out_cfg_path)
    return out_cfg_path


def ensure_stage2_bert_cfg(bert3_cfg_path: str, out_cfg_path: str) -> str:
    cfg = load_yaml(bert3_cfg_path)
    cfg.setdefault("length_control", {})["unit"] = "sentences"
    cfg["length_control"]["max_sentences"] = 3
    cfg.setdefault("optimizer", {})["method"] = "bert"
    dump_yaml(cfg, out_cfg_path)
    return out_cfg_path


def ensure_stage2_fused_cfg(fused3_cfg_path: str, out_cfg_path: str, w_bert: float, alpha: float) -> str:
    cfg = load_yaml(fused3_cfg_path)
    cfg.setdefault("length_control", {})["unit"] = "sentences"
    cfg["length_control"]["max_sentences"] = 3
    cfg.setdefault("optimizer", {})["method"] = "fused"
    cfg.setdefault("fusion", {})
    cfg["fusion"]["w_bert"] = float(w_bert)
    cfg["fusion"]["w_base"] = float(1.0 - float(w_bert))
    cfg.setdefault("redundancy", {})["lambda"] = float(alpha)
    dump_yaml(cfg, out_cfg_path)
    return out_cfg_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="processed jsonl (preprocessed)")
    ap.add_argument("--run_dir", default="runs")
    ap.add_argument("--k1", nargs="*", type=int, default=[20])
    ap.add_argument("--k2", nargs="*", type=int, default=[20])
    ap.add_argument("--cap", nargs="*", type=int, default=[25])
    ap.add_argument(
        "--methods",
        nargs="*",
        default=["fast", "fast_grasp", "fast_nsga2"],
        help="final methods to try (fast|fast_grasp|fast_nsga2)",
    )
    ap.add_argument("--w_bert", nargs="*", type=float, default=[0.5])
    ap.add_argument("--alpha", nargs="*", type=float, default=[0.7])
    ap.add_argument("--optimizer1", default="greedy", help="stage1 optimizer for base route")
    args = ap.parse_args()

    py = os.environ.get("PYTHON", os.sys.executable)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg_dir = os.path.join("configs", "_generated", f"tune-{stamp}")
    os.makedirs(cfg_dir, exist_ok=True)

    # Prepare template paths
    base_cfg_template = "configs/stage1/base/k20.yaml"
    bert_cfg_template = "configs/stage1/llm/bert/k20.yaml"
    bert3_cfg_template = None
    fused3_cfg_template = None
    fast3_cfg_template = "configs/stage2/fast/3sent.yaml"

    def ensure_stage2_fast_cfg(fast3_cfg_path: str, out_cfg_path: str, w_sem: float, alpha: float) -> str:
        cfg = load_yaml(fast3_cfg_path)
        cfg.setdefault("length_control", {})["unit"] = "sentences"
        cfg["length_control"]["max_sentences"] = 3
        cfg.setdefault("optimizer", {})["method"] = "fast"
        cfg.setdefault("fusion", {})
        # reuse w_bert as semantic weight field for consistency in summary
        cfg["fusion"]["w_bert"] = float(w_sem)
        cfg["fusion"]["w_base"] = float(1.0 - float(w_sem))
        cfg.setdefault("redundancy", {})["lambda"] = float(alpha)
        dump_yaml(cfg, out_cfg_path)
        return out_cfg_path

    # Cache stage1 runs
    stage1_base_runs: Dict[int, str] = {}
    stage1_bert_runs: Dict[int, str] = {}

    summary_rows: List[Dict[str, Any]] = []

    for k1 in args.k1:
        # stage1 base run
        cfg1 = ensure_stage1_base_cfg(
            base_cfg_template, os.path.join(cfg_dir, f"base_k{k1}.yaml"), k1, args.optimizer1
        )
        run1_stamp = f"tune1-{args.optimizer1}-k{k1}-{stamp}"
        out1_dir = os.path.join(args.run_dir, run1_stamp)
        if not os.path.exists(os.path.join(out1_dir, "predictions.jsonl")):
            run([py, "-m", "src.pipeline.select_sentences", "--config", cfg1, "--split", "validation", "--input", args.input, "--run_dir", args.run_dir, "--optimizer", args.optimizer1, "--stamp", run1_stamp])
        stage1_base_runs[k1] = os.path.join(out1_dir, "predictions.jsonl")

    for k2 in args.k2:
        # stage1 bert run
        cfg2 = ensure_stage1_bert_cfg(
            bert_cfg_template, os.path.join(cfg_dir, f"bert_k{k2}.yaml"), k2
        )
        run2_stamp = f"tune1-bert-k{k2}-{stamp}"
        out2_dir = os.path.join(args.run_dir, run2_stamp)
        if not os.path.exists(os.path.join(out2_dir, "predictions.jsonl")):
            run([py, "-m", "src.pipeline.select_sentences", "--config", cfg2, "--split", "validation", "--input", args.input, "--run_dir", args.run_dir, "--optimizer", "bert", "--stamp", run2_stamp])
        stage1_bert_runs[k2] = os.path.join(out2_dir, "predictions.jsonl")

    # iterate caps and final methods
    for k1 in args.k1:
        for k2 in args.k2:
            for cap in args.cap:
                union_out = os.path.join(
                    os.path.dirname(args.input), f"validation.stage2.union.k1_{k1}.k2_{k2}.cap_{cap}.{stamp}.jsonl"
                )
                if not os.path.exists(union_out):
                    run([
                        py,
                        "scripts/build_union_stage2.py",
                        "--input",
                        args.input,
                        "--base_pred",
                        stage1_base_runs[k1],
                        "--bert_pred",
                        stage1_bert_runs[k2],
                        "--out",
                        union_out,
                        "--cap",
                        str(cap),
                    ])

                for method in args.methods:
                    if method not in ("fast", "fast_grasp", "fast_nsga2"):
                        print(f"Skipping unsupported Stage2 method: {method}")
                        continue
                    elif method in ("fast", "fast_grasp", "fast_nsga2"):
                        for wb in args.w_bert:
                            for alpha in args.alpha:
                                cfg3 = ensure_stage2_fast_cfg(
                                    fast3_cfg_template,
                                    os.path.join(cfg_dir, f"stage2_{method}_w{wb}_a{alpha}.yaml"),
                                    wb,
                                    alpha,
                                )
                                stamp3 = f"tune2-{method}-k1{k1}-k2{k2}-cap{cap}-wb{wb}-a{alpha}-{stamp}"
                                out3_dir = os.path.join(args.run_dir, stamp3)
                                run([py, "-m", "src.pipeline.select_sentences", "--config", cfg3, "--split", "validation", "--input", union_out, "--run_dir", args.run_dir, "--optimizer", method, "--stamp", stamp3])
                                metrics_path = os.path.join(out3_dir, "metrics.csv")
                                run([py, "-m", "src.pipeline.evaluate", "--pred", os.path.join(out3_dir, "predictions.jsonl"), "--out", metrics_path])
                                m = read_metrics_csv(metrics_path)
                                summary_rows.append({
                                    "base_optimizer": args.optimizer1,
                                    "k1": k1,
                                    "k2": k2,
                                    "cap": cap,
                                    "method": method,
                                    "w_bert": wb,
                                    "alpha": alpha,
                                    **m,
                                    "pred": os.path.join(out3_dir, "predictions.jsonl"),
                                })

    # write summary
    os.makedirs(args.run_dir, exist_ok=True)
    summary_path = os.path.join(args.run_dir, f"tune_summary_{stamp}.csv")
    fields = ["base_optimizer", "k1", "k2", "cap", "method", "w_bert", "alpha", "rouge1", "rouge2", "rougeL", "pred"]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in summary_rows:
            w.writerow({k: row.get(k, "") for k in fields})
    print(f"Summary written to {summary_path}")


def read_metrics_csv(path: str) -> Dict[str, float]:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) >= 2:
                out[row[0]] = float(row[1])
    return out


if __name__ == "__main__":
    main()
