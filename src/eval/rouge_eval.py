import csv


def evaluate_rouge(pred_jsonl_path, out_csv_path):
    try:
        import json
        from pathlib import Path

        preds = [
            json.loads(l)
            for l in Path(pred_jsonl_path).read_text(encoding="utf-8").splitlines()
        ]
        with open(out_csv_path, "w", newline="", encoding="utf-8") as fw:
            w = csv.writer(fw)
            w.writerow(["id", "rouge1", "rouge2", "rougeL"])
            for p in preds:
                w.writerow([p.get("id"), "", "", ""])  # 留空，若安裝 rouge-score 可替換
    except Exception:
        with open(out_csv_path, "w", newline="", encoding="utf-8") as fw:
            csv.writer(fw).writerow(["id", "rouge1", "rouge2", "rougeL"])
