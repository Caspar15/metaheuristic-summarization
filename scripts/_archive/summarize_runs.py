import csv
import glob
import os
from typing import List, Dict


def read_csv(path: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append(row)
    return out


def main():
    rows: List[Dict[str, str]] = []
    for p in glob.glob(os.path.join("runs", "tune_summary_*.csv")):
        rows.extend(read_csv(p))
    if not rows:
        print("No tune_summary_*.csv found under runs/.")
        return
    # normalize keys
    fields = [
        "base_optimizer",
        "k1",
        "k2",
        "cap",
        "method",
        "w_bert",
        "alpha",
        "rouge1",
        "rouge2",
        "rougeL",
        "pred",
    ]
    for r in rows:
        for k in fields:
            r.setdefault(k, "")
    out_path = os.path.join("runs", "summary_all.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"Wrote {out_path} with {len(rows)} rows")


if __name__ == "__main__":
    main()

