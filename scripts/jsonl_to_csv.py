import argparse
import csv
import json


def jsonl_to_csv(input_path: str, out_path: str):
    with open(input_path, "r", encoding="utf-8") as fr, open(out_path, "w", newline="", encoding="utf-8") as fw:
        w = csv.writer(fw)
        w.writerow(["id", "article", "highlights"])
        for line in fr:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            _id = obj.get("id")
            art = obj.get("article", "")
            ref = obj.get("reference") or obj.get("highlights") or ""
            w.writerow([_id, art, ref])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    jsonl_to_csv(args.input, args.out)
    print(f"Wrote CSV to {args.out}")


if __name__ == "__main__":
    main()

