import argparse
import os
import random
import pandas as pd


def split_csv(input_csv: str, out_dir: str, ratios=(0.7, 0.2, 0.1), seed: int = 2024):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    n = len(df)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_train = int(n * ratios[0])
    n_valid = int(n * ratios[1])
    train_idx = idx[:n_train]
    valid_idx = idx[n_train : n_train + n_valid]
    test_idx = idx[n_train + n_valid :]
    df.iloc[train_idx].to_csv(os.path.join(out_dir, "train.csv"), index=False)
    df.iloc[valid_idx].to_csv(os.path.join(out_dir, "validation.csv"), index=False)
    df.iloc[test_idx].to_csv(os.path.join(out_dir, "test.csv"), index=False)
    return {
        "train": os.path.join(out_dir, "train.csv"),
        "validation": os.path.join(out_dir, "validation.csv"),
        "test": os.path.join(out_dir, "test.csv"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="input CSV with id, article, highlights")
    ap.add_argument("--out_dir", default="data/raw", help="output directory for split CSVs")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--ratios", nargs=3, type=float, default=(0.7, 0.2, 0.1))
    args = ap.parse_args()
    paths = split_csv(args.input, args.out_dir, tuple(args.ratios), args.seed)
    print("Saved:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

