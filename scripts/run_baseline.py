import argparse
from src.cli import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    run(**vars(ap.parse_args()))

