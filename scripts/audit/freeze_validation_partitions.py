"""Freeze reference-blind dev/dev-test membership within validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.partitions import freeze_validation_partition_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dev_fraction", type=float, default=0.70)
    args = parser.parse_args()

    manifest = freeze_validation_partition_manifest(
        Path(args.input),
        dataset=args.dataset,
        seed=args.seed,
        dev_fraction=args.dev_fraction,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(output)
    print(
        f"Wrote {output}: dev={manifest['partitions']['dev']['rows']}, "
        f"dev-test={manifest['partitions']['dev-test']['rows']}"
    )


if __name__ == "__main__":
    main()
