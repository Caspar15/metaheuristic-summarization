from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from src.utils.io import ensure_dir, write_jsonl, timestamp


def write_report(
    items: List[Dict[str, Any]],
    out_dir: str = "runs",
) -> Dict[str, str]:
    """Write predictions.jsonl and metrics.csv under runs/{timestamp}/.

    items: list of {id, system, reference, metrics: {rouge1, rouge2, rougeL}}
    Returns written file paths.
    """
    run_dir = f"{out_dir}/{timestamp()}"
    ensure_dir(run_dir)

    # Predictions JSONL
    preds_path = f"{run_dir}/predictions.jsonl"
    write_jsonl(preds_path, (
        {"id": it["id"], "system": it["system"], "reference": it["reference"], "metrics": it.get("metrics", {})}
        for it in items
    ))

    # Metrics CSV (per doc + average row)
    rows = []
    for it in items:
        row = {"id": it["id"]}
        row.update(it.get("metrics", {}))
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        avg = {"id": "avg"}
        for c in df.columns:
            if c == "id":
                continue
            avg[c] = float(df[c].mean())
        df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)
    metrics_path = f"{run_dir}/metrics.csv"
    df.to_csv(metrics_path, index=False)

    return {"predictions": preds_path, "metrics": metrics_path}

