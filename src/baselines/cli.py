"""CLI entry-point for baseline runs.

Deliberately mirrors ``src.pipeline.select_sentences``'s CLI shape and reuses
its helpers directly (``read_jsonl``/``write_jsonl_atomic``/``now_stamp``/
``ensure_dir``, ``validate_experiment_request``/``validate_requested_split``,
``src.data.policy.validate_dataset_policy_request``) so a baseline run
produces the same run-directory layout and provenance artifacts
(``config_used.json``, ``dataset_preflight.json``, ``time_select_seconds.txt``)
as a system run, and can be scored by ``src.pipeline.evaluate`` unchanged.

``baseline_run.json`` is written alongside those, deliberately separate from
``config_used.json``: the latter is a verbatim dump of the config YAML, and
mixing CLI-only invocation flags into it would blur "the config that was
used" with "how this particular run was invoked". ``--baseline``,
``--ordering``, ``--first_k``, and ``--seed`` are CLI arguments, not config
fields -- without a dedicated file, two ``document_order``/``round_robin``
Lead runs against the same config, or two Random runs with different
``--seed``, would be indistinguishable from their run directories alone.

``--seed`` is unrelated to ``cfg.get("seed")`` (read via
``set_global_seed`` below): the latter is this project's existing global
determinism seed for whatever else consumes it (numpy/torch); ``--seed`` is
Random's own per-row seed, deliberately *not* folded into that global,
cumulative-state mechanism -- see
``src.baselines.contract.derive_row_seed`` for why a global seed cannot
give a single row independent reproducibility.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, Mapping, Optional

from tqdm import tqdm

from src.baselines.lead import ORDERINGS, summarize_one_lead
from src.baselines.random_baseline import summarize_one_random
from src.data.policy import validate_dataset_policy_request
from src.pipeline.select_sentences import (
    validate_experiment_request,
    validate_requested_split,
)
from src.utils.io import (
    ensure_dir,
    load_yaml,
    now_stamp,
    read_jsonl,
    set_global_seed,
    write_jsonl_atomic,
)

BASELINE_METHODS = {"lead": summarize_one_lead, "random": summarize_one_random}

# Baselines whose select_fn needs an explicit --seed to be reproducible.
SEEDED_BASELINES = {"random"}


def _validate_baseline_seed_pairing(baseline: str, seed: Optional[int]) -> None:
    """Fail loud in both directions rather than silently dropping a request.

    A seeded baseline run without --seed would produce a result that looks
    reproducible (it ran, it printed indices) but was never actually pinned
    to anything anyone chose -- the same failure shape as
    ``requires_seed`` in ``contract.py``. A deterministic baseline run
    *with* --seed silently ignoring it is the same "requested value
    disappears" problem this repo keeps hitting (``w_bert``,
    ``requested_min_words``, ``requested_max_*``): the user asked for
    something specific and the artifact would give no sign it was ignored.
    """

    if baseline in SEEDED_BASELINES and seed is None:
        raise ValueError(
            f"--baseline {baseline} requires --seed; a silently-defaulted "
            "seed would produce a result that looks reproducible but was "
            "never actually chosen by anyone"
        )
    if baseline not in SEEDED_BASELINES and seed is not None:
        raise ValueError(
            f"--baseline {baseline} is deterministic and does not accept "
            "--seed; silently ignoring it would let a user believe "
            f"{baseline} has randomness it does not have"
        )


def summarize_jsonl_baseline(
    input_path: str,
    predictions_path: str,
    cfg: Mapping,
    requested_split: str,
    *,
    baseline: str,
    ordering: str,
    first_k: int,
    seed: Optional[int] = None,
    dataset_preflight: Dict | None = None,
) -> int:
    """Stream one dataset into a baseline prediction artifact."""

    if baseline not in BASELINE_METHODS:
        raise ValueError(f"unknown baseline {baseline!r}; choose one of {sorted(BASELINE_METHODS)}")
    _validate_baseline_seed_pairing(baseline, seed)
    summarize_one = BASELINE_METHODS[baseline]

    if dataset_preflight is None:
        dataset_preflight = validate_dataset_policy_request(cfg, input_path, requested_split)

    processed = 0

    def prediction_rows():
        nonlocal processed
        for doc in tqdm(read_jsonl(input_path), desc=f"{baseline} baseline"):
            validate_requested_split(doc, requested_split)
            if baseline in SEEDED_BASELINES:
                result = summarize_one(doc, cfg, seed=seed)
            else:
                result = summarize_one(doc, cfg, ordering=ordering, first_k=first_k)
            processed += 1
            yield result
        if processed == 0:
            raise ValueError("input dataset is empty; refusing to write an empty run")

    write_jsonl_atomic(predictions_path, prediction_rows())
    return processed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="lead", choices=sorted(BASELINE_METHODS))
    ap.add_argument("--config", required=True, help="path to config yaml")
    ap.add_argument("--split", required=True, help="dataset split name")
    ap.add_argument("--input", required=True, help="processed jsonl path")
    ap.add_argument("--run_dir", default="runs", help="runs output root")
    ap.add_argument("--stamp", default=None, help="optional fixed stamp for output dir")
    ap.add_argument(
        "--ordering",
        default="document_order",
        choices=ORDERINGS,
        help="multi-document Lead ordering; see src/baselines/lead.py docstring",
    )
    ap.add_argument(
        "--first_k",
        type=int,
        default=3,
        help="sentences per source document for ordering=fabbri_first_k (diagnostic only)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "required for --baseline random (per-row seed is derived from "
            "this plus the document id, see src.baselines.contract."
            "derive_row_seed); must be omitted for --baseline lead, which "
            "is deterministic and does not accept a seed"
        ),
    )
    args = ap.parse_args()

    # Fail before touching the filesystem: validate the baseline/seed
    # pairing ahead of ensure_dir() so a rejected run never leaves behind an
    # empty run directory.
    _validate_baseline_seed_pairing(args.baseline, args.seed)

    cfg = load_yaml(args.config)

    validate_experiment_request(cfg, args.split)
    dataset_preflight = validate_dataset_policy_request(cfg, args.input, args.split)

    set_global_seed(cfg.get("seed"))
    stamp = args.stamp or now_stamp()
    out_dir = os.path.join(args.run_dir, stamp)
    ensure_dir(out_dir)

    preds_path = os.path.join(out_dir, "predictions.jsonl")
    t0 = time.perf_counter()
    summarize_jsonl_baseline(
        args.input,
        preds_path,
        cfg,
        args.split,
        baseline=args.baseline,
        ordering=args.ordering,
        first_k=args.first_k,
        seed=args.seed,
        dataset_preflight=dataset_preflight,
    )
    t1 = time.perf_counter()

    with open(os.path.join(out_dir, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    baseline_run = {
        "baseline": args.baseline,
        "ordering": args.ordering,
        "first_k": args.first_k,
        "seed": args.seed,
        "split": args.split,
        "input": args.input,
    }
    with open(os.path.join(out_dir, "baseline_run.json"), "w", encoding="utf-8") as f:
        json.dump(baseline_run, f, ensure_ascii=False, indent=2)
    if dataset_preflight is not None:
        with open(os.path.join(out_dir, "dataset_preflight.json"), "w", encoding="utf-8") as f:
            json.dump(dataset_preflight, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "time_select_seconds.txt"), "w", encoding="utf-8") as f:
        f.write(f"{t1 - t0:.6f}")
    print(f"Wrote predictions to {preds_path}")


if __name__ == "__main__":
    main()
